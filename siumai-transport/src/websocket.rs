//! Owned WebSocket connection policy and bounded provider sessions.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use http::Method;
use http::header::HeaderMap;
use reqwest::Url;
use siumai_core::{CallOptions, Cancellation, Error, ErrorKind, ResponseDiagnostics};
use tokio::net::TcpStream;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::{CloseFrame, WebSocketConfig};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, client_async_tls_with_config};

use crate::auth::{AuthApplier, AuthContext, AuthRefresh, NoAuth, append_credential_query};
use crate::endpoint::{CredentialAudience, EndpointConfig, Resolver, SystemResolver};
use crate::framing::{WebSocketFrame, WebSocketFramer};
use crate::transport::{ResponseHeaders, effective_deadline, response_limit_error, run_controlled};
use crate::{
    EndpointError, EndpointPolicy, LocalNetworkGrant, RequestBuildError, RequestHeaders,
    TransportConfigError, TransportLimits,
};

const DEFAULT_SESSION_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const DEFAULT_IO_TIMEOUT: Duration = Duration::from_secs(5 * 60);

/// A WebSocket URL bound to the same host and IP policy as HTTP transport.
#[derive(Clone)]
pub struct WebSocketEndpoint {
    url: Url,
    audience: CredentialAudience,
    connector_endpoint: EndpointConfig,
}

impl WebSocketEndpoint {
    pub fn official(
        url: impl AsRef<str>,
        origin: crate::OfficialOrigin,
    ) -> Result<Self, EndpointError> {
        Self::new(url, EndpointPolicy::Official(origin))
    }

    pub fn public_custom(url: impl AsRef<str>) -> Result<Self, EndpointError> {
        Self::new(url, EndpointPolicy::PublicCustom)
    }

    pub fn local_explicit(url: impl AsRef<str>) -> Result<Self, EndpointError> {
        Self::new(
            url,
            EndpointPolicy::LocalExplicit(LocalNetworkGrant::Loopback),
        )
    }

    pub fn private_network_explicit(url: impl AsRef<str>) -> Result<Self, EndpointError> {
        Self::new(
            url,
            EndpointPolicy::LocalExplicit(LocalNetworkGrant::PrivateNetwork),
        )
    }

    pub fn link_local_explicit(url: impl AsRef<str>) -> Result<Self, EndpointError> {
        Self::new(
            url,
            EndpointPolicy::LocalExplicit(LocalNetworkGrant::LinkLocal),
        )
    }

    pub fn new(url: impl AsRef<str>, policy: EndpointPolicy) -> Result<Self, EndpointError> {
        let url = Url::parse(url.as_ref()).map_err(|_| EndpointError::InvalidUrl)?;
        match policy {
            EndpointPolicy::Official(_) | EndpointPolicy::PublicCustom if url.scheme() != "wss" => {
                return Err(EndpointError::SchemeNotAllowed);
            }
            EndpointPolicy::LocalExplicit(_) if !matches!(url.scheme(), "ws" | "wss") => {
                return Err(EndpointError::SchemeNotAllowed);
            }
            _ => {}
        }
        if !url.username().is_empty() || url.password().is_some() {
            return Err(EndpointError::UserInfoNotAllowed);
        }
        if url.fragment().is_some() {
            return Err(EndpointError::FragmentNotAllowed);
        }
        let audience = CredentialAudience::from_url(&url)?;
        let mut connector_url = url.clone();
        connector_url
            .set_scheme(if url.scheme() == "wss" {
                "https"
            } else {
                "http"
            })
            .map_err(|_| EndpointError::SchemeNotAllowed)?;
        let connector_endpoint = EndpointConfig::new(connector_url.as_str(), policy)?;
        Ok(Self {
            url,
            audience,
            connector_endpoint,
        })
    }

    pub fn audience(&self) -> &CredentialAudience {
        &self.audience
    }

    /// Explicit access for persistence and provenance. Connection setup is
    /// deliberately owned by [`WebSocketTransport`].
    pub fn expose_url(&self) -> &Url {
        &self.url
    }
}

impl fmt::Debug for WebSocketEndpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WebSocketEndpoint")
            .field("audience", &self.audience)
            .field("url", &"[REDACTED]")
            .finish()
    }
}

/// Builder for a provider-runtime-level WebSocket transport.
pub struct WebSocketTransportBuilder {
    endpoint: WebSocketEndpoint,
    resolver: Arc<dyn Resolver>,
    auth: Arc<dyn AuthApplier>,
    limits: TransportLimits,
    connect_timeout: Duration,
    session_timeout: Duration,
    io_timeout: Duration,
}

impl WebSocketTransportBuilder {
    pub fn new(endpoint: WebSocketEndpoint) -> Self {
        Self {
            endpoint,
            resolver: Arc::new(SystemResolver),
            auth: Arc::new(NoAuth),
            limits: TransportLimits::default(),
            connect_timeout: Duration::from_secs(10),
            session_timeout: DEFAULT_SESSION_TIMEOUT,
            io_timeout: DEFAULT_IO_TIMEOUT,
        }
    }

    pub fn with_resolver(mut self, resolver: Arc<dyn Resolver>) -> Self {
        self.resolver = resolver;
        self
    }

    pub fn with_auth(mut self, auth: Arc<dyn AuthApplier>) -> Self {
        self.auth = auth;
        self
    }

    pub fn with_limits(mut self, limits: TransportLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    pub fn with_session_timeout(mut self, timeout: Duration) -> Self {
        self.session_timeout = timeout;
        self
    }

    pub fn with_io_timeout(mut self, timeout: Duration) -> Self {
        self.io_timeout = timeout;
        self
    }

    pub fn build(self) -> Result<WebSocketTransport, TransportConfigError> {
        self.limits.validate()?;
        for (name, timeout) in [
            ("connect_timeout", self.connect_timeout),
            ("session_timeout", self.session_timeout),
            ("io_timeout", self.io_timeout),
        ] {
            if timeout.is_zero() {
                return Err(TransportConfigError::ZeroTimeout { name });
            }
            if Instant::now().checked_add(timeout).is_none() {
                return Err(TransportConfigError::TimeoutTooLarge { name });
            }
        }
        let admission_capacity = self
            .limits
            .max_in_flight_requests
            .checked_add(self.limits.max_queued_requests)
            .ok_or(TransportConfigError::CapacityOverflow)?;
        Ok(WebSocketTransport {
            inner: Arc::new(WebSocketTransportInner {
                endpoint: self.endpoint,
                resolver: self.resolver,
                auth: self.auth,
                admission: Arc::new(Semaphore::new(admission_capacity)),
                in_flight: Arc::new(Semaphore::new(self.limits.max_in_flight_requests)),
                limits: self.limits,
                connect_timeout: self.connect_timeout,
                session_timeout: self.session_timeout,
                io_timeout: self.io_timeout,
            }),
        })
    }
}

/// Cloneable transport that owns DNS validation, TCP/TLS, SNI, and handshake limits.
#[derive(Clone)]
pub struct WebSocketTransport {
    inner: Arc<WebSocketTransportInner>,
}

impl WebSocketTransport {
    pub fn builder(endpoint: WebSocketEndpoint) -> WebSocketTransportBuilder {
        WebSocketTransportBuilder::new(endpoint)
    }

    pub fn endpoint(&self) -> &WebSocketEndpoint {
        &self.inner.endpoint
    }

    pub fn limits(&self) -> &TransportLimits {
        &self.inner.limits
    }

    /// Open exactly one bounded session. TCP candidates may be tried before
    /// the handshake, but a failed handshake is never replayed automatically.
    pub async fn connect(
        &self,
        headers: RequestHeaders,
        options: CallOptions,
    ) -> Result<WebSocketConnection, Error> {
        headers
            .validate(&self.inner.limits)
            .map_err(websocket_request_error)?;
        let cancellation = options.cancellation().child();
        let session_deadline = effective_deadline(options.deadline(), self.inner.session_timeout);
        let connect_deadline = effective_deadline(session_deadline, self.inner.connect_timeout);
        let permits = self.acquire(&cancellation, connect_deadline).await?;

        let mut url = self.inner.endpoint.url.clone();
        let method = Method::GET;
        let body = Bytes::new();
        let mut request_headers = headers.clone_inner();
        let patch = run_controlled(
            self.inner.auth.apply(
                AuthContext::new(
                    self.inner.endpoint.audience(),
                    &method,
                    &url,
                    &request_headers,
                    &body,
                ),
                AuthRefresh::Current,
            ),
            &cancellation,
            connect_deadline,
        )
        .await??;
        let (credential_headers, credential_query, _credential_revision) = patch.into_parts();
        for (name, value) in credential_headers {
            let Some(name) = name else {
                return Err(websocket_request_error(RequestBuildError::ProtectedHeader));
            };
            if request_headers.insert(name, value).is_some() {
                return Err(websocket_request_error(RequestBuildError::ProtectedHeader));
            }
        }
        if request_headers.len() > self.inner.limits.max_header_count
            || request_headers
                .values()
                .any(|value| value.as_bytes().len() > self.inner.limits.max_header_value_bytes)
        {
            return Err(websocket_request_error(RequestBuildError::TooManyHeaders));
        }
        append_credential_query(&mut url, credential_query).map_err(websocket_request_error)?;
        if !self.inner.endpoint.audience.matches(&url) {
            return Err(websocket_endpoint_error(EndpointError::AudienceMismatch));
        }

        let addresses = run_controlled(
            self.inner
                .endpoint
                .connector_endpoint
                .validated_addresses(self.inner.resolver.as_ref()),
            &cancellation,
            connect_deadline,
        )
        .await?
        .map_err(websocket_endpoint_error)?;
        let socket = self
            .connect_tcp(&addresses, &cancellation, connect_deadline)
            .await?;
        let remote = socket
            .peer_addr()
            .map_err(|error| websocket_connect_error(Some(error)))?;
        self.inner
            .endpoint
            .connector_endpoint
            .validate_remote(remote)
            .map_err(websocket_endpoint_error)?;
        socket
            .set_nodelay(true)
            .map_err(|error| websocket_connect_error(Some(error)))?;

        let mut request = url
            .as_str()
            .into_client_request()
            .map_err(|_| websocket_request_error(RequestBuildError::InvalidTarget))?;
        merge_handshake_headers(request.headers_mut(), request_headers)?;
        validate_handshake_headers(request.headers(), &self.inner.limits)?;
        let config = WebSocketConfig::default()
            .read_buffer_size(self.inner.limits.max_frame_bytes.min(64 * 1024))
            .write_buffer_size(0)
            .max_write_buffer_size(self.inner.limits.max_frame_bytes.saturating_add(14))
            .max_message_size(Some(self.inner.limits.max_frame_bytes))
            .max_frame_size(Some(self.inner.limits.max_frame_bytes));
        let (stream, response) = run_controlled(
            client_async_tls_with_config(request, socket, Some(config), None),
            &cancellation,
            connect_deadline,
        )
        .await?
        .map_err(websocket_handshake_error)?;
        ResponseHeaders::validate(response.headers(), &self.inner.limits).map_err(|detail| {
            response_limit_error(response.status(), response.headers(), Vec::new(), detail)
        })?;

        let (sender, receiver) = stream.split();
        let state = Arc::new(WebSocketSessionState {
            cancellation,
            deadline: session_deadline,
            io_timeout: self.inner.io_timeout,
            permits: Mutex::new(Some(permits)),
            terminated: AtomicBool::new(false),
        });

        Ok(WebSocketConnection {
            sender: Some(WebSocketSender {
                sink: Some(sender),
                state: state.clone(),
                maximum_outgoing_bytes: self.inner.limits.max_frame_bytes,
            }),
            receiver: Some(WebSocketReceiver {
                stream: Some(receiver),
                framer: WebSocketFramer::new(&self.inner.limits),
                state,
            }),
        })
    }

    async fn acquire(
        &self,
        cancellation: &Cancellation,
        deadline: Option<Instant>,
    ) -> Result<WebSocketPermits, Error> {
        let admission = self
            .inner
            .admission
            .clone()
            .try_acquire_owned()
            .map_err(|_| Error::new(ErrorKind::Transport, "WebSocket queue is full"))?;
        let in_flight = run_controlled(
            self.inner.in_flight.clone().acquire_owned(),
            cancellation,
            deadline,
        )
        .await?
        .map_err(|_| Error::new(ErrorKind::Internal, "WebSocket admission control is closed"))?;
        Ok(WebSocketPermits {
            _admission: admission,
            _in_flight: in_flight,
        })
    }

    async fn connect_tcp(
        &self,
        addresses: &[std::net::SocketAddr],
        cancellation: &Cancellation,
        deadline: Option<Instant>,
    ) -> Result<TcpStream, Error> {
        let mut last_error = None;
        for address in addresses {
            match run_controlled(TcpStream::connect(address), cancellation, deadline).await? {
                Ok(socket) => return Ok(socket),
                Err(error) => last_error = Some(error),
            }
        }
        Err(websocket_connect_error(last_error))
    }
}

impl fmt::Debug for WebSocketTransport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WebSocketTransport")
            .field("endpoint", &self.inner.endpoint)
            .field("limits", &self.inner.limits)
            .field("connect_timeout", &self.inner.connect_timeout)
            .field("session_timeout", &self.inner.session_timeout)
            .field("io_timeout", &self.inner.io_timeout)
            .finish()
    }
}

struct WebSocketTransportInner {
    endpoint: WebSocketEndpoint,
    resolver: Arc<dyn Resolver>,
    auth: Arc<dyn AuthApplier>,
    admission: Arc<Semaphore>,
    in_flight: Arc<Semaphore>,
    limits: TransportLimits,
    connect_timeout: Duration,
    session_timeout: Duration,
    io_timeout: Duration,
}

type RawWebSocket = WebSocketStream<MaybeTlsStream<TcpStream>>;
type RawWebSocketSink = SplitSink<RawWebSocket, Message>;
type RawWebSocketStream = SplitStream<RawWebSocket>;

/// An established bounded WebSocket session with drop cancellation.
pub struct WebSocketConnection {
    sender: Option<WebSocketSender>,
    receiver: Option<WebSocketReceiver>,
}

impl WebSocketConnection {
    pub async fn send(&mut self, frame: WebSocketFrame) -> Result<(), Error> {
        self.sender
            .as_mut()
            .ok_or_else(websocket_closed_error)?
            .send(frame)
            .await
    }

    pub async fn next(&mut self) -> Result<Option<WebSocketFrame>, Error> {
        self.receiver
            .as_mut()
            .ok_or_else(websocket_closed_error)?
            .next()
            .await
    }

    pub async fn close(&mut self) -> Result<(), Error> {
        match self.sender.as_mut() {
            Some(sender) => sender.close().await,
            None => Ok(()),
        }
    }

    /// Split the session into concurrently usable sending and receiving halves.
    /// Dropping either half terminates the shared session.
    pub fn split(mut self) -> (WebSocketSender, WebSocketReceiver) {
        let sender = self.sender.take().expect("a connection owns one sender");
        let receiver = self
            .receiver
            .take()
            .expect("a connection owns one receiver");
        (sender, receiver)
    }
}

impl fmt::Debug for WebSocketConnection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let open = self
            .sender
            .as_ref()
            .is_some_and(|sender| sender.state.is_open());
        formatter
            .debug_struct("WebSocketConnection")
            .field("open", &open)
            .finish()
    }
}

/// Concurrent sending half of a bounded WebSocket session.
pub struct WebSocketSender {
    sink: Option<RawWebSocketSink>,
    state: Arc<WebSocketSessionState>,
    maximum_outgoing_bytes: usize,
}

impl WebSocketSender {
    pub async fn send(&mut self, frame: WebSocketFrame) -> Result<(), Error> {
        if !self.state.is_open() {
            self.sink = None;
            return Err(websocket_closed_error());
        }
        let closes_session = matches!(frame, WebSocketFrame::Close { .. });
        let message = encode_frame(frame);
        if message.len() > self.maximum_outgoing_bytes {
            self.terminate();
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "outgoing WebSocket message exceeds the configured frame limit",
            ));
        }
        let cancellation = self.state.cancellation.clone();
        let deadline = self.state.io_deadline();
        let Some(sink) = self.sink.as_mut() else {
            self.state.terminate();
            return Err(websocket_closed_error());
        };
        let result = run_controlled(sink.send(message), &cancellation, deadline).await;
        match result {
            Ok(Ok(())) => {
                if closes_session {
                    self.terminate();
                }
                Ok(())
            }
            Ok(Err(error)) => {
                self.terminate();
                Err(websocket_io_error(error))
            }
            Err(error) => {
                self.terminate();
                Err(error)
            }
        }
    }

    pub async fn close(&mut self) -> Result<(), Error> {
        if !self.state.is_open() {
            self.sink = None;
            return Ok(());
        }
        let cancellation = self.state.cancellation.clone();
        let deadline = self.state.io_deadline();
        let Some(mut sink) = self.sink.take() else {
            self.state.terminate();
            return Ok(());
        };
        let result = run_controlled(sink.close(), &cancellation, deadline).await;
        self.state.terminate();
        match result {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => Err(websocket_io_error(error)),
            Err(error) => Err(error),
        }
    }

    fn terminate(&mut self) {
        self.sink = None;
        self.state.terminate();
    }
}

impl fmt::Debug for WebSocketSender {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WebSocketSender")
            .field("open", &self.state.is_open())
            .field("maximum_outgoing_bytes", &self.maximum_outgoing_bytes)
            .finish()
    }
}

impl Drop for WebSocketSender {
    fn drop(&mut self) {
        self.terminate();
    }
}

/// Concurrent receiving half of a bounded WebSocket session.
pub struct WebSocketReceiver {
    stream: Option<RawWebSocketStream>,
    framer: WebSocketFramer,
    state: Arc<WebSocketSessionState>,
}

impl WebSocketReceiver {
    pub async fn next(&mut self) -> Result<Option<WebSocketFrame>, Error> {
        if !self.state.is_open() {
            self.stream = None;
            return Ok(None);
        }
        let cancellation = self.state.cancellation.clone();
        let deadline = self.state.io_deadline();
        let Some(stream) = self.stream.as_mut() else {
            self.state.terminate();
            return Ok(None);
        };
        let message = match run_controlled(stream.next(), &cancellation, deadline).await {
            Ok(message) => message,
            Err(error) => {
                self.terminate();
                return Err(error);
            }
        };
        match message {
            Some(Ok(message)) => {
                let frame = match self.framer.decode(message) {
                    Ok(frame) => frame,
                    Err(error) => {
                        self.terminate();
                        return Err(Error::new(
                            ErrorKind::ResponseLimit,
                            "incoming WebSocket message exceeded a transport limit",
                        )
                        .with_source(error));
                    }
                };
                if matches!(frame, WebSocketFrame::Close { .. }) {
                    self.terminate();
                }
                Ok(Some(frame))
            }
            Some(Err(error)) => {
                self.terminate();
                Err(websocket_io_error(error))
            }
            None => {
                self.terminate();
                Ok(None)
            }
        }
    }

    fn terminate(&mut self) {
        self.stream = None;
        self.state.terminate();
    }
}

impl fmt::Debug for WebSocketReceiver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WebSocketReceiver")
            .field("open", &self.state.is_open())
            .field("framer", &self.framer)
            .finish()
    }
}

impl Drop for WebSocketReceiver {
    fn drop(&mut self) {
        self.terminate();
    }
}

struct WebSocketSessionState {
    cancellation: Cancellation,
    deadline: Option<Instant>,
    io_timeout: Duration,
    permits: Mutex<Option<WebSocketPermits>>,
    terminated: AtomicBool,
}

impl WebSocketSessionState {
    fn is_open(&self) -> bool {
        !self.terminated.load(Ordering::Acquire)
    }

    fn io_deadline(&self) -> Option<Instant> {
        effective_deadline(self.deadline, self.io_timeout)
    }

    fn terminate(&self) {
        if self.terminated.swap(true, Ordering::AcqRel) {
            return;
        }
        self.cancellation.cancel();
        let mut permits = self
            .permits
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        permits.take();
    }
}

struct WebSocketPermits {
    _admission: OwnedSemaphorePermit,
    _in_flight: OwnedSemaphorePermit,
}

fn merge_handshake_headers(destination: &mut HeaderMap, source: HeaderMap) -> Result<(), Error> {
    for (name, value) in source {
        let Some(name) = name else {
            return Err(websocket_request_error(RequestBuildError::ProtectedHeader));
        };
        if destination.contains_key(&name) {
            return Err(websocket_request_error(RequestBuildError::ProtectedHeader));
        }
        destination.insert(name, value);
    }
    Ok(())
}

fn validate_handshake_headers(headers: &HeaderMap, limits: &TransportLimits) -> Result<(), Error> {
    if headers.len() > limits.max_header_count {
        return Err(websocket_request_error(RequestBuildError::TooManyHeaders));
    }
    if headers
        .values()
        .any(|value| value.as_bytes().len() > limits.max_header_value_bytes)
    {
        return Err(websocket_request_error(
            RequestBuildError::HeaderValueTooLarge,
        ));
    }
    Ok(())
}

fn encode_frame(frame: WebSocketFrame) -> Message {
    match frame {
        WebSocketFrame::Text(text) => Message::text(text),
        WebSocketFrame::Binary(bytes) => Message::Binary(bytes),
        WebSocketFrame::Ping(bytes) => Message::Ping(bytes),
        WebSocketFrame::Pong(bytes) => Message::Pong(bytes),
        WebSocketFrame::Close { code, reason } => Message::Close(Some(CloseFrame {
            code: code
                .map(tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode::from)
                .unwrap_or(
                    tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode::Normal,
                ),
            reason: reason.into(),
        })),
    }
}

fn websocket_request_error(error: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "WebSocket handshake request is invalid",
    )
    .with_source(error)
}

fn websocket_endpoint_error(error: EndpointError) -> Error {
    Error::new(
        ErrorKind::Configuration,
        "WebSocket endpoint failed security validation",
    )
    .with_source(error)
}

fn websocket_connect_error(source: Option<std::io::Error>) -> Error {
    let error = Error::new(ErrorKind::Transport, "WebSocket TCP connection failed");
    match source {
        Some(source) => error.with_source(source),
        None => error,
    }
}

fn websocket_handshake_error(source: tokio_tungstenite::tungstenite::Error) -> Error {
    let mut error = Error::new(ErrorKind::Transport, "WebSocket handshake failed");
    if let tokio_tungstenite::tungstenite::Error::Http(response) = &source {
        error = error.with_diagnostics(
            ResponseDiagnostics::default().with_status(response.status().as_u16()),
        );
    }
    error.with_source(source)
}

fn websocket_io_error(source: tokio_tungstenite::tungstenite::Error) -> Error {
    Error::new(ErrorKind::Transport, "WebSocket session I/O failed").with_source(source)
}

fn websocket_closed_error() -> Error {
    Error::new(ErrorKind::InvalidInput, "WebSocket session is closed")
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use http::header::{AUTHORIZATION, HeaderValue};
    use tokio_tungstenite::accept_hdr_async;
    use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};

    use super::*;

    #[derive(Debug)]
    struct SecretAuth;

    #[async_trait]
    impl AuthApplier for SecretAuth {
        async fn apply(
            &self,
            _context: AuthContext<'_>,
            _refresh: AuthRefresh,
        ) -> Result<crate::CredentialPatch, Error> {
            crate::CredentialPatch::new()
                .try_insert(
                    AUTHORIZATION,
                    HeaderValue::from_static("Bearer canary-handshake-secret"),
                )
                .map_err(websocket_request_error)
        }
    }

    #[derive(Debug)]
    struct HangingAuth;

    #[async_trait]
    impl AuthApplier for HangingAuth {
        async fn apply(
            &self,
            _context: AuthContext<'_>,
            _refresh: AuthRefresh,
        ) -> Result<crate::CredentialPatch, Error> {
            std::future::pending().await
        }
    }

    #[test]
    fn public_websocket_requires_wss_and_redacts_query() {
        assert_eq!(
            WebSocketEndpoint::new("ws://example.com/live", EndpointPolicy::PublicCustom)
                .unwrap_err(),
            EndpointError::SchemeNotAllowed
        );
        let endpoint = WebSocketEndpoint::new(
            "wss://example.com/live?token=canary-secret",
            EndpointPolicy::PublicCustom,
        )
        .unwrap();
        assert!(!format!("{endpoint:?}").contains("canary-secret"));
    }

    #[test]
    fn websocket_and_http_audiences_are_not_interchangeable() {
        let websocket =
            WebSocketEndpoint::new("wss://example.com/live", EndpointPolicy::PublicCustom).unwrap();
        let http = Url::parse("https://example.com/live").unwrap();
        assert!(!websocket.audience().matches(&http));
    }

    #[test]
    fn local_websocket_still_rejects_userinfo_and_fragments() {
        assert_eq!(
            WebSocketEndpoint::new(
                "ws://user:secret@localhost:8080/live",
                EndpointPolicy::LocalExplicit(LocalNetworkGrant::Loopback),
            )
            .unwrap_err(),
            EndpointError::UserInfoNotAllowed
        );
        assert_eq!(
            WebSocketEndpoint::new(
                "ws://localhost:8080/live#fragment",
                EndpointPolicy::LocalExplicit(LocalNetworkGrant::Loopback),
            )
            .unwrap_err(),
            EndpointError::FragmentNotAllowed
        );
        assert_eq!(
            WebSocketEndpoint::local_explicit("ws://10.0.0.7:8080/live").unwrap_err(),
            EndpointError::AddressNotAllowed
        );
        WebSocketEndpoint::private_network_explicit("ws://10.0.0.7:8080/live").unwrap();
    }

    #[tokio::test]
    #[allow(clippy::result_large_err)] // The tungstenite callback owns this result shape.
    async fn owned_connector_authenticates_and_bounds_a_real_session() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut stream = accept_hdr_async(socket, |request: &Request, response: Response| {
                assert_eq!(
                    request.headers().get(AUTHORIZATION).unwrap(),
                    "Bearer canary-handshake-secret"
                );
                Ok(response)
            })
            .await
            .unwrap();
            stream
                .send(Message::text("canary-frame-payload"))
                .await
                .unwrap();
            let _ = stream.next().await;
        });

        let endpoint = WebSocketEndpoint::local_explicit(format!("ws://{address}/events"))
            .expect("loopback endpoint is explicitly authorized");
        let transport = WebSocketTransport::builder(endpoint)
            .with_auth(Arc::new(SecretAuth))
            .build()
            .unwrap();
        let mut connection = transport
            .connect(RequestHeaders::new(), CallOptions::default())
            .await
            .unwrap();
        let frame = connection.next().await.unwrap().unwrap();
        assert!(matches!(&frame, WebSocketFrame::Text(text) if text == "canary-frame-payload"));
        assert!(!format!("{frame:?}").contains("canary-frame-payload"));
        connection.close().await.unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn split_halves_send_and_receive_concurrently_and_peer_close_is_terminal() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut stream = tokio_tungstenite::accept_async(socket).await.unwrap();
            stream.send(Message::text("server-ready")).await.unwrap();
            let message = stream.next().await.unwrap().unwrap();
            assert_eq!(message.into_text().unwrap(), "client-ready");
            stream
                .send(Message::Close(Some(CloseFrame {
                    code:
                        tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode::Normal,
                    reason: "done".into(),
                })))
                .await
                .unwrap();
        });
        let transport = WebSocketTransport::builder(
            WebSocketEndpoint::local_explicit(format!("ws://{address}/events")).unwrap(),
        )
        .build()
        .unwrap();
        let connection = transport
            .connect(RequestHeaders::new(), CallOptions::default())
            .await
            .unwrap();
        let (mut sender, mut receiver) = connection.split();

        let (sent, received) = tokio::join!(
            sender.send(WebSocketFrame::Text("client-ready".to_owned())),
            receiver.next()
        );
        sent.unwrap();
        assert!(matches!(
            received.unwrap(),
            Some(WebSocketFrame::Text(text)) if text == "server-ready"
        ));
        assert!(matches!(
            receiver.next().await.unwrap(),
            Some(WebSocketFrame::Close {
                code: Some(1000),
                ..
            })
        ));
        assert_eq!(
            sender
                .send(WebSocketFrame::Text("after-close".to_owned()))
                .await
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidInput
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn peer_close_releases_the_shared_session_permit_before_halves_drop() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            for index in 0..2 {
                let (socket, _) = listener.accept().await.unwrap();
                let mut stream = tokio_tungstenite::accept_async(socket).await.unwrap();
                if index == 0 {
                    stream.send(Message::Close(None)).await.unwrap();
                } else {
                    stream.send(Message::text("second-session")).await.unwrap();
                }
            }
        });
        let limits = TransportLimits {
            max_connections: 1,
            max_in_flight_requests: 1,
            max_queued_requests: 1,
            ..TransportLimits::default()
        };
        let transport = WebSocketTransport::builder(
            WebSocketEndpoint::local_explicit(format!("ws://{address}/events")).unwrap(),
        )
        .with_limits(limits)
        .build()
        .unwrap();
        let first = transport
            .connect(RequestHeaders::new(), CallOptions::default())
            .await
            .unwrap();
        let (_sender, mut receiver) = first.split();
        assert!(matches!(
            receiver.next().await.unwrap(),
            Some(WebSocketFrame::Close { .. })
        ));

        let mut second = tokio::time::timeout(
            Duration::from_secs(1),
            transport.connect(RequestHeaders::new(), CallOptions::default()),
        )
        .await
        .expect("peer close must release the session permit")
        .unwrap();
        assert!(matches!(
            second.next().await.unwrap(),
            Some(WebSocketFrame::Text(text)) if text == "second-session"
        ));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn outgoing_frame_limit_terminates_both_halves() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut stream = tokio_tungstenite::accept_async(socket).await.unwrap();
            let _ = stream.next().await;
        });
        let transport = WebSocketTransport::builder(
            WebSocketEndpoint::local_explicit(format!("ws://{address}/events")).unwrap(),
        )
        .with_limits(TransportLimits {
            max_frame_bytes: 4,
            ..TransportLimits::default()
        })
        .build()
        .unwrap();
        let connection = transport
            .connect(RequestHeaders::new(), CallOptions::default())
            .await
            .unwrap();
        let (mut sender, mut receiver) = connection.split();
        let error = sender
            .send(WebSocketFrame::Text("12345".to_owned()))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        assert!(receiver.next().await.unwrap().is_none());
        server.await.unwrap();
    }

    #[tokio::test]
    async fn connect_timeout_covers_authentication_work() {
        let endpoint = WebSocketEndpoint::local_explicit("ws://127.0.0.1:9/events").unwrap();
        let transport = WebSocketTransport::builder(endpoint)
            .with_auth(Arc::new(HangingAuth))
            .with_connect_timeout(Duration::from_millis(30))
            .build()
            .unwrap();
        let error = tokio::time::timeout(
            Duration::from_secs(1),
            transport.connect(RequestHeaders::new(), CallOptions::default()),
        )
        .await
        .expect("connect timeout must cover auth")
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Timeout);
    }

    #[tokio::test]
    #[allow(clippy::result_large_err)] // The tungstenite callback owns this result shape.
    async fn oversized_upgrade_response_headers_are_rejected() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let _ = accept_hdr_async(socket, |_request: &Request, mut response: Response| {
                response.headers_mut().insert(
                    http::header::HeaderName::from_static("x-oversized"),
                    HeaderValue::from_str(&"a".repeat(65)).unwrap(),
                );
                Ok(response)
            })
            .await;
        });
        let transport = WebSocketTransport::builder(
            WebSocketEndpoint::local_explicit(format!("ws://{address}/events")).unwrap(),
        )
        .with_limits(TransportLimits {
            max_header_value_bytes: 64,
            ..TransportLimits::default()
        })
        .build()
        .unwrap();
        let error = transport
            .connect(RequestHeaders::new(), CallOptions::default())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ResponseLimit);
        server.await.unwrap();
    }
}
