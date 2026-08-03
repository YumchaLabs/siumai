//! Owned WebSocket connection policy and bounded provider sessions.

use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use http::Method;
use http::header::HeaderMap;
use reqwest::Url;
use siumai_core::{CallOptions, Cancellation, Error, ErrorKind};
use tokio::net::TcpStream;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::{CloseFrame, WebSocketConfig};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, client_async_tls_with_config};

use crate::auth::{AuthApplier, AuthContext, AuthRefresh, NoAuth};
use crate::endpoint::{CredentialAudience, EndpointConfig, Resolver, SystemResolver};
use crate::framing::{WebSocketFrame, WebSocketFramer};
use crate::transport::{effective_deadline, run_controlled};
use crate::{
    EndpointError, EndpointPolicy, RequestBuildError, RequestHeaders, TransportConfigError,
    TransportLimits,
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
        Self::new(url, EndpointPolicy::LocalExplicit)
    }

    pub fn new(url: impl AsRef<str>, policy: EndpointPolicy) -> Result<Self, EndpointError> {
        let url = Url::parse(url.as_ref()).map_err(|_| EndpointError::InvalidUrl)?;
        match policy {
            EndpointPolicy::Official(_) | EndpointPolicy::PublicCustom if url.scheme() != "wss" => {
                return Err(EndpointError::SchemeNotAllowed);
            }
            EndpointPolicy::LocalExplicit if !matches!(url.scheme(), "ws" | "wss") => {
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
        let deadline = effective_deadline(options.deadline(), self.inner.session_timeout);
        let permits = self.acquire(&cancellation, deadline).await?;

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
            deadline,
        )
        .await??;
        let (credential_headers, credential_query) = patch.into_parts();
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
        if !credential_query.is_empty() {
            let mut pairs = url.query_pairs_mut();
            for (name, value) in credential_query {
                pairs.append_pair(&name, &value);
            }
        }
        if !self.inner.endpoint.audience.matches(&url) {
            return Err(websocket_endpoint_error(EndpointError::AudienceMismatch));
        }

        let addresses = run_controlled(
            self.inner
                .endpoint
                .connector_endpoint
                .validated_addresses(self.inner.resolver.as_ref()),
            &cancellation,
            deadline,
        )
        .await?
        .map_err(websocket_endpoint_error)?;
        let socket = self
            .connect_tcp(&addresses, &cancellation, deadline)
            .await?;
        let remote = socket.peer_addr().map_err(|_| websocket_connect_error())?;
        self.inner
            .endpoint
            .connector_endpoint
            .validate_remote(remote)
            .map_err(websocket_endpoint_error)?;
        socket
            .set_nodelay(true)
            .map_err(|_| websocket_connect_error())?;

        let mut request = url
            .as_str()
            .into_client_request()
            .map_err(|_| websocket_request_error(RequestBuildError::InvalidTarget))?;
        merge_handshake_headers(request.headers_mut(), request_headers)?;
        let config = WebSocketConfig::default()
            .read_buffer_size(self.inner.limits.max_frame_bytes.min(64 * 1024))
            .write_buffer_size(0)
            .max_write_buffer_size(self.inner.limits.max_request_bytes)
            .max_message_size(Some(self.inner.limits.max_frame_bytes))
            .max_frame_size(Some(self.inner.limits.max_frame_bytes));
        let (stream, _response) = run_controlled(
            client_async_tls_with_config(request, socket, Some(config), None),
            &cancellation,
            deadline,
        )
        .await?
        .map_err(|_| websocket_handshake_error())?;

        Ok(WebSocketConnection {
            stream: Some(stream),
            framer: WebSocketFramer::new(&self.inner.limits),
            cancellation,
            deadline,
            permits: Some(permits),
            maximum_outgoing_bytes: self.inner.limits.max_request_bytes,
            io_timeout: self.inner.io_timeout,
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
        for address in addresses {
            let attempt = run_controlled(
                tokio::time::timeout(self.inner.connect_timeout, TcpStream::connect(address)),
                cancellation,
                deadline,
            )
            .await?;
            if let Ok(Ok(socket)) = attempt {
                return Ok(socket);
            }
        }
        Err(websocket_connect_error())
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

/// An established bounded WebSocket session with drop cancellation.
pub struct WebSocketConnection {
    stream: Option<RawWebSocket>,
    framer: WebSocketFramer,
    cancellation: Cancellation,
    deadline: Option<Instant>,
    permits: Option<WebSocketPermits>,
    maximum_outgoing_bytes: usize,
    io_timeout: Duration,
}

impl WebSocketConnection {
    pub async fn send(&mut self, frame: WebSocketFrame) -> Result<(), Error> {
        let message = encode_frame(frame);
        if message.len() > self.maximum_outgoing_bytes {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "outgoing WebSocket message exceeds the configured limit",
            ));
        }
        let cancellation = self.cancellation.clone();
        let deadline = effective_deadline(self.deadline, self.io_timeout);
        let stream = self.stream.as_mut().ok_or_else(websocket_closed_error)?;
        run_controlled(stream.send(message), &cancellation, deadline)
            .await?
            .map_err(|_| websocket_io_error())
    }

    pub async fn next(&mut self) -> Result<Option<WebSocketFrame>, Error> {
        let cancellation = self.cancellation.clone();
        let deadline = effective_deadline(self.deadline, self.io_timeout);
        let stream = self.stream.as_mut().ok_or_else(websocket_closed_error)?;
        let message = run_controlled(stream.next(), &cancellation, deadline).await?;
        match message {
            Some(Ok(message)) => self.framer.decode(message).map(Some).map_err(|_| {
                Error::new(
                    ErrorKind::ResponseLimit,
                    "incoming WebSocket message exceeded a transport limit",
                )
            }),
            Some(Err(_)) => {
                self.terminate();
                Err(websocket_io_error())
            }
            None => {
                self.terminate();
                Ok(None)
            }
        }
    }

    pub async fn close(&mut self) -> Result<(), Error> {
        let cancellation = self.cancellation.clone();
        let deadline = effective_deadline(self.deadline, self.io_timeout);
        let Some(mut stream) = self.stream.take() else {
            self.terminate();
            return Ok(());
        };
        let result = run_controlled(stream.close(None), &cancellation, deadline)
            .await?
            .map_err(|_| websocket_io_error());
        self.terminate();
        result
    }

    fn terminate(&mut self) {
        self.stream = None;
        self.permits = None;
    }
}

impl fmt::Debug for WebSocketConnection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WebSocketConnection")
            .field("open", &self.stream.is_some())
            .field("framer", &self.framer)
            .field("deadline", &self.deadline)
            .finish()
    }
}

impl Drop for WebSocketConnection {
    fn drop(&mut self) {
        self.cancellation.cancel();
        self.terminate();
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

fn websocket_connect_error() -> Error {
    Error::new(ErrorKind::Transport, "WebSocket TCP connection failed")
}

fn websocket_handshake_error() -> Error {
    Error::new(ErrorKind::Transport, "WebSocket handshake failed")
}

fn websocket_io_error() -> Error {
    Error::new(ErrorKind::Transport, "WebSocket session I/O failed")
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
                EndpointPolicy::LocalExplicit,
            )
            .unwrap_err(),
            EndpointError::UserInfoNotAllowed
        );
        assert_eq!(
            WebSocketEndpoint::new(
                "ws://localhost:8080/live#fragment",
                EndpointPolicy::LocalExplicit,
            )
            .unwrap_err(),
            EndpointError::FragmentNotAllowed
        );
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
}
