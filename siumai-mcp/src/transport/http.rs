use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::Arc;

use futures_util::StreamExt;
use futures_util::stream::BoxStream;
use http::header::{ACCEPT, CONTENT_TYPE, HeaderName, HeaderValue, WWW_AUTHENTICATE};
use reqwest::dns::{Addrs, Name, Resolve as ReqwestResolve, Resolving};
use reqwest::redirect;
use rmcp::model::{ClientJsonRpcMessage, JsonRpcMessage, ServerJsonRpcMessage};
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::streamable_http_client::{
    AuthRequiredError, InsufficientScopeError, SseError, StreamableHttpClient,
    StreamableHttpClientTransportConfig, StreamableHttpError, StreamableHttpPostResponse,
};
use siumai_transport::framing::{SseDecoder, SseEvent};
use siumai_transport::{
    EndpointError, HttpTransportRoute, ProxyEndpoint, Resolver, SystemResolver,
    TransportConfigError, TransportLimits,
};
use sse_stream::Sse;

use crate::error::McpSensitiveDetails;

const EVENT_STREAM_MIME_TYPE: &str = "text/event-stream";
const JSON_MIME_TYPE: &str = "application/json";
const HEADER_SESSION_ID: &str = "mcp-session-id";
const HEADER_LAST_EVENT_ID: &str = "last-event-id";

/// HTTP failure retained behind the explicit MCP sensitive-source boundary.
pub(crate) enum McpHttpClientError {
    Request(reqwest::Error),
    Route(TransportConfigError),
    Peer(EndpointError),
    MessageTooLarge {
        maximum: usize,
    },
    HttpStatus {
        status: u16,
        endpoint: Arc<str>,
        body: Arc<[u8]>,
    },
}

impl fmt::Debug for McpHttpClientError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Request(_) => formatter.write_str("McpHttpClientError::Request([REDACTED])"),
            Self::Route(error) => formatter
                .debug_tuple("McpHttpClientError::Route")
                .field(error)
                .finish(),
            Self::Peer(error) => formatter
                .debug_tuple("McpHttpClientError::Peer")
                .field(error)
                .finish(),
            Self::MessageTooLarge { maximum } => formatter
                .debug_struct("McpHttpClientError::MessageTooLarge")
                .field("maximum", maximum)
                .finish(),
            Self::HttpStatus { status, body, .. } => formatter
                .debug_struct("McpHttpClientError::HttpStatus")
                .field("status", status)
                .field("endpoint", &"[REDACTED]")
                .field("body_bytes", &body.len())
                .finish(),
        }
    }
}

impl fmt::Display for McpHttpClientError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Request(_) => formatter.write_str("MCP HTTP request failed"),
            Self::Route(_) => formatter.write_str("MCP HTTP route configuration failed"),
            Self::Peer(_) => formatter.write_str("MCP HTTP peer validation failed"),
            Self::MessageTooLarge { maximum } => {
                write!(formatter, "MCP HTTP message exceeded {maximum} bytes")
            }
            Self::HttpStatus { status, .. } => {
                write!(formatter, "MCP HTTP server returned status {status}")
            }
        }
    }
}

impl std::error::Error for McpHttpClientError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Request(error) => Some(error),
            Self::Route(error) => Some(error),
            Self::Peer(error) => Some(error),
            Self::MessageTooLarge { .. } | Self::HttpStatus { .. } => None,
        }
    }
}

pub(crate) fn sensitive_details(error: &(dyn std::error::Error + 'static)) -> McpSensitiveDetails {
    let mut details = McpSensitiveDetails::default();
    collect_sensitive_details(error, &mut details, 0);
    details
}

fn collect_sensitive_details(
    error: &(dyn std::error::Error + 'static),
    details: &mut McpSensitiveDetails,
    depth: usize,
) {
    if depth >= 8 {
        return;
    }
    if let Some(error) = error.downcast_ref::<McpHttpClientError>() {
        match error {
            McpHttpClientError::Request(error) => {
                if details.endpoint.is_none() {
                    details.endpoint = error.url().map(|url| Arc::from(url.as_str()));
                }
            }
            McpHttpClientError::Route(_) | McpHttpClientError::Peer(_) => {}
            McpHttpClientError::MessageTooLarge { .. } => {}
            McpHttpClientError::HttpStatus { endpoint, body, .. } => {
                details.endpoint.get_or_insert_with(|| endpoint.clone());
                details.response_body.get_or_insert_with(|| body.clone());
            }
        }
        return;
    }
    if let Some(error) = error.downcast_ref::<StreamableHttpError<McpHttpClientError>>() {
        match error {
            StreamableHttpError::Client(source) => {
                collect_sensitive_details(source, details, depth + 1);
            }
            StreamableHttpError::AuthRequired(source) => {
                details
                    .auth_challenge
                    .get_or_insert_with(|| Arc::from(source.www_authenticate_header.as_str()));
            }
            StreamableHttpError::InsufficientScope(source) => {
                details
                    .auth_challenge
                    .get_or_insert_with(|| Arc::from(source.www_authenticate_header.as_str()));
            }
            _ => {}
        }
        return;
    }
    if let Some(error) = error.downcast_ref::<rmcp::transport::DynamicTransportError>() {
        collect_sensitive_details(error.error.as_ref(), details, depth + 1);
        return;
    }
    if let Some(error) = error.downcast_ref::<rmcp::service::ClientInitializeError>() {
        if let rmcp::service::ClientInitializeError::TransportError { error, .. } = error {
            collect_sensitive_details(error.error.as_ref(), details, depth + 1);
        }
        return;
    }
    if let Some(error) = error.downcast_ref::<rmcp::service::ServiceError>() {
        if let rmcp::service::ServiceError::TransportSend(error) = error {
            collect_sensitive_details(error.error.as_ref(), details, depth + 1);
        }
        return;
    }
    if let Some(source) = error.source() {
        collect_sensitive_details(source, details, depth + 1);
    }
}

#[derive(Clone)]
pub(crate) struct BoundedHttpClient {
    client: reqwest::Client,
    network_endpoint: Option<ProxyEndpoint>,
    max_message_bytes: usize,
}

struct McpGuardedDnsResolver {
    endpoint: ProxyEndpoint,
    resolver: Arc<dyn Resolver>,
}

impl ReqwestResolve for McpGuardedDnsResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let endpoint = self.endpoint.clone();
        let resolver = self.resolver.clone();
        let requested_host = name.as_str().to_owned();
        Box::pin(async move {
            let addresses = endpoint
                .resolve_for_connector(&requested_host, resolver.as_ref())
                .await
                .map_err(|error| {
                    Box::new(error) as Box<dyn std::error::Error + Send + Sync + 'static>
                })?;
            Ok(Box::new(addresses.into_iter()) as Addrs)
        })
    }
}

impl BoundedHttpClient {
    #[cfg(test)]
    fn new(max_message_bytes: usize) -> Result<Self, McpHttpClientError> {
        Self::with_route(max_message_bytes, &HttpTransportRoute::Direct)
    }

    fn with_route(
        max_message_bytes: usize,
        route: &HttpTransportRoute,
    ) -> Result<Self, McpHttpClientError> {
        Self::with_builder(
            max_message_bytes,
            route,
            reqwest::Client::builder(),
            Arc::new(SystemResolver),
        )
    }

    fn with_builder(
        max_message_bytes: usize,
        route: &HttpTransportRoute,
        builder: reqwest::ClientBuilder,
        resolver: Arc<dyn Resolver>,
    ) -> Result<Self, McpHttpClientError> {
        let network_endpoint = route.proxy().cloned();
        let mut builder = builder
            .redirect(redirect::Policy::none())
            .referer(false)
            .no_proxy()
            .retry(reqwest::retry::never());
        if let Some(endpoint) = &network_endpoint {
            builder = builder.dns_resolver(McpGuardedDnsResolver {
                endpoint: endpoint.clone(),
                resolver,
            });
        }
        if let Some(proxy) = route
            .build_reqwest_proxy()
            .map_err(McpHttpClientError::Route)?
        {
            builder = builder.proxy(proxy);
        }
        let client = builder
            .build()
            .map_err(|_| McpHttpClientError::Route(TransportConfigError::ClientBuild))?;
        Ok(Self {
            client,
            network_endpoint,
            max_message_bytes,
        })
    }

    async fn send(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<reqwest::Response, McpHttpClientError> {
        let response = request.send().await.map_err(McpHttpClientError::Request)?;
        if let Some(endpoint) = &self.network_endpoint {
            let remote = response
                .remote_addr()
                .ok_or(McpHttpClientError::Peer(EndpointError::AddressNotAllowed))?;
            endpoint
                .validate_remote(remote)
                .map_err(McpHttpClientError::Peer)?;
        }
        Ok(response)
    }

    fn request(
        &self,
        method: reqwest::Method,
        uri: &str,
        session_id: Option<&str>,
        auth_token: Option<&str>,
        last_event_id: Option<&str>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> reqwest::RequestBuilder {
        let mut request = self.client.request(method, uri);
        if let Some(session_id) = session_id {
            request = request.header(HEADER_SESSION_ID, session_id);
        }
        if let Some(auth_token) = auth_token {
            request = request.bearer_auth(auth_token);
        }
        if let Some(last_event_id) = last_event_id {
            request = request.header(HEADER_LAST_EVENT_ID, last_event_id);
        }
        for (name, value) in custom_headers {
            request = request.header(name, value);
        }
        request
    }

    async fn bounded_body(
        &self,
        response: reqwest::Response,
    ) -> Result<Vec<u8>, McpHttpClientError> {
        if response
            .content_length()
            .is_some_and(|length| length > self.max_message_bytes as u64)
        {
            return Err(McpHttpClientError::MessageTooLarge {
                maximum: self.max_message_bytes,
            });
        }
        let mut body = Vec::with_capacity(
            response
                .content_length()
                .and_then(|length| usize::try_from(length).ok())
                .unwrap_or_default()
                .min(self.max_message_bytes),
        );
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(McpHttpClientError::Request)?;
            if body.len().saturating_add(chunk.len()) > self.max_message_bytes {
                return Err(McpHttpClientError::MessageTooLarge {
                    maximum: self.max_message_bytes,
                });
            }
            body.extend_from_slice(&chunk);
        }
        Ok(body)
    }

    async fn status_error(
        &self,
        uri: Arc<str>,
        response: reqwest::Response,
    ) -> Result<McpHttpClientError, McpHttpClientError> {
        let status = response.status().as_u16();
        let body = self.bounded_body(response).await?;
        Ok(McpHttpClientError::HttpStatus {
            status,
            endpoint: uri,
            body: body.into(),
        })
    }
}

impl StreamableHttpClient for BoundedHttpClient {
    type Error = McpHttpClientError;

    async fn post_message(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_token: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<StreamableHttpPostResponse, StreamableHttpError<Self::Error>> {
        self.post_message_with_max_sse_event_size(
            uri,
            message,
            session_id,
            auth_token,
            custom_headers,
            self.max_message_bytes,
        )
        .await
    }

    async fn post_message_with_max_sse_event_size(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_token: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
        max_sse_event_size: usize,
    ) -> Result<StreamableHttpPostResponse, StreamableHttpError<Self::Error>> {
        let session_was_attached = session_id.is_some();
        let encoded = serde_json::to_vec(&message).map_err(StreamableHttpError::Deserialize)?;
        if encoded.len() > self.max_message_bytes {
            return Err(StreamableHttpError::Client(
                McpHttpClientError::MessageTooLarge {
                    maximum: self.max_message_bytes,
                },
            ));
        }
        let request = self
            .request(
                reqwest::Method::POST,
                uri.as_ref(),
                session_id.as_deref(),
                auth_token.as_deref(),
                None,
                custom_headers,
            )
            .header(ACCEPT, [EVENT_STREAM_MIME_TYPE, JSON_MIME_TYPE].join(", "))
            .header(CONTENT_TYPE, JSON_MIME_TYPE)
            .body(encoded);
        let response = self
            .send(request)
            .await
            .map_err(StreamableHttpError::Client)?;

        if let Some(error) = authentication_error(&response)? {
            return Err(error);
        }
        let status = response.status();
        if matches!(
            status,
            reqwest::StatusCode::ACCEPTED | reqwest::StatusCode::NO_CONTENT
        ) {
            return Ok(StreamableHttpPostResponse::Accepted);
        }
        if status == reqwest::StatusCode::NOT_FOUND && session_was_attached {
            return Err(StreamableHttpError::SessionExpired);
        }
        let content_type = content_type(&response);
        let response_session_id = session_id_header(&response);
        if !status.is_success() {
            let body = self
                .bounded_body(response)
                .await
                .map_err(StreamableHttpError::Client)?;
            if content_type.as_deref().is_some_and(is_json_content_type)
                && let Ok(message) = serde_json::from_slice::<ServerJsonRpcMessage>(&body)
                && matches!(message, JsonRpcMessage::Error(_))
            {
                return Ok(StreamableHttpPostResponse::Json(
                    message,
                    response_session_id,
                ));
            }
            return Err(StreamableHttpError::Client(
                McpHttpClientError::HttpStatus {
                    status: status.as_u16(),
                    endpoint: uri,
                    body: body.into(),
                },
            ));
        }
        match content_type.as_deref() {
            Some(value) if is_sse_content_type(value) => Ok(StreamableHttpPostResponse::Sse(
                bounded_sse_stream(response, max_sse_event_size.min(self.max_message_bytes)),
                response_session_id,
            )),
            Some(value) if is_json_content_type(value) => {
                let body = self
                    .bounded_body(response)
                    .await
                    .map_err(StreamableHttpError::Client)?;
                match serde_json::from_slice::<ServerJsonRpcMessage>(&body) {
                    Ok(message) => Ok(StreamableHttpPostResponse::Json(
                        message,
                        response_session_id,
                    )),
                    Err(_) => Ok(StreamableHttpPostResponse::Accepted),
                }
            }
            _ => Err(StreamableHttpError::UnexpectedContentType(content_type)),
        }
    }

    async fn delete_session(
        &self,
        uri: Arc<str>,
        session_id: Arc<str>,
        auth_token: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<(), StreamableHttpError<Self::Error>> {
        let request = self.request(
            reqwest::Method::DELETE,
            uri.as_ref(),
            Some(session_id.as_ref()),
            auth_token.as_deref(),
            None,
            custom_headers,
        );
        let response = self
            .send(request)
            .await
            .map_err(StreamableHttpError::Client)?;
        if response.status() == reqwest::StatusCode::METHOD_NOT_ALLOWED {
            return Ok(());
        }
        if let Some(error) = authentication_error(&response)? {
            return Err(error);
        }
        if response.status().is_success() {
            return Ok(());
        }
        Err(StreamableHttpError::Client(
            self.status_error(uri, response)
                .await
                .map_err(StreamableHttpError::Client)?,
        ))
    }

    async fn get_stream(
        &self,
        uri: Arc<str>,
        session_id: Option<Arc<str>>,
        last_event_id: Option<String>,
        auth_token: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<BoxStream<'static, Result<Sse, SseError>>, StreamableHttpError<Self::Error>> {
        self.get_stream_with_max_sse_event_size(
            uri,
            session_id,
            last_event_id,
            auth_token,
            custom_headers,
            self.max_message_bytes,
        )
        .await
    }

    async fn get_stream_with_max_sse_event_size(
        &self,
        uri: Arc<str>,
        session_id: Option<Arc<str>>,
        last_event_id: Option<String>,
        auth_token: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
        max_sse_event_size: usize,
    ) -> Result<BoxStream<'static, Result<Sse, SseError>>, StreamableHttpError<Self::Error>> {
        let request = self
            .request(
                reqwest::Method::GET,
                uri.as_ref(),
                session_id.as_deref(),
                auth_token.as_deref(),
                last_event_id.as_deref(),
                custom_headers,
            )
            .header(ACCEPT, [EVENT_STREAM_MIME_TYPE, JSON_MIME_TYPE].join(", "));
        let response = self
            .send(request)
            .await
            .map_err(StreamableHttpError::Client)?;
        if response.status() == reqwest::StatusCode::METHOD_NOT_ALLOWED {
            return Err(StreamableHttpError::ServerDoesNotSupportSse);
        }
        if let Some(error) = authentication_error(&response)? {
            return Err(error);
        }
        if !response.status().is_success() {
            return Err(StreamableHttpError::Client(
                self.status_error(uri, response)
                    .await
                    .map_err(StreamableHttpError::Client)?,
            ));
        }
        let content_type = content_type(&response);
        if !content_type
            .as_deref()
            .is_some_and(|value| is_sse_content_type(value) || is_json_content_type(value))
        {
            return Err(StreamableHttpError::UnexpectedContentType(content_type));
        }
        Ok(bounded_sse_stream(
            response,
            max_sse_event_size.min(self.max_message_bytes),
        ))
    }
}

fn authentication_error(
    response: &reqwest::Response,
) -> Result<Option<StreamableHttpError<McpHttpClientError>>, StreamableHttpError<McpHttpClientError>>
{
    let Some(challenge) = response.headers().get(WWW_AUTHENTICATE) else {
        return Ok(None);
    };
    let challenge = challenge.to_str().map_err(|_| {
        StreamableHttpError::UnexpectedServerResponse(Cow::Borrowed(
            "invalid www-authenticate header value",
        ))
    })?;
    match response.status() {
        reqwest::StatusCode::UNAUTHORIZED => Ok(Some(StreamableHttpError::AuthRequired(
            AuthRequiredError::new(challenge.to_owned()),
        ))),
        reqwest::StatusCode::FORBIDDEN => Ok(Some(StreamableHttpError::InsufficientScope(
            InsufficientScopeError::new(challenge.to_owned(), None),
        ))),
        _ => Ok(None),
    }
}

fn content_type(response: &reqwest::Response) -> Option<String> {
    response
        .headers()
        .get(CONTENT_TYPE)
        .map(|value| String::from_utf8_lossy(value.as_bytes()).into_owned())
}

fn session_id_header(response: &reqwest::Response) -> Option<String> {
    response
        .headers()
        .get(HEADER_SESSION_ID)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}

fn is_json_content_type(value: &str) -> bool {
    value.as_bytes().starts_with(JSON_MIME_TYPE.as_bytes())
}

fn is_sse_content_type(value: &str) -> bool {
    value
        .as_bytes()
        .starts_with(EVENT_STREAM_MIME_TYPE.as_bytes())
}

fn bounded_sse_stream(
    response: reqwest::Response,
    maximum: usize,
) -> BoxStream<'static, Result<Sse, SseError>> {
    let byte_stream = response.bytes_stream().boxed();
    let decoder = SseDecoder::new(&TransportLimits {
        max_frame_bytes: maximum,
        max_event_bytes: maximum,
        max_events_per_stream: usize::MAX,
        ..TransportLimits::default()
    });
    let queued = VecDeque::new();
    futures_util::stream::try_unfold(
        (byte_stream, decoder, queued),
        |(mut byte_stream, mut decoder, mut queued)| async move {
            loop {
                if let Some(event) = queued.pop_front() {
                    return Ok(Some((event, (byte_stream, decoder, queued))));
                }
                match byte_stream.next().await {
                    Some(Ok(chunk)) => {
                        queued.extend(
                            decoder
                                .push(&chunk)
                                .map_err(|error| SseError::Body(Box::new(error)))?
                                .into_iter()
                                .map(to_sse),
                        );
                    }
                    Some(Err(error)) => {
                        return Err(SseError::Body(Box::new(McpHttpClientError::Request(error))));
                    }
                    None => {
                        decoder
                            .finish()
                            .map_err(|error| SseError::Body(Box::new(error)))?;
                        return Ok(None);
                    }
                }
            }
        },
    )
    .boxed()
}

fn to_sse(event: SseEvent) -> Sse {
    Sse {
        event: (event.event_type() != "message").then(|| event.event_type().to_owned()),
        data: Some(event.data().to_owned()),
        id: event.id().map(str::to_owned),
        retry: event
            .retry()
            .map(|duration| duration.as_millis().min(u64::MAX as u128) as u64),
    }
}

pub(crate) fn http_transport(
    uri: &str,
    max_message_bytes: usize,
    route: &HttpTransportRoute,
) -> Result<StreamableHttpClientTransport<BoundedHttpClient>, McpHttpClientError> {
    let client = BoundedHttpClient::with_route(max_message_bytes, route)?;
    Ok(StreamableHttpClientTransport::with_client(
        client,
        http_transport_config(uri, max_message_bytes),
    ))
}

fn http_transport_config(
    uri: &str,
    max_message_bytes: usize,
) -> StreamableHttpClientTransportConfig {
    StreamableHttpClientTransportConfig::with_uri(uri)
        .max_sse_event_size(max_message_bytes)
        .reinit_on_expired_session(false)
}
#[cfg(test)]
mod tests {
    use std::net::SocketAddr;
    use std::process::Command;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use async_trait::async_trait;
    use base64::Engine as _;
    use rmcp::model::{CallToolRequestParams, ClientRequest, PingRequest, RequestId};
    use rmcp::service::ServiceExt;
    use siumai_transport::{ProxyBasicCredential, ProxyEndpoint};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio_rustls::TlsAcceptor;
    use tokio_rustls::rustls::ServerConfig;
    use tokio_rustls::rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};

    const DIRECT_PROXY_ENV_HELPER: &str = "SIUMAI_MCP_DIRECT_PROXY_ENV_HELPER";

    // Offline `.example.test` certificate fixtures; these are not credentials.
    const TEST_CA_DER: &str = "MIIBpzCCAU2gAwIBAgIUbTYlr1376Yr/+ZGcs/7LTVRAn9owCgYIKoZIzj0EAwIwITEfMB0GA1UEAwwWU2l1bWFpIE9mZmxpbmUgVGVzdCBDQTAeFw0yNjA4MTcwNTQ5MzZaFw0zNjA4MTQwNTQ5MzZaMCExHzAdBgNVBAMMFlNpdW1haSBPZmZsaW5lIFRlc3QgQ0EwWTATBgcqhkjOPQIBBggqhkjOPQMBBwNCAATS9UwJf41omyTuG43+oHO6FXHq3G1Msddvu66IHfd3+PDaPqSiY8XDX+Ey5d3EAgKWxzILQUfOPT0QCt9RacK6o2MwYTAdBgNVHQ4EFgQUEpo8dhQlLwIWZeaCSQlAUGUGuu0wHwYDVR0jBBgwFoAUEpo8dhQlLwIWZeaCSQlAUGUGuu0wDwYDVR0TAQH/BAUwAwEB/zAOBgNVHQ8BAf8EBAMCAQYwCgYIKoZIzj0EAwIDSAAwRQIhAO8iTQUm3hTksiBftChN8ziP91dYm3pH5iYJyqa4e5pOAiBbPFSbx5UM0wSdVkmEYs7+B2Dw7gZBCoB1OwphmBlrLg==";
    const PROXY_CERT_DER: &str = "MIIB1zCCAXygAwIBAgIUeauQEUJf7Pl280FQrQf3FSHAFGYwCgYIKoZIzj0EAwIwITEfMB0GA1UEAwwWU2l1bWFpIE9mZmxpbmUgVGVzdCBDQTAeFw0yNjA4MTcwNTQ5MzZaFw0zNjA4MTQwNTQ5MzZaMB0xGzAZBgNVBAMMEnByb3h5LmV4YW1wbGUudGVzdDBZMBMGByqGSM49AgEGCCqGSM49AwEHA0IABJOBb+14giMbsCKDztTFG65FCL39GGl9sIl8C/I8AkR5aqjzEB1U9XtFjTHUeabjyodO5JTv3ZB/MhF+EQV6K/WjgZUwgZIwHQYDVR0RBBYwFIIScHJveHkuZXhhbXBsZS50ZXN0MAwGA1UdEwEB/wQCMAAwDgYDVR0PAQH/BAQDAgeAMBMGA1UdJQQMMAoGCCsGAQUFBwMBMB0GA1UdDgQWBBRFpAwYyf389PxKkdIFuETkW59wIzAfBgNVHSMEGDAWgBQSmjx2FCUvAhZl5oJJCUBQZQa67TAKBggqhkjOPQQDAgNJADBGAiEArYfyumOCRVrLAJosZ9O0ecknTdbOe3aCRbcthT7s58sCIQDuq995l+HAm+PoJsx0DhVeKpjPDDkJLuxT8/eOXLEMcg==";
    const PROXY_KEY_DER: &str = "MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgYlViovjr5cdWq8MDvDUlYGhAPFNKp5+ExoX0wpYHyCqhRANCAASTgW/teIIjG7Aig87UxRuuRQi9/RhpfbCJfAvyPAJEeWqo8xAdVPV7RY0x1Hmm48qHTuSU792QfzIRfhEFeiv1";
    const ORIGIN_CERT_DER: &str = "MIIB3TCCAYKgAwIBAgIUeauQEUJf7Pl280FQrQf3FSHAFGcwCgYIKoZIzj0EAwIwITEfMB0GA1UEAwwWU2l1bWFpIE9mZmxpbmUgVGVzdCBDQTAeFw0yNjA4MTcwNTQ5MzZaFw0zNjA4MTQwNTQ5MzZaMCAxHjAcBgNVBAMMFXByb3ZpZGVyLmV4YW1wbGUudGVzdDBZMBMGByqGSM49AgEGCCqGSM49AwEHA0IABNrtKG7bQ83W+iw+kj39Wuwp5mE01ezAJbR+9NZeCTvg2YYMSmXnXwDniIviSyJwK3dh8MfSbUIx2GEs81eobQWjgZgwgZUwIAYDVR0RBBkwF4IVcHJvdmlkZXIuZXhhbXBsZS50ZXN0MAwGA1UdEwEB/wQCMAAwDgYDVR0PAQH/BAQDAgeAMBMGA1UdJQQMMAoGCCsGAQUFBwMBMB0GA1UdDgQWBBQ2L3md9NSIZIaY4RUoxlP0eziwaTAfBgNVHSMEGDAWgBQSmjx2FCUvAhZl5oJJCUBQZQa67TAKBggqhkjOPQQDAgNJADBGAiEAiU7zCZp7WT9f1w2OJ6z84Cj4Lo3yASg0RbqGucZTzrkCIQC9Q+REa8I+UT2EVXZl5JobOhaFxYvDrZmyKGZxBEdtiQ==";
    const ORIGIN_KEY_DER: &str = "MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgw91aPqnWKZ/kNPOkZC8uPc/RncGLJZDegIMETxlFlTihRANCAATa7Shu20PN1vosPpI9/VrsKeZhNNXswCW0fvTWXgk74NmGDEpl518A54iL4ksicCt3YfDH0m1CMdhhLPNXqG0F";

    struct FixtureResolver(SocketAddr);

    #[async_trait]
    impl Resolver for FixtureResolver {
        async fn resolve(&self, _host: &str, _port: u16) -> Result<Vec<SocketAddr>, EndpointError> {
            Ok(vec![self.0])
        }
    }

    fn ping() -> ClientJsonRpcMessage {
        ClientJsonRpcMessage::request(
            ClientRequest::PingRequest(PingRequest::default()),
            RequestId::Number(1),
        )
    }

    fn routed_client(
        maximum: usize,
        route: &HttpTransportRoute,
        proxy_address: SocketAddr,
    ) -> BoundedHttpClient {
        let certificate = reqwest::Certificate::from_der(&decode(TEST_CA_DER)).unwrap();
        BoundedHttpClient::with_builder(
            maximum,
            route,
            reqwest::Client::builder()
                .tls_certs_only([certificate])
                .connect_timeout(std::time::Duration::from_secs(2))
                .read_timeout(std::time::Duration::from_secs(2)),
            Arc::new(FixtureResolver(proxy_address)),
        )
        .unwrap()
    }

    async fn spawn_nested_tls_proxy(
        responses: Vec<Vec<u8>>,
    ) -> (SocketAddr, tokio::task::JoinHandle<Vec<(String, String)>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let proxy_acceptor = tls_acceptor(PROXY_CERT_DER, PROXY_KEY_DER);
        let origin_acceptor = tls_acceptor(ORIGIN_CERT_DER, ORIGIN_KEY_DER);
        let server = tokio::spawn(async move {
            let mut exchanges = Vec::with_capacity(responses.len());
            for response in responses {
                let (socket, _) = listener.accept().await.unwrap();
                let mut proxy_tls = proxy_acceptor.accept(socket).await.unwrap();
                let connect = read_http_head(&mut proxy_tls).await;
                proxy_tls
                    .write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                    .await
                    .unwrap();
                let mut origin_tls = origin_acceptor.accept(proxy_tls).await.unwrap();
                let origin_request = read_http_head(&mut origin_tls).await;
                origin_tls.write_all(&response).await.unwrap();
                exchanges.push((connect, origin_request));
            }
            exchanges
        });
        (address, server)
    }

    async fn spawn_plain_proxy(
        response: Option<Vec<u8>>,
    ) -> (SocketAddr, tokio::task::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let connect = read_http_head(&mut socket).await;
            if let Some(response) = response {
                socket.write_all(&response).await.unwrap();
            }
            connect
        });
        (address, server)
    }

    fn tls_acceptor(certificate: &str, private_key: &str) -> TlsAcceptor {
        let certificate = CertificateDer::from(decode(certificate));
        let private_key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(decode(private_key)));
        let config = ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![certificate], private_key)
            .unwrap();
        TlsAcceptor::from(Arc::new(config))
    }

    fn decode(value: &str) -> Vec<u8> {
        base64::engine::general_purpose::STANDARD
            .decode(value)
            .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(value))
            .unwrap()
    }

    async fn read_http_head<S>(stream: &mut S) -> String
    where
        S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
    {
        let mut bytes = Vec::new();
        loop {
            if let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                return String::from_utf8(bytes[..end + 4].to_vec()).unwrap();
            }
            assert!(
                bytes.len() <= 16 * 1024,
                "fixture HTTP head exceeded its bound"
            );
            let mut chunk = [0_u8; 1024];
            let read = stream.read(&mut chunk).await.unwrap();
            assert!(read > 0, "fixture connection ended before the HTTP head");
            bytes.extend_from_slice(&chunk[..read]);
        }
    }

    fn header<'a>(head: &'a str, expected: &str) -> Option<&'a str> {
        head.lines().skip(1).find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case(expected).then(|| value.trim())
        })
    }

    async fn serve_once(response: Vec<u8>) -> Arc<str> {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let header_end = loop {
                if let Some(index) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
                    break index + 4;
                }
                let mut chunk = [0_u8; 4096];
                let count = stream.read(&mut chunk).await.unwrap();
                if count == 0 {
                    return;
                }
                request.extend_from_slice(&chunk[..count]);
            };
            let headers = String::from_utf8_lossy(&request[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap_or_default();
            while request.len().saturating_sub(header_end) < content_length {
                let mut chunk = [0_u8; 4096];
                let count = stream.read(&mut chunk).await.unwrap();
                if count == 0 {
                    return;
                }
                request.extend_from_slice(&chunk[..count]);
            }
            stream.write_all(&response).await.unwrap();
        });
        Arc::from(format!("http://{address}/mcp"))
    }

    fn fixed_response(status: u16, content_type: &str, body: &[u8]) -> Vec<u8> {
        let mut response = format!(
            "HTTP/1.1 {status} Test\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        response.extend_from_slice(body);
        response
    }

    fn chunked_response(status: u16, content_type: &str, chunks: &[&[u8]]) -> Vec<u8> {
        let mut response = format!(
            "HTTP/1.1 {status} Test\r\nContent-Type: {content_type}\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
        )
        .into_bytes();
        for chunk in chunks {
            response.extend_from_slice(format!("{:x}\r\n", chunk.len()).as_bytes());
            response.extend_from_slice(chunk);
            response.extend_from_slice(b"\r\n");
        }
        response.extend_from_slice(b"0\r\n\r\n");
        response
    }

    #[test]
    fn direct_mcp_ignores_common_proxy_environment_variables() {
        if std::env::var_os(DIRECT_PROXY_ENV_HELPER).is_some() {
            return;
        }
        let proxy = "http://127.0.0.1:9";
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "transport::http::tests::direct_mcp_proxy_environment_helper",
                "--nocapture",
            ])
            .env(DIRECT_PROXY_ENV_HELPER, "1")
            .env("HTTP_PROXY", proxy)
            .env("HTTPS_PROXY", proxy)
            .env("ALL_PROXY", proxy)
            .env("http_proxy", proxy)
            .env("https_proxy", proxy)
            .env("all_proxy", proxy)
            .env("NO_PROXY", "")
            .env("no_proxy", "")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "child output:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn direct_mcp_proxy_environment_helper() {
        if std::env::var_os(DIRECT_PROXY_ENV_HELPER).is_none() {
            return;
        }
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let uri = serve_once(fixed_response(202, JSON_MIME_TYPE, b"")).await;
                let response = BoundedHttpClient::new(1024)
                    .unwrap()
                    .post_message(uri, ping(), None, None, HashMap::new())
                    .await
                    .unwrap();
                assert!(matches!(response, StreamableHttpPostResponse::Accepted));
            });
    }

    #[tokio::test]
    async fn trusted_route_applies_to_post_get_and_delete_with_separate_credentials() {
        let responses = vec![
            fixed_response(202, JSON_MIME_TYPE, b""),
            fixed_response(200, EVENT_STREAM_MIME_TYPE, b"data: route-ok\n\n"),
            fixed_response(204, JSON_MIME_TYPE, b""),
        ];
        let (proxy_address, server) = spawn_nested_tls_proxy(responses).await;
        let proxy = ProxyEndpoint::local_explicit(format!(
            "https://proxy.example.test:{}",
            proxy_address.port()
        ))
        .unwrap();
        let route = HttpTransportRoute::trusted_connect(proxy)
            .with_basic_auth(ProxyBasicCredential::new("proxy-user", "proxy-secret").unwrap())
            .unwrap();
        let client = routed_client(1024, &route, proxy_address);
        let uri: Arc<str> =
            Arc::from("https://provider.example.test/mcp?token=origin-query-secret".to_string());
        let bearer = Some("mcp-bearer-secret".to_string());

        let response = client
            .post_message(uri.clone(), ping(), None, bearer.clone(), HashMap::new())
            .await
            .unwrap();
        assert!(matches!(response, StreamableHttpPostResponse::Accepted));

        let mut stream = client
            .get_stream(
                uri.clone(),
                Some(Arc::from("session-1")),
                Some("event-1".to_string()),
                bearer.clone(),
                HashMap::new(),
            )
            .await
            .unwrap();
        assert_eq!(
            stream.next().await.unwrap().unwrap().data.as_deref(),
            Some("route-ok")
        );

        client
            .delete_session(uri, Arc::from("session-1"), bearer, HashMap::new())
            .await
            .unwrap();

        let exchanges = server.await.unwrap();
        assert_eq!(exchanges.len(), 3);
        let methods = ["POST", "GET", "DELETE"];
        for ((connect, origin), method) in exchanges.iter().zip(methods) {
            assert!(connect.starts_with("CONNECT provider.example.test:443 HTTP/1.1\r\n"));
            assert_eq!(
                header(connect, "proxy-authorization"),
                Some("Basic cHJveHktdXNlcjpwcm94eS1zZWNyZXQ=")
            );
            assert!(header(connect, "authorization").is_none());
            assert!(!connect.contains("origin-query-secret"));
            assert!(!connect.contains("mcp-bearer-secret"));

            assert!(origin.starts_with(&format!(
                "{method} /mcp?token=origin-query-secret HTTP/1.1\r\n"
            )));
            assert_eq!(
                header(origin, "authorization"),
                Some("Bearer mcp-bearer-secret")
            );
            assert!(header(origin, "proxy-authorization").is_none());
        }
    }

    #[tokio::test]
    async fn trusted_route_preserves_message_bounds() {
        let body = serde_json::to_vec(&serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": { "value": "x".repeat(128) }
        }))
        .unwrap();
        let oversized_sse = format!(
            "data: {}\ndata: {}\ndata: {}\n\n",
            "event-secret-a".repeat(2),
            "event-secret-b".repeat(2),
            "event-secret-c".repeat(2),
        );
        let (proxy_address, server) = spawn_nested_tls_proxy(vec![
            fixed_response(200, JSON_MIME_TYPE, &body),
            fixed_response(200, EVENT_STREAM_MIME_TYPE, oversized_sse.as_bytes()),
        ])
        .await;
        let route = HttpTransportRoute::trusted_connect(
            ProxyEndpoint::local_explicit(format!(
                "https://proxy.example.test:{}",
                proxy_address.port()
            ))
            .unwrap(),
        );
        let client = routed_client(64, &route, proxy_address);
        let error = client
            .post_message(
                Arc::from("https://provider.example.test/mcp"),
                ping(),
                None,
                None,
                HashMap::new(),
            )
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            StreamableHttpError::Client(McpHttpClientError::MessageTooLarge { maximum: 64 })
        ));

        let response = client
            .post_message(
                Arc::from("https://provider.example.test/mcp"),
                ping(),
                None,
                None,
                HashMap::new(),
            )
            .await
            .unwrap();
        let StreamableHttpPostResponse::Sse(mut stream, _) = response else {
            panic!("expected an SSE response");
        };
        let error = stream.next().await.unwrap().unwrap_err();
        assert!(error.to_string().contains("event limit"));
        assert!(!format!("{error:?} | {error}").contains("event-secret"));
        assert_eq!(server.await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn trusted_route_rejects_a_disallowed_proxy_peer_before_connect() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_address = listener.local_addr().unwrap();
        let route = HttpTransportRoute::trusted_connect(
            ProxyEndpoint::https(format!(
                "https://proxy.example.test:{}",
                proxy_address.port()
            ))
            .unwrap(),
        );
        let client = BoundedHttpClient::with_builder(
            1024,
            &route,
            reqwest::Client::builder().connect_timeout(std::time::Duration::from_secs(1)),
            Arc::new(FixtureResolver(proxy_address)),
        )
        .unwrap();
        let error = client
            .post_message(
                Arc::from("https://provider.example.test/mcp"),
                ping(),
                None,
                None,
                HashMap::new(),
            )
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            StreamableHttpError::Client(McpHttpClientError::Request(_))
        ));
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(100), listener.accept())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn proxy_connect_failures_are_sanitized_and_never_replayed() {
        for response in [
            Some(b"HTTP/1.1 407 Proxy Authentication Required\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()),
            None,
        ] {
            let (proxy_address, server) = spawn_plain_proxy(response).await;
            let route = HttpTransportRoute::trusted_connect(
                ProxyEndpoint::local_explicit(format!(
                    "http://proxy.example.test:{}",
                    proxy_address.port()
                ))
                .unwrap(),
            );
            let client = BoundedHttpClient::with_builder(
                1024,
                &route,
                reqwest::Client::builder()
                    .connect_timeout(std::time::Duration::from_secs(1))
                    .read_timeout(std::time::Duration::from_secs(1)),
                Arc::new(FixtureResolver(proxy_address)),
            )
            .unwrap();
            let error = client
                .post_message(
                    Arc::from("https://provider.example.test/mcp?token=query-secret"),
                    ping(),
                    None,
                    Some("mcp-secret".to_string()),
                    HashMap::new(),
                )
                .await
                .unwrap_err();
            assert!(matches!(
                error,
                StreamableHttpError::Client(McpHttpClientError::Request(_))
            ));
            let diagnostics = format!("{error:?} | {error}");
            assert!(!diagnostics.contains("query-secret"));
            assert!(!diagnostics.contains("mcp-secret"));
            let connect = server.await.unwrap();
            assert!(connect.starts_with("CONNECT provider.example.test:443 HTTP/1.1\r\n"));
        }
    }

    #[derive(Clone)]
    struct ExpiredSessionClient {
        tool_calls: Arc<AtomicUsize>,
    }

    impl StreamableHttpClient for ExpiredSessionClient {
        type Error = std::io::Error;

        async fn post_message(
            &self,
            _uri: Arc<str>,
            message: ClientJsonRpcMessage,
            session_id: Option<Arc<str>>,
            _auth_token: Option<String>,
            _custom_headers: HashMap<HeaderName, HeaderValue>,
        ) -> Result<StreamableHttpPostResponse, StreamableHttpError<Self::Error>> {
            let value = serde_json::to_value(&message).unwrap();
            match value.get("method").and_then(serde_json::Value::as_str) {
                Some("initialize") => {
                    let response = serde_json::from_value(serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": value["id"].clone(),
                        "result": {
                            "protocolVersion": value["params"]["protocolVersion"].clone(),
                            "capabilities": {},
                            "serverInfo": { "name": "replay-probe", "version": "1" }
                        }
                    }))
                    .unwrap();
                    Ok(StreamableHttpPostResponse::Json(
                        response,
                        Some("session-1".to_owned()),
                    ))
                }
                Some("tools/call") => {
                    assert!(session_id.is_some());
                    self.tool_calls.fetch_add(1, Ordering::SeqCst);
                    Err(StreamableHttpError::SessionExpired)
                }
                _ => Ok(StreamableHttpPostResponse::Accepted),
            }
        }

        async fn delete_session(
            &self,
            _uri: Arc<str>,
            _session_id: Arc<str>,
            _auth_token: Option<String>,
            _custom_headers: HashMap<HeaderName, HeaderValue>,
        ) -> Result<(), StreamableHttpError<Self::Error>> {
            Ok(())
        }

        async fn get_stream(
            &self,
            _uri: Arc<str>,
            _session_id: Option<Arc<str>>,
            _last_event_id: Option<String>,
            _auth_token: Option<String>,
            _custom_headers: HashMap<HeaderName, HeaderValue>,
        ) -> Result<BoxStream<'static, Result<Sse, SseError>>, StreamableHttpError<Self::Error>>
        {
            Err(StreamableHttpError::ServerDoesNotSupportSse)
        }
    }

    #[tokio::test]
    async fn expired_session_does_not_reinitialize_or_replay_tool_call() {
        let tool_calls = Arc::new(AtomicUsize::new(0));
        let client = ExpiredSessionClient {
            tool_calls: tool_calls.clone(),
        };
        let transport = StreamableHttpClientTransport::with_client(
            client,
            http_transport_config("https://example.invalid/mcp", 1024),
        );
        let service = ().serve(transport).await.unwrap();

        let result = service
            .peer()
            .call_tool(CallToolRequestParams::new("side_effect"))
            .await;

        assert!(result.is_err());
        assert_eq!(tool_calls.load(Ordering::SeqCst), 1);
        let _ = service.cancel().await;
    }

    #[tokio::test]
    async fn json_success_is_rejected_from_content_length_before_decode() {
        let body = serde_json::to_vec(&serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": { "value": "x".repeat(128) }
        }))
        .unwrap();
        let uri = serve_once(fixed_response(200, JSON_MIME_TYPE, &body)).await;
        let client = BoundedHttpClient::new(64).unwrap();

        let error = client
            .post_message(uri, ping(), None, None, HashMap::new())
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            StreamableHttpError::Client(McpHttpClientError::MessageTooLarge { maximum: 64 })
        ));
    }

    #[tokio::test]
    async fn chunked_json_rpc_error_is_rejected_before_decode() {
        let body = serde_json::to_vec(&serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "error": { "code": -32603, "message": "x".repeat(128) }
        }))
        .unwrap();
        let midpoint = body.len() / 2;
        let uri = serve_once(chunked_response(
            500,
            JSON_MIME_TYPE,
            &[&body[..midpoint], &body[midpoint..]],
        ))
        .await;
        let client = BoundedHttpClient::new(64).unwrap();

        let error = client
            .post_message(uri, ping(), None, None, HashMap::new())
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            StreamableHttpError::Client(McpHttpClientError::MessageTooLarge { maximum: 64 })
        ));
    }

    #[tokio::test]
    async fn plain_error_body_is_rejected_at_the_raw_limit() {
        let body = b"private-error-body".repeat(16);
        let uri = serve_once(fixed_response(500, "text/plain", &body)).await;
        let client = BoundedHttpClient::new(64).unwrap();

        let error = client
            .post_message(uri, ping(), None, None, HashMap::new())
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            StreamableHttpError::Client(McpHttpClientError::MessageTooLarge { maximum: 64 })
        ));
    }

    #[tokio::test]
    async fn sse_event_is_rejected_before_protocol_decode() {
        let body = format!(
            "data: {}\ndata: {}\ndata: {}\n\n",
            "x".repeat(24),
            "y".repeat(24),
            "z".repeat(24),
        );
        let uri = serve_once(fixed_response(200, EVENT_STREAM_MIME_TYPE, body.as_bytes())).await;
        let client = BoundedHttpClient::new(64).unwrap();

        let response = client
            .post_message(uri, ping(), None, None, HashMap::new())
            .await
            .unwrap();
        let StreamableHttpPostResponse::Sse(mut stream, _) = response else {
            panic!("expected an SSE response");
        };
        let error = stream.next().await.unwrap().unwrap_err();

        assert!(error.to_string().contains("event limit"));
    }

    #[tokio::test]
    async fn http_error_details_are_sensitive_by_default() {
        let base = serve_once(fixed_response(500, "text/plain", b"body-canary")).await;
        let uri: Arc<str> = Arc::from(format!("{base}?token=query-canary"));
        let expected_endpoint = uri.to_string();
        let source = BoundedHttpClient::new(64)
            .unwrap()
            .post_message(uri, ping(), None, None, HashMap::new())
            .await
            .unwrap_err();
        let details = sensitive_details(&source);
        let error = crate::McpError::Connect(
            siumai_core::Error::new(siumai_core::ErrorKind::Transport, "MCP HTTP request failed")
                .with_source(crate::error::McpBackendSource::new(source, details)),
        );

        let debug = format!("{error:?}");
        let display = error.to_string();
        assert!(!debug.contains("query-canary"));
        assert!(!debug.contains("body-canary"));
        assert!(!display.contains("query-canary"));
        assert!(!display.contains("body-canary"));
        assert_eq!(error.sensitive_endpoint(), Some(expected_endpoint.as_str()));
        assert_eq!(
            error.sensitive_response_body(),
            Some(b"body-canary".as_slice())
        );
    }

    #[tokio::test]
    async fn auth_challenge_requires_explicit_sensitive_source_traversal() {
        let response = b"HTTP/1.1 401 Unauthorized\r\nWWW-Authenticate: Bearer realm=\"auth-challenge-canary\"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec();
        let uri = serve_once(response).await;
        let source = BoundedHttpClient::new(64)
            .unwrap()
            .post_message(uri, ping(), None, None, HashMap::new())
            .await
            .unwrap_err();
        let details = sensitive_details(&source);
        let error = crate::McpError::Connect(
            siumai_core::Error::new(
                siumai_core::ErrorKind::Transport,
                "MCP HTTP authentication failed",
            )
            .with_source(crate::error::McpBackendSource::new(source, details)),
        );
        let default_chain = std::iter::successors(
            Some(&error as &(dyn std::error::Error + 'static)),
            |error| error.source(),
        )
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(" | ");
        let explicit_source =
            error.sensitive_source().unwrap().expose() as &(dyn std::error::Error + 'static);
        let explicit_chain = std::iter::successors(Some(explicit_source), |error| error.source())
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" | ");

        assert!(!default_chain.contains("auth-challenge-canary"));
        assert!(explicit_chain.contains("auth-challenge-canary"));
        assert_eq!(
            error.sensitive_auth_challenge(),
            Some("Bearer realm=\"auth-challenge-canary\"")
        );
    }
}
