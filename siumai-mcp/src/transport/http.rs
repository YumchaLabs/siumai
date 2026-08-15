use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::Arc;

use futures_util::StreamExt;
use futures_util::stream::BoxStream;
use http::header::{ACCEPT, CONTENT_TYPE, HeaderName, HeaderValue, WWW_AUTHENTICATE};
use rmcp::model::{ClientJsonRpcMessage, JsonRpcMessage, ServerJsonRpcMessage};
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::streamable_http_client::{
    AuthRequiredError, InsufficientScopeError, SseError, StreamableHttpClient,
    StreamableHttpClientTransportConfig, StreamableHttpError, StreamableHttpPostResponse,
};
use siumai_transport::TransportLimits;
use siumai_transport::framing::{SseDecoder, SseEvent};
use sse_stream::Sse;

use crate::error::McpSensitiveDetails;

const EVENT_STREAM_MIME_TYPE: &str = "text/event-stream";
const JSON_MIME_TYPE: &str = "application/json";
const HEADER_SESSION_ID: &str = "mcp-session-id";
const HEADER_LAST_EVENT_ID: &str = "last-event-id";

/// HTTP failure retained behind the explicit MCP sensitive-source boundary.
pub(crate) enum McpHttpClientError {
    Request(reqwest::Error),
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
    max_message_bytes: usize,
}

impl BoundedHttpClient {
    fn new(max_message_bytes: usize) -> Result<Self, McpHttpClientError> {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .referer(false)
            .no_proxy()
            .retry(reqwest::retry::never())
            .build()
            .map_err(McpHttpClientError::Request)?;
        Ok(Self {
            client,
            max_message_bytes,
        })
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
        let response = self
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
            .body(encoded)
            .send()
            .await
            .map_err(McpHttpClientError::Request)
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
        let response = self
            .request(
                reqwest::Method::DELETE,
                uri.as_ref(),
                Some(session_id.as_ref()),
                auth_token.as_deref(),
                None,
                custom_headers,
            )
            .send()
            .await
            .map_err(McpHttpClientError::Request)
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
        let response = self
            .request(
                reqwest::Method::GET,
                uri.as_ref(),
                session_id.as_deref(),
                auth_token.as_deref(),
                last_event_id.as_deref(),
                custom_headers,
            )
            .header(ACCEPT, [EVENT_STREAM_MIME_TYPE, JSON_MIME_TYPE].join(", "))
            .send()
            .await
            .map_err(McpHttpClientError::Request)
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
) -> Result<StreamableHttpClientTransport<BoundedHttpClient>, McpHttpClientError> {
    let client = BoundedHttpClient::new(max_message_bytes)?;
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use rmcp::model::{CallToolRequestParams, ClientRequest, PingRequest, RequestId};
    use rmcp::service::ServiceExt;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    fn ping() -> ClientJsonRpcMessage {
        ClientJsonRpcMessage::request(
            ClientRequest::PingRequest(PingRequest::default()),
            RequestId::Number(1),
        )
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
