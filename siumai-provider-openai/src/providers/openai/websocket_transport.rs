//! OpenAI WebSocket transport for Responses streaming.
//!
//! This implements Siumai's `HttpTransport` abstraction and mirrors the Vercel AI SDK pattern:
//! a drop-in "fetch/transport" replacement that only reroutes streaming `POST /responses`
//! requests through OpenAI's WebSocket mode. All other requests fall back to standard HTTP.
//!
//! Design notes:
//! - We intentionally emit SSE-framed `data: <json>\n\n` chunks from WebSocket messages so we can
//!   reuse the existing Responses SSE converter (`OpenAiResponsesEventConverter`) unchanged.
//! - The WebSocket connection is kept open for reuse across sequential requests. If concurrent
//!   streams happen, the transport may open additional connections (best-effort reuse).

use crate::error::LlmError;
use crate::execution::http::transport::{
    HttpTransport, HttpTransportRequest, HttpTransportResponse, HttpTransportStreamBody,
    HttpTransportStreamResponse,
};
use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use reqwest::header::{HeaderMap, HeaderValue};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http;
use tokio_tungstenite::tungstenite::protocol::Message;

type WsStream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

const OPENAI_WS_BETA_HEADER_VALUE: &str = "responses_websockets=2026-02-06";

#[derive(Debug, Default)]
struct WsResponsesEventMeta {
    terminal: bool,
    cacheable: bool,
    completed_response_id: Option<String>,
}

#[derive(Debug)]
struct CachedWs {
    ws: WsStream,
    last_completed_response_id: Option<String>,
    created_at: Instant,
    last_used: Instant,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct WsCacheKey {
    ws_url: String,
    headers_hash: u64,
}

#[derive(Default)]
struct WsState {
    /// Idle WebSocket connections (per key) for reuse across sequential streams.
    idle: HashMap<WsCacheKey, Vec<CachedWs>>,
}

/// OpenAI WebSocket transport.
///
/// This transport:
/// - Uses HTTP for non-streaming requests.
/// - For streaming Responses API requests (`/responses` with `stream: true`),
///   uses OpenAI WebSocket mode and emits SSE-framed chunks.
#[derive(Clone)]
pub struct OpenAiWebSocketTransport {
    http: reqwest::Client,
    state: Arc<Mutex<WsState>>,
    max_idle_connections: usize,
    stateful_previous_response_id: bool,
    emit_done_marker: bool,
    max_connection_age: Option<Duration>,
    idle_ttl: Option<Duration>,
}

impl OpenAiWebSocketTransport {
    /// Create a new transport using the provided HTTP client for fallback requests.
    pub fn new(http: reqwest::Client) -> Self {
        Self {
            http,
            state: Arc::new(Mutex::new(WsState::default())),
            max_idle_connections: 1,
            stateful_previous_response_id: false,
            emit_done_marker: false,
            // OpenAI WebSocket connections are time-limited; proactively avoid reusing old connections.
            max_connection_age: Some(Duration::from_secs(55 * 60)),
            idle_ttl: None,
        }
    }

    /// Configure how many idle connections to keep per `(ws_url, auth headers)` key.
    pub fn with_max_idle_connections(mut self, max_idle_connections: usize) -> Self {
        self.max_idle_connections = max_idle_connections;
        self
    }

    /// Enable stateful incremental continuation by auto-injecting `previous_response_id`
    /// for streaming `/responses` requests when the field is not already present.
    pub fn with_stateful_previous_response_id(mut self, enabled: bool) -> Self {
        self.stateful_previous_response_id = enabled;
        self
    }

    /// Emit an extra SSE done marker (`data: [DONE]\n\n`) after a terminal WebSocket event.
    ///
    /// OpenAI Responses WebSocket mode terminates via JSON events like `response.completed` / `error`.
    /// Emitting a `[DONE]` marker is optional and only needed for compatibility with SSE consumers
    /// that expect Chat Completions-style done semantics.
    pub fn with_emit_done_marker(mut self, enabled: bool) -> Self {
        self.emit_done_marker = enabled;
        self
    }

    /// Set a maximum connection age; cached connections older than this are not reused.
    pub fn with_max_connection_age(mut self, age: Duration) -> Self {
        self.max_connection_age = Some(age);
        self
    }

    /// Disable maximum connection age checks (not recommended for long-running agents).
    pub fn without_max_connection_age(mut self) -> Self {
        self.max_connection_age = None;
        self
    }

    /// Set an idle TTL; cached connections unused longer than this are not reused.
    pub fn with_idle_ttl(mut self, ttl: Duration) -> Self {
        self.idle_ttl = Some(ttl);
        self
    }

    /// Disable idle TTL checks.
    pub fn without_idle_ttl(mut self) -> Self {
        self.idle_ttl = None;
        self
    }

    /// Close the cached WebSocket connection, if any.
    pub async fn close(&self) {
        let mut st = self.state.lock().await;
        st.idle.clear();
    }

    fn is_stale(&self, c: &CachedWs, now: Instant) -> bool {
        if let Some(max_age) = self.max_connection_age
            && now.duration_since(c.created_at) >= max_age
        {
            return true;
        }
        if let Some(ttl) = self.idle_ttl
            && now.duration_since(c.last_used) >= ttl
        {
            return true;
        }
        false
    }

    fn is_responses_stream_request(request: &HttpTransportRequest) -> bool {
        let url = request.url.trim_end();
        if !url.ends_with("/responses") && !url.contains("/responses?") {
            return false;
        }
        request
            .body
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }

    fn model_from_body(request: &HttpTransportRequest) -> Option<String> {
        request
            .body
            .get("model")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    fn to_ws_url(http_url: &str) -> Result<String, LlmError> {
        let u = http_url.trim_end();
        let (scheme, rest) = if let Some(r) = u.strip_prefix("https://") {
            ("wss://", r)
        } else if let Some(r) = u.strip_prefix("http://") {
            ("ws://", r)
        } else if let Some(r) = u.strip_prefix("wss://") {
            ("wss://", r)
        } else if let Some(r) = u.strip_prefix("ws://") {
            ("ws://", r)
        } else {
            return Err(LlmError::ProviderError {
                provider: "openai".to_string(),
                message: format!("Unsupported URL scheme for WebSocket transport: {http_url}"),
                error_code: Some("websocket_unsupported_url_scheme".to_string()),
            });
        };

        Ok(format!("{scheme}{rest}"))
    }

    fn build_ws_handshake_request(
        ws_url: &str,
        base_headers: &HeaderMap,
    ) -> Result<http::Request<()>, LlmError> {
        let mut req =
            ws_url
                .to_string()
                .into_client_request()
                .map_err(|e| LlmError::ProviderError {
                    provider: "openai".to_string(),
                    message: format!("Invalid WebSocket URL: {e}"),
                    error_code: Some("websocket_invalid_url".to_string()),
                })?;

        // Copy relevant headers from the original HTTP request.
        //
        // Filter out HTTP/SSE-only headers that could interfere with the WS handshake.
        for (name, v) in base_headers.iter() {
            let name_lc = name.as_str().to_ascii_lowercase();
            if matches!(
                name_lc.as_str(),
                "accept"
                    | "content-type"
                    | "content-length"
                    | "cache-control"
                    | "connection"
                    | "keep-alive"
                    | "transfer-encoding"
                    | "accept-encoding"
            ) {
                continue;
            }
            req.headers_mut().insert(name, v.clone());
        }

        // Match OpenAI's official WebSocket mode requirement.
        if req.headers().get("openai-beta").is_none() {
            req.headers_mut().insert(
                reqwest::header::HeaderName::from_static("openai-beta"),
                reqwest::header::HeaderValue::from_static(OPENAI_WS_BETA_HEADER_VALUE),
            );
        }

        Ok(req)
    }

    async fn connect_ws(ws_url: &str, headers: &HeaderMap) -> Result<WsStream, LlmError> {
        let req = Self::build_ws_handshake_request(ws_url, headers)?;
        let (ws, _resp) =
            tokio_tungstenite::connect_async(req)
                .await
                .map_err(|e| LlmError::ProviderError {
                    provider: "openai".to_string(),
                    message: format!("WebSocket connect failed: {e}"),
                    error_code: Some("websocket_connect_failed".to_string()),
                })?;
        Ok(ws)
    }

    async fn take_or_connect_ws(
        &self,
        key: &WsCacheKey,
        headers: &HeaderMap,
    ) -> Result<CachedWs, LlmError> {
        let now = Instant::now();
        loop {
            let cached = {
                let mut st = self.state.lock().await;
                st.idle.get_mut(key).and_then(|v| v.pop())
            };

            if let Some(mut cached_ws) = cached {
                if self.is_stale(&cached_ws, now) {
                    // Best-effort close; ignore errors.
                    let _ = cached_ws.ws.close(None).await;
                    continue;
                }
                cached_ws.last_used = now;
                return Ok(cached_ws);
            }
            break;
        }

        let ws = Self::connect_ws(&key.ws_url, headers).await?;
        Ok(CachedWs {
            ws,
            last_completed_response_id: None,
            created_at: now,
            last_used: now,
        })
    }

    fn inspect_ws_responses_event(json_text: &str) -> WsResponsesEventMeta {
        let mut meta = WsResponsesEventMeta {
            terminal: false,
            cacheable: true,
            completed_response_id: None,
        };

        let Ok(v) = serde_json::from_str::<serde_json::Value>(json_text) else {
            return meta;
        };
        let Some(t) = v.get("type").and_then(|x| x.as_str()) else {
            return meta;
        };

        meta.terminal = matches!(
            t,
            "error"
                | "response.completed"
                | "response.incomplete"
                | "response.failed"
                | "response.canceled"
                | "response.cancelled"
        );

        // Conservatively avoid reusing connections that emitted `error` events, because
        // server-side cache semantics become ambiguous.
        if t == "error" {
            meta.cacheable = false;
        }

        if t == "response.completed" {
            meta.completed_response_id = v
                .get("response")
                .and_then(|r| r.get("id"))
                .and_then(|id| id.as_str())
                .map(|s| s.to_string());
        }

        meta
    }

    fn hash_ws_headers(headers: &HeaderMap) -> u64 {
        use std::hash::{Hash, Hasher};

        fn pick<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
            headers.get(name).and_then(|v| v.to_str().ok())
        }

        let mut h = std::collections::hash_map::DefaultHasher::new();
        pick(headers, "authorization").hash(&mut h);
        pick(headers, "openai-beta")
            .unwrap_or(OPENAI_WS_BETA_HEADER_VALUE)
            .hash(&mut h);
        pick(headers, "openai-organization").hash(&mut h);
        pick(headers, "openai-project").hash(&mut h);
        h.finish()
    }

    fn sse_data_frame(json_text: &str) -> Vec<u8> {
        // Minimal SSE framing: each WebSocket message becomes one SSE `data:` payload.
        // The Responses SSE converter expects JSON in `event.data`.
        let mut out = Vec::with_capacity(json_text.len() + 16);
        out.extend_from_slice(b"data: ");
        out.extend_from_slice(json_text.as_bytes());
        out.extend_from_slice(b"\n\n");
        out
    }

    fn sse_done_frame() -> Vec<u8> {
        b"data: [DONE]\n\n".to_vec()
    }

    fn build_response_create_message(
        body: serde_json::Value,
    ) -> Result<serde_json::Value, LlmError> {
        // The caller set `stream: true` to indicate streaming; WebSocket mode is inherently
        // streaming, so remove `stream` to match the official WS request shape.
        // `background` is also transport-specific and not used in WebSocket mode.
        let serde_json::Value::Object(mut obj) = body else {
            return Err(LlmError::InvalidParameter(
                "WebSocket transport expects JSON object body for response.create".to_string(),
            ));
        };
        obj.remove("stream");
        obj.remove("background");

        // WebSocket mode uses `{ "type": "response.create", ...create_body_fields }`.
        let mut out = serde_json::Map::new();
        out.insert(
            "type".to_string(),
            serde_json::Value::String("response.create".to_string()),
        );
        for (k, v) in obj {
            if k == "type" {
                continue;
            }
            out.insert(k, v);
        }
        Ok(serde_json::Value::Object(out))
    }
}

impl Default for OpenAiWebSocketTransport {
    fn default() -> Self {
        Self::new(reqwest::Client::new())
    }
}

#[async_trait]
impl HttpTransport for OpenAiWebSocketTransport {
    async fn execute_json(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        let resp = self
            .http
            .post(&request.url)
            .headers(request.headers)
            .json(&request.body)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("HTTP request failed: {e}")))?;

        let status = resp.status().as_u16();
        let headers = resp.headers().clone();
        let body = resp
            .bytes()
            .await
            .map_err(|e| LlmError::HttpError(format!("HTTP body read failed: {e}")))?
            .to_vec();

        Ok(HttpTransportResponse {
            status,
            headers,
            body,
        })
    }

    async fn execute_stream(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportStreamResponse, LlmError> {
        // Only reroute streaming Responses requests. Everything else falls back to HTTP streaming.
        if !Self::is_responses_stream_request(&request) {
            let resp = self
                .http
                .post(&request.url)
                .headers(request.headers)
                .json(&request.body)
                .send()
                .await
                .map_err(|e| LlmError::HttpError(format!("HTTP stream request failed: {e}")))?;

            let status = resp.status().as_u16();
            let headers = resp.headers().clone();
            let byte_stream = resp.bytes_stream().map(|r| {
                r.map(|b| b.to_vec())
                    .map_err(|e| LlmError::StreamError(format!("HTTP byte stream error: {e}")))
            });

            return Ok(HttpTransportStreamResponse {
                status,
                headers,
                body: HttpTransportStreamBody::from_stream(byte_stream),
            });
        }

        let _model = Self::model_from_body(&request).ok_or_else(|| {
            LlmError::InvalidParameter(
                "OpenAI WebSocket transport requires `model` in the request body".to_string(),
            )
        })?;
        let ws_url = Self::to_ws_url(&request.url)?;

        let key = WsCacheKey {
            ws_url: ws_url.clone(),
            headers_hash: Self::hash_ws_headers(&request.headers),
        };

        let cached_ws = self.take_or_connect_ws(&key, &request.headers).await?;
        let ws_created_at = cached_ws.created_at;

        let mut body = request.body;
        if self.stateful_previous_response_id
            && let serde_json::Value::Object(obj) = &mut body
            && obj.get("previous_response_id").is_none()
            && let Some(prev) = cached_ws.last_completed_response_id.clone()
        {
            obj.insert(
                "previous_response_id".to_string(),
                serde_json::Value::String(prev),
            );
        }

        let ws = cached_ws.ws;
        let create_msg = Self::build_response_create_message(body)?;
        let create_text = serde_json::to_string(&create_msg).map_err(|e| {
            LlmError::InvalidParameter(format!("Failed to serialize response.create: {e}"))
        })?;

        let state = self.state.clone();
        let max_idle = self.max_idle_connections;
        let stateful_previous = self.stateful_previous_response_id;
        let emit_done_marker = self.emit_done_marker;
        let key_for_state = key.clone();
        let ws_created_at_for_cache = ws_created_at;
        let max_connection_age = self.max_connection_age;
        let idle_ttl = self.idle_ttl;

        let mut headers = HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        );

        let out_stream = async_stream::stream! {
            let mut ws = ws;
            // Send response.create
            if let Err(e) = ws.send(Message::Text(create_text.into())).await {
                yield Err(LlmError::ProviderError {
                    provider: "openai".to_string(),
                    message: format!("WebSocket send failed: {e}"),
                    error_code: Some("websocket_send_failed".to_string()),
                });
                return;
            }

            let mut terminal_seen = false;
            let mut cacheable = true;
            let mut completed_id: Option<String> = None;
            while let Some(msg) = ws.next().await {
                let msg = match msg {
                    Ok(m) => m,
                    Err(e) => {
                        yield Err(LlmError::ProviderError {
                            provider: "openai".to_string(),
                            message: format!("WebSocket recv failed: {e}"),
                            error_code: Some("websocket_recv_failed".to_string()),
                        });
                        return;
                    }
                };

                match msg {
                    Message::Text(text) => {
                        let meta = Self::inspect_ws_responses_event(&text);
                        terminal_seen = meta.terminal;
                        if completed_id.is_none() {
                            completed_id = meta.completed_response_id;
                        }
                        if !meta.cacheable {
                            cacheable = false;
                        }
                        yield Ok(Self::sse_data_frame(&text));
                        if terminal_seen {
                            break;
                        }
                    }
                    Message::Binary(bin) => {
                        match String::from_utf8(bin.to_vec()) {
                            Ok(text) => {
                                let meta = Self::inspect_ws_responses_event(&text);
                                terminal_seen = meta.terminal;
                                if completed_id.is_none() {
                                    completed_id = meta.completed_response_id;
                                }
                                if !meta.cacheable {
                                    cacheable = false;
                                }
                                yield Ok(Self::sse_data_frame(&text));
                                if terminal_seen {
                                    break;
                                }
                            }
                            Err(e) => {
                                yield Err(LlmError::ParseError(format!("WebSocket binary frame is not valid UTF-8: {e}")));
                                return;
                            }
                        }
                    }
                    Message::Ping(payload) => {
                        if let Err(e) = ws.send(Message::Pong(payload)).await {
                            yield Err(LlmError::ProviderError {
                                provider: "openai".to_string(),
                                message: format!("WebSocket pong send failed: {e}"),
                                error_code: Some("websocket_pong_send_failed".to_string()),
                            });
                            return;
                        }
                    }
                    Message::Pong(_) => {}
                    Message::Close(_) => {
                        break;
                    }
                    Message::Frame(_) => {}
                }
            }

            // Cache the connection for reuse when we saw a clean terminal event.
            if terminal_seen && cacheable && max_idle > 0 {
                let mut st = state.lock().await;
                let v = st.idle.entry(key_for_state.clone()).or_default();
                let now = Instant::now();

                // Prune stale cached connections to keep the pool healthy.
                v.retain(|c| {
                    let stale_age = max_connection_age.is_some_and(|d| now.duration_since(c.created_at) >= d);
                    let stale_idle = idle_ttl.is_some_and(|d| now.duration_since(c.last_used) >= d);
                    !(stale_age || stale_idle)
                });

                if v.len() < max_idle {
                    v.push(CachedWs {
                        ws,
                        last_completed_response_id: if stateful_previous {
                            completed_id.clone()
                        } else {
                            None
                        },
                        created_at: ws_created_at_for_cache,
                        last_used: now,
                    });
                }
            }

            if terminal_seen && emit_done_marker {
                yield Ok(Self::sse_done_frame());
            }
        };

        Ok(HttpTransportStreamResponse {
            status: 200,
            headers,
            body: HttpTransportStreamBody::from_stream(out_stream),
        })
    }
}

#[cfg(test)]
#[cfg(feature = "openai-websocket")]
mod tests {
    use super::*;
    use crate::execution::http::interceptor::HttpRequestContext;
    use crate::streaming::StreamFactory;
    use futures_util::TryStreamExt;
    use std::time::Duration;
    use tokio::net::TcpListener;
    use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};

    fn ctx(url: &str) -> HttpRequestContext {
        HttpRequestContext {
            request_id: "test".to_string(),
            provider_id: "openai".to_string(),
            url: url.to_string(),
            stream: true,
        }
    }

    #[tokio::test]
    async fn websocket_stream_emits_sse_framed_json() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let http_url = format!("http://{addr}/v1/responses");

        let server = tokio::spawn(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let mut ws =
                tokio_tungstenite::accept_hdr_async(tcp, |req: &Request, resp: Response| {
                    // Ensure Authorization is present (copied from the original request headers).
                    assert_eq!(
                        req.headers()
                            .get("authorization")
                            .and_then(|v| v.to_str().ok()),
                        Some("Bearer test")
                    );
                    assert_eq!(
                        req.headers()
                            .get("openai-beta")
                            .and_then(|v| v.to_str().ok()),
                        Some(OPENAI_WS_BETA_HEADER_VALUE)
                    );
                    Ok(resp)
                })
                .await
                .unwrap();

            // Receive response.create
            let first = ws.next().await.unwrap().unwrap();
            let Message::Text(txt) = first else {
                panic!("expected text message");
            };
            let v: serde_json::Value = serde_json::from_str(&txt).unwrap();
            assert_eq!(v["type"], "response.create");
            assert_eq!(v["model"], "gpt-test");
            // Ensure `stream` was stripped.
            assert!(v.get("stream").is_none());

            // Emit a minimal Responses event sequence.
            ws.send(Message::Text(
                serde_json::json!({
                    "type": "response.created",
                    "response": { "id": "resp_1", "model": "gpt-test", "created_at": 0 }
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
            ws.send(Message::Text(
                serde_json::json!({
                    "type": "response.output_text.delta",
                    "delta": "hi"
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
            ws.send(Message::Text(
                serde_json::json!({
                    "type": "response.completed",
                    "response": { "id": "resp_1", "model": "gpt-test", "created_at": 0, "output": [] }
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
        });

        let transport = OpenAiWebSocketTransport::new(reqwest::Client::new());
        let mut headers = HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            HeaderValue::from_static("Bearer test"),
        );

        let request = HttpTransportRequest {
            ctx: ctx(&http_url),
            url: http_url,
            headers,
            body: serde_json::json!({
                "model": "gpt-test",
                "stream": true,
                "input": "hello"
            }),
        };

        let resp = transport.execute_stream(request).await.unwrap();
        assert_eq!(resp.status, 200);
        assert!(
            resp.headers
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .contains("text/event-stream")
        );

        let raw = resp.body.into_stream().try_concat().await.unwrap();
        let raw_text = String::from_utf8_lossy(&raw).to_string();
        assert!(!raw_text.contains("data: [DONE]\n\n"));

        // Ensure the returned byte stream can be consumed by the standard Responses SSE converter.
        let converter =
            crate::standards::openai::responses_sse::OpenAiResponsesEventConverter::new();
        let chat_stream = StreamFactory::stream_from_byte_stream_with_sse_fallback(
            resp.headers,
            futures_util::stream::iter([Ok(raw)]),
            converter,
        )
        .await
        .unwrap();

        let events: Vec<_> = chat_stream.try_collect().await.unwrap();
        assert!(events.iter().any(|e| matches!(e, crate::types::ChatStreamEvent::ContentDelta { delta, .. } if delta == "hi")));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, crate::types::ChatStreamEvent::StreamEnd { .. }))
        );

        server.await.unwrap();
    }

    #[tokio::test]
    async fn websocket_stream_emits_optional_done_marker() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let http_url = format!("http://{addr}/v1/responses");

        let server = tokio::spawn(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let mut ws =
                tokio_tungstenite::accept_hdr_async(tcp, |req: &Request, resp: Response| {
                    assert_eq!(
                        req.headers()
                            .get("authorization")
                            .and_then(|v| v.to_str().ok()),
                        Some("Bearer test")
                    );
                    assert_eq!(
                        req.headers()
                            .get("openai-beta")
                            .and_then(|v| v.to_str().ok()),
                        Some(OPENAI_WS_BETA_HEADER_VALUE)
                    );
                    Ok(resp)
                })
                .await
                .unwrap();

            let first = ws.next().await.unwrap().unwrap();
            let Message::Text(_txt) = first else {
                panic!("expected text message");
            };

            ws.send(Message::Text(
                serde_json::json!({
                    "type": "response.created",
                    "response": { "id": "resp_1", "model": "gpt-test", "created_at": 0 }
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
            ws.send(Message::Text(
                serde_json::json!({
                    "type": "response.completed",
                    "response": { "id": "resp_1", "model": "gpt-test", "created_at": 0, "output": [] }
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
        });

        let transport =
            OpenAiWebSocketTransport::new(reqwest::Client::new()).with_emit_done_marker(true);
        let mut headers = HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            HeaderValue::from_static("Bearer test"),
        );

        let request = HttpTransportRequest {
            ctx: ctx(&http_url),
            url: http_url,
            headers,
            body: serde_json::json!({
                "model": "gpt-test",
                "stream": true,
                "input": "hello"
            }),
        };

        let resp = transport.execute_stream(request).await.unwrap();
        let raw = resp.body.into_stream().try_concat().await.unwrap();
        let raw_text = String::from_utf8_lossy(&raw).to_string();
        assert!(raw_text.contains("data: [DONE]\n\n"));

        server.await.unwrap();
    }

    #[tokio::test]
    async fn websocket_stream_stateful_previous_response_id_injected() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let http_url = format!("http://{addr}/v1/responses");

        let server = tokio::spawn(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let mut ws =
                tokio_tungstenite::accept_hdr_async(tcp, |req: &Request, resp: Response| {
                    assert_eq!(
                        req.headers()
                            .get("authorization")
                            .and_then(|v| v.to_str().ok()),
                        Some("Bearer test")
                    );
                    assert_eq!(
                        req.headers()
                            .get("openai-beta")
                            .and_then(|v| v.to_str().ok()),
                        Some(OPENAI_WS_BETA_HEADER_VALUE)
                    );
                    Ok(resp)
                })
                .await
                .unwrap();

            // First response.create should not include previous_response_id.
            let first = ws.next().await.unwrap().unwrap();
            let Message::Text(txt1) = first else {
                panic!("expected text message");
            };
            let v1: serde_json::Value = serde_json::from_str(&txt1).unwrap();
            assert_eq!(v1["type"], "response.create");
            assert!(v1.get("previous_response_id").is_none());

            ws.send(Message::Text(
                serde_json::json!({
                    "type": "response.created",
                    "response": { "id": "resp_1", "model": "gpt-test", "created_at": 0 }
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
            ws.send(Message::Text(
                serde_json::json!({
                    "type": "response.completed",
                    "response": { "id": "resp_1", "model": "gpt-test", "created_at": 0, "output": [] }
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();

            // Second response.create should include previous_response_id=resp_1.
            let second = ws.next().await.unwrap().unwrap();
            let Message::Text(txt2) = second else {
                panic!("expected text message");
            };
            let v2: serde_json::Value = serde_json::from_str(&txt2).unwrap();
            assert_eq!(v2["type"], "response.create");
            assert_eq!(v2["previous_response_id"], "resp_1");

            ws.send(Message::Text(
                serde_json::json!({
                    "type": "response.created",
                    "response": { "id": "resp_2", "model": "gpt-test", "created_at": 0 }
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
            ws.send(Message::Text(
                serde_json::json!({
                    "type": "response.completed",
                    "response": { "id": "resp_2", "model": "gpt-test", "created_at": 0, "output": [] }
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
        });

        let transport = OpenAiWebSocketTransport::new(reqwest::Client::new())
            .with_stateful_previous_response_id(true);

        let mut headers = HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            HeaderValue::from_static("Bearer test"),
        );

        let mk_req = |request_id: &str| HttpTransportRequest {
            ctx: HttpRequestContext {
                request_id: request_id.to_string(),
                provider_id: "openai".to_string(),
                url: http_url.clone(),
                stream: true,
            },
            url: http_url.clone(),
            headers: headers.clone(),
            body: serde_json::json!({
                "model": "gpt-test",
                "stream": true,
                "input": "hello"
            }),
        };

        // Drive the returned stream to completion so the connection can be returned to the pool
        // and the completed response id can be recorded.
        let r1 = transport.execute_stream(mk_req("req_1")).await.unwrap();
        let _: Vec<Vec<u8>> = r1.body.into_stream().try_collect().await.unwrap();

        let r2 = transport.execute_stream(mk_req("req_2")).await.unwrap();
        let _: Vec<Vec<u8>> = r2.body.into_stream().try_collect().await.unwrap();

        server.await.unwrap();
    }

    #[tokio::test]
    async fn websocket_stream_does_not_reuse_stale_connection() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let http_url = format!("http://{addr}/v1/responses");

        let server = tokio::spawn(async move {
            for _ in 0..2 {
                let (tcp, _) = tokio::time::timeout(Duration::from_secs(2), listener.accept())
                    .await
                    .expect("timed out waiting for ws connection")
                    .unwrap();
                let mut ws =
                    tokio_tungstenite::accept_hdr_async(tcp, |_req: &Request, resp: Response| {
                        Ok(resp)
                    })
                    .await
                    .unwrap();

                let first = tokio::time::timeout(Duration::from_secs(2), ws.next())
                    .await
                    .expect("timed out waiting for response.create")
                    .unwrap()
                    .unwrap();
                let Message::Text(txt) = first else {
                    panic!("expected text message");
                };
                let v: serde_json::Value = serde_json::from_str(&txt).unwrap();
                assert_eq!(v["type"], "response.create");
                // If the connection is not reused, no previous_response_id is injected.
                assert!(v.get("previous_response_id").is_none());

                ws.send(Message::Text(
                    serde_json::json!({
                        "type": "response.created",
                        "response": { "id": "resp_1", "model": "gpt-test", "created_at": 0 }
                    })
                    .to_string()
                    .into(),
                ))
                .await
                .unwrap();
                ws.send(Message::Text(
                    serde_json::json!({
                        "type": "response.completed",
                        "response": { "id": "resp_1", "model": "gpt-test", "created_at": 0, "output": [] }
                    })
                    .to_string()
                    .into(),
                ))
                .await
                .unwrap();
            }
        });

        let transport = OpenAiWebSocketTransport::new(reqwest::Client::new())
            .with_max_idle_connections(1)
            .with_stateful_previous_response_id(true)
            // Always treat cached connections as stale to force a reconnect.
            .with_max_connection_age(Duration::from_secs(0));

        let mut headers = HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            HeaderValue::from_static("Bearer test"),
        );

        let mk_req = |request_id: &str| HttpTransportRequest {
            ctx: HttpRequestContext {
                request_id: request_id.to_string(),
                provider_id: "openai".to_string(),
                url: http_url.clone(),
                stream: true,
            },
            url: http_url.clone(),
            headers: headers.clone(),
            body: serde_json::json!({
                "model": "gpt-test",
                "stream": true,
                "input": "hello"
            }),
        };

        let r1 = transport.execute_stream(mk_req("req_1")).await.unwrap();
        let _: Vec<Vec<u8>> = r1.body.into_stream().try_collect().await.unwrap();

        let r2 = transport.execute_stream(mk_req("req_2")).await.unwrap();
        let _: Vec<Vec<u8>> = r2.body.into_stream().try_collect().await.unwrap();

        server.await.unwrap();
    }
}
