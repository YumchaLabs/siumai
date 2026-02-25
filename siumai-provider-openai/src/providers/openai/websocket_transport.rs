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
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http;
use tokio_tungstenite::tungstenite::protocol::Message;

type WsStream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

#[derive(Debug)]
struct CachedWs {
    ws: WsStream,
    last_completed_response_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct WsCacheKey {
    ws_url: String,
    headers_hash: u64,
}

struct WsState {
    /// Idle WebSocket connections (per key) for reuse across sequential streams.
    idle: HashMap<WsCacheKey, Vec<CachedWs>>,
}

impl Default for WsState {
    fn default() -> Self {
        Self {
            idle: HashMap::new(),
        }
    }
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
}

impl OpenAiWebSocketTransport {
    /// Create a new transport using the provided HTTP client for fallback requests.
    pub fn new(http: reqwest::Client) -> Self {
        Self {
            http,
            state: Arc::new(Mutex::new(WsState::default())),
            max_idle_connections: 1,
            stateful_previous_response_id: false,
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

    /// Close the cached WebSocket connection, if any.
    pub async fn close(&self) {
        let mut st = self.state.lock().await;
        st.idle.clear();
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
        let cached = {
            let mut st = self.state.lock().await;
            st.idle.get_mut(key).and_then(|v| v.pop())
        };

        if let Some(cached_ws) = cached {
            return Ok(cached_ws);
        }

        Ok(CachedWs {
            ws: Self::connect_ws(&key.ws_url, headers).await?,
            last_completed_response_id: None,
        })
    }

    fn completed_response_id_from_event(json_text: &str) -> Option<String> {
        let v: serde_json::Value = serde_json::from_str(json_text).ok()?;
        let ty = v.get("type")?.as_str()?;
        if ty != "response.completed" {
            return None;
        }
        v.get("response")?
            .get("id")?
            .as_str()
            .map(|s| s.to_string())
    }

    fn hash_ws_headers(headers: &HeaderMap) -> u64 {
        use std::hash::{Hash, Hasher};

        fn pick<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
            headers.get(name).and_then(|v| v.to_str().ok())
        }

        let mut h = std::collections::hash_map::DefaultHasher::new();
        pick(headers, "authorization").hash(&mut h);
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

    fn is_terminal_responses_event(json_text: &str) -> bool {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(json_text) else {
            return false;
        };
        let Some(t) = v.get("type").and_then(|x| x.as_str()) else {
            return false;
        };
        matches!(
            t,
            "error"
                | "response.completed"
                | "response.incomplete"
                | "response.failed"
                | "response.canceled"
                | "response.cancelled"
        )
    }

    fn is_connection_limit_error(json_text: &str) -> bool {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(json_text) else {
            return false;
        };
        if v.get("type").and_then(|x| x.as_str()) != Some("error") {
            return false;
        }
        v.get("error")
            .and_then(|e| e.get("code"))
            .and_then(|c| c.as_str())
            .is_some_and(|c| c == "websocket_connection_limit_reached")
    }

    fn is_error_event(json_text: &str) -> bool {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(json_text) else {
            return false;
        };
        v.get("type").and_then(|x| x.as_str()) == Some("error")
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

        let mut body = request.body;
        if self.stateful_previous_response_id {
            if let serde_json::Value::Object(obj) = &mut body {
                if obj.get("previous_response_id").is_none() {
                    if let Some(prev) = cached_ws.last_completed_response_id.clone() {
                        obj.insert(
                            "previous_response_id".to_string(),
                            serde_json::Value::String(prev),
                        );
                    }
                }
            }
        }

        let ws = cached_ws.ws;
        let create_msg = Self::build_response_create_message(body)?;
        let create_text = serde_json::to_string(&create_msg).map_err(|e| {
            LlmError::InvalidParameter(format!("Failed to serialize response.create: {e}"))
        })?;

        let state = self.state.clone();
        let max_idle = self.max_idle_connections;
        let stateful_previous = self.stateful_previous_response_id;
        let key_for_state = key.clone();

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
                        terminal_seen = Self::is_terminal_responses_event(&text);
                        if completed_id.is_none() {
                            completed_id = Self::completed_response_id_from_event(&text);
                        }
                        if Self::is_error_event(&text) || Self::is_connection_limit_error(&text) {
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
                                terminal_seen = Self::is_terminal_responses_event(&text);
                                if completed_id.is_none() {
                                    completed_id = Self::completed_response_id_from_event(&text);
                                }
                                if Self::is_error_event(&text) || Self::is_connection_limit_error(&text) {
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
                if v.len() < max_idle {
                    v.push(CachedWs {
                        ws,
                        last_completed_response_id: if stateful_previous {
                            completed_id.clone()
                        } else {
                            None
                        },
                    });
                }
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

        // Ensure the returned byte stream can be consumed by the standard Responses SSE converter.
        let converter =
            crate::standards::openai::responses_sse::OpenAiResponsesEventConverter::new();
        let chat_stream = StreamFactory::stream_from_byte_stream_with_sse_fallback(
            resp.headers,
            resp.body.into_stream(),
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
}
