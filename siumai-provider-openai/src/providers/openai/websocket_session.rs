//! OpenAI WebSocket session helpers.
//!
//! This module provides a provider-specific "session" wrapper optimized for agentic workflows:
//! - single WebSocket connection reuse (no ambiguity)
//! - optional connection warm-up (`generate: false`)
//! - connection-local incremental continuation (`previous_response_id`)

use crate::error::LlmError;
use crate::providers::openai::OpenAiWebSocketTransport;
use crate::streaming::{ChatStream, ChatStreamHandle};
use crate::traits::ChatCapability;
use crate::types::{ChatMessage, ChatRequest, ChatResponse, ProviderOptionsMap, Tool};
use async_trait::async_trait;
use futures_util::StreamExt;
use std::ops::Deref;
use std::sync::Arc;
use tokio::sync::Semaphore;

use super::{OpenAiBuilder, OpenAiClient};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenAiWebSocketRecoveryConfig {
    /// Whether to fall back to HTTP (SSE) streaming when WebSocket streaming fails.
    pub allow_http_fallback: bool,
    /// How many times to retry WebSocket streaming on a fresh connection for recoverable WS errors.
    pub max_ws_retries: u8,
}

impl Default for OpenAiWebSocketRecoveryConfig {
    fn default() -> Self {
        Self {
            allow_http_fallback: true,
            max_ws_retries: 1,
        }
    }
}

/// A single-connection OpenAI WebSocket session (Responses API streaming).
///
/// Notes:
/// - This wrapper is intentionally provider-specific and does not attempt to be generic.
/// - It configures the underlying WebSocket transport to keep exactly 1 idle connection
///   and enable connection-local `previous_response_id` continuation.
#[derive(Clone)]
pub struct OpenAiWebSocketSession {
    client: OpenAiClient,
    http_fallback_client: OpenAiClient,
    transport: OpenAiWebSocketTransport,
    gate: Arc<Semaphore>,
    recovery: OpenAiWebSocketRecoveryConfig,
}

impl std::fmt::Debug for OpenAiWebSocketSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiWebSocketSession")
            .finish_non_exhaustive()
    }
}

impl OpenAiWebSocketSession {
    /// Create a session from an OpenAI builder.
    ///
    /// This injects a WebSocket-backed transport and builds an `OpenAiClient`.
    pub async fn from_builder(builder: OpenAiBuilder) -> Result<Self, LlmError> {
        let transport = OpenAiWebSocketTransport::default()
            .with_max_idle_connections(1)
            .with_stateful_previous_response_id(true);

        let client = builder.fetch(Arc::new(transport.clone())).build().await?;
        let http_fallback_client = client.clone_without_http_transport();

        Ok(Self {
            client,
            http_fallback_client,
            transport,
            gate: Arc::new(Semaphore::new(1)),
            recovery: OpenAiWebSocketRecoveryConfig::default(),
        })
    }

    pub fn recovery_config(&self) -> OpenAiWebSocketRecoveryConfig {
        self.recovery
    }

    pub fn with_recovery_config(mut self, cfg: OpenAiWebSocketRecoveryConfig) -> Self {
        self.recovery = cfg;
        self
    }

    /// Access the underlying OpenAI client.
    pub fn client(&self) -> &OpenAiClient {
        &self.client
    }

    /// Access the underlying WebSocket transport.
    pub fn transport(&self) -> &OpenAiWebSocketTransport {
        &self.transport
    }

    /// Close the session (closes cached WebSocket connection).
    pub async fn close(&self) {
        self.transport.close().await;
    }

    /// Warm up the WebSocket connection by running a streaming Responses request with `generate=false`.
    ///
    /// This is intended to:
    /// - establish DNS/TCP/TLS/WS
    /// - "prime" the connection-local cache (tools + instructions) for subsequent continuation
    pub async fn warm_up(&self, mut request: ChatRequest) -> Result<(), LlmError> {
        let _permit =
            self.gate.clone().acquire_owned().await.map_err(|_| {
                LlmError::InternalError("WebSocket session gate closed".to_string())
            })?;

        request.stream = true;

        let mut overrides = ProviderOptionsMap::new();
        overrides.insert(
            "openai",
            serde_json::json!({
                "responsesApi": {
                    "enabled": true,
                    "generate": false,
                    "store": false
                }
            }),
        );
        request.provider_options_map.merge_overrides(overrides);

        let mut stream = self.client.chat_stream_request(request).await?;
        while let Some(item) = stream.next().await {
            let ev = item?;
            if let crate::types::ChatStreamEvent::Error { error } = ev {
                return Err(LlmError::StreamError(error));
            }
        }
        Ok(())
    }

    /// Warm up with Responses `instructions` (stored on the connection for incremental continuation).
    pub async fn warm_up_with_instructions(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        instructions: impl Into<String>,
    ) -> Result<(), LlmError> {
        let mut req = ChatRequest::new(messages).with_streaming(true);
        if let Some(t) = tools {
            req = req.with_tools(t);
        }
        req = req.with_provider_option(
            "openai",
            serde_json::json!({
                "responsesApi": {
                    "enabled": true,
                    "instructions": instructions.into()
                }
            }),
        );
        self.warm_up(req).await
    }

    /// Convenience warm-up: just provide messages/tools.
    pub async fn warm_up_messages(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<(), LlmError> {
        let mut req = ChatRequest::new(messages).with_streaming(true);
        if let Some(t) = tools {
            req = req.with_tools(t);
        }
        self.warm_up(req).await
    }
}

impl Deref for OpenAiWebSocketSession {
    type Target = OpenAiClient;
    fn deref(&self) -> &Self::Target {
        &self.client
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamRecoveryAction {
    RetryWsFresh,
    FallbackHttp,
}

fn is_ws_connect_error(err: &LlmError) -> bool {
    fn has_ws_hint(msg: &str) -> bool {
        let msg_lc = msg.to_ascii_lowercase();
        msg_lc.contains("websocket")
            || msg_lc.contains("tungstenite")
            || msg_lc.contains("ws handshake")
            || msg_lc.contains("ws connect")
    }

    match err {
        LlmError::ProviderError {
            provider,
            error_code: Some(code),
            ..
        } => {
            provider == "openai"
                && matches!(
                    code.as_str(),
                    "websocket_connect_failed"
                        | "websocket_send_failed"
                        | "websocket_recv_failed"
                        | "websocket_pong_send_failed"
                        | "websocket_invalid_url"
                        | "websocket_unsupported_url_scheme"
                )
        }
        LlmError::ConnectionError(msg) => has_ws_hint(msg),
        LlmError::StreamError(msg) => has_ws_hint(msg),
        _ => false,
    }
}

fn openai_ws_provider_error_code(err: &LlmError) -> Option<&str> {
    match err {
        LlmError::ProviderError {
            provider,
            error_code: Some(code),
            ..
        } if provider == "openai" => Some(code.as_str()),
        _ => None,
    }
}

fn is_ws_configuration_error(err: &LlmError) -> bool {
    matches!(
        openai_ws_provider_error_code(err),
        Some("websocket_invalid_url" | "websocket_unsupported_url_scheme")
    )
}

fn ws_recovery_custom_event(
    stage: &'static str,
    action: &'static str,
    reason: serde_json::Value,
) -> crate::types::ChatStreamEvent {
    crate::types::ChatStreamEvent::Custom {
        event_type: "openai:ws-recovery".to_string(),
        data: serde_json::json!({
            "type": "ws-recovery",
            "stage": stage,
            "action": action,
            "reason": reason,
        }),
    }
}

fn openai_error_code_from_event(ev: &crate::types::ChatStreamEvent) -> Option<String> {
    let crate::types::ChatStreamEvent::Custom { event_type, data } = ev else {
        return None;
    };
    if event_type != "openai:error" {
        return None;
    }

    let code = data
        .pointer("/error/error/code")
        .and_then(|v| v.as_str())
        .or_else(|| data.pointer("/error/code").and_then(|v| v.as_str()));
    code.map(|s| s.to_string())
}

fn recovery_action_from_openai_error_code(code: &str) -> Option<StreamRecoveryAction> {
    match code {
        "websocket_connection_limit_reached" => Some(StreamRecoveryAction::FallbackHttp),
        // This error typically happens when `previous_response_id` references a response that is not
        // available in the current WebSocket connection-local cache. Retrying on a fresh connection
        // (without implicit `previous_response_id`) is a safe, conservative recovery attempt.
        //
        // Official docs mention `previous_response_not_found`, but we also accept the older/more
        // explicit variant for forward/backward compatibility.
        "previous_response_not_found" | "websocket_previous_response_id_not_found" => {
            Some(StreamRecoveryAction::RetryWsFresh)
        }
        _ => None,
    }
}

#[async_trait]
impl ChatCapability for OpenAiWebSocketSession {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        self.client.chat_with_tools(messages, tools).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        let gate = Arc::clone(&self.gate);
        let client = self.client.clone();
        let http_fallback_client = self.http_fallback_client.clone();
        let transport = self.transport.clone();
        let recovery = self.recovery;

        let s = async_stream::stream! {
            let _permit = gate
                .acquire_owned()
                .await
                .map_err(|_| LlmError::InternalError("WebSocket session gate closed".to_string()))?;

            let messages_retry = messages.clone();
            let tools_retry = tools.clone();

            let mut inner = match client.chat_stream(messages, tools).await {
                Ok(s) => s,
                Err(e) if is_ws_connect_error(&e) && recovery.allow_http_fallback => {
                    if is_ws_configuration_error(&e) {
                        yield Err(e);
                        return;
                    }
                    tracing::warn!(
                        target: "siumai::openai::websocket_session",
                        error = %e,
                        "WebSocket streaming failed during connect; falling back to HTTP (SSE) streaming for this request"
                    );
                    yield Ok(ws_recovery_custom_event(
                        "connect",
                        "fallback_http",
                        serde_json::json!({ "kind": "connect_error", "error": e.to_string() }),
                    ));
                    http_fallback_client
                        .chat_stream(messages_retry.clone(), tools_retry.clone())
                        .await?
                }
                Err(e) => {
                    if is_ws_connect_error(&e) && !recovery.allow_http_fallback {
                        tracing::warn!(
                            target: "siumai::openai::websocket_session",
                            error = %e,
                            "WebSocket streaming failed during connect; HTTP fallback disabled"
                        );
                    }
                    yield Err(e);
                    return;
                }
            };

            let mut ws_retries_left = recovery.max_ws_retries;
            let mut emitted_any = false;
            let mut pending_action: Option<StreamRecoveryAction> = None;
            let mut pending_error_code: Option<String> = None;
            let mut buffered: Vec<Result<crate::types::ChatStreamEvent, LlmError>> = Vec::new();

            while let Some(item) = inner.next().await {
                if emitted_any {
                    yield item;
                    continue;
                }

                if let Ok(ev) = item.as_ref() {
                    if let Some(code) = openai_error_code_from_event(ev) {
                        pending_error_code = Some(code.clone());
                        pending_action = recovery_action_from_openai_error_code(&code);
                    }
                }

                let is_errorish = matches!(item, Err(_) | Ok(crate::types::ChatStreamEvent::Error { .. }))
                    || matches!(
                        item,
                        Ok(crate::types::ChatStreamEvent::Custom {
                            ref event_type,
                            ..
                        })
                            if event_type == "openai:error"
                    );

                if is_errorish {
                    buffered.push(item);

                    let action = pending_action.or_else(|| {
                        buffered.iter().find_map(|it| match it {
                            Ok(crate::types::ChatStreamEvent::Custom { event_type, data }) => {
                                if event_type == "openai:error" {
                                    let code = data
                                        .pointer("/error/error/code")
                                        .and_then(|v| v.as_str())
                                        .or_else(|| data.pointer("/error/code").and_then(|v| v.as_str()))?;
                                    recovery_action_from_openai_error_code(code)
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        })
                    });

                    let should_recover = action.is_some()
                        || buffered.iter().any(|it| matches!(it, Err(e) if is_ws_connect_error(e)));

                    if should_recover && (ws_retries_left > 0 || recovery.allow_http_fallback) {
                        buffered.clear();

                        if matches!(action, Some(StreamRecoveryAction::RetryWsFresh)) && ws_retries_left > 0
                        {
                            ws_retries_left -= 1;
                            tracing::warn!(
                                target: "siumai::openai::websocket_session",
                                error_code = pending_error_code.as_deref().unwrap_or("unknown"),
                                ws_retries_left = ws_retries_left,
                                "Recovering WebSocket stream by rebuilding the connection and retrying once (connection-local cache will be lost)"
                            );
                            yield Ok(ws_recovery_custom_event(
                                "stream",
                                "retry_ws_fresh",
                                serde_json::json!({
                                    "kind": "openai_error",
                                    "code": pending_error_code.as_deref().unwrap_or("unknown"),
                                    "wsRetriesLeft": ws_retries_left
                                }),
                            ));
                            transport.close().await;
                            inner = client.chat_stream(messages_retry.clone(), tools_retry.clone()).await?;
                            pending_action = None;
                            pending_error_code = None;
                            continue;
                        }

                        if recovery.allow_http_fallback {
                            tracing::warn!(
                                target: "siumai::openai::websocket_session",
                                error_code = pending_error_code.as_deref().unwrap_or("unknown"),
                                "Recovering WebSocket stream by falling back to HTTP (SSE) streaming for this request"
                            );
                            yield Ok(ws_recovery_custom_event(
                                "stream",
                                "fallback_http",
                                serde_json::json!({
                                    "kind": "openai_error",
                                    "code": pending_error_code.as_deref().unwrap_or("unknown")
                                }),
                            ));
                            inner = http_fallback_client
                                .chat_stream(messages_retry.clone(), tools_retry.clone())
                                .await?;
                            pending_action = None;
                            pending_error_code = None;
                            continue;
                        }
                    }

                    // No recovery (or already retried): flush buffered items.
                    for it in buffered.drain(..) {
                        yield it;
                    }
                    emitted_any = true;
                    continue;
                }

                // First non-error-ish event: flush buffer then continue normally.
                for it in buffered.drain(..) {
                    yield it;
                }
                yield item;
                emitted_any = true;
            }

            if !emitted_any {
                for it in buffered.drain(..) {
                    yield it;
                }
            }
        };
        Ok(Box::pin(s))
    }

    async fn chat_stream_with_cancel(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStreamHandle, LlmError> {
        let this = self.clone();
        Ok(
            crate::utils::cancel::make_cancellable_stream_handle_from_future(async move {
                this.chat_stream(messages, tools).await
            }),
        )
    }

    async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.client.chat_request(request).await
    }

    async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        let gate = Arc::clone(&self.gate);
        let client = self.client.clone();
        let http_fallback_client = self.http_fallback_client.clone();
        let transport = self.transport.clone();
        let recovery = self.recovery;

        let s = async_stream::stream! {
            let _permit = gate
                .acquire_owned()
                .await
                .map_err(|_| LlmError::InternalError("WebSocket session gate closed".to_string()))?;

            let request_retry = request.clone();
            let mut inner = match client.chat_stream_request(request).await {
                Ok(s) => s,
                Err(e) if is_ws_connect_error(&e) && recovery.allow_http_fallback => {
                    if is_ws_configuration_error(&e) {
                        yield Err(e);
                        return;
                    }
                    tracing::warn!(
                        target: "siumai::openai::websocket_session",
                        error = %e,
                        "WebSocket streaming failed during connect; falling back to HTTP (SSE) streaming for this request"
                    );
                    yield Ok(ws_recovery_custom_event(
                        "connect",
                        "fallback_http",
                        serde_json::json!({ "kind": "connect_error", "error": e.to_string() }),
                    ));
                    http_fallback_client
                        .chat_stream_request(request_retry.clone())
                        .await?
                }
                Err(e) => {
                    if is_ws_connect_error(&e) && !recovery.allow_http_fallback {
                        tracing::warn!(
                            target: "siumai::openai::websocket_session",
                            error = %e,
                            "WebSocket streaming failed during connect; HTTP fallback disabled"
                        );
                    }
                    yield Err(e);
                    return;
                }
            };

            let mut ws_retries_left = recovery.max_ws_retries;
            let mut emitted_any = false;
            let mut pending_action: Option<StreamRecoveryAction> = None;
            let mut pending_error_code: Option<String> = None;
            let mut buffered: Vec<Result<crate::types::ChatStreamEvent, LlmError>> = Vec::new();

            while let Some(item) = inner.next().await {
                if emitted_any {
                    yield item;
                    continue;
                }

                if let Ok(ev) = item.as_ref() {
                    if let Some(code) = openai_error_code_from_event(ev) {
                        pending_error_code = Some(code.clone());
                        pending_action = recovery_action_from_openai_error_code(&code);
                    }
                }

                let is_errorish = matches!(item, Err(_) | Ok(crate::types::ChatStreamEvent::Error { .. }))
                    || matches!(
                        item,
                        Ok(crate::types::ChatStreamEvent::Custom {
                            ref event_type,
                            ..
                        })
                            if event_type == "openai:error"
                    );

                if is_errorish {
                    buffered.push(item);

                    let action = pending_action.or_else(|| {
                        buffered.iter().find_map(|it| match it {
                            Ok(crate::types::ChatStreamEvent::Custom { event_type, data }) => {
                                if event_type == "openai:error" {
                                    let code = data
                                        .pointer("/error/error/code")
                                        .and_then(|v| v.as_str())
                                        .or_else(|| data.pointer("/error/code").and_then(|v| v.as_str()))?;
                                    recovery_action_from_openai_error_code(code)
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        })
                    });

                    let should_recover = action.is_some()
                        || buffered.iter().any(|it| matches!(it, Err(e) if is_ws_connect_error(e)));

                    if should_recover && (ws_retries_left > 0 || recovery.allow_http_fallback) {
                        buffered.clear();

                        if matches!(action, Some(StreamRecoveryAction::RetryWsFresh)) && ws_retries_left > 0
                        {
                            ws_retries_left -= 1;
                            tracing::warn!(
                                target: "siumai::openai::websocket_session",
                                error_code = pending_error_code.as_deref().unwrap_or("unknown"),
                                ws_retries_left = ws_retries_left,
                                "Recovering WebSocket stream by rebuilding the connection and retrying once (connection-local cache will be lost)"
                            );
                            yield Ok(ws_recovery_custom_event(
                                "stream",
                                "retry_ws_fresh",
                                serde_json::json!({
                                    "kind": "openai_error",
                                    "code": pending_error_code.as_deref().unwrap_or("unknown"),
                                    "wsRetriesLeft": ws_retries_left
                                }),
                            ));
                            transport.close().await;
                            inner = client.chat_stream_request(request_retry.clone()).await?;
                            pending_action = None;
                            pending_error_code = None;
                            continue;
                        }

                        if recovery.allow_http_fallback {
                            tracing::warn!(
                                target: "siumai::openai::websocket_session",
                                error_code = pending_error_code.as_deref().unwrap_or("unknown"),
                                "Recovering WebSocket stream by falling back to HTTP (SSE) streaming for this request"
                            );
                            yield Ok(ws_recovery_custom_event(
                                "stream",
                                "fallback_http",
                                serde_json::json!({
                                    "kind": "openai_error",
                                    "code": pending_error_code.as_deref().unwrap_or("unknown")
                                }),
                            ));
                            inner = http_fallback_client
                                .chat_stream_request(request_retry.clone())
                                .await?;
                            pending_action = None;
                            pending_error_code = None;
                            continue;
                        }
                    }

                    for it in buffered.drain(..) {
                        yield it;
                    }
                    emitted_any = true;
                    continue;
                }

                for it in buffered.drain(..) {
                    yield it;
                }
                yield item;
                emitted_any = true;
            }

            if !emitted_any {
                for it in buffered.drain(..) {
                    yield it;
                }
            }
        };
        Ok(Box::pin(s))
    }

    async fn chat_stream_request_with_cancel(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStreamHandle, LlmError> {
        let this = self.clone();
        Ok(
            crate::utils::cancel::make_cancellable_stream_handle_from_future(async move {
                this.chat_stream_request(request).await
            }),
        )
    }
}

#[cfg(test)]
#[cfg(feature = "openai-websocket")]
mod tests {
    use super::*;
    use crate::types::ChatStreamEvent;
    use futures_util::SinkExt;
    use futures_util::StreamExt;
    use std::sync::atomic::{AtomicU32, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;
    use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};
    use tokio_tungstenite::tungstenite::protocol::Message;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn session_gate_prevents_concurrent_second_connection() {
        let accept_count = Arc::new(AtomicU32::new(0));
        let (finish_first_tx, finish_first_rx) = oneshot::channel::<()>();

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let accept_count_server = Arc::clone(&accept_count);
        let server = tokio::spawn(async move {
            let (tcp1, _) = listener.accept().await.unwrap();
            accept_count_server.fetch_add(1, Ordering::SeqCst);

            // Start accepting a potential second connection while the first stream is in-flight.
            let accept_count_server_2 = Arc::clone(&accept_count_server);
            let accept2 = tokio::spawn(async move {
                if let Ok(Ok((tcp2, _))) =
                    tokio::time::timeout(std::time::Duration::from_millis(250), listener.accept())
                        .await
                {
                    accept_count_server_2.fetch_add(1, Ordering::SeqCst);
                    // Best-effort: complete a minimal WS handshake so the client doesn't hang forever.
                    let mut ws2 = tokio_tungstenite::accept_hdr_async(
                        tcp2,
                        |_req: &Request, resp: Response| Ok(resp),
                    )
                    .await
                    .unwrap();
                    let _ = ws2.next().await;
                }
            });

            let mut ws1 =
                tokio_tungstenite::accept_hdr_async(tcp1, |req: &Request, resp: Response| {
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

            // First response.create
            let first = ws1.next().await.unwrap().unwrap();
            let Message::Text(_txt1) = first else {
                panic!("expected text message");
            };
            ws1.send(Message::Text(
                serde_json::json!({
                    "type": "response.created",
                    "response": { "id": "resp_1", "model": "gpt-test", "created_at": 0 }
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
            ws1.send(Message::Text(
                serde_json::json!({
                    "type": "response.output_text.delta",
                    "delta": "A"
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();

            let _ = finish_first_rx.await;

            ws1.send(Message::Text(
                serde_json::json!({
                    "type": "response.completed",
                    "response": { "id": "resp_1", "model": "gpt-test", "created_at": 0, "output": [] }
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();

            // Second response.create (must arrive on the same connection, after first completes)
            let second = tokio::time::timeout(std::time::Duration::from_secs(1), ws1.next())
                .await
                .ok()
                .flatten()
                .and_then(Result::ok);
            if let Some(Message::Text(_txt2)) = second {
                ws1.send(Message::Text(
                    serde_json::json!({
                        "type": "response.created",
                        "response": { "id": "resp_2", "model": "gpt-test", "created_at": 0 }
                    })
                    .to_string()
                    .into(),
                ))
                .await
                .unwrap();
                ws1.send(Message::Text(
                    serde_json::json!({
                        "type": "response.output_text.delta",
                        "delta": "B"
                    })
                    .to_string()
                    .into(),
                ))
                .await
                .unwrap();
                ws1.send(Message::Text(
                    serde_json::json!({
                        "type": "response.completed",
                        "response": { "id": "resp_2", "model": "gpt-test", "created_at": 0, "output": [] }
                    })
                    .to_string()
                    .into(),
                ))
                .await
                .unwrap();
            }

            accept2.await.unwrap();
        });

        let session = OpenAiWebSocketSession::from_builder(
            crate::providers::openai::OpenAiBuilder::new(crate::builder::BuilderBase::default())
                .api_key("test")
                .base_url(format!("http://{addr}/v1"))
                .model("gpt-test"),
        )
        .await
        .unwrap();

        let mut s1 = session
            .chat_stream_request(
                ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_streaming(true),
            )
            .await
            .unwrap();

        let first_delta = loop {
            let ev = s1.next().await.unwrap().unwrap();
            if let ChatStreamEvent::ContentDelta { delta, .. } = ev {
                break delta;
            }
        };
        assert_eq!(first_delta, "A");

        let session2 = session.clone();
        let second_task = tokio::spawn(async move {
            let mut s2 = session2
                .chat_stream_request(
                    ChatRequest::new(vec![ChatMessage::user("hi2").build()]).with_streaming(true),
                )
                .await
                .unwrap();
            let mut out = String::new();
            while let Some(item) = s2.next().await {
                let ev = item.unwrap();
                if let ChatStreamEvent::ContentDelta { delta, .. } = ev {
                    out.push_str(&delta);
                }
            }
            out
        });

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        assert_eq!(
            accept_count.load(Ordering::SeqCst),
            1,
            "session should not establish a second connection while the first stream is in-flight"
        );

        let _ = finish_first_tx.send(());
        while s1.next().await.is_some() {}

        let out = tokio::time::timeout(std::time::Duration::from_secs(2), second_task)
            .await
            .expect("second stream should complete")
            .expect("task ok");
        assert_eq!(out, "B");

        assert_eq!(
            accept_count.load(Ordering::SeqCst),
            1,
            "session should reuse the same connection"
        );

        server.await.unwrap();
    }

    #[tokio::test]
    async fn session_falls_back_to_http_when_ws_handshake_fails() {
        let server = MockServer::start().await;

        // WebSocket handshake uses GET; return a normal HTTP response so tungstenite fails the handshake.
        Mock::given(method("GET"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let sse = concat!(
            "event: response.created\n",
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-test\",\"created_at\":0}}\n\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"OK\"}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-test\",\"created_at\":0,\"output\":[]}}\n\n",
        );

        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_raw(sse, "text/event-stream"),
            )
            .mount(&server)
            .await;

        let session = OpenAiWebSocketSession::from_builder(
            crate::providers::openai::OpenAiBuilder::new(crate::builder::BuilderBase::default())
                .api_key("test")
                .base_url(format!("{}/v1", server.uri()))
                .model("gpt-test"),
        )
        .await
        .unwrap();

        let mut s = session
            .chat_stream(vec![ChatMessage::user("hi").build()], None)
            .await
            .unwrap();

        let first = s.next().await.unwrap().unwrap();
        let ChatStreamEvent::Custom { event_type, .. } = first else {
            panic!("expected openai:ws-recovery custom event");
        };
        assert_eq!(event_type, "openai:ws-recovery");

        let mut out = String::new();
        while let Some(item) = s.next().await {
            let ev = item.unwrap();
            if let ChatStreamEvent::ContentDelta { delta, .. } = ev {
                out.push_str(&delta);
            }
        }
        assert_eq!(out, "OK");
    }

    #[tokio::test]
    async fn session_falls_back_to_http_on_ws_connection_limit_error() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let sse = concat!(
            "event: response.created\n",
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-test\",\"created_at\":0}}\n\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"OK\"}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-test\",\"created_at\":0,\"output\":[]}}\n\n",
        );

        let server = tokio::spawn(async move {
            for _ in 0..2 {
                let (mut tcp, _) = listener.accept().await.unwrap();
                let mut peek = [0u8; 4];
                let _ = tcp.peek(&mut peek).await.unwrap();

                if peek.starts_with(b"GET ") {
                    let mut ws = tokio_tungstenite::accept_hdr_async(
                        tcp,
                        |req: &Request, resp: Response| {
                            assert_eq!(req.uri().path(), "/v1/responses");
                            Ok(resp)
                        },
                    )
                    .await
                    .unwrap();

                    // Wait for `response.create`, then return a WS-specific OpenAI error code.
                    let _ = ws.next().await;
                    ws.send(Message::Text(
                        serde_json::json!({
                            "type": "error",
                            "error": {
                                "code": "websocket_connection_limit_reached",
                                "message": "limit"
                            }
                        })
                        .to_string()
                        .into(),
                    ))
                    .await
                    .unwrap();
                    let _ = ws.close(None).await;
                    continue;
                }

                // HTTP fallback: minimal POST handler returning SSE.
                let mut buf = Vec::<u8>::new();
                let mut tmp = [0u8; 1024];
                loop {
                    let n = tcp.read(&mut tmp).await.unwrap();
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                        break;
                    }
                }

                let header_end = buf
                    .windows(4)
                    .position(|w| w == b"\r\n\r\n")
                    .map(|p| p + 4)
                    .unwrap_or(buf.len());
                let header_str = String::from_utf8_lossy(&buf[..header_end]);
                let content_len = header_str
                    .lines()
                    .find_map(|l| l.strip_prefix("Content-Length: "))
                    .and_then(|v| v.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                let already = buf.len().saturating_sub(header_end);
                if content_len > already {
                    let mut remaining = vec![0u8; content_len - already];
                    tcp.read_exact(&mut remaining).await.unwrap();
                }

                let body = sse.as_bytes();
                let resp = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\n\r\n",
                    body.len()
                );
                tcp.write_all(resp.as_bytes()).await.unwrap();
                tcp.write_all(body).await.unwrap();
                let _ = tcp.shutdown().await;
            }
        });

        let session = OpenAiWebSocketSession::from_builder(
            crate::providers::openai::OpenAiBuilder::new(crate::builder::BuilderBase::default())
                .api_key("test")
                .base_url(format!("http://{addr}/v1"))
                .model("gpt-test"),
        )
        .await
        .unwrap();

        let mut s = session
            .chat_stream(vec![ChatMessage::user("hi").build()], None)
            .await
            .unwrap();

        let first = s.next().await.unwrap().unwrap();
        let ChatStreamEvent::Custom { event_type, .. } = first else {
            panic!("expected openai:ws-recovery custom event");
        };
        assert_eq!(event_type, "openai:ws-recovery");

        let mut out = String::new();
        while let Some(item) = s.next().await {
            let ev = item.unwrap();
            if let ChatStreamEvent::ContentDelta { delta, .. } = ev {
                out.push_str(&delta);
            }
        }
        assert_eq!(out, "OK");

        server.await.unwrap();
    }

    #[tokio::test]
    async fn session_retries_ws_on_previous_response_not_found() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            // Connection A: succeed once (to seed transport's connection-local previous_response_id),
            // then fail the next request with previous_response_not_found.
            let (tcp_a, _) = listener.accept().await.unwrap();
            let mut ws_a =
                tokio_tungstenite::accept_hdr_async(tcp_a, |req: &Request, resp: Response| {
                    assert_eq!(req.uri().path(), "/v1/responses");
                    Ok(resp)
                })
                .await
                .unwrap();

            // Request 1: no previous_response_id expected.
            let req1 = ws_a.next().await.unwrap().unwrap();
            let Message::Text(t1) = req1 else {
                panic!("expected text request");
            };
            let v1: serde_json::Value = serde_json::from_str(&t1).unwrap();
            assert_eq!(v1["type"], "response.create");
            assert!(v1.get("previous_response_id").is_none());
            ws_a.send(Message::Text(
                serde_json::json!({
                    "type": "response.created",
                    "response": { "id": "resp_1", "model": "gpt-test", "created_at": 0 }
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
            ws_a.send(Message::Text(
                serde_json::json!({
                    "type": "response.output_text.delta",
                    "delta": "A"
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
            ws_a
                .send(Message::Text(
                    serde_json::json!({
                        "type": "response.completed",
                        "response": { "id": "resp_1", "model": "gpt-test", "created_at": 0, "output": [] }
                    })
                    .to_string()
                    .into(),
                ))
                .await
                .unwrap();

            // Request 2 (same connection): previous_response_id should be injected; respond with the WS error.
            let req2 = ws_a.next().await.unwrap().unwrap();
            let Message::Text(t2) = req2 else {
                panic!("expected text request");
            };
            let v2: serde_json::Value = serde_json::from_str(&t2).unwrap();
            assert_eq!(v2["type"], "response.create");
            assert_eq!(v2["previous_response_id"], "resp_1");
            ws_a.send(Message::Text(
                serde_json::json!({
                    "type": "error",
                    "error": {
                        "code": "previous_response_not_found",
                        "message": "not found"
                    }
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
            let _ = ws_a.close(None).await;

            // Connection B: recovery retry must open a fresh connection and succeed (no previous_response_id).
            let (tcp_b, _) = listener.accept().await.unwrap();
            let mut ws_b =
                tokio_tungstenite::accept_hdr_async(tcp_b, |req: &Request, resp: Response| {
                    assert_eq!(req.uri().path(), "/v1/responses");
                    Ok(resp)
                })
                .await
                .unwrap();
            let req3 = ws_b.next().await.unwrap().unwrap();
            let Message::Text(t3) = req3 else {
                panic!("expected text request");
            };
            let v3: serde_json::Value = serde_json::from_str(&t3).unwrap();
            assert_eq!(v3["type"], "response.create");
            assert!(v3.get("previous_response_id").is_none());
            ws_b.send(Message::Text(
                serde_json::json!({
                    "type": "response.created",
                    "response": { "id": "resp_2", "model": "gpt-test", "created_at": 0 }
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
            ws_b.send(Message::Text(
                serde_json::json!({
                    "type": "response.output_text.delta",
                    "delta": "OK"
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
            ws_b
                .send(Message::Text(
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

        let session = OpenAiWebSocketSession::from_builder(
            crate::providers::openai::OpenAiBuilder::new(crate::builder::BuilderBase::default())
                .api_key("test")
                .base_url(format!("http://{addr}/v1"))
                .model("gpt-test"),
        )
        .await
        .unwrap();

        // Run 2 sequential streams:
        // - #1 seeds the cached connection-local previous_response_id
        // - #2 triggers the recoverable WS error then succeeds after retry
        let mut s1 = session
            .chat_stream(vec![ChatMessage::user("seed").build()], None)
            .await
            .unwrap();
        while s1.next().await.is_some() {}

        let mut s = session
            .chat_stream(vec![ChatMessage::user("retry").build()], None)
            .await
            .unwrap();

        let first = s.next().await.unwrap().unwrap();
        let ChatStreamEvent::Custom { event_type, .. } = first else {
            panic!("expected openai:ws-recovery custom event");
        };
        assert_eq!(event_type, "openai:ws-recovery");

        let mut out = String::new();
        while let Some(item) = s.next().await {
            let ev = item.unwrap();
            if let ChatStreamEvent::ContentDelta { delta, .. } = ev {
                out.push_str(&delta);
            }
        }
        assert_eq!(out, "OK");

        server.await.unwrap();
    }

    #[tokio::test]
    async fn session_does_not_fallback_when_disabled() {
        let server = MockServer::start().await;

        // WebSocket handshake uses GET; return a normal HTTP response so tungstenite fails the handshake.
        Mock::given(method("GET"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        // If fallback were enabled, this would be used. With fallback disabled we should surface the WS error.
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let session = OpenAiWebSocketSession::from_builder(
            crate::providers::openai::OpenAiBuilder::new(crate::builder::BuilderBase::default())
                .api_key("test")
                .base_url(format!("{}/v1", server.uri()))
                .model("gpt-test"),
        )
        .await
        .unwrap()
        .with_recovery_config(OpenAiWebSocketRecoveryConfig {
            allow_http_fallback: false,
            max_ws_retries: 0,
        });

        let mut s = session
            .chat_stream(vec![ChatMessage::user("hi").build()], None)
            .await
            .unwrap();

        let first = s.next().await.expect("should yield an error");
        assert!(
            first.is_err(),
            "expected WebSocket error without HTTP fallback"
        );
        while s.next().await.is_some() {}
    }

    #[tokio::test]
    async fn session_surfaces_unsupported_ws_url_scheme_when_fallback_disabled() {
        let server = MockServer::start().await;

        let session = OpenAiWebSocketSession::from_builder(
            crate::providers::openai::OpenAiBuilder::new(crate::builder::BuilderBase::default())
                .api_key("test")
                .base_url(format!("ftp://{}", server.address()))
                .model("gpt-test"),
        )
        .await
        .unwrap()
        .with_recovery_config(OpenAiWebSocketRecoveryConfig {
            allow_http_fallback: false,
            max_ws_retries: 0,
        });

        let mut s = session
            .chat_stream(vec![ChatMessage::user("hi").build()], None)
            .await
            .unwrap();

        let first = s.next().await.expect("should yield an error");
        let err = first.expect_err("expected error without HTTP fallback");
        let LlmError::ProviderError {
            provider,
            error_code,
            ..
        } = err
        else {
            panic!("expected ProviderError");
        };
        assert_eq!(provider, "openai");
        assert_eq!(
            error_code.as_deref(),
            Some("websocket_unsupported_url_scheme")
        );
        while s.next().await.is_some() {}
    }

    #[tokio::test]
    async fn session_surfaces_unsupported_ws_url_scheme_even_when_fallback_enabled() {
        let server = MockServer::start().await;

        let session = OpenAiWebSocketSession::from_builder(
            crate::providers::openai::OpenAiBuilder::new(crate::builder::BuilderBase::default())
                .api_key("test")
                .base_url(format!("ftp://{}", server.address()))
                .model("gpt-test"),
        )
        .await
        .unwrap();

        let mut s = session
            .chat_stream(vec![ChatMessage::user("hi").build()], None)
            .await
            .unwrap();

        let first = s.next().await.expect("should yield an error");
        let err = first.expect_err("expected error for invalid base_url scheme");
        let LlmError::ProviderError {
            provider,
            error_code,
            ..
        } = err
        else {
            panic!("expected ProviderError");
        };
        assert_eq!(provider, "openai");
        assert_eq!(
            error_code.as_deref(),
            Some("websocket_unsupported_url_scheme")
        );
        while s.next().await.is_some() {}
    }
}
