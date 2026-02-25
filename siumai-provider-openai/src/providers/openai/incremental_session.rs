//! Incremental WebSocket session helpers for OpenAI Responses streaming.
//!
//! This module provides a higher-level API over [`OpenAiWebSocketSession`] for agentic tool loops:
//! - Warm up a connection with tools/instructions (connection-local cache).
//! - Send only incremental messages per step.
//! - On `RetryWsFresh` recovery, optionally re-run the warm-up request to restore cache.

use crate::error::LlmError;
use crate::streaming::ChatStreamHandle;
use crate::traits::ChatCapability;
use crate::types::{ChatMessage, ChatRequest, Tool};

#[cfg(feature = "openai-websocket")]
use super::OpenAiWebSocketSession;

/// A helper that runs incremental multi-turn streaming over a single WebSocket session.
///
/// Recommended for tool loops where each step only needs to send:
/// - the new user message(s), and
/// - tool results from the previous step.
///
/// When `cache_defaults_on_connection` is enabled via [`Self::cache_defaults_on_connection`],
/// the session will:
/// - warm up the WebSocket connection with `generate=false`, and
/// - configure reconnect warm-up so fresh-retry rebuilds the connection-local cache.
#[cfg(feature = "openai-websocket")]
#[derive(Clone)]
pub struct OpenAiIncrementalWebSocketSession {
    session: OpenAiWebSocketSession,
    cache_defaults_on_connection: bool,
    default_tools: Option<Vec<Tool>>,
    default_instructions: Option<String>,
    pending: Vec<ChatMessage>,
}

#[cfg(feature = "openai-websocket")]
impl std::fmt::Debug for OpenAiIncrementalWebSocketSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiIncrementalWebSocketSession")
            .field(
                "cache_defaults_on_connection",
                &self.cache_defaults_on_connection,
            )
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "openai-websocket")]
impl OpenAiIncrementalWebSocketSession {
    pub fn new(session: OpenAiWebSocketSession) -> Self {
        Self {
            session,
            cache_defaults_on_connection: false,
            default_tools: None,
            default_instructions: None,
            pending: Vec::new(),
        }
    }

    pub fn session(&self) -> &OpenAiWebSocketSession {
        &self.session
    }

    pub fn into_inner(self) -> OpenAiWebSocketSession {
        self.session
    }

    /// Enable caching tools/instructions on the WebSocket connection via a warm-up request.
    ///
    /// After this is called:
    /// - subsequent step requests can omit tools/instructions, and
    /// - fresh-retry recovery will best-effort rerun warm-up automatically.
    pub async fn cache_defaults_on_connection(
        mut self,
        tools: Option<Vec<Tool>>,
        instructions: Option<String>,
    ) -> Result<Self, LlmError> {
        self.cache_defaults_on_connection = true;
        self.default_tools = tools;
        self.default_instructions = instructions;

        let warm_up_request = self.build_defaults_request(/*stream*/ true);
        self.session = self
            .session
            .clone()
            .with_reconnect_warm_up_request(warm_up_request.clone());
        self.session.warm_up(warm_up_request).await?;
        Ok(self)
    }

    /// Append a message to the pending input for the next step.
    pub fn push(&mut self, message: ChatMessage) {
        self.pending.push(message);
    }

    /// Convenience helper for pushing a single user text message.
    pub fn push_user_text(&mut self, text: impl Into<String>) {
        self.push(ChatMessage::user(text).build());
    }

    /// Drain and return all pending messages (for advanced use).
    pub fn take_pending(&mut self) -> Vec<ChatMessage> {
        std::mem::take(&mut self.pending)
    }

    /// Stream the next step using the current pending messages.
    ///
    /// On success, pending messages are cleared.
    pub async fn stream_next_with_cancel(&mut self) -> Result<ChatStreamHandle, LlmError> {
        let messages = self.pending.clone();
        let handle = self.stream_messages_with_cancel(messages).await?;
        self.pending.clear();
        Ok(handle)
    }

    /// Stream an explicit message list as an incremental step.
    pub async fn stream_messages_with_cancel(
        &self,
        messages: Vec<ChatMessage>,
    ) -> Result<ChatStreamHandle, LlmError> {
        let mut req = ChatRequest::new(messages).with_streaming(true);

        // Ensure Responses API is enabled even if the underlying client defaults differ.
        req = req.with_provider_option(
            "openai",
            serde_json::json!({
                "responsesApi": {
                    "enabled": true
                }
            }),
        );

        // If defaults are not cached on the connection, include them on every request.
        if !self.cache_defaults_on_connection {
            if let Some(t) = self.default_tools.clone() {
                req = req.with_tools(t);
            }
            if let Some(instructions) = self.default_instructions.clone() {
                req = req.with_provider_option(
                    "openai",
                    serde_json::json!({
                        "responsesApi": {
                            "instructions": instructions
                        }
                    }),
                );
            }
        }

        self.session.chat_stream_request_with_cancel(req).await
    }

    fn build_defaults_request(&self, stream: bool) -> ChatRequest {
        let mut req = ChatRequest::new(Vec::new()).with_streaming(stream);

        if let Some(t) = self.default_tools.clone() {
            req = req.with_tools(t);
        }

        let mut openai_obj = serde_json::json!({
            "responsesApi": {
                "enabled": true
            }
        });

        if let Some(instructions) = self.default_instructions.clone()
            && let Some(m) = openai_obj
                .get_mut("responsesApi")
                .and_then(|v| v.as_object_mut())
        {
            m.insert(
                "instructions".to_string(),
                serde_json::Value::String(instructions),
            );
        }

        req.with_provider_option("openai", openai_obj)
    }
}

#[cfg(not(feature = "openai-websocket"))]
#[allow(dead_code)]
fn _feature_gate_note() -> Result<(), LlmError> {
    Err(LlmError::UnsupportedOperation(
        "OpenAiIncrementalWebSocketSession requires the `openai-websocket` feature".to_string(),
    ))
}
