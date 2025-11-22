//! Anthropic Messages Chat Standard (initial version)
//!
//! Goals:
//! - Provide a provider-agnostic Anthropic Chat API standard mapping (request/response/streaming)
//! - Depend only on `siumai-core` abstractions; concrete providers inject differences via adapter traits
//!
//! Current implementation:
//! - Only defines structures and interfaces (`AnthropicChatStandard` + `AnthropicChatAdapter`);
//!   the internal conversion logic will gradually be migrated from the aggregator crate.

use serde::Deserialize;
use siumai_core::error::LlmError;
use siumai_core::execution::chat::{
    ChatInput, ChatRequestTransformer, ChatResponseTransformer, ChatResult,
};
use siumai_core::execution::streaming::{ChatStreamEventConverterCore, ChatStreamEventCore};
use siumai_core::types::FinishReasonCore;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Anthropic Chat standard entrypoint
#[derive(Clone, Default)]
pub struct AnthropicChatStandard {
    adapter: Option<Arc<dyn AnthropicChatAdapter>>,
}

impl AnthropicChatStandard {
    pub fn new() -> Self {
        Self { adapter: None }
    }

    pub fn with_adapter(adapter: Arc<dyn AnthropicChatAdapter>) -> Self {
        Self {
            adapter: Some(adapter),
        }
    }

    /// Create a request transformer (core ChatInput -> Anthropic JSON).
    pub fn create_request_transformer(&self, provider_id: &str) -> Arc<dyn ChatRequestTransformer> {
        Arc::new(AnthropicChatRequestTx {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        })
    }

    /// Create a response transformer (Anthropic JSON -> core ChatResult).
    pub fn create_response_transformer(
        &self,
        provider_id: &str,
    ) -> Arc<dyn ChatResponseTransformer> {
        Arc::new(AnthropicChatResponseTx {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        })
    }

    /// Create a streaming event transformer (SSE -> ChatStreamEventCore).
    pub fn create_stream_converter(
        &self,
        provider_id: &str,
    ) -> Arc<dyn ChatStreamEventConverterCore> {
        Arc::new(AnthropicChatStreamConv {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
            ended: Arc::new(AtomicBool::new(false)),
        })
    }
}

/// Anthropic Chat provider adapter used to inject small provider-specific differences.
pub trait AnthropicChatAdapter: Send + Sync {
    /// Request JSON adjustments (model aliases, special parameters, etc.).
    fn transform_request(
        &self,
        _input: &ChatInput,
        _body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    /// Response JSON adjustments (compatibility with different versions/fields).
    fn transform_response(&self, _resp: &mut serde_json::Value) -> Result<(), LlmError> {
        Ok(())
    }

    /// SSE event JSON adjustments.
    fn transform_sse_event(&self, _event: &mut serde_json::Value) -> Result<(), LlmError> {
        Ok(())
    }

    /// Messages endpoint path (default `/v1/messages`).
    fn messages_endpoint(&self) -> &str {
        "/v1/messages"
    }
}

/// Default Anthropic Chat adapter.
///
/// Currently very lightweight: it only renames Anthropic-specific values
/// injected by the aggregator via `ChatInput::extra` into protocol fields:
///
/// - `anthropic_thinking` → `thinking`
/// - `anthropic_response_format` → `response_format`
///
/// The concrete JSON structures are still determined by the typed options
/// mapping logic in the aggregator.
#[derive(Clone, Default)]
pub struct AnthropicDefaultChatAdapter;

impl AnthropicChatAdapter for AnthropicDefaultChatAdapter {
    fn transform_request(
        &self,
        _input: &ChatInput,
        body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        if let Some(obj) = body.as_object_mut() {
            if let Some(thinking) = obj.remove("anthropic_thinking") {
                obj.insert("thinking".to_string(), thinking);
            }
            if let Some(response_format) = obj.remove("anthropic_response_format") {
                obj.insert("response_format".to_string(), response_format);
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
struct AnthropicChatRequestTx {
    provider_id: String,
    adapter: Option<Arc<dyn AnthropicChatAdapter>>,
}

impl ChatRequestTransformer for AnthropicChatRequestTx {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_chat(&self, input: &ChatInput) -> Result<serde_json::Value, LlmError> {
        if input.model.as_deref().unwrap_or("").is_empty() {
            return Err(LlmError::InvalidParameter("Model must be specified".into()));
        }

        // Use utils to build Anthropic Messages structure + optional system prompt.
        let (messages, system) = crate::anthropic::utils::build_messages_payload(input)?;

        let mut body = serde_json::json!({
            "model": input.model.clone().unwrap_or_default(),
            "messages": messages,
        });

        if let Some(sys) = system {
            body["system"] = serde_json::json!(sys);
        }
        if let Some(mt) = input.max_tokens {
            body["max_tokens"] = serde_json::json!(mt);
        }
        if let Some(t) = input.temperature {
            body["temperature"] = serde_json::json!(t);
        }
        if let Some(tp) = input.top_p {
            body["top_p"] = serde_json::json!(tp);
        }

        if let Some(obj) = body.as_object_mut() {
            for (k, v) in &input.extra {
                obj.insert(k.clone(), v.clone());
            }
        }

        if let Some(adapter) = &self.adapter {
            adapter.transform_request(input, &mut body)?;
        }
        Ok(body)
    }
}

#[derive(Clone)]
struct AnthropicChatResponseTx {
    provider_id: String,
    adapter: Option<Arc<dyn AnthropicChatAdapter>>,
}

impl ChatResponseTransformer for AnthropicChatResponseTx {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResult, LlmError> {
        let mut resp = raw.clone();
        if let Some(adapter) = &self.adapter {
            adapter.transform_response(&mut resp)?;
        }

        // Use utils to parse a minimal ChatResult (can be enhanced in utils later).
        Ok(crate::anthropic::utils::parse_minimal_chat_result(&resp))
    }
}

#[derive(Clone)]
struct AnthropicChatStreamConv {
    provider_id: String,
    adapter: Option<Arc<dyn AnthropicChatAdapter>>,
    ended: Arc<AtomicBool>,
}

impl ChatStreamEventConverterCore for AnthropicChatStreamConv {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn convert_event(
        &self,
        event: eventsource_stream::Event,
    ) -> Vec<Result<ChatStreamEventCore, LlmError>> {
        #[derive(Debug, Clone, Deserialize, serde::Serialize)]
        struct AnthropicStreamEvent {
            #[serde(rename = "type")]
            kind: String,
            #[serde(default)]
            delta: Option<AnthropicDelta>,
            #[serde(default)]
            usage: Option<AnthropicUsage>,
        }

        #[derive(Debug, Clone, Deserialize, serde::Serialize)]
        struct AnthropicDelta {
            #[serde(default)]
            text: Option<String>,
            #[serde(default)]
            thinking: Option<String>,
            #[serde(default)]
            stop_reason: Option<String>,
        }

        #[derive(Debug, Clone, Deserialize, serde::Serialize)]
        struct AnthropicUsage {
            #[serde(default)]
            input_tokens: Option<u32>,
            #[serde(default)]
            output_tokens: Option<u32>,
        }

        let mut out = Vec::new();
        let data = event.data.trim();
        if data.is_empty() || data == "[DONE]" {
            return out;
        }

        // First parse into raw JSON to allow the adapter to modify or detect error events.
        let mut raw: serde_json::Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(e) => {
                return vec![Err(LlmError::ParseError(format!(
                    "Anthropic SSE parse error: {}",
                    e
                )))];
            }
        };

        // Allow the adapter to adjust the underlying JSON (for proxies/variants, etc.).
        if let Some(adapter) = &self.adapter
            && let Err(e) = adapter.transform_sse_event(&mut raw)
        {
            return vec![Err(e)];
        }

        // If the event represents an error (no type field, only an error object),
        // map it directly to an Error event.
        if let Some(err_obj) = raw.get("error") {
            let msg = err_obj
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Anthropic streaming error")
                .to_string();
            return vec![Ok(ChatStreamEventCore::Error { error: msg })];
        }

        // Try to parse into the standard streaming event structure; on failure,
        // return a Custom event to aid debugging.
        let evt: AnthropicStreamEvent = match serde_json::from_value(raw.clone()) {
            Ok(v) => v,
            Err(_) => {
                return vec![Ok(ChatStreamEventCore::Custom {
                    event_type: "anthropic:unknown_chunk".into(),
                    data: raw,
                })];
            }
        };

        match evt.kind.as_str() {
            "message_start" => {
                // Emit a StreamStart event; the aggregator will inject provider metadata.
                out.push(Ok(ChatStreamEventCore::StreamStart {}));
            }
            "message_delta" | "content_block_delta" | "message_delta_input_json_delta" => {
                if let Some(d) = evt.delta {
                    if let Some(text) = d.text
                        && !text.is_empty()
                    {
                        out.push(Ok(ChatStreamEventCore::ContentDelta {
                            delta: text,
                            index: None,
                        }));
                    }
                    if let Some(th) = d.thinking
                        && !th.is_empty()
                    {
                        out.push(Ok(ChatStreamEventCore::ThinkingDelta { delta: th }));
                    }

                    // When a delta carries a stop_reason, emit a StreamEnd with the concrete reason.
                    if let Some(stop_reason) = d.stop_reason.as_deref() {
                        let finish_reason =
                            crate::anthropic::utils::parse_finish_reason_core(Some(stop_reason))
                                .unwrap_or(FinishReasonCore::Other(stop_reason.to_string()));
                        self.ended.store(true, Ordering::Relaxed);
                        out.push(Ok(ChatStreamEventCore::StreamEnd {
                            finish_reason: Some(finish_reason),
                        }));
                    }
                }

                // Some implementations carry usage in message_delta (e.g., fixtures);
                // emit a UsageUpdate in that case.
                if let Some(u) = evt.usage {
                    let prompt_tokens = u.input_tokens.unwrap_or(0);
                    let completion_tokens = u.output_tokens.unwrap_or(0);
                    let total_tokens = prompt_tokens.saturating_add(completion_tokens);
                    out.push(Ok(ChatStreamEventCore::UsageUpdate {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                    }));
                }
            }
            "message_stop" | "message_end" => {
                if let Some(u) = evt.usage {
                    let prompt_tokens = u.input_tokens.unwrap_or(0);
                    let completion_tokens = u.output_tokens.unwrap_or(0);
                    let total_tokens = prompt_tokens.saturating_add(completion_tokens);
                    out.push(Ok(ChatStreamEventCore::UsageUpdate {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                    }));
                }

                // message_stop/message_end also represent a natural end. If we haven't
                // parsed a stop_reason from deltas before, synthesize a default Stop here.
                self.ended.store(true, Ordering::Relaxed);
                out.push(Ok(ChatStreamEventCore::StreamEnd {
                    finish_reason: Some(FinishReasonCore::Stop),
                }));
            }
            _ => {}
        }

        if out.is_empty() {
            // Forward unknown events as Custom to aid higher-level debugging.
            let raw: serde_json::Value =
                serde_json::from_str(data).unwrap_or_else(|_| serde_json::json!({}));
            out.push(Ok(ChatStreamEventCore::Custom {
                event_type: "anthropic:unknown_chunk".into(),
                data: raw,
            }));
        }
        out
    }

    fn handle_stream_end(&self) -> Option<Result<ChatStreamEventCore, LlmError>> {
        // Avoid emitting a second StreamEnd if one has already been sent.
        if self
            .ended
            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return None;
        }

        Some(Ok(ChatStreamEventCore::StreamEnd {
            finish_reason: None,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_converter_maps_refusal_stop_reason_to_content_filter() {
        let std = AnthropicChatStandard::new();
        let conv = std.create_stream_converter("anthropic");

        let evt = eventsource_stream::Event {
            id: String::new(),
            event: String::new(),
            data: r#"{
                "type": "message_delta",
                "delta": {
                    "text": "blocked content",
                    "stop_reason": "refusal"
                },
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5
                }
            }"#
            .to_string(),
            retry: None,
        };

        let out = conv.convert_event(evt);
        // Expect at least one StreamEnd with ContentFilter
        let end = out
            .into_iter()
            .find_map(|e| match e {
                Ok(ChatStreamEventCore::StreamEnd { finish_reason }) => finish_reason,
                _ => None,
            })
            .expect("expected StreamEnd from refusal stop_reason");

        assert_eq!(end, FinishReasonCore::ContentFilter);
    }

    #[test]
    fn stream_converter_emits_stop_on_message_stop() {
        let std = AnthropicChatStandard::new();
        let conv = std.create_stream_converter("anthropic");

        let evt = eventsource_stream::Event {
            id: String::new(),
            event: String::new(),
            data: r#"{
                "type": "message_stop",
                "usage": {
                    "input_tokens": 2,
                    "output_tokens": 3
                }
            }"#
            .to_string(),
            retry: None,
        };

        let out = conv.convert_event(evt);
        let end = out
            .into_iter()
            .find_map(|e| match e {
                Ok(ChatStreamEventCore::StreamEnd { finish_reason }) => finish_reason,
                _ => None,
            })
            .expect("expected StreamEnd from message_stop");

        assert_eq!(end, FinishReasonCore::Stop);
    }
}
