//! OpenAI Chat Completions Standard (external, minimal – no streaming)
//!
//! Converts core ChatInput/ChatResult to/from OpenAI's Chat Completions JSON.

use siumai_core::error::LlmError;
use siumai_core::execution::chat::{
    ChatInput, ChatParsedContentCore, ChatParsedToolCallCore, ChatRequestTransformer,
    ChatResponseTransformer, ChatResult, ChatRole, ChatUsage,
};
use siumai_core::execution::streaming::ChatStreamEventCore;
use siumai_core::types::FinishReasonCore;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone)]
pub struct OpenAiChatStandard {
    adapter: Option<Arc<dyn OpenAiChatAdapter>>,
}

impl OpenAiChatStandard {
    pub fn new() -> Self {
        Self { adapter: None }
    }
    pub fn with_adapter(adapter: Arc<dyn OpenAiChatAdapter>) -> Self {
        Self {
            adapter: Some(adapter),
        }
    }

    pub fn create_request_transformer(&self, provider_id: &str) -> Arc<dyn ChatRequestTransformer> {
        Arc::new(OpenAiChatRequestTransformer {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        })
    }
    pub fn create_response_transformer(
        &self,
        provider_id: &str,
    ) -> Arc<dyn ChatResponseTransformer> {
        Arc::new(OpenAiChatResponseTransformer {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        })
    }
    pub fn create_stream_converter(
        &self,
        provider_id: &str,
    ) -> Arc<dyn siumai_core::execution::streaming::ChatStreamEventConverterCore> {
        Arc::new(OpenAiChatStreamConverter::new(
            provider_id.to_string(),
            self.adapter.clone(),
        ))
    }
}

impl Default for OpenAiChatStandard {
    fn default() -> Self {
        Self::new()
    }
}

/// Adapter for provider-specific diffs
pub trait OpenAiChatAdapter: Send + Sync {
    fn transform_request(
        &self,
        _req: &ChatInput,
        _body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        Ok(())
    }
    fn transform_response(&self, _resp: &mut serde_json::Value) -> Result<(), LlmError> {
        Ok(())
    }
    fn chat_endpoint(&self) -> &str {
        "/chat/completions"
    }
    fn transform_sse_event(&self, _event: &mut serde_json::Value) -> Result<(), LlmError> {
        Ok(())
    }
}

/// Default OpenAI Chat adapter used by the provider crates.
///
/// This adapter is intentionally minimal and only consumes a subset of
/// `ChatInput::extra` keys that are populated by the aggregator:
/// - `openai_reasoning_effort`: serialized `ReasoningEffort`
/// - `openai_service_tier`: serialized `ServiceTier`
///
/// Additional OpenAI-specific options can be moved here gradually so
/// that JSON shaping logic lives close to the standard instead of the
/// aggregator crate.
#[derive(Clone, Default)]
pub struct OpenAiDefaultChatAdapter;

impl OpenAiChatAdapter for OpenAiDefaultChatAdapter {
    fn transform_request(
        &self,
        input: &ChatInput,
        body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        // Reasoning effort (o1/o3 models)
        if let Some(v) = input.extra.get("openai_reasoning_effort") {
            body["reasoning_effort"] = v.clone();
        }

        // Service tier preference
        if let Some(v) = input.extra.get("openai_service_tier") {
            body["service_tier"] = v.clone();
        }

        // Modalities (e.g., ["text","audio"])
        if let Some(v) = input.extra.get("openai_modalities") {
            body["modalities"] = v.clone();
        }

        // Audio configuration
        if let Some(v) = input.extra.get("openai_audio") {
            body["audio"] = v.clone();
        }

        // Prediction content
        if let Some(v) = input.extra.get("openai_prediction") {
            body["prediction"] = v.clone();
        }

        // Web search options
        if let Some(v) = input.extra.get("openai_web_search_options") {
            body["web_search_options"] = v.clone();
        }

        Ok(())
    }
}

#[derive(Clone)]
struct OpenAiChatStreamConverter {
    provider_id: String,
    adapter: Option<Arc<dyn OpenAiChatAdapter>>,
    started: Arc<AtomicBool>,
    ended: Arc<AtomicBool>,
}

impl OpenAiChatStreamConverter {
    fn new(provider_id: String, adapter: Option<Arc<dyn OpenAiChatAdapter>>) -> Self {
        Self {
            provider_id,
            adapter,
            started: Arc::new(AtomicBool::new(false)),
            ended: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl siumai_core::execution::streaming::ChatStreamEventConverterCore for OpenAiChatStreamConverter {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }
    fn convert_event(
        &self,
        event: eventsource_stream::Event,
    ) -> Vec<Result<siumai_core::execution::streaming::ChatStreamEventCore, LlmError>> {
        let mut out = Vec::new();
        let data = event.data;
        let trimmed = data.trim();
        if trimmed == "[DONE]" {
            return out;
        }

        // Emit a StreamStart event once per stream, on the first non-[DONE] chunk.
        if self
            .started
            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            out.push(Ok(
                siumai_core::execution::streaming::ChatStreamEventCore::StreamStart {},
            ));
        }

        let mut v: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                return vec![Err(LlmError::ParseError(format!(
                    "Invalid SSE JSON: {}",
                    e
                )))];
            }
        };
        if let Some(adapter) = &self.adapter {
            if let Err(e) = adapter.transform_sse_event(&mut v) {
                return vec![Err(e)];
            }
        }
        if let Some(choices) = v.get("choices").and_then(|c| c.as_array()) {
            for (i, ch) in choices.iter().enumerate() {
                if let Some(delta) = ch.get("delta") {
                    if let Some(text) = delta.get("content").and_then(|s| s.as_str()) {
                        out.push(Ok(ChatStreamEventCore::ContentDelta {
                            delta: text.to_string(),
                            index: Some(i),
                        }));
                    }
                    if let Some(tc_arr) = delta.get("tool_calls").and_then(|a| a.as_array()) {
                        for tc in tc_arr.iter() {
                            let id = tc.get("id").and_then(|s| s.as_str()).map(|s| s.to_string());
                            let func = tc.get("function").cloned().unwrap_or(serde_json::json!({}));
                            let name = func
                                .get("name")
                                .and_then(|s| s.as_str())
                                .map(|s| s.to_string());
                            let args = func
                                .get("arguments")
                                .and_then(|s| s.as_str())
                                .map(|s| s.to_string());
                            out.push(Ok(ChatStreamEventCore::ToolCallDelta {
                                id,
                                function_name: name,
                                arguments_delta: args,
                                index: Some(i),
                            }));
                        }
                    }
                }
            }
        }
        if let Some(usage) = v.get("usage") {
            let pt = usage
                .get("prompt_tokens")
                .and_then(|n| n.as_u64())
                .unwrap_or(0) as u32;
            let ct = usage
                .get("completion_tokens")
                .and_then(|n| n.as_u64())
                .unwrap_or(0) as u32;
            let tt = usage
                .get("total_tokens")
                .and_then(|n| n.as_u64())
                .unwrap_or(pt as u64 + ct as u64) as u32;
            out.push(Ok(ChatStreamEventCore::UsageUpdate {
                prompt_tokens: pt,
                completion_tokens: ct,
                total_tokens: tt,
            }));
        }

        // Emit a StreamEnd event when finish_reason is present in the chunk.
        if let Some(choices) = v.get("choices").and_then(|c| c.as_array()) {
            if let Some(first) = choices.first() {
                if let Some(reason_str) = first.get("finish_reason").and_then(|r| r.as_str()) {
                    let finish_reason = FinishReasonCore::from_str(Some(reason_str));
                    // Mark that we have emitted a StreamEnd for this stream.
                    self.ended.store(true, Ordering::Relaxed);
                    out.push(Ok(ChatStreamEventCore::StreamEnd { finish_reason }));
                }
            }
        }

        if out.is_empty() {
            out.push(Ok(
                siumai_core::execution::streaming::ChatStreamEventCore::Custom {
                    event_type: "openai:unknown_chunk".into(),
                    data: v,
                },
            ));
        }
        out
    }

    fn handle_stream_end(
        &self,
    ) -> Option<Result<siumai_core::execution::streaming::ChatStreamEventCore, LlmError>> {
        // If a StreamEnd was already emitted for this stream, do not emit another.
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

#[derive(Clone)]
struct OpenAiChatRequestTransformer {
    provider_id: String,
    adapter: Option<Arc<dyn OpenAiChatAdapter>>,
}

impl ChatRequestTransformer for OpenAiChatRequestTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }
    fn transform_chat(&self, req: &ChatInput) -> Result<serde_json::Value, LlmError> {
        // Map messages
        let messages = req
            .messages
            .iter()
            .map(|m| match m.role {
                ChatRole::System => serde_json::json!({ "role": "system", "content": m.content }),
                ChatRole::User => serde_json::json!({ "role": "user", "content": m.content }),
                ChatRole::Assistant => {
                    serde_json::json!({ "role": "assistant", "content": m.content })
                }
            })
            .collect::<Vec<_>>();

        let mut body = serde_json::json!({ "messages": messages });
        if let Some(model) = &req.model {
            body["model"] = serde_json::json!(model);
        }
        if let Some(n) = req.max_tokens {
            body["max_tokens"] = serde_json::json!(n);
        }
        if let Some(t) = req.temperature {
            body["temperature"] = serde_json::json!(t);
        }
        if let Some(t) = req.top_p {
            body["top_p"] = serde_json::json!(t);
        }
        if let Some(p) = req.presence_penalty {
            body["presence_penalty"] = serde_json::json!(p);
        }
        if let Some(f) = req.frequency_penalty {
            body["frequency_penalty"] = serde_json::json!(f);
        }
        if let Some(stop) = &req.stop {
            body["stop"] = serde_json::json!(stop);
        }

        if let Some(obj) = body.as_object_mut() {
            for (k, v) in &req.extra {
                obj.insert(k.clone(), v.clone());
            }
        }

        if let Some(adapter) = &self.adapter {
            adapter.transform_request(req, &mut body)?;
        }
        Ok(body)
    }
}

#[derive(Clone)]
struct OpenAiChatResponseTransformer {
    provider_id: String,
    adapter: Option<Arc<dyn OpenAiChatAdapter>>,
}

impl ChatResponseTransformer for OpenAiChatResponseTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }
    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResult, LlmError> {
        let mut resp = raw.clone();
        if let Some(adapter) = &self.adapter {
            adapter.transform_response(&mut resp)?;
        }

        // Parse minimal OpenAI chat response
        let choice0 = resp
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));

        let message = choice0
            .get("message")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));

        let content = message
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let finish_reason_str = choice0.get("finish_reason").and_then(|v| v.as_str());
        let finish_reason = FinishReasonCore::from_str(finish_reason_str);
        let usage = if let Some(u) = resp.get("usage") {
            Some(ChatUsage {
                prompt_tokens: u["prompt_tokens"].as_u64().unwrap_or(0) as u32,
                completion_tokens: u["completion_tokens"].as_u64().unwrap_or(0) as u32,
                total_tokens: u["total_tokens"].as_u64().unwrap_or(0) as u32,
            })
        } else {
            None
        };

        // Build parsed_content with text + tool calls + optional reasoning/thinking.
        let mut parsed = ChatParsedContentCore::default();
        parsed.text = content.clone();

        // Tool calls (function/tool_call schema)
        if let Some(tcs) = message.get("tool_calls").and_then(|v| v.as_array()) {
            for tc in tcs {
                let id = tc.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());
                let func = tc.get("function").cloned().unwrap_or(serde_json::json!({}));
                let name = func
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let arguments = func
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
                    .unwrap_or_else(|| serde_json::json!({}));

                if !name.is_empty() {
                    parsed.tool_calls.push(ChatParsedToolCallCore {
                        id,
                        name,
                        arguments,
                    });
                }
            }
        }

        // Reasoning / thinking content (best-effort, provider-specific)
        if let Some(reasoning) = message
            .get("reasoning")
            .or_else(|| message.get("thinking"))
            .and_then(|v| v.as_str())
        {
            if !reasoning.is_empty() {
                parsed.thinking = Some(reasoning.to_string());
            }
        }

        Ok(ChatResult {
            content,
            finish_reason,
            usage,
            metadata: Default::default(),
            parsed_content: Some(parsed),
        })
    }
}
