//! Gemini Chat standard (core-level skeleton).
//!
//! This module provides a minimal, provider-agnostic Gemini Chat standard
//! built on top of `siumai-core` execution traits. The implementation is
//! intentionally small and will be expanded during later refactor phases.

use siumai_core::error::LlmError;
use siumai_core::execution::chat::{
    ChatInput, ChatRequestTransformer, ChatResponseTransformer, ChatResult, ChatRole, ChatUsage,
};
use siumai_core::execution::streaming::{ChatStreamEventConverterCore, ChatStreamEventCore};
use siumai_core::types::FinishReasonCore;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Core-level Gemini Chat standard.
#[derive(Clone, Default)]
pub struct GeminiChatStandard {
    adapter: Option<Arc<dyn GeminiChatAdapter>>,
}

impl GeminiChatStandard {
    /// Create a new standard with no adapter.
    pub fn new() -> Self {
        Self { adapter: None }
    }

    /// Create a new standard with a provider-specific adapter.
    pub fn with_adapter(adapter: Arc<dyn GeminiChatAdapter>) -> Self {
        Self {
            adapter: Some(adapter),
        }
    }

    /// Create a request transformer for the given provider id.
    pub fn create_request_transformer(&self, provider_id: &str) -> Arc<dyn ChatRequestTransformer> {
        Arc::new(GeminiChatRequestTx {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        })
    }

    /// Create a response transformer for the given provider id.
    pub fn create_response_transformer(
        &self,
        provider_id: &str,
    ) -> Arc<dyn ChatResponseTransformer> {
        Arc::new(GeminiChatResponseTx {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        })
    }

    /// Create a streaming event converter for the given provider id.
    pub fn create_stream_converter(
        &self,
        provider_id: &str,
    ) -> Arc<dyn ChatStreamEventConverterCore> {
        Arc::new(GeminiChatStreamConv {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
            ended: Arc::new(AtomicBool::new(false)),
        })
    }
}

/// Adapter hook for provider-specific Gemini behaviour.
///
/// The initial skeleton leaves all hooks as no-ops so that the standard
/// can be wired gradually from the aggregator and provider crates.
pub trait GeminiChatAdapter: Send + Sync {
    /// Transform request JSON after the standard mapping.
    fn transform_request(
        &self,
        _input: &ChatInput,
        _body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    /// Transform response JSON before mapping into core types.
    fn transform_response(&self, _resp: &mut serde_json::Value) -> Result<(), LlmError> {
        Ok(())
    }

    /// Transform SSE event payload prior to conversion.
    fn transform_sse_event(&self, _event: &mut serde_json::Value) -> Result<(), LlmError> {
        Ok(())
    }
}

/// Default Gemini Chat adapter.
///
/// Reads Gemini-specific configuration from `ChatInput::extra` and injects it
/// into the final request JSON:
/// - `gemini_code_execution` → `code_execution` entry in `tools`
/// - `gemini_search_grounding` → `google_search` entry in `tools`
/// - `gemini_file_search` → `file_search` entry in `tools`
/// - `gemini_response_mime_type` → `generationConfig.response_mime_type`
#[derive(Clone, Default)]
pub struct GeminiDefaultChatAdapter;

impl GeminiChatAdapter for GeminiDefaultChatAdapter {
    fn transform_request(
        &self,
        input: &ChatInput,
        body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        // Tools array
        let mut tools = body
            .get("tools")
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default();

        if let Some(v) = input.extra.get("gemini_code_execution") {
            if v.get("enabled").and_then(|b| b.as_bool()).unwrap_or(false) {
                tools.push(serde_json::json!({ "code_execution": {} }));
            }
        }

        if let Some(v) = input.extra.get("gemini_search_grounding") {
            if v.get("enabled").and_then(|b| b.as_bool()).unwrap_or(false) {
                let mut tool = serde_json::json!({ "google_search": {} });
                if let Some(dr) = v.get("dynamic_retrieval_config") {
                    tool["google_search"]["dynamic_retrieval_config"] = dr.clone();
                }
                tools.push(tool);
            }
        }

        if let Some(v) = input.extra.get("gemini_file_search") {
            if let Some(stores) = v.get("file_search_store_names")
                && stores.is_array()
            {
                tools.push(serde_json::json!({
                    "file_search": {
                        "file_search_store_names": stores.clone()
                    }
                }));
            }
        }

        if !tools.is_empty() {
            body["tools"] = serde_json::Value::Array(tools);
        }

        if let Some(mime) = input
            .extra
            .get("gemini_response_mime_type")
            .and_then(|v| v.as_str())
        {
            body["generationConfig"]["response_mime_type"] = serde_json::json!(mime);
        }

        Ok(())
    }
}

#[derive(Clone)]
struct GeminiChatRequestTx {
    provider_id: String,
    adapter: Option<Arc<dyn GeminiChatAdapter>>,
}

impl ChatRequestTransformer for GeminiChatRequestTx {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_chat(&self, input: &ChatInput) -> Result<serde_json::Value, LlmError> {
        // Naive mapping: map core chat messages into a simple Gemini-like
        // structure. This is intentionally minimal and will be refined.
        let contents = input
            .messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    ChatRole::System => "user", // Gemini has user/model; system will be treated as user content.
                    ChatRole::User => "user",
                    ChatRole::Assistant => "model",
                };
                serde_json::json!({
                    "role": role,
                    "parts": [{
                        "text": m.content.clone()
                    }],
                })
            })
            .collect::<Vec<_>>();

        let mut body = serde_json::json!({
            "contents": contents,
        });

        if let Some(model) = &input.model {
            body["model"] = serde_json::json!(model);
        }
        if let Some(max) = input.max_tokens {
            body["generationConfig"]["maxOutputTokens"] = serde_json::json!(max);
        }
        if let Some(t) = input.temperature {
            body["generationConfig"]["temperature"] = serde_json::json!(t);
        }
        if let Some(tp) = input.top_p {
            body["generationConfig"]["topP"] = serde_json::json!(tp);
        }
        if let Some(stops) = &input.stop {
            body["generationConfig"]["stopSequences"] = serde_json::json!(stops);
        }

        if let Some(adapter) = &self.adapter {
            adapter.transform_request(input, &mut body)?;
        }

        Ok(body)
    }
}

#[derive(Clone)]
struct GeminiChatResponseTx {
    provider_id: String,
    adapter: Option<Arc<dyn GeminiChatAdapter>>,
}

impl ChatResponseTransformer for GeminiChatResponseTx {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResult, LlmError> {
        let mut resp = raw.clone();
        if let Some(adapter) = &self.adapter {
            adapter.transform_response(&mut resp)?;
        }

        // Use the unified content parser to build the core content model.
        let parsed = crate::gemini::utils::parse_content_core(&resp);

        // Canonicalize finishReason into a minimal string representation, then map
        // into the core-level FinishReasonCore enum.
        let finish_reason_str = resp
            .get("candidates")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|cand| cand.get("finishReason"))
            .and_then(|v| v.as_str())
            .map(|s| match s {
                // Gemini enums: https://ai.google.dev/api/generate-content#FinishReason
                "STOP" => "stop".to_string(),
                "MAX_TOKENS" => "length".to_string(),
                "SAFETY" | "RECITATION" => "content_filter".to_string(),
                other => other.to_lowercase(),
            });
        let finish_reason = FinishReasonCore::from_str(finish_reason_str.as_deref());

        let usage = resp.get("usageMetadata").map(|u| ChatUsage {
            prompt_tokens: u
                .get("promptTokenCount")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            completion_tokens: u
                .get("candidatesTokenCount")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            total_tokens: u
                .get("totalTokenCount")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
        });

        // Populate the core parsed content model so higher layers can
        // reconstruct richer MessageContent (text + tool calls + thinking)
        // without re-parsing provider JSON.
        let parsed_content = siumai_core::execution::chat::ChatParsedContentCore {
            text: parsed.text.clone(),
            tool_calls: parsed
                .tool_calls
                .iter()
                .map(|t| siumai_core::execution::chat::ChatParsedToolCallCore {
                    id: None,
                    name: t.name.clone(),
                    arguments: t.arguments.clone(),
                })
                .collect(),
            thinking: parsed.thinking.clone(),
        };

        Ok(ChatResult {
            content: parsed.text,
            finish_reason,
            usage,
            metadata: Default::default(),
            parsed_content: Some(parsed_content),
        })
    }
}

#[derive(Clone)]
struct GeminiChatStreamConv {
    provider_id: String,
    adapter: Option<Arc<dyn GeminiChatAdapter>>,
    ended: Arc<AtomicBool>,
}

impl ChatStreamEventConverterCore for GeminiChatStreamConv {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn convert_event(
        &self,
        event: eventsource_stream::Event,
    ) -> Vec<Result<ChatStreamEventCore, LlmError>> {
        use serde::Deserialize;

        let data = event.data;
        let trimmed = data.trim();
        if trimmed.is_empty() || trimmed == "[DONE]" {
            return Vec::new();
        }

        // Parse into a minimal Gemini stream response structure.
        #[derive(Debug, Clone, Deserialize)]
        struct GeminiStreamResponseCore {
            candidates: Option<Vec<GeminiCandidateCore>>,
            #[serde(rename = "usageMetadata")]
            usage_metadata: Option<GeminiUsageMetadataCore>,
        }

        #[derive(Debug, Clone, Deserialize)]
        struct GeminiCandidateCore {
            content: Option<GeminiContentCore>,
            #[serde(rename = "finishReason")]
            finish_reason: Option<String>,
        }

        #[derive(Debug, Clone, Deserialize)]
        struct GeminiContentCore {
            parts: Option<Vec<GeminiPartCore>>,
            #[allow(dead_code)]
            role: Option<String>,
        }

        #[derive(Debug, Clone, Deserialize)]
        struct GeminiPartCore {
            text: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            thought: Option<bool>,
        }

        #[derive(Debug, Clone, Deserialize)]
        struct GeminiUsageMetadataCore {
            #[serde(rename = "promptTokenCount")]
            prompt_token_count: Option<u32>,
            #[serde(rename = "candidatesTokenCount")]
            candidates_token_count: Option<u32>,
            #[serde(rename = "totalTokenCount")]
            total_token_count: Option<u32>,
        }

        let mut v: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                return vec![Err(LlmError::ParseError(format!(
                    "Invalid Gemini SSE JSON: {e}"
                )))];
            }
        };

        // Allow adapter to mutate raw JSON before standard parsing.
        if let Some(adapter) = &self.adapter {
            if let Err(e) = adapter.transform_sse_event(&mut v) {
                return vec![Err(e)];
            }
        }

        let resp: GeminiStreamResponseCore = match serde_json::from_value(v.clone()) {
            Ok(r) => r,
            Err(e) => {
                // If structure does not match expected shape, fall back to a Custom event.
                return vec![Ok(ChatStreamEventCore::Custom {
                    event_type: "gemini:unknown_chunk".into(),
                    data: v,
                })];
            }
        };

        let mut out = Vec::new();

        // Content deltas: accumulate all non-empty text parts.
        if let Some(candidates) = &resp.candidates {
            for (cand_idx, cand) in candidates.iter().enumerate() {
                if let Some(content) = &cand.content {
                    if let Some(parts) = &content.parts {
                        for (part_idx, part) in parts.iter().enumerate() {
                            if let Some(text) = &part.text {
                                if !text.is_empty() && !part.thought.unwrap_or(false) {
                                    out.push(Ok(ChatStreamEventCore::ContentDelta {
                                        delta: text.clone(),
                                        index: Some(cand_idx.max(part_idx)),
                                    }));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Thinking delta: first part with thought == true.
        if let Some(candidates) = &resp.candidates {
            if let Some(first) = candidates.first() {
                if let Some(content) = &first.content {
                    if let Some(parts) = &content.parts {
                        if let Some(thinking) = parts.iter().find_map(|p| {
                            if p.thought.unwrap_or(false) {
                                p.text.clone()
                            } else {
                                None
                            }
                        }) {
                            out.push(Ok(ChatStreamEventCore::ThinkingDelta { delta: thinking }));
                        }
                    }
                }
            }
        }

        // Usage update from usageMetadata.
        if let Some(meta) = &resp.usage_metadata {
            let prompt_tokens = meta.prompt_token_count.unwrap_or(0);
            let completion_tokens = meta.candidates_token_count.unwrap_or(0);
            let total_tokens = meta
                .total_token_count
                .unwrap_or(prompt_tokens.saturating_add(completion_tokens));
            out.push(Ok(ChatStreamEventCore::UsageUpdate {
                prompt_tokens,
                completion_tokens,
                total_tokens,
            }));
        }

        // Emit StreamEnd when finishReason is present on the first candidate.
        if let Some(candidates) = &resp.candidates {
            if let Some(first) = candidates.first() {
                if let Some(fr) = &first.finish_reason {
                    let canon = match fr.as_str() {
                        "STOP" => "stop".to_string(),
                        "MAX_TOKENS" => "length".to_string(),
                        "SAFETY" | "RECITATION" => "content_filter".to_string(),
                        other => other.to_lowercase(),
                    };
                    let finish_reason = FinishReasonCore::from_str(Some(&canon));
                    // Mark that a StreamEnd has been emitted for this stream.
                    self.ended.store(true, Ordering::Relaxed);
                    out.push(Ok(ChatStreamEventCore::StreamEnd { finish_reason }));
                }
            }
        }

        if out.is_empty() {
            out.push(Ok(ChatStreamEventCore::Custom {
                event_type: "gemini:unknown_chunk".into(),
                data: v,
            }));
        }

        out
    }

    fn handle_stream_end(&self) -> Option<Result<ChatStreamEventCore, LlmError>> {
        // Avoid emitting duplicate StreamEnd events.
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
