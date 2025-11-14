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
use std::sync::Arc;

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

        // Minimal Gemini response parsing based on candidates[0].content.parts[*].text.
        let text = resp
            .get("candidates")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|cand| cand.get("content"))
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.as_array())
            .and_then(|parts| {
                parts
                    .iter()
                    .find_map(|part| part.get("text").and_then(|t| t.as_str()))
            })
            .unwrap_or("")
            .to_string();

        // Canonicalize finishReason into a minimal string representation.
        let finish_reason = resp
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

        Ok(ChatResult {
            content: text,
            // The aggregator is responsible for mapping this canonical string
            // into its own FinishReason enum.
            finish_reason,
            usage,
            metadata: Default::default(),
        })
    }
}

#[derive(Clone)]
struct GeminiChatStreamConv {
    provider_id: String,
    adapter: Option<Arc<dyn GeminiChatAdapter>>,
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

        if out.is_empty() {
            out.push(Ok(ChatStreamEventCore::Custom {
                event_type: "gemini:unknown_chunk".into(),
                data: v,
            }));
        }

        out
    }
}
