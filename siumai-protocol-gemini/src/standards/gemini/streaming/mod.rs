//! Gemini streaming implementation using eventsource-stream (protocol layer)
//!
//! Provides SSE event conversion for Gemini streaming responses.
//! The legacy GeminiStreaming client has been removed in favor of the unified HttpChatExecutor.

use super::types::GeminiConfig;
use crate::error::LlmError;
use crate::streaming::SseEventConverter;
use crate::streaming::{
    ChatStreamEvent, LanguageModelV3Source, LanguageModelV3StreamPart, SharedV3ProviderMetadata,
    StreamStateTracker, V3UnsupportedPartBehavior,
};
use crate::types::Usage;
use crate::types::{
    ChatResponse, ChatStreamFileData, ChatStreamFinishInfo, ChatStreamPart, FinishReason,
    MessageContent, ResponseMetadata,
};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Debug, Default, Clone)]
struct GeminiFunctionCallSerializeState {
    name: Option<String>,
    arguments: String,
    last_emitted_args_json: Option<String>,
}

#[derive(Debug, Default, Clone)]
struct GeminiSerializeState {
    function_calls_by_id: std::collections::HashMap<String, GeminiFunctionCallSerializeState>,
    active_reasoning_provider_metadata: Option<SharedV3ProviderMetadata>,
    terminal_emitted: bool,
}

/// Gemini stream response structure
#[derive(Debug, Clone, Deserialize)]
struct GeminiStreamResponse {
    candidates: Option<Vec<GeminiCandidate>>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsageMetadata>,
    #[serde(rename = "promptFeedback")]
    prompt_feedback: Option<super::types::PromptFeedback>,
    #[serde(rename = "serviceTier")]
    service_tier: Option<String>,
    /// Gateway-only metadata used for stable tool call id propagation in re-serialized streams.
    #[serde(default)]
    siumai: Option<GeminiSiumaiMetadata>,
}

/// Gemini candidate structure
#[derive(Debug, Clone, Deserialize)]
struct GeminiCandidate {
    content: Option<GeminiContent>,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
    #[serde(rename = "finishMessage")]
    finish_message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "groundingMetadata")]
    grounding_metadata: Option<super::types::GroundingMetadata>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "urlContextMetadata")]
    url_context_metadata: Option<super::types::UrlContextMetadata>,
    #[serde(default, rename = "safetyRatings")]
    safety_ratings: Vec<super::types::SafetyRating>,
}

/// Gemini content structure
#[derive(Debug, Clone, Deserialize)]
struct GeminiContent {
    parts: Option<Vec<GeminiPart>>,
    #[allow(dead_code)]
    // Role appears in some responses but is not required by our unified event model
    role: Option<String>,
}

/// Gemini part structure
#[derive(Debug, Clone, Deserialize)]
struct GeminiPart {
    text: Option<String>,
    /// Optional. Whether this is a thought summary (for thinking models)
    #[serde(skip_serializing_if = "Option::is_none")]
    thought: Option<bool>,
    /// Optional. An opaque signature for the thought so it can be reused in subsequent requests.
    #[serde(skip_serializing_if = "Option::is_none", rename = "thoughtSignature")]
    thought_signature: Option<String>,
    #[serde(rename = "executableCode", skip_serializing_if = "Option::is_none")]
    executable_code: Option<GeminiExecutableCode>,
    #[serde(
        rename = "codeExecutionResult",
        skip_serializing_if = "Option::is_none"
    )]
    code_execution_result: Option<GeminiCodeExecutionResult>,
    #[serde(rename = "functionCall", skip_serializing_if = "Option::is_none")]
    function_call: Option<GeminiFunctionCall>,
    #[serde(rename = "functionResponse", skip_serializing_if = "Option::is_none")]
    function_response: Option<GeminiFunctionResponse>,
    #[serde(rename = "inlineData", skip_serializing_if = "Option::is_none")]
    inline_data: Option<GeminiInlineData>,
}

#[derive(Debug, Clone, Deserialize)]
struct GeminiInlineData {
    #[serde(rename = "mimeType")]
    mime_type: String,
    data: String,
}

#[derive(Debug, Clone, Deserialize)]
struct GeminiExecutableCode {
    language: Option<String>,
    code: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct GeminiCodeExecutionResult {
    outcome: Option<String>,
    output: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct GeminiFunctionCall {
    name: String,
    args: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct GeminiFunctionResponse {
    name: String,
    response: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize)]
struct GeminiSiumaiMetadata {
    #[serde(rename = "toolCallId")]
    tool_call_id: Option<String>,
}

/// Gemini usage metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiUsageMetadata {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "promptTokenCount"
    )]
    prompt_token_count: Option<u32>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "cachedContentTokenCount"
    )]
    cached_content_token_count: Option<u32>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "candidatesTokenCount"
    )]
    candidates_token_count: Option<u32>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "totalTokenCount"
    )]
    total_token_count: Option<u32>,
    /// Number of tokens used for thinking (only for thinking models)
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "thoughtsTokenCount"
    )]
    thoughts_token_count: Option<u32>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "trafficType"
    )]
    traffic_type: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "promptTokensDetails"
    )]
    prompt_tokens_details: Option<Vec<GeminiTokenDetail>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "candidatesTokensDetails"
    )]
    candidates_tokens_details: Option<Vec<GeminiTokenDetail>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiTokenDetail {
    modality: String,
    #[serde(rename = "tokenCount")]
    token_count: u32,
}

fn gemini_finish_provider_metadata(
    grounding_metadata: Option<&super::types::GroundingMetadata>,
    url_context_metadata: Option<&super::types::UrlContextMetadata>,
    safety_ratings: &[super::types::SafetyRating],
    prompt_feedback: Option<&super::types::PromptFeedback>,
    usage_metadata: Option<&GeminiUsageMetadata>,
    finish_message: Option<&str>,
    service_tier: Option<&str>,
) -> std::collections::HashMap<String, serde_json::Value> {
    let mut meta = std::collections::HashMap::new();
    meta.insert(
        "promptFeedback".to_string(),
        prompt_feedback
            .and_then(|value| serde_json::to_value(value).ok())
            .unwrap_or(serde_json::Value::Null),
    );
    meta.insert(
        "groundingMetadata".to_string(),
        grounding_metadata
            .and_then(|value| serde_json::to_value(value).ok())
            .unwrap_or(serde_json::Value::Null),
    );
    meta.insert(
        "urlContextMetadata".to_string(),
        url_context_metadata
            .and_then(|value| serde_json::to_value(value).ok())
            .unwrap_or(serde_json::Value::Null),
    );
    meta.insert(
        "safetyRatings".to_string(),
        if safety_ratings.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::to_value(safety_ratings).unwrap_or(serde_json::Value::Null)
        },
    );
    meta.insert(
        "usageMetadata".to_string(),
        usage_metadata
            .and_then(|value| serde_json::to_value(value).ok())
            .unwrap_or(serde_json::Value::Null),
    );
    meta.insert(
        "finishMessage".to_string(),
        finish_message
            .map(|value| serde_json::json!(value))
            .unwrap_or(serde_json::Value::Null),
    );
    meta.insert(
        "serviceTier".to_string(),
        service_tier
            .map(|value| serde_json::json!(value))
            .unwrap_or(serde_json::Value::Null),
    );

    meta
}

/// Gemini event converter
#[derive(Clone)]
pub struct GeminiEventConverter {
    config: GeminiConfig,
    include_raw_chunks: bool,
    /// Track if StreamStart has been emitted
    state_tracker: StreamStateTracker,
    /// Deduplicate sources across stream chunks
    seen_source_keys: Arc<Mutex<std::collections::HashSet<String>>>,
    /// Monotonic id counter for emitted stable text/reasoning blocks
    next_block_id: Arc<AtomicU64>,
    /// Pair executableCode -> codeExecutionResult across chunks
    pending_code_execution_id: Arc<Mutex<Option<String>>>,
    /// Track the active text block id for stable text parts
    current_text_block_id: Arc<Mutex<Option<String>>>,
    /// Track the active reasoning block id for stable reasoning parts
    current_reasoning_block_id: Arc<Mutex<Option<String>>>,
    /// Preserve the latest usage snapshot so finish parts can carry usage even
    /// if the terminal chunk omits usageMetadata.
    latest_usage: Arc<Mutex<Option<Usage>>>,

    serialize_state: Arc<Mutex<GeminiSerializeState>>,
    v3_unsupported_part_behavior: V3UnsupportedPartBehavior,
    emit_function_response_tool_results: bool,
}

impl GeminiEventConverter {
    pub fn new(config: GeminiConfig) -> Self {
        Self {
            config,
            include_raw_chunks: false,
            state_tracker: StreamStateTracker::new(),
            seen_source_keys: Arc::new(Mutex::new(std::collections::HashSet::new())),
            next_block_id: Arc::new(AtomicU64::new(0)),
            pending_code_execution_id: Arc::new(Mutex::new(None)),
            current_text_block_id: Arc::new(Mutex::new(None)),
            current_reasoning_block_id: Arc::new(Mutex::new(None)),
            latest_usage: Arc::new(Mutex::new(None)),
            serialize_state: Arc::new(Mutex::new(GeminiSerializeState::default())),
            v3_unsupported_part_behavior: V3UnsupportedPartBehavior::default(),
            emit_function_response_tool_results: false,
        }
    }

    pub fn with_v3_unsupported_part_behavior(
        mut self,
        behavior: V3UnsupportedPartBehavior,
    ) -> Self {
        self.v3_unsupported_part_behavior = behavior;
        self
    }

    pub fn with_include_raw_chunks(mut self, include_raw_chunks: bool) -> Self {
        self.include_raw_chunks = include_raw_chunks;
        self
    }

    /// Emit Gemini `functionResponse` frames for v3 `tool-result` parts when serializing.
    ///
    /// Notes:
    /// - This is primarily a gateway/proxy feature: Gemini responses usually don't contain
    ///   functionResponse parts; they are commonly sent in the *next request*.
    /// - When enabled, Siumai will include a `siumai.toolCallId` field in the JSON frame so the
    ///   decoder can associate results with the original tool call id.
    pub fn with_emit_function_response_tool_results(mut self, enabled: bool) -> Self {
        self.emit_function_response_tool_results = enabled;
        self
    }

    fn provider_metadata_key(&self) -> &'static str {
        if let Some(override_key) = self.config.provider_metadata_key.as_deref() {
            let key = override_key.trim().to_ascii_lowercase();
            if key.contains("vertex") {
                return "vertex";
            }
            if key.contains("google") {
                return "google";
            }
        }

        if self.config.base_url.contains("aiplatform.googleapis.com")
            || self.config.base_url.contains("vertex")
        {
            "vertex"
        } else {
            "google"
        }
    }

    fn generate_id(&self) -> String {
        self.config.generate_id()
    }

    fn thought_signature_provider_metadata(
        &self,
        sig: Option<&String>,
    ) -> Option<crate::types::StreamProviderMetadata> {
        let sig = sig?;
        if sig.trim().is_empty() {
            return None;
        }
        let key = self.provider_metadata_key();
        let mut inner = serde_json::Map::new();
        inner.insert(
            "thoughtSignature".to_string(),
            serde_json::Value::String(sig.clone()),
        );

        let mut provider_metadata = crate::types::StreamProviderMetadata::new();
        provider_metadata.insert(key.to_string(), serde_json::Value::Object(inner));
        Some(provider_metadata)
    }

    fn take_reasoning_block_id(&self) -> Option<String> {
        let mut lock = self
            .current_reasoning_block_id
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        lock.take()
    }

    fn take_text_block_id(&self) -> Option<String> {
        let mut lock = self
            .current_text_block_id
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        lock.take()
    }

    fn current_text_block_id(&self) -> Option<String> {
        self.current_text_block_id
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn next_block_id_string(&self) -> String {
        self.next_block_id
            .fetch_add(1, Ordering::Relaxed)
            .to_string()
    }

    fn open_text_lane(
        &self,
        provider_metadata: Option<crate::types::StreamProviderMetadata>,
    ) -> (Vec<LanguageModelV3StreamPart>, String) {
        let mut parts = Vec::new();

        if let Some(id) = self.take_reasoning_block_id() {
            parts.push(LanguageModelV3StreamPart::ReasoningEnd {
                id,
                provider_metadata: None,
            });
        }

        let id = {
            let mut lock = self
                .current_text_block_id
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(id) = lock.as_ref() {
                id.clone()
            } else {
                let id = self.next_block_id_string();
                *lock = Some(id.clone());
                parts.push(LanguageModelV3StreamPart::TextStart {
                    id: id.clone(),
                    provider_metadata,
                });
                id
            }
        };

        (parts, id)
    }

    fn open_reasoning_lane(
        &self,
        provider_metadata: Option<crate::types::StreamProviderMetadata>,
    ) -> (Vec<LanguageModelV3StreamPart>, String) {
        let mut parts = Vec::new();

        if let Some(id) = self.take_text_block_id() {
            parts.push(LanguageModelV3StreamPart::TextEnd {
                id,
                provider_metadata: None,
            });
        }

        let id = {
            let mut lock = self
                .current_reasoning_block_id
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(id) = lock.as_ref() {
                id.clone()
            } else {
                let id = self.next_block_id_string();
                *lock = Some(id.clone());
                parts.push(LanguageModelV3StreamPart::ReasoningStart {
                    id: id.clone(),
                    provider_metadata,
                });
                id
            }
        };

        (parts, id)
    }

    fn close_active_content_parts(&self) -> Vec<LanguageModelV3StreamPart> {
        let mut parts = Vec::new();

        if let Some(id) = self.take_text_block_id() {
            parts.push(LanguageModelV3StreamPart::TextEnd {
                id,
                provider_metadata: None,
            });
        }

        if let Some(id) = self.take_reasoning_block_id() {
            parts.push(LanguageModelV3StreamPart::ReasoningEnd {
                id,
                provider_metadata: None,
            });
        }

        parts
    }

    fn remember_usage(&self, usage: &Usage) {
        if let Ok(mut lock) = self.latest_usage.lock() {
            *lock = Some(usage.clone());
        }
    }

    fn latest_usage(&self) -> Option<Usage> {
        self.latest_usage
            .lock()
            .ok()
            .and_then(|usage| usage.clone())
    }

    fn append_stream_start_events(&self, out: &mut Vec<ChatStreamEvent>) {
        if !self.needs_stream_start() {
            return;
        }

        out.push(ChatStreamEvent::StreamStart {
            metadata: self.create_stream_start_metadata(),
        });
        out.push(ChatStreamEvent::Part {
            part: ChatStreamPart::StreamStart { warnings: vec![] },
        });
    }

    fn inject_raw_chunk(
        &self,
        mut events: Vec<ChatStreamEvent>,
        raw_value: serde_json::Value,
    ) -> Vec<ChatStreamEvent> {
        if !self.include_raw_chunks {
            return events;
        }

        let raw_event = ChatStreamEvent::Part {
            part: ChatStreamPart::Raw { raw_value },
        };

        let insert_at = events
            .iter()
            .position(|event| {
                matches!(
                    event,
                    ChatStreamEvent::Part {
                        part: ChatStreamPart::StreamStart { .. }
                    }
                )
            })
            .map(|index| index + 1)
            .unwrap_or(0);

        events.insert(insert_at, raw_event);
        events
    }

    fn build_parse_error_events(
        &self,
        message: String,
        raw_value: Option<serde_json::Value>,
    ) -> Vec<Result<ChatStreamEvent, LlmError>> {
        let mut events = Vec::new();
        self.append_stream_start_events(&mut events);

        if let Some(raw_value) = raw_value {
            events = self.inject_raw_chunk(events, raw_value);
        }

        events.push(ChatStreamEvent::Part {
            part: ChatStreamPart::Error {
                error: serde_json::Value::String(message),
            },
        });

        events.into_iter().map(Ok).collect()
    }

    fn build_error_payload_events(
        &self,
        error: serde_json::Value,
        raw_value: serde_json::Value,
    ) -> Vec<Result<ChatStreamEvent, LlmError>> {
        let mut events = Vec::new();
        self.append_stream_start_events(&mut events);
        let mut events = self.inject_raw_chunk(events, raw_value);
        events.push(ChatStreamEvent::Part {
            part: ChatStreamPart::Error { error },
        });
        events.into_iter().map(Ok).collect()
    }

    fn add_gemini_stream_part(
        &self,
        builder: crate::streaming::EventBuilder,
        part: LanguageModelV3StreamPart,
    ) -> crate::streaming::EventBuilder {
        match part.to_part_event() {
            ChatStreamEvent::Part { part } => builder.add_part(part),
            other => {
                debug_assert!(
                    false,
                    "typed stream parts should convert to ChatStreamEvent::Part, got {other:?}"
                );
                builder
            }
        }
    }

    /// Convert Gemini stream response to multiple ChatStreamEvents
    async fn convert_gemini_response_async(
        &self,
        response: GeminiStreamResponse,
    ) -> Vec<ChatStreamEvent> {
        use crate::streaming::EventBuilder;

        let mut builder = EventBuilder::new();

        // Check if we need to emit StreamStart first
        if self.needs_stream_start() {
            builder = builder
                .add_stream_start(self.create_stream_start_metadata())
                .add_part(ChatStreamPart::StreamStart { warnings: vec![] });
        }

        // Emit normalized sources before content, matching the audited upstream ordering.
        for part in self.extract_source_parts(&response) {
            builder = builder.add_part(part);
        }

        if let Some(candidates) = response.candidates.as_ref() {
            for candidate in candidates {
                let Some(content) = candidate.content.as_ref() else {
                    continue;
                };
                let Some(parts) = content.parts.as_ref() else {
                    continue;
                };

                for part in parts {
                    let provider_metadata =
                        self.thought_signature_provider_metadata(part.thought_signature.as_ref());

                    if let Some(text) = part.text.as_ref() {
                        if text.is_empty() {
                            if provider_metadata.is_some()
                                && !part.thought.unwrap_or(false)
                                && let Some(id) = self.current_text_block_id()
                            {
                                builder = self.add_gemini_stream_part(
                                    builder,
                                    LanguageModelV3StreamPart::TextDelta {
                                        id,
                                        delta: String::new(),
                                        provider_metadata,
                                    },
                                );
                            }
                            continue;
                        }

                        if part.thought.unwrap_or(false) {
                            let (lane_parts, id) =
                                self.open_reasoning_lane(provider_metadata.clone());
                            for lane_part in lane_parts {
                                builder = self.add_gemini_stream_part(builder, lane_part);
                            }
                            builder = self.add_gemini_stream_part(
                                builder,
                                LanguageModelV3StreamPart::ReasoningDelta {
                                    id,
                                    delta: text.clone(),
                                    provider_metadata,
                                },
                            );
                        } else {
                            let (lane_parts, id) = self.open_text_lane(provider_metadata.clone());
                            for lane_part in lane_parts {
                                builder = self.add_gemini_stream_part(builder, lane_part);
                            }
                            builder = self.add_gemini_stream_part(
                                builder,
                                LanguageModelV3StreamPart::TextDelta {
                                    id,
                                    delta: text.clone(),
                                    provider_metadata,
                                },
                            );
                        }
                        continue;
                    }

                    if let Some(inline_data) = part.inline_data.as_ref() {
                        for lane_part in self.close_active_content_parts() {
                            builder = self.add_gemini_stream_part(builder, lane_part);
                        }

                        let file_part = crate::types::ChatStreamFilePart {
                            media_type: inline_data.mime_type.clone(),
                            data: ChatStreamFileData::Base64(inline_data.data.clone()),
                            provider_metadata,
                        };

                        builder = builder.add_part(if part.thought.unwrap_or(false) {
                            ChatStreamPart::ReasoningFile(file_part)
                        } else {
                            ChatStreamPart::File(file_part)
                        });
                    }
                }
            }
        }

        // Process provider-executed tool parts.
        for part in self.extract_code_execution_parts(&response) {
            builder = builder.add_part(part);
        }

        // Process client-executed tool calls (functionCall parts).
        for (tool_name, args_json, thought_sig) in self.extract_function_call_events(&response) {
            let call_id = self.generate_id();

            builder = builder.add_part(crate::types::ChatStreamPart::ToolCall(
                crate::types::ChatStreamToolCall {
                    tool_call_id: call_id,
                    tool_name,
                    input: args_json,
                    provider_executed: None,
                    dynamic: None,
                    provider_metadata: self
                        .thought_signature_provider_metadata(thought_sig.as_ref()),
                },
            ));
        }

        // Process client-provided tool results (functionResponse parts).
        for (tool_name, result, thought_sig) in self.extract_function_response_events(&response) {
            let tool_call_id = response
                .siumai
                .as_ref()
                .and_then(|m| m.tool_call_id.clone())
                .unwrap_or_else(|| self.generate_id());

            builder = builder.add_part(crate::types::ChatStreamPart::ToolResult(
                crate::types::ChatStreamToolResult {
                    tool_call_id,
                    tool_name,
                    result,
                    is_error: None,
                    preliminary: None,
                    dynamic: None,
                    provider_metadata: self
                        .thought_signature_provider_metadata(thought_sig.as_ref()),
                },
            ));
        }

        // Process usage update if available
        if let Some(usage) = self.extract_usage(&response) {
            self.remember_usage(&usage);
        }

        // Handle completion/finish reason
        if let Some(end_response) = self.extract_completion(&response) {
            for lane_part in self.close_active_content_parts() {
                builder = self.add_gemini_stream_part(builder, lane_part);
            }

            let finish_reason = end_response
                .finish_reason
                .clone()
                .unwrap_or(FinishReason::Unknown);
            let finish_usage = end_response
                .usage
                .clone()
                .unwrap_or_else(|| Usage::builder().build());

            builder = builder
                .add_part(ChatStreamPart::Finish {
                    usage: finish_usage,
                    finish_reason: ChatStreamFinishInfo {
                        unified: finish_reason,
                        raw: end_response.raw_finish_reason.clone(),
                    },
                    provider_metadata: end_response.provider_metadata.clone(),
                })
                .add_stream_end(end_response);
        }

        builder.build()
    }

    /// Check if StreamStart event needs to be emitted
    fn needs_stream_start(&self) -> bool {
        self.state_tracker.needs_stream_start()
    }

    fn extract_function_call_events(
        &self,
        response: &GeminiStreamResponse,
    ) -> Vec<(String, String, Option<String>)> {
        let mut out = Vec::new();
        let Some(candidates) = response.candidates.as_ref() else {
            return out;
        };

        for cand in candidates {
            let Some(content) = cand.content.as_ref() else {
                continue;
            };
            let Some(parts) = content.parts.as_ref() else {
                continue;
            };

            for part in parts {
                let Some(call) = part.function_call.as_ref() else {
                    continue;
                };

                let args = call.args.clone().unwrap_or_else(|| serde_json::json!({}));
                let args_json = serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string());
                out.push((call.name.clone(), args_json, part.thought_signature.clone()));
            }
        }

        out
    }

    fn extract_function_response_events(
        &self,
        response: &GeminiStreamResponse,
    ) -> Vec<(String, serde_json::Value, Option<String>)> {
        let mut out = Vec::new();
        let Some(candidates) = response.candidates.as_ref() else {
            return out;
        };

        for cand in candidates {
            let Some(content) = cand.content.as_ref() else {
                continue;
            };
            let Some(parts) = content.parts.as_ref() else {
                continue;
            };

            for part in parts {
                let Some(res) = part.function_response.as_ref() else {
                    continue;
                };

                out.push((
                    res.name.clone(),
                    res.response.clone(),
                    part.thought_signature.clone(),
                ));
            }
        }

        out
    }

    fn has_function_call_parts(&self, response: &GeminiStreamResponse) -> bool {
        let Some(candidates) = response.candidates.as_ref() else {
            return false;
        };
        for cand in candidates {
            if let Some(content) = cand.content.as_ref()
                && let Some(parts) = content.parts.as_ref()
                && parts.iter().any(|p| p.function_call.is_some())
            {
                return true;
            }
        }
        false
    }

    fn extract_code_execution_parts(
        &self,
        response: &GeminiStreamResponse,
    ) -> Vec<crate::types::ChatStreamPart> {
        let mut out: Vec<crate::types::ChatStreamPart> = Vec::new();

        let Some(candidates) = response.candidates.as_ref() else {
            return out;
        };

        for candidate in candidates {
            let Some(content) = candidate.content.as_ref() else {
                continue;
            };
            let Some(parts) = content.parts.as_ref() else {
                continue;
            };

            for part in parts {
                if let Some(exec) = part.executable_code.as_ref() {
                    let id = {
                        let id = self.generate_id();
                        if let Ok(mut lock) = self.pending_code_execution_id.lock() {
                            *lock = Some(id.clone());
                        }
                        id
                    };

                    let input = serde_json::json!({
                        "language": exec.language.clone().unwrap_or_else(|| "PYTHON".to_string()),
                        "code": exec.code.clone().unwrap_or_default()
                    });

                    out.push(crate::types::ChatStreamPart::ToolCall(
                        crate::types::ChatStreamToolCall {
                            tool_call_id: id,
                            tool_name: "code_execution".to_string(),
                            input: serde_json::to_string(&input)
                                .unwrap_or_else(|_| "{}".to_string()),
                            provider_executed: Some(true),
                            dynamic: None,
                            provider_metadata: None,
                        },
                    ));
                }

                if let Some(res) = part.code_execution_result.as_ref() {
                    let id = if let Ok(mut lock) = self.pending_code_execution_id.lock() {
                        lock.take().unwrap_or_else(|| self.generate_id())
                    } else {
                        self.generate_id()
                    };

                    out.push(crate::types::ChatStreamPart::ToolResult(
                        crate::types::ChatStreamToolResult {
                            tool_call_id: id,
                            tool_name: "code_execution".to_string(),
                            result: serde_json::json!({
                                "outcome": res.outcome.clone().unwrap_or_else(|| "OUTCOME_OK".to_string()),
                                "output": res.output.clone().unwrap_or_default()
                            }),
                            is_error: None,
                            preliminary: None,
                            dynamic: None,
                            provider_metadata: None,
                        },
                    ));
                }
            }
        }

        out
    }

    /// Extract normalized source parts from grounding metadata (deduplicated across stream).
    fn extract_source_parts(
        &self,
        response: &GeminiStreamResponse,
    ) -> Vec<crate::types::ChatStreamPart> {
        let mut out: Vec<crate::types::ChatStreamPart> = Vec::new();

        let Some(candidates) = response.candidates.as_ref() else {
            return out;
        };

        let mut seen = self
            .seen_source_keys
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        for cand in candidates {
            let sources = super::sources::extract_sources_with_generate_id(
                cand.grounding_metadata.as_ref(),
                || self.generate_id(),
            );
            for source in sources {
                let key = super::sources::source_key(&source);
                if !seen.insert(key) {
                    continue;
                }

                let source_part = match source.source_type.as_str() {
                    "url" => {
                        let Some(url) = source.url else {
                            continue;
                        };
                        crate::types::SourcePart::Url {
                            url,
                            title: source.title,
                        }
                    }
                    "document" => {
                        let Some(media_type) = source.media_type else {
                            continue;
                        };
                        crate::types::SourcePart::Document {
                            media_type,
                            title: source
                                .title
                                .unwrap_or_else(|| "Unknown Document".to_string()),
                            filename: source.filename,
                        }
                    }
                    _ => continue,
                };

                out.push(crate::types::ChatStreamPart::Source {
                    id: source.id,
                    source: source_part,
                    provider_metadata: None,
                });
            }
        }

        out
    }

    /// Extract completion information
    fn extract_completion(&self, response: &GeminiStreamResponse) -> Option<ChatResponse> {
        let candidate = response.candidates.as_ref()?.first()?;

        if let Some(finish_reason) = &candidate.finish_reason {
            let finish_reason = match finish_reason.as_str() {
                "STOP" => {
                    if self.has_function_call_parts(response) {
                        FinishReason::ToolCalls
                    } else {
                        FinishReason::Stop
                    }
                }
                "MAX_TOKENS" => FinishReason::Length,
                "IMAGE_SAFETY" | "RECITATION" | "SAFETY" | "BLOCKLIST" | "PROHIBITED_CONTENT"
                | "SPII" => FinishReason::ContentFilter,
                "MALFORMED_FUNCTION_CALL" => FinishReason::Error,
                other => FinishReason::Other(other.to_string()),
            };

            // Mark that StreamEnd is being emitted
            self.state_tracker.mark_stream_ended();

            let provider_metadata = {
                let provider_key = self.provider_metadata_key();
                let mut meta = gemini_finish_provider_metadata(
                    candidate.grounding_metadata.as_ref(),
                    candidate.url_context_metadata.as_ref(),
                    &candidate.safety_ratings,
                    response.prompt_feedback.as_ref(),
                    response.usage_metadata.as_ref(),
                    candidate.finish_message.as_deref(),
                    response.service_tier.as_deref(),
                );

                let sources = super::sources::extract_sources_with_generate_id(
                    candidate.grounding_metadata.as_ref(),
                    || self.generate_id(),
                );
                if !sources.is_empty()
                    && let Ok(v) = serde_json::to_value(sources)
                {
                    meta.insert("sources".to_string(), v);
                }

                Some(
                    crate::types::provider_metadata::provider_metadata_from_object(
                        provider_key,
                        meta,
                    ),
                )
            };

            let response = ChatResponse {
                id: None,
                model: None,
                content: MessageContent::Text("".to_string()),
                usage: self.extract_usage(response).or_else(|| self.latest_usage()),
                finish_reason: Some(finish_reason),
                raw_finish_reason: candidate.finish_reason.clone(),
                audio: None,
                system_fingerprint: None,
                service_tier: response.service_tier.clone(),
                warnings: None,
                request: None,
                provider_metadata,
                response: None,
            };

            Some(response)
        } else {
            None
        }
    }

    /// Extract usage information
    fn extract_usage(&self, response: &GeminiStreamResponse) -> Option<Usage> {
        if let Some(meta) = &response.usage_metadata {
            let prompt_tokens = meta.prompt_token_count;
            let text_tokens = meta.candidates_token_count;
            let total_tokens = meta.total_token_count;
            let cached_tokens = meta.cached_content_token_count;
            let reasoning_tokens = meta.thoughts_token_count;
            let completion_tokens = text_tokens
                .zip(reasoning_tokens)
                .map(|(text, reasoning)| text.saturating_add(reasoning))
                .or(text_tokens)
                .or_else(|| {
                    total_tokens
                        .zip(prompt_tokens)
                        .map(|(total, prompt)| total.saturating_sub(prompt))
                });
            let output_text_tokens = text_tokens.or_else(|| {
                completion_tokens.map(|total| total.saturating_sub(reasoning_tokens.unwrap_or(0)))
            });

            let mut builder = Usage::builder().with_raw_usage_value(
                serde_json::to_value(meta).unwrap_or(serde_json::Value::Null),
            );

            if let Some(prompt_tokens) = prompt_tokens {
                builder = builder
                    .prompt_tokens(prompt_tokens)
                    .with_input_total_tokens(prompt_tokens)
                    .with_input_no_cache_tokens(
                        prompt_tokens.saturating_sub(cached_tokens.unwrap_or(0)),
                    );
            }
            if let Some(completion_tokens) = completion_tokens {
                builder = builder
                    .completion_tokens(completion_tokens)
                    .with_output_total_tokens(completion_tokens);
            }
            if let Some(output_text_tokens) = output_text_tokens {
                builder = builder.with_output_text_tokens(output_text_tokens);
            }
            if let Some(total_tokens) = total_tokens {
                builder = builder.total_tokens(total_tokens);
            }
            if let Some(cached_tokens) = cached_tokens {
                builder = builder
                    .with_cached_tokens(cached_tokens)
                    .with_input_cache_read_tokens(cached_tokens);
            }

            if let Some(reasoning_tokens) = reasoning_tokens {
                builder = builder
                    .with_reasoning_tokens(reasoning_tokens)
                    .with_output_reasoning_tokens(reasoning_tokens);
            }

            return Some(builder.build());
        }
        None
    }

    /// Create StreamStart metadata
    fn create_stream_start_metadata(&self) -> ResponseMetadata {
        ResponseMetadata {
            id: None,                               // Gemini doesn't provide ID in stream events
            model: Some(self.config.model.clone()), // Use model from config
            created: Some(chrono::Utc::now()),
            provider: "gemini".to_string(),
            request_id: None,
            headers: None,
            body: None,
        }
    }
}

impl SseEventConverter for GeminiEventConverter {
    fn convert_event(
        &self,
        event: eventsource_stream::Event,
    ) -> Pin<Box<dyn Future<Output = Vec<Result<ChatStreamEvent, LlmError>>> + Send + Sync + '_>>
    {
        Box::pin(async move {
            // Skip done marker or empty events
            if event.data.trim() == "[DONE]" || event.data.trim().is_empty() {
                return vec![];
            }

            let parsed_value: Result<serde_json::Value, _> = serde_json::from_str(&event.data);

            match parsed_value {
                Ok(raw_json) => {
                    if let Some(error) = raw_json.get("error").cloned() {
                        return self.build_error_payload_events(error, raw_json);
                    }

                    match serde_json::from_value::<GeminiStreamResponse>(raw_json.clone()) {
                        Ok(gemini_response) => self
                            .inject_raw_chunk(
                                self.convert_gemini_response_async(gemini_response).await,
                                raw_json,
                            )
                            .into_iter()
                            .map(Ok)
                            .collect(),
                        Err(e) => self.build_parse_error_events(
                            format!("Failed to parse Gemini SSE JSON: {e}"),
                            Some(raw_json),
                        ),
                    }
                }
                Err(e) => self.build_parse_error_events(
                    format!("Failed to parse Gemini SSE JSON: {e}"),
                    Some(serde_json::Value::String(event.data.clone())),
                ),
            }
        })
    }

    fn handle_stream_end(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
        // Gemini normally emits finish_reason in the stream (handled in extract_completion).
        // If we reach here without seeing finish_reason, the model has not transmitted
        // a finish reason (e.g., connection lost, server error, client cancelled).
        // Always emit StreamEnd with Unknown reason so users can detect this.

        // Check if StreamEnd was already emitted
        if !self.state_tracker.needs_stream_end() {
            return None; // StreamEnd already emitted
        }

        let response = ChatResponse {
            id: None,
            model: None,
            content: MessageContent::Text("".to_string()),
            usage: None,
            finish_reason: Some(FinishReason::Unknown),
            raw_finish_reason: None,
            audio: None,
            system_fingerprint: None,
            service_tier: None,
            warnings: None,
            request: None,
            provider_metadata: None,
            response: None,
        };
        Some(Ok(ChatStreamEvent::StreamEnd { response }))
    }

    fn handle_stream_end_events(&self) -> Vec<Result<ChatStreamEvent, LlmError>> {
        if !self.state_tracker.needs_stream_end() {
            return vec![];
        }

        self.state_tracker.mark_stream_ended();

        let usage = self
            .latest_usage()
            .unwrap_or_else(|| Usage::builder().build());
        let mut out = Vec::new();

        for lane_part in self.close_active_content_parts() {
            out.push(Ok(lane_part.to_part_event()));
        }

        out.push(Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::Finish {
                usage: usage.clone(),
                finish_reason: ChatStreamFinishInfo {
                    unified: FinishReason::Unknown,
                    raw: None,
                },
                provider_metadata: None,
            },
        }));

        out.push(Ok(ChatStreamEvent::StreamEnd {
            response: ChatResponse {
                id: None,
                model: None,
                content: MessageContent::Text("".to_string()),
                usage: Some(usage),
                finish_reason: Some(FinishReason::Unknown),
                raw_finish_reason: None,
                audio: None,
                system_fingerprint: None,
                service_tier: None,
                warnings: None,
                request: None,
                provider_metadata: None,
                response: None,
            },
        }));

        out
    }

    fn serialize_event(&self, event: &ChatStreamEvent) -> Result<Vec<u8>, LlmError> {
        self.serialize_event_impl(event)
    }
}

// Legacy GeminiStreaming client has been removed in favor of the unified HttpChatExecutor.
// The GeminiEventConverter is still used for SSE event conversion in tests.

mod serialize;

impl GeminiEventConverter {
    fn serialize_event_impl(&self, event: &ChatStreamEvent) -> Result<Vec<u8>, LlmError> {
        serialize::serialize_event(self, event)
    }
}

#[cfg(test)]
mod tests;
