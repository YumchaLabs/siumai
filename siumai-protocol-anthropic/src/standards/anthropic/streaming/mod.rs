//! Anthropic streaming implementation using eventsource-stream
//!
//! Provides SSE event conversion for Anthropic streaming responses.
//! The legacy AnthropicStreaming client has been removed in favor of the unified HttpChatExecutor.

use super::params::AnthropicParams;
use super::params::StructuredOutputMode;
use super::provider_metadata::AnthropicSource;
use super::server_tools;
use crate::error::LlmError;
use crate::streaming::SseEventConverter;
use crate::streaming::{
    ChatStreamEvent, LanguageModelV3StreamPart, StreamStateTracker, V3UnsupportedPartBehavior,
};
use crate::types::{ChatResponse, FinishReason, MessageContent, ResponseMetadata, Usage};
use eventsource_stream::Event;
use serde::Deserialize;

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Debug, Default, Clone)]
struct AnthropicSerializeState {
    message_id: Option<String>,
    model: Option<String>,
    message_start_emitted: bool,
    next_block_index: usize,
    text_block_index: Option<usize>,
    thinking_block_index: Option<usize>,
    tool_block_index_by_id: std::collections::HashMap<String, usize>,
    started_block_indices: Vec<usize>,
    latest_usage: Option<Usage>,
}

/// Citation document metadata extracted from the prompt (Vercel-aligned).
///
/// Anthropic streaming citation deltas reference documents by index, where the index is the
/// position of the cited document in the prompt's user file parts that have citations enabled.
#[derive(Debug, Clone)]
pub struct AnthropicCitationDocument {
    pub title: String,
    pub filename: Option<String>,
    pub media_type: String,
}

/// Anthropic stream event structure
/// This structure is flexible to handle different event types from Anthropic's SSE stream
#[derive(Debug, Clone, Deserialize)]
struct AnthropicStreamEvent {
    r#type: String,
    #[serde(default)]
    message: Option<AnthropicMessage>,
    #[serde(default)]
    delta: Option<AnthropicDelta>,
    #[serde(default)]
    error: Option<serde_json::Value>,
    #[serde(default)]
    usage: Option<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    // Some Anthropic events may include context management information at the top-level.
    context_management: Option<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    // Kept for forward-compatibility with Anthropic SSE payloads (serde needs the field even if unused)
    index: Option<usize>,
    #[serde(default)]
    #[allow(dead_code)]
    // Some Anthropic events provide a content_block object we don't consume yet; retained to avoid parse failures
    content_block: Option<serde_json::Value>,
}

/// Anthropic message structure
#[derive(Debug, Clone, Deserialize)]
struct AnthropicMessage {
    id: Option<String>,
    model: Option<String>,
    #[allow(dead_code)]
    // Role is not consumed by our unified event model, but appears in message_start payloads
    role: Option<String>,
    #[allow(dead_code)]
    // Raw content blocks not needed for our delta-based pipeline; retain for serde compatibility
    content: Option<Vec<AnthropicContent>>,
    #[allow(dead_code)]
    // Final stop reason may appear on message events; parsing handled elsewhere
    stop_reason: Option<String>,
    #[allow(dead_code)]
    // Stop sequence may appear on message events; retained for finish metadata
    stop_sequence: Option<String>,
    #[allow(dead_code)]
    // Usage may be attached to message_start; retained for finish metadata
    usage: Option<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    // Container metadata may be attached to message_start for code execution / skills.
    container: Option<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    // Context management metadata may be attached to message_start.
    context_management: Option<serde_json::Value>,
}

/// Anthropic content structure
#[derive(Debug, Clone, Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    #[allow(dead_code)]
    // Different content block types exist; we only consume text via deltas
    content_type: String,
    #[allow(dead_code)]
    // Some events carry full text here; our converter aggregates from deltas instead
    text: Option<String>,
}

/// Anthropic delta structure
/// Supports different delta types: text_delta, input_json_delta, thinking_delta, etc.
#[derive(Debug, Clone, Deserialize)]
struct AnthropicDelta {
    #[serde(rename = "type")]
    #[serde(default)]
    #[allow(dead_code)]
    // Delta subtype (text_delta, input_json_delta, etc.); not required for our current transformations
    delta_type: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    // Partial JSON chunks for tool inputs; not emitted as separate events in our model yet
    partial_json: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    signature: Option<String>,
    #[serde(default)]
    citation: Option<serde_json::Value>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    // Stop sequence token for deltas; usage is reflected via finish events elsewhere
    stop_sequence: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    // Container updates may be attached to message_delta events for agent skills.
    container: Option<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    // Context management updates may be attached to message_delta events.
    context_management: Option<serde_json::Value>,
}

/// Anthropic event converter
#[derive(Clone)]
pub struct AnthropicEventConverter {
    #[allow(dead_code)]
    // Retained for potential future behavior toggles; not read in the current converter
    config: AnthropicParams,
    state_tracker: StreamStateTracker,
    tool_use_ids_by_index: Arc<Mutex<std::collections::HashMap<usize, String>>>,
    content_block_type_by_index: Arc<Mutex<std::collections::HashMap<usize, String>>>,
    thinking_signature_by_index: Arc<Mutex<std::collections::HashMap<usize, String>>>,
    redacted_thinking_data_by_index: Arc<Mutex<std::collections::HashMap<usize, String>>>,
    citation_documents: Vec<AnthropicCitationDocument>,
    sources_by_id: Arc<Mutex<std::collections::HashMap<String, AnthropicSource>>>,
    tool_names_by_id: Arc<Mutex<std::collections::HashMap<String, String>>>,
    mcp_server_name_by_id: Arc<Mutex<std::collections::HashMap<String, String>>>,
    json_tool_seen: Arc<AtomicBool>,
    vercel_stream_start_emitted: Arc<AtomicBool>,
    seen_error: Arc<AtomicBool>,
    vercel_response_id: Arc<Mutex<Option<String>>>,
    vercel_model_id: Arc<Mutex<Option<String>>>,
    vercel_stop_sequence: Arc<Mutex<Option<String>>>,
    vercel_usage: Arc<Mutex<Option<serde_json::Value>>>,
    vercel_container: Arc<Mutex<Option<serde_json::Value>>>,
    vercel_context_management: Arc<Mutex<Option<serde_json::Value>>>,
    serialize_state: Arc<Mutex<AnthropicSerializeState>>,
    v3_unsupported_part_behavior: V3UnsupportedPartBehavior,
}

impl AnthropicEventConverter {
    pub fn new(config: AnthropicParams) -> Self {
        Self {
            config,
            state_tracker: StreamStateTracker::new(),
            tool_use_ids_by_index: Arc::new(Mutex::new(std::collections::HashMap::new())),
            content_block_type_by_index: Arc::new(Mutex::new(std::collections::HashMap::new())),
            thinking_signature_by_index: Arc::new(Mutex::new(std::collections::HashMap::new())),
            redacted_thinking_data_by_index: Arc::new(Mutex::new(std::collections::HashMap::new())),
            citation_documents: Vec::new(),
            sources_by_id: Arc::new(Mutex::new(std::collections::HashMap::new())),
            tool_names_by_id: Arc::new(Mutex::new(std::collections::HashMap::new())),
            mcp_server_name_by_id: Arc::new(Mutex::new(std::collections::HashMap::new())),
            json_tool_seen: Arc::new(AtomicBool::new(false)),
            vercel_stream_start_emitted: Arc::new(AtomicBool::new(false)),
            seen_error: Arc::new(AtomicBool::new(false)),
            vercel_response_id: Arc::new(Mutex::new(None)),
            vercel_model_id: Arc::new(Mutex::new(None)),
            vercel_stop_sequence: Arc::new(Mutex::new(None)),
            vercel_usage: Arc::new(Mutex::new(None)),
            vercel_container: Arc::new(Mutex::new(None)),
            vercel_context_management: Arc::new(Mutex::new(None)),
            serialize_state: Arc::new(Mutex::new(AnthropicSerializeState::default())),
            v3_unsupported_part_behavior: V3UnsupportedPartBehavior::default(),
        }
    }

    pub fn with_citation_documents(mut self, docs: Vec<AnthropicCitationDocument>) -> Self {
        self.citation_documents = docs;
        self
    }

    pub fn with_v3_unsupported_part_behavior(
        mut self,
        behavior: V3UnsupportedPartBehavior,
    ) -> Self {
        self.v3_unsupported_part_behavior = behavior;
        self
    }

    fn record_source(&self, source: AnthropicSource) {
        if let Ok(mut map) = self.sources_by_id.lock() {
            map.insert(source.id.clone(), source);
        }
    }

    fn record_content_block_type(&self, index: usize, block_type: String) {
        if let Ok(mut map) = self.content_block_type_by_index.lock() {
            map.insert(index, block_type);
        }
    }

    fn get_content_block_type(&self, index: usize) -> Option<String> {
        self.content_block_type_by_index
            .lock()
            .ok()
            .and_then(|map| map.get(&index).cloned())
    }

    fn append_thinking_signature(&self, index: usize, delta: String) {
        if delta.is_empty() {
            return;
        }
        if let Ok(mut map) = self.thinking_signature_by_index.lock() {
            let entry = map.entry(index).or_default();
            entry.push_str(&delta);
        }
    }

    fn record_redacted_thinking_data(&self, index: usize, data: String) {
        if data.is_empty() {
            return;
        }
        if let Ok(mut map) = self.redacted_thinking_data_by_index.lock() {
            map.insert(index, data);
        }
    }

    fn build_stream_provider_metadata(
        &self,
    ) -> Option<
        std::collections::HashMap<String, std::collections::HashMap<String, serde_json::Value>>,
    > {
        let mut anthropic = std::collections::HashMap::new();

        if let Ok(v) = self.vercel_container.lock()
            && let Some(container) = v.clone()
            && let Some(mapped) = super::utils::map_container_provider_metadata(&container)
        {
            anthropic.insert("container".to_string(), mapped);
        }

        if let Ok(v) = self.vercel_context_management.lock()
            && let Some(cm) = v.clone()
            && let Some(mapped) = super::utils::map_context_management_provider_metadata(&cm)
        {
            anthropic.insert("contextManagement".to_string(), mapped);
        }

        if let Ok(map) = self.thinking_signature_by_index.lock()
            && let Some((_, sig)) = map.iter().min_by_key(|(k, _)| *k)
            && !sig.is_empty()
        {
            anthropic.insert("thinking_signature".to_string(), serde_json::json!(sig));
        }

        if let Ok(map) = self.redacted_thinking_data_by_index.lock()
            && let Some((_, data)) = map.iter().min_by_key(|(k, _)| *k)
            && !data.is_empty()
        {
            anthropic.insert(
                "redacted_thinking_data".to_string(),
                serde_json::json!(data),
            );
        }

        if let Ok(map) = self.sources_by_id.lock()
            && !map.is_empty()
        {
            let mut sources: Vec<_> = map.values().cloned().collect();
            sources.sort_by(|a, b| a.id.cmp(&b.id));
            if let Ok(v) = serde_json::to_value(sources) {
                anthropic.insert("sources".to_string(), v);
            }
        }

        if anthropic.is_empty() {
            None
        } else {
            let mut all = std::collections::HashMap::new();
            all.insert("anthropic".to_string(), anthropic);
            Some(all)
        }
    }

    fn record_vercel_message_start(&self, message: &AnthropicMessage) {
        if let Ok(mut id) = self.vercel_response_id.lock() {
            *id = message.id.clone();
        }
        if let Ok(mut model) = self.vercel_model_id.lock() {
            *model = message.model.clone();
        }
        if let Ok(mut stop_seq) = self.vercel_stop_sequence.lock() {
            *stop_seq = message.stop_sequence.clone();
        }
        if let Ok(mut usage) = self.vercel_usage.lock() {
            *usage = message.usage.clone();
        }
        if let Ok(mut container) = self.vercel_container.lock() {
            *container = message.container.clone();
        }
        if let Ok(mut cm) = self.vercel_context_management.lock() {
            *cm = message.context_management.clone();
        }
    }

    fn merge_vercel_usage(&self, patch: &serde_json::Value) {
        let Ok(mut usage) = self.vercel_usage.lock() else {
            return;
        };

        let Some(patch_obj) = patch.as_object() else {
            return;
        };

        match usage.as_mut() {
            Some(serde_json::Value::Object(base)) => {
                for (k, v) in patch_obj {
                    base.insert(k.clone(), v.clone());
                }
            }
            None => {
                *usage = Some(serde_json::Value::Object(patch_obj.clone()));
            }
            Some(_) => {}
        }
    }

    fn finish_reason_unified(reason: &FinishReason) -> String {
        match reason {
            FinishReason::Stop => "stop".to_string(),
            FinishReason::StopSequence => "stop".to_string(),
            FinishReason::Length => "length".to_string(),
            FinishReason::ToolCalls => "tool-calls".to_string(),
            FinishReason::ContentFilter => "content-filter".to_string(),
            FinishReason::Error => "error".to_string(),
            FinishReason::Unknown => "unknown".to_string(),
            FinishReason::Other(s) => s.clone(),
        }
    }

    fn vercel_stream_start_event(&self) -> Option<ChatStreamEvent> {
        if self
            .vercel_stream_start_emitted
            .swap(true, Ordering::Relaxed)
        {
            return None;
        }

        Some(ChatStreamEvent::Custom {
            event_type: "anthropic:stream-start".to_string(),
            data: serde_json::json!({
                "type": "stream-start",
                "warnings": [],
            }),
        })
    }

    fn vercel_response_metadata_event(&self) -> Option<ChatStreamEvent> {
        let id = self
            .vercel_response_id
            .lock()
            .ok()
            .and_then(|v| v.clone())?;
        let model_id = self.vercel_model_id.lock().ok().and_then(|v| v.clone())?;

        Some(ChatStreamEvent::Custom {
            event_type: "anthropic:response-metadata".to_string(),
            data: serde_json::json!({
                "type": "response-metadata",
                "id": id,
                "modelId": model_id,
            }),
        })
    }

    fn vercel_text_start_event(id: usize) -> ChatStreamEvent {
        ChatStreamEvent::Custom {
            event_type: "anthropic:text-start".to_string(),
            data: serde_json::json!({
                "type": "text-start",
                "id": id.to_string(),
            }),
        }
    }

    #[allow(dead_code)]
    fn vercel_text_delta_event(id: usize, delta: String) -> ChatStreamEvent {
        ChatStreamEvent::Custom {
            event_type: "anthropic:text-delta".to_string(),
            data: serde_json::json!({
                "type": "text-delta",
                "id": id.to_string(),
                "delta": delta,
            }),
        }
    }

    fn vercel_text_end_event(id: usize) -> ChatStreamEvent {
        ChatStreamEvent::Custom {
            event_type: "anthropic:text-end".to_string(),
            data: serde_json::json!({
                "type": "text-end",
                "id": id.to_string(),
            }),
        }
    }

    fn should_stream_json_tool_as_text(&self) -> bool {
        self.config.structured_output_mode == Some(StructuredOutputMode::JsonTool)
    }

    fn vercel_finish_event(
        &self,
        raw_stop_reason: Option<&str>,
        finish_reason: &FinishReason,
    ) -> ChatStreamEvent {
        let unified = Self::finish_reason_unified(finish_reason);

        let usage_raw = self
            .vercel_usage
            .lock()
            .ok()
            .and_then(|v| v.clone())
            .unwrap_or_else(|| serde_json::json!({}));

        let input_tokens = usage_raw
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let output_tokens = usage_raw
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cache_creation_input_tokens = usage_raw
            .get("cache_creation_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cache_read_input_tokens = usage_raw
            .get("cache_read_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let input_no_cache = input_tokens
            .saturating_sub(cache_creation_input_tokens)
            .saturating_sub(cache_read_input_tokens);

        let stop_sequence = self
            .vercel_stop_sequence
            .lock()
            .ok()
            .and_then(|v| v.clone());

        let container = self
            .vercel_container
            .lock()
            .ok()
            .and_then(|v| v.clone())
            .and_then(|v| super::utils::map_container_provider_metadata(&v))
            .unwrap_or(serde_json::Value::Null);

        let context_management = self
            .vercel_context_management
            .lock()
            .ok()
            .and_then(|v| v.clone())
            .and_then(|v| super::utils::map_context_management_provider_metadata(&v))
            .unwrap_or(serde_json::Value::Null);

        ChatStreamEvent::Custom {
            event_type: "anthropic:finish".to_string(),
            data: serde_json::json!({
                "type": "finish",
                "finishReason": {
                    "raw": raw_stop_reason.map(|s| serde_json::json!(s)).unwrap_or(serde_json::Value::Null),
                    "unified": unified,
                },
                "providerMetadata": {
                    "anthropic": {
                        "cacheCreationInputTokens": cache_creation_input_tokens,
                        "container": container,
                        "contextManagement": context_management,
                        "stopSequence": stop_sequence.map(|s| serde_json::json!(s)).unwrap_or(serde_json::Value::Null),
                        "usage": usage_raw,
                    }
                },
                "usage": {
                    "inputTokens": {
                        "total": input_tokens,
                        "cacheRead": cache_read_input_tokens,
                        "cacheWrite": cache_creation_input_tokens,
                        "noCache": input_no_cache,
                    },
                    "outputTokens": {
                        "total": output_tokens,
                    },
                    "raw": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cache_creation_input_tokens": cache_creation_input_tokens,
                        "cache_read_input_tokens": cache_read_input_tokens,
                    }
                },
            }),
        }
    }

    /// Convert Anthropic stream event to one or more ChatStreamEvents
    fn convert_anthropic_event(&self, event: AnthropicStreamEvent) -> Vec<ChatStreamEvent> {
        use crate::streaming::EventBuilder;

        match event.r#type.as_str() {
            "error" => {
                self.seen_error.store(true, Ordering::Relaxed);
                let msg = event
                    .error
                    .as_ref()
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| {
                        event.error.as_ref().and_then(|e| {
                            serde_json::to_string(e)
                                .ok()
                                .map(|json| format!("Anthropic streaming error: {json}"))
                        })
                    })
                    .unwrap_or_else(|| "Anthropic streaming error (unknown)".to_string());

                vec![ChatStreamEvent::Error { error: msg }]
            }
            "message_start" => {
                if let Some(message) = event.message {
                    self.record_vercel_message_start(&message);
                    let metadata = ResponseMetadata {
                        id: message.id,
                        model: message.model,
                        created: Some(chrono::Utc::now()),
                        provider: "anthropic".to_string(),
                        request_id: None,
                    };
                    let mut out: Vec<ChatStreamEvent> = Vec::new();
                    out.push(ChatStreamEvent::StreamStart { metadata });
                    if let Some(evt) = self.vercel_stream_start_event() {
                        out.push(evt);
                    }
                    if let Some(evt) = self.vercel_response_metadata_event() {
                        out.push(evt);
                    }
                    out
                } else {
                    vec![]
                }
            }
            "content_block_start" => {
                let Some(content_block) = event.content_block else {
                    return vec![];
                };

                let block_type = content_block
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                let effective_block_type = if block_type == "tool_use"
                    && content_block.get("name").and_then(|v| v.as_str()) == Some("json")
                {
                    self.json_tool_seen.store(true, Ordering::Relaxed);
                    "json_tool_use"
                } else {
                    block_type
                };

                if let Some(idx) = event.index {
                    self.record_content_block_type(idx, effective_block_type.to_string());
                }

                match effective_block_type {
                    "text" => {
                        if self.should_stream_json_tool_as_text() {
                            return vec![];
                        }
                        if let Some(idx) = event.index {
                            vec![Self::vercel_text_start_event(idx)]
                        } else {
                            vec![]
                        }
                    }
                    "thinking" => {
                        if let Some(idx) = event.index {
                            vec![ChatStreamEvent::Custom {
                                event_type: "anthropic:reasoning-start".to_string(),
                                data: serde_json::json!({
                                    "type": "reasoning-start",
                                    "contentBlockIndex": idx as u64,
                                }),
                            }]
                        } else {
                            vec![]
                        }
                    }
                    "redacted_thinking" => {
                        if let Some(idx) = event.index {
                            let data = content_block
                                .get("data")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            if !data.is_empty() {
                                self.record_redacted_thinking_data(idx, data.clone());
                            }

                            vec![ChatStreamEvent::Custom {
                                event_type: "anthropic:reasoning-start".to_string(),
                                data: serde_json::json!({
                                    "type": "reasoning-start",
                                    "contentBlockIndex": idx as u64,
                                    "redactedData": if data.is_empty() { serde_json::Value::Null } else { serde_json::json!(data) },
                                }),
                            }]
                        } else {
                            vec![]
                        }
                    }
                    "tool_use" => {
                        let tool_call_id = content_block
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let tool_name = content_block
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();

                        if tool_call_id.is_empty() || tool_name.is_empty() {
                            return vec![];
                        }

                        if let Some(idx) = event.index
                            && let Ok(mut map) = self.tool_use_ids_by_index.lock()
                        {
                            map.insert(idx, tool_call_id.clone());
                        }

                        vec![ChatStreamEvent::ToolCallDelta {
                            id: tool_call_id,
                            function_name: Some(tool_name),
                            arguments_delta: None,
                            index: event.index,
                        }]
                    }
                    "json_tool_use" => {
                        let tool_call_id = content_block
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();

                        if tool_call_id.is_empty() {
                            return vec![];
                        }

                        if let Some(idx) = event.index
                            && let Ok(mut map) = self.tool_use_ids_by_index.lock()
                        {
                            map.insert(idx, tool_call_id);
                        }

                        // Vercel-aligned: do not emit tool-call deltas for the reserved `json` tool.
                        // The corresponding `input_json_delta` chunks are emitted as ContentDelta.
                        if self.should_stream_json_tool_as_text()
                            && let Some(idx) = event.index
                        {
                            vec![Self::vercel_text_start_event(idx)]
                        } else {
                            vec![]
                        }
                    }
                    "server_tool_use" => {
                        let tool_call_id = content_block
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let tool_name_raw = content_block
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let input = content_block
                            .get("input")
                            .cloned()
                            .unwrap_or_else(|| serde_json::json!({}));

                        if tool_call_id.is_empty() || tool_name_raw.is_empty() {
                            return vec![];
                        }

                        // Vercel-aligned: map provider tool names back to stable custom names.
                        let tool_name =
                            server_tools::normalize_server_tool_name(&tool_name_raw).to_string();
                        let input =
                            server_tools::normalize_server_tool_input(&tool_name_raw, input);

                        vec![ChatStreamEvent::Custom {
                            event_type: "anthropic:tool-call".to_string(),
                            data: serde_json::json!({
                                "type": "tool-call",
                                "toolCallId": tool_call_id,
                                "toolName": tool_name,
                                "input": input,
                                "providerExecuted": true,
                                "contentBlockIndex": event.index.map(|i| i as u64),
                                "rawContentBlock": content_block,
                            }),
                        }]
                    }
                    "mcp_tool_use" => {
                        let tool_call_id = content_block
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let tool_name = content_block
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let server_name = content_block
                            .get("server_name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let input = content_block
                            .get("input")
                            .cloned()
                            .unwrap_or_else(|| serde_json::json!({}));

                        if tool_call_id.is_empty() || tool_name.is_empty() {
                            return vec![];
                        }

                        if let Some(idx) = event.index
                            && let Ok(mut map) = self.tool_use_ids_by_index.lock()
                        {
                            map.insert(idx, tool_call_id.clone());
                        }

                        if let Ok(mut map) = self.tool_names_by_id.lock() {
                            map.insert(tool_call_id.clone(), tool_name.clone());
                        }

                        if !server_name.is_empty()
                            && let Ok(mut map) = self.mcp_server_name_by_id.lock()
                        {
                            map.insert(tool_call_id.clone(), server_name.clone());
                        }

                        let server_name_json = if server_name.is_empty() {
                            serde_json::Value::Null
                        } else {
                            serde_json::json!(server_name)
                        };

                        vec![ChatStreamEvent::Custom {
                            event_type: "anthropic:tool-call".to_string(),
                            data: serde_json::json!({
                                "type": "tool-call",
                                "toolCallId": tool_call_id,
                                "toolName": tool_name,
                                "input": input,
                                "providerExecuted": true,
                                // Vercel-aligned: MCP tool parts are emitted as dynamic.
                                "dynamic": true,
                                "providerMetadata": {
                                    "anthropic": {
                                        "type": "mcp-tool-use",
                                        "serverName": server_name_json,
                                    }
                                },
                                "contentBlockIndex": event.index.map(|i| i as u64),
                                "rawContentBlock": content_block,
                                // Back-compat shim (preferred shape is providerMetadata.anthropic.serverName).
                                "serverName": server_name_json,
                            }),
                        }]
                    }
                    t if t.ends_with("_tool_result") => {
                        let tool_call_id = content_block
                            .get("tool_use_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();

                        if tool_call_id.is_empty() {
                            return vec![];
                        }

                        let tool_name = if t == "mcp_tool_result" {
                            self.tool_names_by_id
                                .lock()
                                .ok()
                                .and_then(|map| map.get(&tool_call_id).cloned())
                                .unwrap_or_else(|| "mcp".to_string())
                        } else {
                            match t {
                                "tool_search_tool_result" => "tool_search".to_string(),
                                "text_editor_code_execution_tool_result"
                                | "bash_code_execution_tool_result" => "code_execution".to_string(),
                                _ => t.strip_suffix("_tool_result").unwrap_or(t).to_string(),
                            }
                        };
                        let raw_result = content_block
                            .get("content")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null);

                        // Vercel-aligned: normalize provider-hosted tool results when possible
                        // (keep `rawContentBlock` for lossless access).
                        let (result, is_error) = if t == "web_search_tool_result" {
                            if let Some(arr) = raw_result.as_array() {
                                let normalized = arr
                                    .iter()
                                    .filter_map(|item| item.as_object())
                                    .map(|obj| {
                                        serde_json::json!({
                                            "type": obj.get("type").cloned().unwrap_or_else(|| serde_json::json!("web_search_result")),
                                            "url": obj.get("url").cloned().unwrap_or(serde_json::json!(null)),
                                            "title": obj.get("title").cloned().unwrap_or(serde_json::json!(null)),
                                            "pageAge": obj.get("page_age").cloned().unwrap_or(serde_json::json!(null)),
                                            "encryptedContent": obj.get("encrypted_content").cloned().unwrap_or(serde_json::json!(null)),
                                        })
                                    })
                                    .collect::<Vec<_>>();
                                (serde_json::Value::Array(normalized), false)
                            } else if let Some(obj) = raw_result.as_object() {
                                let error_code = obj
                                    .get("error_code")
                                    .cloned()
                                    .unwrap_or(serde_json::json!(null));
                                (
                                    serde_json::json!({
                                        "type": "web_search_tool_result_error",
                                        "errorCode": error_code,
                                    }),
                                    true,
                                )
                            } else {
                                (
                                    serde_json::json!({
                                        "type": "web_search_tool_result_error",
                                        "errorCode": serde_json::Value::Null,
                                    }),
                                    true,
                                )
                            }
                        } else if t == "web_fetch_tool_result" {
                            if let Some(obj) = raw_result.as_object() {
                                let tpe = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");

                                if tpe == "web_fetch_result" {
                                    let url =
                                        obj.get("url").cloned().unwrap_or(serde_json::json!(null));
                                    let retrieved_at = obj
                                        .get("retrieved_at")
                                        .cloned()
                                        .unwrap_or(serde_json::json!(null));
                                    let content = obj.get("content").and_then(|v| v.as_object());

                                    let mut out_content = serde_json::Map::new();
                                    if let Some(content) = content {
                                        if let Some(v) = content.get("type") {
                                            out_content.insert("type".to_string(), v.clone());
                                        }
                                        if let Some(v) = content.get("title") {
                                            out_content.insert("title".to_string(), v.clone());
                                        }
                                        if let Some(v) = content.get("citations") {
                                            out_content.insert("citations".to_string(), v.clone());
                                        }
                                        if let Some(source) =
                                            content.get("source").and_then(|v| v.as_object())
                                        {
                                            let mut out_source = serde_json::Map::new();
                                            if let Some(v) = source.get("type") {
                                                out_source.insert("type".to_string(), v.clone());
                                            }
                                            if let Some(v) = source
                                                .get("media_type")
                                                .or_else(|| source.get("mediaType"))
                                            {
                                                out_source
                                                    .insert("mediaType".to_string(), v.clone());
                                            }
                                            if let Some(v) = source.get("data") {
                                                out_source.insert("data".to_string(), v.clone());
                                            }
                                            out_content.insert(
                                                "source".to_string(),
                                                serde_json::Value::Object(out_source),
                                            );
                                        }
                                    }

                                    (
                                        serde_json::json!({
                                            "type": "web_fetch_result",
                                            "url": url,
                                            "retrievedAt": retrieved_at,
                                            "content": serde_json::Value::Object(out_content),
                                        }),
                                        false,
                                    )
                                } else if tpe == "web_fetch_tool_result_error" {
                                    let error_code = obj
                                        .get("error_code")
                                        .cloned()
                                        .unwrap_or(serde_json::json!(null));
                                    (
                                        serde_json::json!({
                                            "type": "web_fetch_tool_result_error",
                                            "errorCode": error_code,
                                        }),
                                        true,
                                    )
                                } else {
                                    (raw_result.clone(), false)
                                }
                            } else {
                                (raw_result.clone(), false)
                            }
                        } else if t == "tool_search_tool_result" {
                            if let Some(obj) = raw_result.as_object() {
                                let tpe = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");

                                if tpe == "tool_search_tool_search_result" {
                                    let refs = obj
                                        .get("tool_references")
                                        .and_then(|v| v.as_array())
                                        .cloned()
                                        .unwrap_or_default()
                                        .into_iter()
                                        .filter_map(|v| v.as_object().cloned())
                                        .map(|ref_obj| {
                                            serde_json::json!({
                                                "type": ref_obj.get("type").cloned().unwrap_or_else(|| serde_json::json!("tool_reference")),
                                                "toolName": ref_obj.get("tool_name").cloned().unwrap_or(serde_json::Value::Null),
                                            })
                                        })
                                        .collect::<Vec<_>>();
                                    (serde_json::Value::Array(refs), false)
                                } else {
                                    let error_code = obj
                                        .get("error_code")
                                        .cloned()
                                        .unwrap_or(serde_json::Value::Null);
                                    (
                                        serde_json::json!({
                                            "type": "tool_search_tool_result_error",
                                            "errorCode": error_code,
                                        }),
                                        true,
                                    )
                                }
                            } else {
                                (raw_result.clone(), false)
                            }
                        } else if t == "code_execution_tool_result" {
                            if let Some(obj) = raw_result.as_object() {
                                let tpe = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");

                                if tpe == "code_execution_result" {
                                    let mut out = serde_json::json!({
                                            "type": "code_execution_result",
                                            "stdout": obj.get("stdout").cloned().unwrap_or(serde_json::Value::Null),
                                            "stderr": obj.get("stderr").cloned().unwrap_or(serde_json::Value::Null),
                                            "return_code": obj.get("return_code").cloned().unwrap_or(serde_json::Value::Null),
                                    });
                                    if let Some(v) = obj.get("content") {
                                        out["content"] = v.clone();
                                    }
                                    (out, false)
                                } else if tpe == "code_execution_tool_result_error" {
                                    let error_code = obj
                                        .get("error_code")
                                        .cloned()
                                        .unwrap_or(serde_json::Value::Null);
                                    (
                                        serde_json::json!({
                                            "type": "code_execution_tool_result_error",
                                            "errorCode": error_code,
                                        }),
                                        true,
                                    )
                                } else {
                                    (raw_result.clone(), false)
                                }
                            } else {
                                (raw_result.clone(), false)
                            }
                        } else {
                            (raw_result.clone(), false)
                        };

                        let server_name = if t == "mcp_tool_result" {
                            self.mcp_server_name_by_id
                                .lock()
                                .ok()
                                .and_then(|map| map.get(&tool_call_id).cloned())
                        } else {
                            None
                        };

                        let server_name_json = if t == "mcp_tool_result" {
                            serde_json::json!(server_name.clone())
                        } else {
                            serde_json::Value::Null
                        };

                        let mut data = serde_json::Map::new();
                        data.insert("type".to_string(), serde_json::json!("tool-result"));
                        data.insert("toolCallId".to_string(), serde_json::json!(tool_call_id));
                        data.insert("toolName".to_string(), serde_json::json!(tool_name));
                        data.insert("result".to_string(), result);
                        data.insert("providerExecuted".to_string(), serde_json::json!(true));
                        data.insert("isError".to_string(), serde_json::json!(is_error));
                        data.insert(
                            "contentBlockIndex".to_string(),
                            serde_json::json!(event.index.map(|i| i as u64)),
                        );
                        data.insert("rawContentBlock".to_string(), content_block.clone());
                        // Back-compat shim (preferred shape is providerMetadata.anthropic.serverName).
                        data.insert("serverName".to_string(), server_name_json.clone());

                        if t == "mcp_tool_result" {
                            data.insert("dynamic".to_string(), serde_json::json!(true));
                            data.insert(
                                "providerMetadata".to_string(),
                                serde_json::json!({
                                    "anthropic": {
                                        "type": "mcp-tool-use",
                                        "serverName": server_name_json,
                                    }
                                }),
                            );
                        }

                        let mut events = vec![ChatStreamEvent::Custom {
                            event_type: "anthropic:tool-result".to_string(),
                            data: serde_json::Value::Object(data),
                        }];

                        // Vercel-aligned: emit sources for web search results
                        if t == "web_search_tool_result"
                            && let Some(arr) =
                                content_block.get("content").and_then(|v| v.as_array())
                        {
                            for (i, item) in arr.iter().enumerate() {
                                let Some(obj) = item.as_object() else {
                                    continue;
                                };
                                let Some(url) = obj.get("url").and_then(|v| v.as_str()) else {
                                    continue;
                                };
                                let title = obj.get("title").and_then(|v| v.as_str());
                                let page_age = obj.get("page_age").and_then(|v| v.as_str());
                                let encrypted_content =
                                    obj.get("encrypted_content").and_then(|v| v.as_str());

                                self.record_source(AnthropicSource {
                                    id: format!("{tool_call_id}:{i}"),
                                    source_type: "url".to_string(),
                                    url: Some(url.to_string()),
                                    title: title.map(|s| s.to_string()),
                                    media_type: None,
                                    filename: None,
                                    page_age: page_age.map(|s| s.to_string()),
                                    encrypted_content: encrypted_content.map(|s| s.to_string()),
                                    tool_call_id: Some(tool_call_id.clone()),
                                    provider_metadata: None,
                                });

                                events.push(ChatStreamEvent::Custom {
                                    event_type: "anthropic:source".to_string(),
                                    data: serde_json::json!({
                                        "type": "source",
                                        "sourceType": "url",
                                        "id": format!("{tool_call_id}:{i}"),
                                        "url": url,
                                        "title": title,
                                        "toolCallId": tool_call_id,
                                        "providerMetadata": {
                                            "anthropic": {
                                                "pageAge": page_age,
                                                "encryptedContent": encrypted_content,
                                            }
                                        }
                                    }),
                                });
                            }
                        }

                        events
                    }
                    _ => vec![],
                }
            }
            "content_block_delta" => {
                let mut builder = EventBuilder::new();
                if let Some(delta) = event.delta {
                    match delta.delta_type.as_deref() {
                        Some("text_delta") => {
                            if let Some(text) = delta.text
                                && !self.should_stream_json_tool_as_text()
                            {
                                builder = builder.add_content_delta(text.clone(), None);
                                if let Some(idx) = event.index
                                    && self.get_content_block_type(idx).as_deref() == Some("text")
                                {
                                    builder = builder.add_custom_event(
                                        "anthropic:text-delta".to_string(),
                                        serde_json::json!({
                                            "type": "text-delta",
                                            "id": idx.to_string(),
                                            "delta": text,
                                        }),
                                    );
                                }
                            }
                        }
                        Some("thinking_delta") => {
                            if let Some(thinking) = delta.thinking {
                                builder = builder.add_thinking_delta(thinking);
                            }
                        }
                        Some("signature_delta") => {
                            if let (Some(idx), Some(sig_delta)) = (event.index, delta.signature)
                                && self
                                    .get_content_block_type(idx)
                                    .is_some_and(|t| t == "thinking")
                            {
                                self.append_thinking_signature(idx, sig_delta.clone());
                                builder = builder.add_custom_event(
                                    "anthropic:thinking-signature-delta".to_string(),
                                    serde_json::json!({
                                        "type": "thinking-signature-delta",
                                        "contentBlockIndex": idx as u64,
                                        "signatureDelta": sig_delta,
                                    }),
                                );
                            }
                        }
                        Some("citations_delta") => {
                            if let (Some(idx), Some(citation)) = (event.index, delta.citation) {
                                // Vercel-aligned: citations deltas are converted into `source` events when possible.
                                let Some(obj) = citation.as_object() else {
                                    return builder.build();
                                };
                                let Some(kind) = obj.get("type").and_then(|v| v.as_str()) else {
                                    return builder.build();
                                };

                                match kind {
                                    "page_location" => {
                                        let doc_index = obj
                                            .get("document_index")
                                            .and_then(|v| v.as_u64())
                                            .map(|u| u as usize);
                                        let Some(doc_index) = doc_index else {
                                            return builder.build();
                                        };
                                        let Some(doc) = self.citation_documents.get(doc_index)
                                        else {
                                            return builder.build();
                                        };

                                        let cited_text =
                                            obj.get("cited_text").and_then(|v| v.as_str());
                                        let start_page =
                                            obj.get("start_page_number").and_then(|v| v.as_u64());
                                        let end_page =
                                            obj.get("end_page_number").and_then(|v| v.as_u64());
                                        let title = obj
                                            .get("document_title")
                                            .and_then(|v| v.as_str())
                                            .filter(|s| !s.is_empty())
                                            .map(|s| s.to_string())
                                            .unwrap_or_else(|| doc.title.clone());

                                        let id = format!(
                                            "doc:{doc_index}:page:{}-{}",
                                            start_page.unwrap_or(0),
                                            end_page.unwrap_or(0)
                                        );

                                        builder = builder.add_custom_event(
                                            "anthropic:source".to_string(),
                                            serde_json::json!({
                                                "type": "source",
                                                "sourceType": "document",
                                                "id": id,
                                                "mediaType": doc.media_type,
                                                "title": title,
                                                "filename": doc.filename,
                                                "providerMetadata": {
                                                    "anthropic": {
                                                        "citedText": cited_text,
                                                        "startPageNumber": start_page,
                                                        "endPageNumber": end_page,
                                                    }
                                                }
                                            }),
                                        );

                                        self.record_source(AnthropicSource {
                                            id,
                                            source_type: "document".to_string(),
                                            url: None,
                                            title: Some(title),
                                            media_type: Some(doc.media_type.clone()),
                                            filename: doc.filename.clone(),
                                            page_age: None,
                                            encrypted_content: None,
                                            tool_call_id: None,
                                            provider_metadata: Some(serde_json::json!({
                                                "citedText": cited_text,
                                                "startPageNumber": start_page,
                                                "endPageNumber": end_page,
                                            })),
                                        });
                                    }
                                    "char_location" => {
                                        let doc_index = obj
                                            .get("document_index")
                                            .and_then(|v| v.as_u64())
                                            .map(|u| u as usize);
                                        let Some(doc_index) = doc_index else {
                                            return builder.build();
                                        };
                                        let Some(doc) = self.citation_documents.get(doc_index)
                                        else {
                                            return builder.build();
                                        };

                                        let cited_text =
                                            obj.get("cited_text").and_then(|v| v.as_str());
                                        let start_char =
                                            obj.get("start_char_index").and_then(|v| v.as_u64());
                                        let end_char =
                                            obj.get("end_char_index").and_then(|v| v.as_u64());
                                        let title = obj
                                            .get("document_title")
                                            .and_then(|v| v.as_str())
                                            .filter(|s| !s.is_empty())
                                            .map(|s| s.to_string())
                                            .unwrap_or_else(|| doc.title.clone());

                                        let id = format!(
                                            "doc:{doc_index}:char:{}-{}",
                                            start_char.unwrap_or(0),
                                            end_char.unwrap_or(0)
                                        );

                                        builder = builder.add_custom_event(
                                            "anthropic:source".to_string(),
                                            serde_json::json!({
                                                "type": "source",
                                                "sourceType": "document",
                                                "id": id,
                                                "mediaType": doc.media_type,
                                                "title": title,
                                                "filename": doc.filename,
                                                "providerMetadata": {
                                                    "anthropic": {
                                                        "citedText": cited_text,
                                                        "startCharIndex": start_char,
                                                        "endCharIndex": end_char,
                                                    }
                                                }
                                            }),
                                        );

                                        self.record_source(AnthropicSource {
                                            id,
                                            source_type: "document".to_string(),
                                            url: None,
                                            title: Some(title),
                                            media_type: Some(doc.media_type.clone()),
                                            filename: doc.filename.clone(),
                                            page_age: None,
                                            encrypted_content: None,
                                            tool_call_id: None,
                                            provider_metadata: Some(serde_json::json!({
                                                "citedText": cited_text,
                                                "startCharIndex": start_char,
                                                "endCharIndex": end_char,
                                            })),
                                        });
                                    }
                                    _ => {
                                        // Vercel-aligned: ignore other citation kinds in the stream converter.
                                        let _ = idx;
                                    }
                                }
                            }
                        }
                        Some("input_json_delta") => {
                            if let Some(partial_json) = delta.partial_json
                                && !partial_json.is_empty()
                                && let Some(idx) = event.index
                            {
                                if self.get_content_block_type(idx).as_deref()
                                    == Some("json_tool_use")
                                {
                                    builder = builder.add_content_delta(partial_json.clone(), None);
                                    if self.should_stream_json_tool_as_text() {
                                        builder = builder.add_custom_event(
                                            "anthropic:text-delta".to_string(),
                                            serde_json::json!({
                                                "type": "text-delta",
                                                "id": idx.to_string(),
                                                "delta": partial_json,
                                            }),
                                        );
                                    }
                                } else if let Ok(map) = self.tool_use_ids_by_index.lock()
                                    && let Some(tool_call_id) = map.get(&idx)
                                {
                                    builder = builder.add_tool_call_delta(
                                        tool_call_id.clone(),
                                        None,
                                        Some(partial_json),
                                        Some(idx),
                                    );
                                }
                            }
                        }
                        _ => {
                            if let Some(text) = delta.text {
                                builder = builder.add_content_delta(text, None);
                            }
                            if let Some(thinking) = delta.thinking {
                                builder = builder.add_thinking_delta(thinking);
                            }
                            if let Some(partial_json) = delta.partial_json
                                && !partial_json.is_empty()
                                && let Some(idx) = event.index
                            {
                                if self.get_content_block_type(idx).as_deref()
                                    == Some("json_tool_use")
                                {
                                    builder = builder.add_content_delta(partial_json.clone(), None);
                                    if self.should_stream_json_tool_as_text() {
                                        builder = builder.add_custom_event(
                                            "anthropic:text-delta".to_string(),
                                            serde_json::json!({
                                                "type": "text-delta",
                                                "id": idx.to_string(),
                                                "delta": partial_json,
                                            }),
                                        );
                                    }
                                } else if let Ok(map) = self.tool_use_ids_by_index.lock()
                                    && let Some(tool_call_id) = map.get(&idx)
                                {
                                    builder = builder.add_tool_call_delta(
                                        tool_call_id.clone(),
                                        None,
                                        Some(partial_json),
                                        Some(idx),
                                    );
                                }
                            }
                        }
                    };
                }
                builder.build()
            }
            "content_block_stop" => {
                let Some(idx) = event.index else {
                    return vec![];
                };

                match self.get_content_block_type(idx).as_deref() {
                    Some("text") => {
                        if self.should_stream_json_tool_as_text() {
                            vec![]
                        } else {
                            vec![Self::vercel_text_end_event(idx)]
                        }
                    }
                    Some("json_tool_use") => {
                        if self.should_stream_json_tool_as_text() {
                            vec![Self::vercel_text_end_event(idx)]
                        } else {
                            vec![]
                        }
                    }
                    Some("thinking") | Some("redacted_thinking") => vec![ChatStreamEvent::Custom {
                        event_type: "anthropic:reasoning-end".to_string(),
                        data: serde_json::json!({
                            "type": "reasoning-end",
                            "contentBlockIndex": idx as u64,
                        }),
                    }],
                    _ => vec![],
                }
            }
            "message_delta" => {
                let mut builder = EventBuilder::new();

                // Thinking (if present)
                if let Some(delta) = &event.delta
                    && let Some(thinking) = &delta.thinking
                    && !thinking.is_empty()
                {
                    builder = builder.add_thinking_delta(thinking.clone());
                }

                // Usage update
                if let Some(usage) = &event.usage {
                    self.merge_vercel_usage(usage);

                    let prompt_tokens = usage
                        .get("input_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as u32;
                    let completion_tokens = usage
                        .get("output_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as u32;
                    builder =
                        builder.add_usage_update(Usage::new(prompt_tokens, completion_tokens));
                }

                if let Some(cm) = event
                    .delta
                    .as_ref()
                    .and_then(|d| d.context_management.as_ref())
                    .or(event.context_management.as_ref())
                    && let Ok(mut v) = self.vercel_context_management.lock()
                {
                    *v = Some(cm.clone());
                }

                // Finish reason -> StreamEnd
                if let Some(delta) = &event.delta
                    && let Some(stop_reason) = &delta.stop_reason
                {
                    if let Some(container) = &delta.container
                        && let Ok(mut v) = self.vercel_container.lock()
                    {
                        *v = Some(container.clone());
                    }

                    let reason = match stop_reason.as_str() {
                        "end_turn" => FinishReason::Stop,
                        "max_tokens" => FinishReason::Length,
                        "stop_sequence" => FinishReason::StopSequence,
                        "tool_use" => {
                            if self.json_tool_seen.load(Ordering::Relaxed) {
                                FinishReason::Stop
                            } else {
                                FinishReason::ToolCalls
                            }
                        }
                        "refusal" => FinishReason::ContentFilter,
                        _ => FinishReason::Stop,
                    };

                    if let Some(stop_sequence) = &delta.stop_sequence
                        && !stop_sequence.is_empty()
                        && let Ok(mut v) = self.vercel_stop_sequence.lock()
                    {
                        *v = Some(stop_sequence.clone());
                    }

                    if self.state_tracker.needs_stream_end() {
                        if let ChatStreamEvent::Custom { event_type, data } =
                            self.vercel_finish_event(Some(stop_reason.as_str()), &reason)
                        {
                            builder = builder.add_custom_event(event_type, data);
                        }

                        let response = ChatResponse {
                            id: None,
                            model: None,
                            content: MessageContent::Text("".to_string()),
                            usage: None, // usage already emitted as UsageUpdate above if present
                            finish_reason: Some(reason),
                            audio: None,
                            system_fingerprint: None,
                            service_tier: None,
                            warnings: None,
                            provider_metadata: self.build_stream_provider_metadata(),
                        };
                        builder = builder.add_stream_end(response);
                    }
                }

                builder.build()
            }
            "message_stop" => {
                let response = ChatResponse {
                    id: None,
                    model: None,
                    content: MessageContent::Text("".to_string()),
                    usage: None,
                    finish_reason: Some(FinishReason::Stop),
                    audio: None,
                    system_fingerprint: None,
                    service_tier: None,
                    warnings: None,
                    provider_metadata: self.build_stream_provider_metadata(),
                };
                if !self.state_tracker.needs_stream_end() {
                    return vec![];
                }

                let mut out = Vec::new();
                if let ChatStreamEvent::Custom { event_type, data } =
                    self.vercel_finish_event(None, &FinishReason::Stop)
                {
                    out.push(ChatStreamEvent::Custom { event_type, data });
                }
                out.push(ChatStreamEvent::StreamEnd { response });
                out
            }
            _ => vec![],
        }
    }
}

impl SseEventConverter for AnthropicEventConverter {
    fn convert_event(
        &self,
        event: Event,
    ) -> Pin<Box<dyn Future<Output = Vec<Result<ChatStreamEvent, LlmError>>> + Send + Sync + '_>>
    {
        Box::pin(async move {
            // Log the raw event data for debugging
            tracing::debug!(
                "Anthropic SSE event: {} (event={})",
                event.data,
                event.event
            );

            // Handle special cases first
            if event.data.trim() == "[DONE]" {
                return vec![];
            }

            if event.event.trim().eq_ignore_ascii_case("error") {
                self.seen_error.store(true, Ordering::Relaxed);
            }

            // Try to parse as standard Anthropic event
            match serde_json::from_str::<AnthropicStreamEvent>(&event.data) {
                Ok(anthropic_event) => self
                    .convert_anthropic_event(anthropic_event)
                    .into_iter()
                    .map(Ok)
                    .collect(),
                Err(e) => {
                    // Enhanced error reporting with event data
                    tracing::warn!("Failed to parse Anthropic SSE event: {}", e);
                    tracing::warn!("Raw event data: {}", event.data);

                    // Try to parse as a generic JSON to see if it's a different format
                    if let Ok(generic_json) = serde_json::from_str::<serde_json::Value>(&event.data)
                    {
                        tracing::warn!("Event parsed as generic JSON: {:#}", generic_json);

                        // Check if this looks like an error response
                        if let Some(error_obj) = generic_json.get("error") {
                            let error_message = error_obj
                                .get("message")
                                .and_then(|m| m.as_str())
                                .unwrap_or("Unknown error");

                            self.seen_error.store(true, Ordering::Relaxed);
                            return vec![Ok(ChatStreamEvent::Error {
                                error: format!("Anthropic API error: {}", error_message),
                            })];
                        }
                    }

                    vec![Err(LlmError::ParseError(format!(
                        "Failed to parse Anthropic event: {}. Raw data: {}",
                        e, event.data
                    )))]
                }
            }
        })
    }

    fn handle_stream_end(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
        // Anthropic normally emits StreamEnd via message_stop event in convert_event.
        // If we reach here without seeing message_stop, the model has not transmitted
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
            finish_reason: Some(if self.seen_error.load(Ordering::Relaxed) {
                FinishReason::Error
            } else {
                FinishReason::Unknown
            }),
            audio: None,
            system_fingerprint: None,
            service_tier: None,
            warnings: None,
            provider_metadata: self.build_stream_provider_metadata(),
        };

        Some(Ok(ChatStreamEvent::StreamEnd { response }))
    }

    fn serialize_event(&self, event: &ChatStreamEvent) -> Result<Vec<u8>, LlmError> {
        self.serialize_event_impl(event)
    }
}

// Legacy AnthropicStreaming client has been removed in favor of the unified HttpChatExecutor.
// The AnthropicEventConverter is still used for SSE event conversion in tests.

mod serialize;

impl AnthropicEventConverter {
    fn serialize_event_impl(&self, event: &ChatStreamEvent) -> Result<Vec<u8>, LlmError> {
        serialize::serialize_event(self, event)
    }
}

#[cfg(test)]
mod tests;
