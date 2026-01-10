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
use crate::streaming::{ChatStreamEvent, StreamStateTracker};
use crate::types::{ChatResponse, FinishReason, MessageContent, ResponseMetadata, Usage};
use eventsource_stream::Event;
use serde::Deserialize;

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

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
    vercel_response_id: Arc<Mutex<Option<String>>>,
    vercel_model_id: Arc<Mutex<Option<String>>>,
    vercel_stop_sequence: Arc<Mutex<Option<String>>>,
    vercel_usage: Arc<Mutex<Option<serde_json::Value>>>,
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
            vercel_response_id: Arc::new(Mutex::new(None)),
            vercel_model_id: Arc::new(Mutex::new(None)),
            vercel_stop_sequence: Arc::new(Mutex::new(None)),
            vercel_usage: Arc::new(Mutex::new(None)),
        }
    }

    pub fn with_citation_documents(mut self, docs: Vec<AnthropicCitationDocument>) -> Self {
        self.citation_documents = docs;
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
                        "container": serde_json::Value::Null,
                        "contextManagement": serde_json::Value::Null,
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

                // Finish reason -> StreamEnd
                if let Some(delta) = &event.delta
                    && let Some(stop_reason) = &delta.stop_reason
                {
                    let reason = match stop_reason.as_str() {
                        "end_turn" => FinishReason::Stop,
                        "max_tokens" => FinishReason::Length,
                        "stop_sequence" => FinishReason::Stop,
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
            tracing::debug!("Anthropic SSE event: {}", event.data);

            // Handle special cases first
            if event.data.trim() == "[DONE]" {
                return vec![];
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
            finish_reason: Some(FinishReason::Unknown),
            audio: None,
            system_fingerprint: None,
            service_tier: None,
            warnings: None,
            provider_metadata: self.build_stream_provider_metadata(),
        };

        Some(Ok(ChatStreamEvent::StreamEnd { response }))
    }
}

// Legacy AnthropicStreaming client has been removed in favor of the unified HttpChatExecutor.
// The AnthropicEventConverter is still used for SSE event conversion in tests.

#[cfg(test)]
mod tests {
    use super::super::params::AnthropicParams;
    use super::*;
    use eventsource_stream::Event;

    fn create_test_config() -> AnthropicParams {
        AnthropicParams::default()
    }

    #[tokio::test]
    async fn test_anthropic_streaming_conversion() {
        let config = create_test_config();
        let converter = AnthropicEventConverter::new(config);

        // Test content delta conversion
        let event = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };

        let result = converter.convert_event(event).await;
        assert!(!result.is_empty());

        if let Some(Ok(ChatStreamEvent::ContentDelta { delta, .. })) = result.first() {
            assert_eq!(delta, "Hello");
        } else {
            panic!("Expected ContentDelta event");
        }
    }

    #[tokio::test]
    async fn test_anthropic_streaming_error_event_is_exposed() {
        let config = create_test_config();
        let converter = AnthropicEventConverter::new(config);

        let event = Event {
            event: "".to_string(),
            data:
                r#"{"type":"error","error":{"type":"overloaded_error","message":"rate limited"}}"#
                    .to_string(),
            id: "".to_string(),
            retry: None,
        };

        let result = converter.convert_event(event).await;
        let err = result
            .iter()
            .find(|e| matches!(e, Ok(ChatStreamEvent::Error { .. })));
        match err {
            Some(Ok(ChatStreamEvent::Error { error })) => {
                assert!(error.contains("rate limited"));
            }
            other => panic!("Expected Error event, got: {other:?}"),
        }
    }

    // Removed legacy merge-provider-params test; behavior now covered by transformers

    #[tokio::test]
    async fn test_anthropic_stream_end() {
        let config = create_test_config();
        let converter = AnthropicEventConverter::new(config);

        let result = converter.handle_stream_end();
        assert!(result.is_some());

        if let Some(Ok(ChatStreamEvent::StreamEnd { .. })) = result {
            // Success
        } else {
            panic!("Expected StreamEnd event");
        }
    }

    #[tokio::test]
    async fn emits_custom_events_for_server_tool_use_and_results() {
        let config = create_test_config();
        let converter = AnthropicEventConverter::new(config);

        let tool_call_event = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"rust"}}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };

        let evs = converter.convert_event(tool_call_event).await;
        assert_eq!(evs.len(), 1);
        match evs.first().unwrap().as_ref().unwrap() {
            ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "anthropic:tool-call");
                assert_eq!(data["toolCallId"], serde_json::json!("srvtoolu_1"));
                assert_eq!(data["toolName"], serde_json::json!("web_search"));
                assert_eq!(data["providerExecuted"], serde_json::json!(true));
            }
            other => panic!("Expected Custom event, got {:?}", other),
        }

        let tool_result_event = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":1,"content_block":{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"type":"web_search_result","title":"Rust","url":"https://www.rust-lang.org","encrypted_content":"..."}]}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };

        let evs = converter.convert_event(tool_result_event).await;
        assert_eq!(evs.len(), 2);
        match evs.first().unwrap().as_ref().unwrap() {
            ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "anthropic:tool-result");
                assert_eq!(data["toolCallId"], serde_json::json!("srvtoolu_1"));
                assert_eq!(data["toolName"], serde_json::json!("web_search"));
                assert_eq!(data["providerExecuted"], serde_json::json!(true));
                assert!(data["result"].is_array());
            }
            other => panic!("Expected Custom event, got {:?}", other),
        }

        match evs.get(1).unwrap().as_ref().unwrap() {
            ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "anthropic:source");
                assert_eq!(data["sourceType"], serde_json::json!("url"));
                assert_eq!(data["url"], serde_json::json!("https://www.rust-lang.org"));
            }
            other => panic!("Expected source Custom event, got {:?}", other),
        }

        let web_fetch_result_event = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":2,"content_block":{"type":"web_fetch_tool_result","tool_use_id":"srvtoolu_2","content":{"type":"web_fetch_result","url":"https://example.com","retrieved_at":"2025-01-01T00:00:00Z","content":{"type":"document","title":"Example","citations":{"enabled":true},"source":{"type":"text","media_type":"text/plain","data":"hello"}}}}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };

        let evs = converter.convert_event(web_fetch_result_event).await;
        assert_eq!(evs.len(), 1);
        match evs.first().unwrap().as_ref().unwrap() {
            ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "anthropic:tool-result");
                assert_eq!(data["toolCallId"], serde_json::json!("srvtoolu_2"));
                assert_eq!(data["toolName"], serde_json::json!("web_fetch"));
                assert_eq!(data["isError"], serde_json::json!(false));
                assert_eq!(
                    data["result"]["type"],
                    serde_json::json!("web_fetch_result")
                );
                assert_eq!(
                    data["result"]["retrievedAt"],
                    serde_json::json!("2025-01-01T00:00:00Z")
                );
                assert_eq!(
                    data["result"]["content"]["source"]["mediaType"],
                    serde_json::json!("text/plain")
                );
            }
            other => panic!("Expected tool-result Custom event, got {:?}", other),
        }

        let tool_search_call_event = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":3,"content_block":{"type":"server_tool_use","id":"srvtoolu_3","name":"tool_search_tool_regex","input":{"pattern":"weather","limit":2}}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };

        let evs = converter.convert_event(tool_search_call_event).await;
        assert_eq!(evs.len(), 1);
        match evs.first().unwrap().as_ref().unwrap() {
            ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "anthropic:tool-call");
                assert_eq!(data["toolCallId"], serde_json::json!("srvtoolu_3"));
                assert_eq!(data["toolName"], serde_json::json!("tool_search"));
                assert_eq!(data["input"]["pattern"], serde_json::json!("weather"));
            }
            other => panic!("Expected tool-call Custom event, got {:?}", other),
        }

        let tool_search_result_event = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":4,"content_block":{"type":"tool_search_tool_result","tool_use_id":"srvtoolu_3","content":{"type":"tool_search_tool_search_result","tool_references":[{"type":"tool_reference","tool_name":"get_weather"}]}}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };

        let evs = converter.convert_event(tool_search_result_event).await;
        assert_eq!(evs.len(), 1);
        match evs.first().unwrap().as_ref().unwrap() {
            ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "anthropic:tool-result");
                assert_eq!(data["toolCallId"], serde_json::json!("srvtoolu_3"));
                assert_eq!(data["toolName"], serde_json::json!("tool_search"));
                assert_eq!(data["isError"], serde_json::json!(false));
                assert!(data["result"].is_array());
                assert_eq!(
                    data["result"][0]["toolName"],
                    serde_json::json!("get_weather")
                );
            }
            other => panic!("Expected tool-result Custom event, got {:?}", other),
        }

        let code_exec_call_event = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":5,"content_block":{"type":"server_tool_use","id":"srvtoolu_4","name":"code_execution","input":{"code":"print(1+1)"}}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };

        let evs = converter.convert_event(code_exec_call_event).await;
        assert_eq!(evs.len(), 1);
        match evs.first().unwrap().as_ref().unwrap() {
            ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "anthropic:tool-call");
                assert_eq!(data["toolCallId"], serde_json::json!("srvtoolu_4"));
                assert_eq!(data["toolName"], serde_json::json!("code_execution"));
            }
            other => panic!("Expected tool-call Custom event, got {:?}", other),
        }

        let code_exec_result_event = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":6,"content_block":{"type":"code_execution_tool_result","tool_use_id":"srvtoolu_4","content":{"type":"code_execution_result","stdout":"2\n","stderr":"","return_code":0}}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };

        let evs = converter.convert_event(code_exec_result_event).await;
        assert_eq!(evs.len(), 1);
        match evs.first().unwrap().as_ref().unwrap() {
            ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "anthropic:tool-result");
                assert_eq!(data["toolCallId"], serde_json::json!("srvtoolu_4"));
                assert_eq!(data["toolName"], serde_json::json!("code_execution"));
                assert_eq!(data["isError"], serde_json::json!(false));
                assert_eq!(
                    data["result"]["type"],
                    serde_json::json!("code_execution_result")
                );
                assert_eq!(data["result"]["return_code"], serde_json::json!(0));
            }
            other => panic!("Expected tool-result Custom event, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn emits_tool_call_delta_for_local_tool_use_input_json_delta() {
        let config = create_test_config();
        let converter = AnthropicEventConverter::new(config);

        let start = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"location":"tokyo"}}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };

        let evs = converter.convert_event(start).await;
        assert_eq!(evs.len(), 1);
        match evs.first().unwrap().as_ref().unwrap() {
            ChatStreamEvent::ToolCallDelta {
                id,
                function_name,
                arguments_delta,
                index,
            } => {
                assert_eq!(id, "toolu_1");
                assert_eq!(function_name.as_deref(), Some("get_weather"));
                assert!(arguments_delta.is_none());
                assert_eq!(*index, Some(0));
            }
            other => panic!("Expected ToolCallDelta, got {:?}", other),
        }

        let delta = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"unit\":\"c\"}"}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };

        let evs = converter.convert_event(delta).await;
        assert_eq!(evs.len(), 1);
        match evs.first().unwrap().as_ref().unwrap() {
            ChatStreamEvent::ToolCallDelta {
                id,
                function_name,
                arguments_delta,
                index,
            } => {
                assert_eq!(id, "toolu_1");
                assert!(function_name.is_none());
                assert_eq!(arguments_delta.as_deref(), Some("{\"unit\":\"c\"}"));
                assert_eq!(*index, Some(0));
            }
            other => panic!("Expected ToolCallDelta, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn captures_thinking_signature_delta_and_exposes_in_stream_end() {
        let config = create_test_config();
        let converter = AnthropicEventConverter::new(config);

        let thinking_start = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };
        let _ = converter.convert_event(thinking_start).await;

        let sig_delta = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-1"}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };
        let evs = converter.convert_event(sig_delta).await;
        assert_eq!(evs.len(), 1);
        match evs.first().unwrap().as_ref().unwrap() {
            ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "anthropic:thinking-signature-delta");
                assert_eq!(data["signatureDelta"], serde_json::json!("sig-1"));
            }
            other => panic!("Expected signature delta Custom event, got {:?}", other),
        }

        let stop = Event {
            event: "".to_string(),
            data: r#"{"type":"message_stop"}"#.to_string(),
            id: "".to_string(),
            retry: None,
        };
        let evs = converter.convert_event(stop).await;
        let end = evs
            .into_iter()
            .find(|e| matches!(e, Ok(ChatStreamEvent::StreamEnd { .. })))
            .expect("stream end");
        match end.unwrap() {
            ChatStreamEvent::StreamEnd { response } => {
                let meta = response.provider_metadata.expect("provider_metadata");
                assert_eq!(
                    meta.get("anthropic")
                        .unwrap()
                        .get("thinking_signature")
                        .unwrap(),
                    &serde_json::json!("sig-1")
                );
            }
            _ => unreachable!(),
        }
    }

    #[tokio::test]
    async fn captures_redacted_thinking_data_in_stream_end() {
        let config = create_test_config();
        let converter = AnthropicEventConverter::new(config);

        let redacted_start = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"abc123"}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };
        let evs = converter.convert_event(redacted_start).await;
        assert_eq!(evs.len(), 1);
        match evs.first().unwrap().as_ref().unwrap() {
            ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "anthropic:reasoning-start");
                assert_eq!(data["redactedData"], serde_json::json!("abc123"));
            }
            other => panic!("Expected reasoning-start Custom event, got {:?}", other),
        }

        let stop = Event {
            event: "".to_string(),
            data: r#"{"type":"message_stop"}"#.to_string(),
            id: "".to_string(),
            retry: None,
        };
        let evs = converter.convert_event(stop).await;
        let end = evs
            .into_iter()
            .find(|e| matches!(e, Ok(ChatStreamEvent::StreamEnd { .. })))
            .expect("stream end");
        match end.unwrap() {
            ChatStreamEvent::StreamEnd { response } => {
                let meta = response.provider_metadata.expect("provider_metadata");
                assert_eq!(
                    meta.get("anthropic")
                        .unwrap()
                        .get("redacted_thinking_data")
                        .unwrap(),
                    &serde_json::json!("abc123")
                );
            }
            _ => unreachable!(),
        }
    }

    #[tokio::test]
    async fn emits_source_event_for_citations_delta_with_document_location() {
        let config = create_test_config();
        let converter = AnthropicEventConverter::new(config).with_citation_documents(vec![
            AnthropicCitationDocument {
                title: "Doc A".to_string(),
                filename: Some("a.pdf".to_string()),
                media_type: "application/pdf".to_string(),
            },
        ]);

        let ev = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"page_location","cited_text":"hello","document_index":0,"document_title":null,"start_page_number":1,"end_page_number":1}}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };

        let out = converter.convert_event(ev).await;
        assert_eq!(out.len(), 1);
        match out.first().unwrap().as_ref().unwrap() {
            ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "anthropic:source");
                assert_eq!(data["sourceType"], serde_json::json!("document"));
                assert_eq!(data["mediaType"], serde_json::json!("application/pdf"));
                assert_eq!(data["title"], serde_json::json!("Doc A"));
                assert_eq!(data["filename"], serde_json::json!("a.pdf"));
                assert_eq!(
                    data["providerMetadata"]["anthropic"]["startPageNumber"],
                    serde_json::json!(1)
                );
            }
            other => panic!("Expected source Custom event, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn accumulates_sources_into_stream_end_provider_metadata() {
        let config = create_test_config();
        let converter = AnthropicEventConverter::new(config).with_citation_documents(vec![
            AnthropicCitationDocument {
                title: "Doc A".to_string(),
                filename: Some("a.pdf".to_string()),
                media_type: "application/pdf".to_string(),
            },
        ]);

        let ev = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"page_location","cited_text":"hello","document_index":0,"document_title":null,"start_page_number":1,"end_page_number":1}}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };
        let _ = converter.convert_event(ev).await;

        let stop = Event {
            event: "".to_string(),
            data: r#"{"type":"message_stop"}"#.to_string(),
            id: "".to_string(),
            retry: None,
        };
        let out = converter.convert_event(stop).await;
        let end = out
            .into_iter()
            .find(|e| matches!(e, Ok(ChatStreamEvent::StreamEnd { .. })))
            .expect("stream end");
        match end.unwrap() {
            ChatStreamEvent::StreamEnd { response } => {
                let meta = response.provider_metadata.expect("provider_metadata");
                let anthropic = meta.get("anthropic").expect("anthropic");
                let sources = anthropic
                    .get("sources")
                    .and_then(|v| v.as_array())
                    .expect("sources array");
                assert_eq!(sources.len(), 1);
                assert_eq!(sources[0]["source_type"], serde_json::json!("document"));
                assert_eq!(
                    sources[0]["media_type"],
                    serde_json::json!("application/pdf")
                );
                assert_eq!(sources[0]["filename"], serde_json::json!("a.pdf"));
            }
            _ => unreachable!(),
        }
    }
}
