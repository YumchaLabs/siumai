//! Anthropic streaming implementation using eventsource-stream
//!
//! Provides SSE event conversion for Anthropic streaming responses.
//! The legacy AnthropicStreaming client has been removed in favor of the unified HttpChatExecutor.

use super::params::AnthropicParams;
use super::params::StructuredOutputMode;
use super::provider_metadata::AnthropicSource;
use super::server_tools;
use super::utils::parse_finish_reason;
use crate::error::LlmError;
use crate::streaming::SseEventConverter;
use crate::streaming::{
    ChatStreamEvent, StreamStateTracker, TypedStreamPart, UnsupportedStreamPartBehavior,
};
use crate::types::{
    ChatResponse, ChatStreamFinishInfo, ChatStreamPart, ChatStreamToolCall, ChatStreamToolResult,
    ContentPart, FinishReason, MessageContent, ResponseMetadata, SourcePart, Usage,
};
use eventsource_stream::Event;
use serde::Deserialize;

use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Debug, Default, Clone)]
struct AnthropicSerializeState {
    message_id: Option<String>,
    model: Option<String>,
    message_start_emitted: bool,
    next_block_index: usize,
    active_block: Option<AnthropicSerializeBlock>,
    latest_usage: Option<Usage>,
    terminal_emitted: bool,
    ignore_next_stream_end: bool,
    seen_tool_call_ids: HashSet<String>,
    provider_executed_tool_input_ids: HashSet<String>,
    provider_executed_tool_call_ids: HashSet<String>,
    provider_raw_server_tool_names_by_id: std::collections::HashMap<String, String>,
    provider_mcp_server_names_by_id: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AnthropicSerializeBlock {
    index: usize,
    kind: AnthropicSerializeBlockKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AnthropicSerializeBlockKind {
    Text,
    Compaction,
    Thinking,
    RedactedThinking,
    Tool { id: String },
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
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    input: Option<serde_json::Value>,
    #[serde(default)]
    caller: Option<serde_json::Value>,
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
    content: Option<String>,
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
    include_raw_chunks: bool,
    custom_provider_metadata_key: Option<String>,
    state_tracker: StreamStateTracker,
    tool_use_ids_by_index: Arc<Mutex<std::collections::HashMap<usize, String>>>,
    content_block_type_by_index: Arc<Mutex<std::collections::HashMap<usize, String>>>,
    text_content_by_index: Arc<Mutex<std::collections::HashMap<usize, String>>>,
    thinking_content_by_index: Arc<Mutex<std::collections::HashMap<usize, String>>>,
    thinking_signature_by_index: Arc<Mutex<std::collections::HashMap<usize, String>>>,
    redacted_thinking_data_by_index: Arc<Mutex<std::collections::HashMap<usize, String>>>,
    citation_documents: Vec<AnthropicCitationDocument>,
    next_source_index: Arc<AtomicUsize>,
    sources_by_id: Arc<Mutex<std::collections::HashMap<String, AnthropicSource>>>,
    tool_names_by_id: Arc<Mutex<std::collections::HashMap<String, String>>>,
    tool_callers_by_id: Arc<Mutex<std::collections::HashMap<String, serde_json::Value>>>,
    tool_input_json_by_id: Arc<Mutex<std::collections::HashMap<String, String>>>,
    server_tool_name_by_id: Arc<Mutex<std::collections::HashMap<String, String>>>,
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
    unsupported_stream_part_behavior: UnsupportedStreamPartBehavior,
}

impl AnthropicEventConverter {
    pub fn new(config: AnthropicParams) -> Self {
        Self {
            config,
            include_raw_chunks: false,
            custom_provider_metadata_key: None,
            state_tracker: StreamStateTracker::new(),
            tool_use_ids_by_index: Arc::new(Mutex::new(std::collections::HashMap::new())),
            content_block_type_by_index: Arc::new(Mutex::new(std::collections::HashMap::new())),
            text_content_by_index: Arc::new(Mutex::new(std::collections::HashMap::new())),
            thinking_content_by_index: Arc::new(Mutex::new(std::collections::HashMap::new())),
            thinking_signature_by_index: Arc::new(Mutex::new(std::collections::HashMap::new())),
            redacted_thinking_data_by_index: Arc::new(Mutex::new(std::collections::HashMap::new())),
            citation_documents: Vec::new(),
            next_source_index: Arc::new(AtomicUsize::new(0)),
            sources_by_id: Arc::new(Mutex::new(std::collections::HashMap::new())),
            tool_names_by_id: Arc::new(Mutex::new(std::collections::HashMap::new())),
            tool_callers_by_id: Arc::new(Mutex::new(std::collections::HashMap::new())),
            tool_input_json_by_id: Arc::new(Mutex::new(std::collections::HashMap::new())),
            server_tool_name_by_id: Arc::new(Mutex::new(std::collections::HashMap::new())),
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
            unsupported_stream_part_behavior: UnsupportedStreamPartBehavior::default(),
        }
    }

    pub fn with_citation_documents(mut self, docs: Vec<AnthropicCitationDocument>) -> Self {
        self.citation_documents = docs;
        self
    }

    pub fn with_unsupported_stream_part_behavior(
        mut self,
        behavior: UnsupportedStreamPartBehavior,
    ) -> Self {
        self.unsupported_stream_part_behavior = behavior;
        self
    }

    pub fn with_include_raw_chunks(mut self, include_raw_chunks: bool) -> Self {
        self.include_raw_chunks = include_raw_chunks;
        self
    }

    pub fn with_provider_metadata_key(mut self, key: impl Into<String>) -> Self {
        self.custom_provider_metadata_key =
            super::transformers::normalize_custom_anthropic_provider_key(&key.into());
        self
    }

    fn dynamic_code_execution_flag(&self, raw_server_tool_name: &str) -> Option<bool> {
        (self.config.should_mark_code_execution_dynamic()
            && raw_server_tool_name == "code_execution")
            .then_some(true)
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

    fn append_text_content(&self, index: usize, delta: &str) {
        if delta.is_empty() {
            return;
        }
        if let Ok(mut map) = self.text_content_by_index.lock() {
            let entry = map.entry(index).or_default();
            entry.push_str(delta);
        }
    }

    fn append_thinking_content(&self, index: usize, delta: &str) {
        if delta.is_empty() {
            return;
        }
        if let Ok(mut map) = self.thinking_content_by_index.lock() {
            let entry = map.entry(index).or_default();
            entry.push_str(delta);
        }
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

    fn record_tool_name(&self, tool_call_id: &str, tool_name: &str) {
        if tool_call_id.is_empty() || tool_name.is_empty() {
            return;
        }
        if let Ok(mut map) = self.tool_names_by_id.lock() {
            map.insert(tool_call_id.to_string(), tool_name.to_string());
        }
    }

    fn tool_name_for_id(&self, tool_call_id: &str) -> Option<String> {
        self.tool_names_by_id
            .lock()
            .ok()
            .and_then(|map| map.get(tool_call_id).cloned())
    }

    fn tool_use_id_for_index(&self, index: usize) -> Option<String> {
        self.tool_use_ids_by_index
            .lock()
            .ok()
            .and_then(|map| map.get(&index).cloned())
    }

    fn map_tool_caller_provider_metadata(caller: &serde_json::Value) -> Option<serde_json::Value> {
        let caller = caller.as_object()?;
        let mut mapped = serde_json::Map::new();
        mapped.insert("type".to_string(), caller.get("type")?.clone());
        if let Some(tool_id) = caller.get("tool_id").or_else(|| caller.get("toolId"))
            && !tool_id.is_null()
        {
            mapped.insert("toolId".to_string(), tool_id.clone());
        }
        Some(serde_json::Value::Object(mapped))
    }

    fn record_tool_caller(&self, tool_call_id: &str, caller: &serde_json::Value) {
        if tool_call_id.is_empty() {
            return;
        }
        let Some(mapped) = Self::map_tool_caller_provider_metadata(caller) else {
            return;
        };
        if let Ok(mut map) = self.tool_callers_by_id.lock() {
            map.insert(tool_call_id.to_string(), mapped);
        }
    }

    fn tool_caller_for_id(&self, tool_call_id: &str) -> Option<serde_json::Value> {
        self.tool_callers_by_id
            .lock()
            .ok()
            .and_then(|map| map.get(tool_call_id).cloned())
    }

    fn set_tool_input_json(&self, tool_call_id: &str, input: String) {
        if tool_call_id.is_empty() {
            return;
        }
        if let Ok(mut map) = self.tool_input_json_by_id.lock() {
            map.insert(tool_call_id.to_string(), input);
        }
    }

    fn record_server_tool_name(&self, tool_call_id: &str, raw_tool_name: &str) {
        if tool_call_id.is_empty() || raw_tool_name.is_empty() {
            return;
        }
        if let Ok(mut map) = self.server_tool_name_by_id.lock() {
            map.insert(tool_call_id.to_string(), raw_tool_name.to_string());
        }
    }

    fn server_tool_name_for_id(&self, tool_call_id: &str) -> Option<String> {
        self.server_tool_name_by_id
            .lock()
            .ok()
            .and_then(|map| map.get(tool_call_id).cloned())
    }

    fn append_tool_input_json(&self, tool_call_id: &str, delta: &str) {
        if tool_call_id.is_empty() || delta.is_empty() {
            return;
        }
        if let Ok(mut map) = self.tool_input_json_by_id.lock() {
            let entry = map.entry(tool_call_id.to_string()).or_default();
            entry.push_str(delta);
        }
    }

    fn take_tool_input_json(&self, tool_call_id: &str) -> Option<String> {
        self.tool_input_json_by_id
            .lock()
            .ok()
            .and_then(|mut map| map.remove(tool_call_id))
    }

    fn encode_non_empty_json(value: Option<&serde_json::Value>) -> Option<String> {
        let value = value?;
        if value.is_null() {
            return None;
        }
        if value.as_object().is_some_and(|obj| obj.is_empty()) {
            return None;
        }
        serde_json::to_string(value).ok()
    }

    fn next_source_id(&self) -> String {
        format!(
            "id-{}",
            self.next_source_index.fetch_add(1, Ordering::Relaxed)
        )
    }

    fn build_stream_provider_metadata(
        &self,
    ) -> Option<std::collections::HashMap<String, serde_json::Value>> {
        let mut anthropic = std::collections::HashMap::new();

        let usage = self.current_vercel_usage();
        anthropic.insert(
            "usage".to_string(),
            usage.clone().unwrap_or(serde_json::Value::Null),
        );
        anthropic.insert(
            "cacheCreationInputTokens".to_string(),
            usage
                .as_ref()
                .and_then(|usage| usage.get("cache_creation_input_tokens"))
                .cloned()
                .unwrap_or(serde_json::Value::Null),
        );
        anthropic.insert(
            "iterations".to_string(),
            usage
                .as_ref()
                .map(super::utils::map_usage_iterations_provider_metadata)
                .unwrap_or(serde_json::Value::Null),
        );

        if let Ok(v) = self.vercel_stop_sequence.lock() {
            anthropic.insert(
                "stopSequence".to_string(),
                v.clone()
                    .map(serde_json::Value::String)
                    .unwrap_or(serde_json::Value::Null),
            );
        }

        let container = self
            .vercel_container
            .lock()
            .ok()
            .and_then(|v| v.clone())
            .and_then(|value| super::utils::map_container_provider_metadata(&value))
            .unwrap_or(serde_json::Value::Null);
        anthropic.insert("container".to_string(), container);

        let context_management = self
            .vercel_context_management
            .lock()
            .ok()
            .and_then(|v| v.clone())
            .and_then(|value| super::utils::map_context_management_provider_metadata(&value))
            .unwrap_or(serde_json::Value::Null);
        anthropic.insert("contextManagement".to_string(), context_management);

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
            Some(super::transformers::anthropic_provider_metadata_map(
                anthropic,
                self.custom_provider_metadata_key.as_deref(),
            ))
        }
    }

    fn build_stream_content(&self) -> MessageContent {
        let mut ordered_types: Vec<(usize, String)> = self
            .content_block_type_by_index
            .lock()
            .map(|map| map.iter().map(|(idx, ty)| (*idx, ty.clone())).collect())
            .unwrap_or_default();
        ordered_types.sort_by_key(|(idx, _)| *idx);

        let text_blocks = self
            .text_content_by_index
            .lock()
            .map(|map| map.clone())
            .unwrap_or_default();
        let thinking_blocks = self
            .thinking_content_by_index
            .lock()
            .map(|map| map.clone())
            .unwrap_or_default();

        let mut parts = Vec::new();
        let mut text_buffer = String::new();

        for (idx, block_type) in ordered_types {
            match block_type.as_str() {
                "text" | "json_tool_use" | "compaction" => {
                    if let Some(text) = text_blocks.get(&idx)
                        && !text.is_empty()
                    {
                        text_buffer.push_str(text);
                        parts.push(ContentPart::text(text.clone()));
                    }
                }
                "thinking" => {
                    if let Some(thinking) = thinking_blocks.get(&idx)
                        && !thinking.is_empty()
                    {
                        parts.push(ContentPart::reasoning(thinking.clone()));
                    }
                }
                _ => {}
            }
        }

        if parts.len() == 1 && parts[0].is_text() {
            MessageContent::Text(text_buffer)
        } else if !parts.is_empty() {
            MessageContent::MultiModal(parts)
        } else if !text_buffer.is_empty() {
            MessageContent::Text(text_buffer)
        } else {
            MessageContent::Text(String::new())
        }
    }

    fn current_vercel_usage(&self) -> Option<serde_json::Value> {
        self.vercel_usage.lock().ok().and_then(|v| v.clone())
    }

    fn build_stream_response(
        &self,
        finish_reason: FinishReason,
        raw_finish_reason: Option<String>,
    ) -> ChatResponse {
        let usage_raw = self.current_vercel_usage();

        ChatResponse {
            id: self.vercel_response_id.lock().ok().and_then(|v| v.clone()),
            model: self.vercel_model_id.lock().ok().and_then(|v| v.clone()),
            content: self.build_stream_content(),
            usage: super::utils::create_usage_from_json_value(usage_raw.as_ref()),
            finish_reason: Some(finish_reason),
            raw_finish_reason,
            audio: None,
            system_fingerprint: None,
            service_tier: usage_raw
                .as_ref()
                .and_then(|usage| usage.get("service_tier"))
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned),
            warnings: None,
            request: None,
            response: None,
            provider_metadata: self.build_stream_provider_metadata(),
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

    fn anthropic_stream_provider_metadata(
        value: serde_json::Value,
    ) -> crate::types::StreamProviderMetadata {
        let mut provider_metadata = crate::types::StreamProviderMetadata::new();
        provider_metadata.insert("anthropic".to_string(), value);
        provider_metadata
    }

    fn anthropic_response_provider_metadata(
        &self,
        value: serde_json::Value,
    ) -> crate::types::StreamProviderMetadata {
        super::transformers::anthropic_provider_metadata_map_from_value(
            value,
            self.custom_provider_metadata_key.as_deref(),
        )
    }

    fn anthropic_content_block_provider_metadata(
        index: usize,
    ) -> crate::types::StreamProviderMetadata {
        Self::anthropic_stream_provider_metadata(serde_json::json!({
            "contentBlockIndex": index as u64,
        }))
    }

    fn anthropic_compaction_provider_metadata(
        index: usize,
    ) -> crate::types::StreamProviderMetadata {
        Self::anthropic_content_block_provider_metadata_with(Some(index), |anthropic| {
            anthropic.insert("type".to_string(), serde_json::json!("compaction"));
        })
        .unwrap_or_else(|| Self::anthropic_content_block_provider_metadata(index))
    }

    fn anthropic_json_tool_provider_metadata(
        index: usize,
        tool_call_id: &str,
    ) -> crate::types::StreamProviderMetadata {
        Self::anthropic_content_block_provider_metadata_with(Some(index), |anthropic| {
            anthropic.insert("type".to_string(), serde_json::json!("jsonTool"));
            anthropic.insert("toolCallId".to_string(), serde_json::json!(tool_call_id));
        })
        .unwrap_or_else(|| Self::anthropic_content_block_provider_metadata(index))
    }

    fn anthropic_tool_call_provider_metadata(
        index: Option<usize>,
        caller: Option<&serde_json::Value>,
    ) -> Option<crate::types::StreamProviderMetadata> {
        Self::anthropic_content_block_provider_metadata_with(index, |anthropic| {
            if let Some(caller) = caller
                && let Some(mapped) = Self::map_tool_caller_provider_metadata(caller)
            {
                anthropic.insert("caller".to_string(), mapped);
            }
        })
    }

    fn anthropic_content_block_provider_metadata_with<F>(
        index: Option<usize>,
        build_extra: F,
    ) -> Option<crate::types::StreamProviderMetadata>
    where
        F: FnOnce(&mut serde_json::Map<String, serde_json::Value>),
    {
        let mut anthropic = serde_json::Map::new();
        if let Some(index) = index {
            anthropic.insert(
                "contentBlockIndex".to_string(),
                serde_json::json!(index as u64),
            );
        }
        build_extra(&mut anthropic);
        (!anthropic.is_empty())
            .then(|| Self::anthropic_stream_provider_metadata(serde_json::Value::Object(anthropic)))
    }

    fn vercel_stream_start_event(&self) -> Option<ChatStreamEvent> {
        if self
            .vercel_stream_start_emitted
            .swap(true, Ordering::Relaxed)
        {
            return None;
        }

        Some(ChatStreamEvent::Part {
            part: ChatStreamPart::StreamStart { warnings: vec![] },
        })
    }

    fn fallback_stream_start_metadata(&self) -> ResponseMetadata {
        ResponseMetadata {
            id: self.vercel_response_id.lock().ok().and_then(|v| v.clone()),
            model: self.vercel_model_id.lock().ok().and_then(|v| v.clone()),
            created: Some(chrono::Utc::now()),
            provider: "anthropic".to_string(),
            request_id: None,
            headers: None,
            body: None,
        }
    }

    fn append_stream_start_events(
        &self,
        out: &mut Vec<ChatStreamEvent>,
        metadata: ResponseMetadata,
    ) {
        if !self.state_tracker.needs_stream_start() {
            return;
        }

        out.push(ChatStreamEvent::StreamStart { metadata });
        if let Some(evt) = self.vercel_stream_start_event() {
            out.push(evt);
        }
    }

    fn vercel_response_metadata_event(&self) -> Option<ChatStreamEvent> {
        let id = self
            .vercel_response_id
            .lock()
            .ok()
            .and_then(|v| v.clone())?;
        let model_id = self.vercel_model_id.lock().ok().and_then(|v| v.clone())?;

        Some(ChatStreamEvent::Part {
            part: ChatStreamPart::ResponseMetadata(ResponseMetadata {
                id: Some(id),
                model: Some(model_id),
                created: None,
                provider: "anthropic".to_string(),
                request_id: None,
                headers: None,
                body: None,
            }),
        })
    }

    fn vercel_text_start_event(id: usize) -> ChatStreamEvent {
        ChatStreamEvent::Part {
            part: ChatStreamPart::TextStart {
                id: id.to_string(),
                provider_metadata: Some(Self::anthropic_content_block_provider_metadata(id)),
            },
        }
    }

    fn vercel_text_end_event(id: usize) -> ChatStreamEvent {
        ChatStreamEvent::Part {
            part: ChatStreamPart::TextEnd {
                id: id.to_string(),
                provider_metadata: Some(Self::anthropic_content_block_provider_metadata(id)),
            },
        }
    }

    fn add_text_delta_part(
        builder: crate::streaming::EventBuilder,
        index: Option<usize>,
        delta: String,
        provider_metadata: Option<crate::types::StreamProviderMetadata>,
    ) -> crate::streaming::EventBuilder {
        if delta.is_empty() {
            return builder;
        }

        builder.add_part(ChatStreamPart::TextDelta {
            id: index
                .map(|idx| idx.to_string())
                .unwrap_or_else(|| "text".to_string()),
            delta,
            provider_metadata,
        })
    }

    fn add_reasoning_delta_part(
        builder: crate::streaming::EventBuilder,
        index: Option<usize>,
        delta: String,
        provider_metadata: Option<crate::types::StreamProviderMetadata>,
    ) -> crate::streaming::EventBuilder {
        if delta.is_empty() {
            return builder;
        }

        builder.add_part(ChatStreamPart::ReasoningDelta {
            id: index
                .map(|idx| idx.to_string())
                .unwrap_or_else(|| "reasoning".to_string()),
            delta,
            provider_metadata,
        })
    }

    fn vercel_json_tool_text_start_event(id: usize, tool_call_id: &str) -> ChatStreamEvent {
        ChatStreamEvent::Part {
            part: ChatStreamPart::TextStart {
                id: id.to_string(),
                provider_metadata: Some(Self::anthropic_json_tool_provider_metadata(
                    id,
                    tool_call_id,
                )),
            },
        }
    }

    fn vercel_json_tool_text_end_event(id: usize, tool_call_id: &str) -> ChatStreamEvent {
        ChatStreamEvent::Part {
            part: ChatStreamPart::TextEnd {
                id: id.to_string(),
                provider_metadata: Some(Self::anthropic_json_tool_provider_metadata(
                    id,
                    tool_call_id,
                )),
            },
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

        let usage =
            super::utils::create_usage_from_json_value(Some(&usage_raw)).unwrap_or_else(|| {
                let input_total = u32::try_from(
                    input_tokens
                        .saturating_add(cache_creation_input_tokens)
                        .saturating_add(cache_read_input_tokens),
                )
                .unwrap_or(u32::MAX);
                let input_no_cache = u32::try_from(input_tokens).unwrap_or(u32::MAX);
                let output_total = u32::try_from(output_tokens).unwrap_or(u32::MAX);
                let cache_read = u32::try_from(cache_read_input_tokens).unwrap_or(u32::MAX);
                let cache_write = u32::try_from(cache_creation_input_tokens).unwrap_or(u32::MAX);

                Usage::builder()
                    .prompt_tokens(input_no_cache)
                    .completion_tokens(output_total)
                    .total_tokens(input_no_cache.saturating_add(output_total))
                    .with_input_tokens(crate::types::UsageInputTokens {
                        total: Some(input_total),
                        no_cache: Some(input_no_cache),
                        cache_read: Some(cache_read),
                        cache_write: Some(cache_write),
                    })
                    .with_output_tokens(crate::types::UsageOutputTokens {
                        total: Some(output_total),
                        text: Some(output_total),
                        reasoning: None,
                    })
                    .with_raw_usage_value(usage_raw.clone())
                    .build()
            });

        ChatStreamEvent::Part {
            part: ChatStreamPart::Finish {
                usage,
                finish_reason: ChatStreamFinishInfo {
                    unified: finish_reason.clone(),
                    raw: raw_stop_reason.map(ToOwned::to_owned),
                },
                provider_metadata: Some(self.anthropic_response_provider_metadata(
                    serde_json::json!({
                        "cacheCreationInputTokens": cache_creation_input_tokens,
                        "container": container,
                        "contextManagement": context_management,
                        "iterations": super::utils::map_usage_iterations_provider_metadata(&usage_raw),
                        "stopSequence": stop_sequence,
                        "usage": usage_raw,
                    }),
                )),
            },
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
                    let preloaded_content = message.content.clone();
                    let metadata = ResponseMetadata {
                        id: message.id.clone(),
                        model: message.model.clone(),
                        created: Some(chrono::Utc::now()),
                        provider: "anthropic".to_string(),
                        request_id: None,
                        headers: None,
                        body: None,
                    };
                    let mut out: Vec<ChatStreamEvent> = Vec::new();
                    self.append_stream_start_events(&mut out, metadata);
                    if let Some(evt) = self.vercel_response_metadata_event() {
                        out.push(evt);
                    }
                    if let Some(content) = preloaded_content {
                        for (content_index, part) in content.into_iter().enumerate() {
                            if part.content_type != "tool_use" {
                                continue;
                            }

                            let Some(tool_call_id) = part.id else {
                                continue;
                            };
                            let Some(tool_name) = part.name else {
                                continue;
                            };

                            self.record_content_block_type(content_index, "tool_use".to_string());
                            self.record_tool_name(&tool_call_id, &tool_name);
                            if let Some(caller) = part.caller.as_ref() {
                                self.record_tool_caller(&tool_call_id, caller);
                            }

                            let input = part.input.unwrap_or_else(|| serde_json::json!({}));
                            let input_json =
                                serde_json::to_string(&input).unwrap_or_else(|_| "{}".to_string());
                            let block_provider_metadata = Some(
                                Self::anthropic_content_block_provider_metadata(content_index),
                            );
                            let tool_call_provider_metadata =
                                Self::anthropic_tool_call_provider_metadata(
                                    Some(content_index),
                                    part.caller.as_ref(),
                                );

                            out.push(ChatStreamEvent::Part {
                                part: ChatStreamPart::ToolInputStart {
                                    id: tool_call_id.clone(),
                                    tool_name: tool_name.clone(),
                                    provider_metadata: block_provider_metadata.clone(),
                                    provider_executed: None,
                                    dynamic: None,
                                    title: None,
                                },
                            });
                            out.push(ChatStreamEvent::Part {
                                part: ChatStreamPart::ToolInputDelta {
                                    id: tool_call_id.clone(),
                                    delta: input_json.clone(),
                                    provider_metadata: block_provider_metadata.clone(),
                                },
                            });
                            out.push(ChatStreamEvent::Part {
                                part: ChatStreamPart::ToolInputEnd {
                                    id: tool_call_id.clone(),
                                    provider_metadata: block_provider_metadata,
                                },
                            });
                            out.push(ChatStreamEvent::Part {
                                part: ChatStreamPart::ToolCall(ChatStreamToolCall {
                                    tool_call_id,
                                    tool_name,
                                    input: input_json,
                                    provider_executed: None,
                                    dynamic: None,
                                    provider_metadata: tool_call_provider_metadata,
                                }),
                            });
                        }
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
                    match effective_block_type {
                        "text" => {
                            if let Some(text) = content_block.get("text").and_then(|v| v.as_str()) {
                                self.append_text_content(idx, text);
                            }
                        }
                        "thinking" => {
                            if let Some(thinking) =
                                content_block.get("thinking").and_then(|v| v.as_str())
                            {
                                self.append_thinking_content(idx, thinking);
                            }
                        }
                        _ => {}
                    }
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
                    "compaction" => {
                        if let Some(idx) = event.index {
                            vec![ChatStreamEvent::Part {
                                part: ChatStreamPart::TextStart {
                                    id: idx.to_string(),
                                    provider_metadata: Some(
                                        Self::anthropic_compaction_provider_metadata(idx),
                                    ),
                                },
                            }]
                        } else {
                            vec![]
                        }
                    }
                    "thinking" => {
                        if let Some(idx) = event.index {
                            vec![ChatStreamEvent::Part {
                                part: ChatStreamPart::ReasoningStart {
                                    id: idx.to_string(),
                                    provider_metadata: Some(
                                        Self::anthropic_content_block_provider_metadata(idx),
                                    ),
                                },
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

                            vec![ChatStreamEvent::Part {
                                part: ChatStreamPart::ReasoningStart {
                                    id: idx.to_string(),
                                    provider_metadata: Some(
                                        Self::anthropic_stream_provider_metadata(
                                            serde_json::json!({
                                                "contentBlockIndex": idx as u64,
                                                "redactedData": if data.is_empty() { serde_json::Value::Null } else { serde_json::json!(data) },
                                            }),
                                        ),
                                    ),
                                },
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
                        let caller = content_block.get("caller").cloned();

                        if tool_call_id.is_empty() || tool_name.is_empty() {
                            return vec![];
                        }

                        if let Some(idx) = event.index
                            && let Ok(mut map) = self.tool_use_ids_by_index.lock()
                        {
                            map.insert(idx, tool_call_id.clone());
                        }
                        self.record_tool_name(&tool_call_id, &tool_name);
                        if let Some(caller) = caller.as_ref() {
                            self.record_tool_caller(&tool_call_id, caller);
                        }

                        let initial_input = Self::encode_non_empty_json(content_block.get("input"));
                        self.set_tool_input_json(
                            &tool_call_id,
                            initial_input.clone().unwrap_or_default(),
                        );

                        let provider_metadata = event
                            .index
                            .map(Self::anthropic_content_block_provider_metadata);
                        let mut events = vec![ChatStreamEvent::Part {
                            part: ChatStreamPart::ToolInputStart {
                                id: tool_call_id.clone(),
                                tool_name,
                                provider_metadata: provider_metadata.clone(),
                                provider_executed: None,
                                dynamic: None,
                                title: None,
                            },
                        }];

                        if let Some(initial_input) = initial_input {
                            events.push(ChatStreamEvent::Part {
                                part: ChatStreamPart::ToolInputDelta {
                                    id: tool_call_id,
                                    delta: initial_input,
                                    provider_metadata,
                                },
                            });
                        }

                        events
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
                            map.insert(idx, tool_call_id.clone());
                        }

                        // Vercel-aligned: do not emit tool-call deltas for the reserved `json` tool.
                        // The input chunks are exposed as text, with Anthropic metadata so gateways
                        // can replay the original reserved tool block.
                        if let Some(idx) = event.index {
                            vec![Self::vercel_json_tool_text_start_event(idx, &tool_call_id)]
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

                        let tool_name =
                            server_tools::normalize_server_tool_name(&tool_name_raw).to_string();
                        let input =
                            server_tools::normalize_server_tool_input(&tool_name_raw, input);
                        let dynamic = self.dynamic_code_execution_flag(&tool_name_raw);

                        if let Some(idx) = event.index
                            && let Ok(mut map) = self.tool_use_ids_by_index.lock()
                        {
                            map.insert(idx, tool_call_id.clone());
                        }
                        self.record_tool_name(&tool_call_id, &tool_name);
                        self.record_server_tool_name(&tool_call_id, &tool_name_raw);

                        let initial_input = Self::encode_non_empty_json(Some(&input));
                        self.set_tool_input_json(
                            &tool_call_id,
                            initial_input.clone().unwrap_or_default(),
                        );

                        let provider_metadata =
                            Self::anthropic_content_block_provider_metadata_with(
                                event.index,
                                |anthropic| {
                                    if tool_name_raw != tool_name {
                                        anthropic.insert(
                                            "serverToolName".to_string(),
                                            serde_json::json!(tool_name_raw),
                                        );
                                    }
                                },
                            );
                        let mut events = vec![ChatStreamEvent::Part {
                            part: ChatStreamPart::ToolInputStart {
                                id: tool_call_id.clone(),
                                tool_name,
                                provider_metadata: provider_metadata.clone(),
                                provider_executed: Some(true),
                                dynamic,
                                title: None,
                            },
                        }];

                        if let Some(initial_input) = initial_input {
                            events.push(ChatStreamEvent::Part {
                                part: ChatStreamPart::ToolInputDelta {
                                    id: tool_call_id,
                                    delta: initial_input,
                                    provider_metadata,
                                },
                            });
                        }

                        events
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

                        let input_json = Self::encode_non_empty_json(Some(&input))
                            .unwrap_or_else(|| "{}".to_string());
                        if !server_name.is_empty()
                            && let Ok(mut map) = self.mcp_server_name_by_id.lock()
                        {
                            map.insert(tool_call_id.clone(), server_name.clone());
                        }

                        let provider_metadata =
                            Self::anthropic_content_block_provider_metadata_with(
                                event.index,
                                |anthropic| {
                                    anthropic.insert(
                                        "type".to_string(),
                                        serde_json::json!("mcp-tool-use"),
                                    );
                                    if !server_name.is_empty() {
                                        anthropic.insert(
                                            "serverName".to_string(),
                                            serde_json::json!(server_name),
                                        );
                                    }
                                },
                            );

                        vec![ChatStreamEvent::Part {
                            part: ChatStreamPart::ToolCall(ChatStreamToolCall {
                                tool_call_id,
                                tool_name,
                                input: input_json,
                                provider_executed: Some(true),
                                dynamic: Some(true),
                                provider_metadata,
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
                            server_tools::normalize_server_tool_result_name(t).to_string()
                        };
                        let raw_result = content_block
                            .get("content")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null);

                        let (result, mut is_error) =
                            server_tools::normalize_server_tool_result(t, &raw_result)
                                .unwrap_or_else(|| (raw_result.clone(), false));

                        let server_name = if t == "mcp_tool_result" {
                            self.mcp_server_name_by_id
                                .lock()
                                .ok()
                                .and_then(|map| map.get(&tool_call_id).cloned())
                        } else {
                            None
                        };

                        if !is_error {
                            is_error = content_block
                                .get("is_error")
                                .and_then(|value| value.as_bool())
                                .unwrap_or(false);
                        }

                        let provider_metadata =
                            Self::anthropic_content_block_provider_metadata_with(
                                event.index,
                                |anthropic| {
                                    if t == "mcp_tool_result" {
                                        anthropic.insert(
                                            "type".to_string(),
                                            serde_json::json!("mcp-tool-use"),
                                        );
                                        if let Some(server_name) = server_name.as_ref() {
                                            anthropic.insert(
                                                "serverName".to_string(),
                                                serde_json::json!(server_name),
                                            );
                                        }
                                    } else if let Some(raw_server_tool_name) =
                                        self.server_tool_name_for_id(&tool_call_id)
                                        && raw_server_tool_name != tool_name
                                    {
                                        anthropic.insert(
                                            "serverToolName".to_string(),
                                            serde_json::json!(raw_server_tool_name),
                                        );
                                    }
                                },
                            );

                        let mut events = vec![ChatStreamEvent::Part {
                            part: ChatStreamPart::ToolResult(ChatStreamToolResult {
                                tool_call_id: tool_call_id.clone(),
                                tool_name,
                                result,
                                is_error: Some(is_error),
                                preliminary: None,
                                dynamic: (t == "mcp_tool_result").then_some(true),
                                provider_metadata,
                            }),
                        }];

                        // Vercel-aligned: emit sources for web search results
                        if t == "web_search_tool_result"
                            && let Some(arr) =
                                content_block.get("content").and_then(|v| v.as_array())
                        {
                            for item in arr {
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
                                let source_id = self.next_source_id();

                                self.record_source(AnthropicSource {
                                    id: source_id.clone(),
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

                                events.push(ChatStreamEvent::Part {
                                    part: ChatStreamPart::Source {
                                        id: source_id,
                                        source: SourcePart::Url {
                                            url: url.to_string(),
                                            title: title.map(ToOwned::to_owned),
                                        },
                                        provider_metadata: Some(
                                            Self::anthropic_stream_provider_metadata(
                                                serde_json::json!({
                                                    "pageAge": page_age,
                                                }),
                                            ),
                                        ),
                                    },
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
                                if let Some(idx) = event.index {
                                    self.append_text_content(idx, &text);
                                }
                                if let Some(idx) = event.index
                                    && self.get_content_block_type(idx).as_deref() == Some("text")
                                {
                                    builder = Self::add_text_delta_part(
                                        builder,
                                        Some(idx),
                                        text,
                                        Some(Self::anthropic_content_block_provider_metadata(idx)),
                                    );
                                } else {
                                    builder =
                                        Self::add_text_delta_part(builder, event.index, text, None);
                                }
                            }
                        }
                        Some("thinking_delta") => {
                            if let Some(thinking) = delta.thinking {
                                if let Some(idx) = event.index {
                                    self.append_thinking_content(idx, &thinking);
                                    builder = builder.add_part(ChatStreamPart::ReasoningDelta {
                                        id: idx.to_string(),
                                        delta: thinking,
                                        provider_metadata: Some(
                                            Self::anthropic_content_block_provider_metadata(idx),
                                        ),
                                    });
                                } else {
                                    builder = Self::add_reasoning_delta_part(
                                        builder,
                                        event.index,
                                        thinking,
                                        None,
                                    );
                                }
                            }
                        }
                        Some("compaction_delta") => {
                            if let Some(content) = delta.content
                                && let Some(idx) = event.index
                            {
                                self.append_text_content(idx, &content);
                                if self.get_content_block_type(idx).as_deref() == Some("compaction")
                                {
                                    builder = builder.add_part(ChatStreamPart::TextDelta {
                                        id: idx.to_string(),
                                        delta: content,
                                        provider_metadata: Some(
                                            Self::anthropic_compaction_provider_metadata(idx),
                                        ),
                                    });
                                } else {
                                    builder = Self::add_text_delta_part(
                                        builder,
                                        Some(idx),
                                        content,
                                        Some(Self::anthropic_content_block_provider_metadata(idx)),
                                    );
                                }
                            }
                        }
                        Some("signature_delta") => {
                            if let (Some(idx), Some(sig_delta)) = (event.index, delta.signature)
                                && self
                                    .get_content_block_type(idx)
                                    .is_some_and(|t| t == "thinking")
                            {
                                self.append_thinking_signature(idx, sig_delta.clone());
                                builder = builder.add_part(ChatStreamPart::ReasoningDelta {
                                    id: idx.to_string(),
                                    delta: String::new(),
                                    provider_metadata:
                                        Self::anthropic_content_block_provider_metadata_with(
                                            Some(idx),
                                            |anthropic| {
                                                anthropic.insert(
                                                    "signature".to_string(),
                                                    serde_json::json!(sig_delta),
                                                );
                                            },
                                        ),
                                });
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

                                        let id = self.next_source_id();

                                        builder = builder.add_part(ChatStreamPart::Source {
                                            id: id.clone(),
                                            source: SourcePart::Document {
                                                media_type: doc.media_type.clone(),
                                                title: title.clone(),
                                                filename: doc.filename.clone(),
                                            },
                                            provider_metadata: Some(
                                                Self::anthropic_stream_provider_metadata(
                                                    serde_json::json!({
                                                        "citedText": cited_text,
                                                        "startPageNumber": start_page,
                                                        "endPageNumber": end_page,
                                                    }),
                                                ),
                                            ),
                                        });

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

                                        let id = self.next_source_id();

                                        builder = builder.add_part(ChatStreamPart::Source {
                                            id: id.clone(),
                                            source: SourcePart::Document {
                                                media_type: doc.media_type.clone(),
                                                title: title.clone(),
                                                filename: doc.filename.clone(),
                                            },
                                            provider_metadata: Some(
                                                Self::anthropic_stream_provider_metadata(
                                                    serde_json::json!({
                                                        "citedText": cited_text,
                                                        "startCharIndex": start_char,
                                                        "endCharIndex": end_char,
                                                    }),
                                                ),
                                            ),
                                        });

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
                                    let provider_metadata = self
                                        .tool_use_id_for_index(idx)
                                        .map(|tool_call_id| {
                                            Self::anthropic_json_tool_provider_metadata(
                                                idx,
                                                &tool_call_id,
                                            )
                                        })
                                        .unwrap_or_else(|| {
                                            Self::anthropic_content_block_provider_metadata(idx)
                                        });
                                    builder = builder.add_part(ChatStreamPart::TextDelta {
                                        id: idx.to_string(),
                                        delta: partial_json.clone(),
                                        provider_metadata: Some(provider_metadata),
                                    });
                                    if self.should_stream_json_tool_as_text() {
                                        self.append_text_content(idx, &partial_json);
                                    }
                                } else if let Ok(map) = self.tool_use_ids_by_index.lock()
                                    && let Some(tool_call_id) = map.get(&idx)
                                {
                                    self.append_tool_input_json(tool_call_id, &partial_json);
                                    builder = builder.add_part(ChatStreamPart::ToolInputDelta {
                                        id: tool_call_id.clone(),
                                        delta: partial_json,
                                        provider_metadata: Some(
                                            Self::anthropic_content_block_provider_metadata(idx),
                                        ),
                                    });
                                }
                            }
                        }
                        _ => {
                            if let Some(text) = delta.text {
                                if let Some(idx) = event.index
                                    && self.get_content_block_type(idx).as_deref() == Some("text")
                                {
                                    self.append_text_content(idx, &text);
                                }
                                if let Some(idx) = event.index
                                    && self.get_content_block_type(idx).as_deref() == Some("text")
                                {
                                    builder = Self::add_text_delta_part(
                                        builder,
                                        Some(idx),
                                        text,
                                        Some(Self::anthropic_content_block_provider_metadata(idx)),
                                    );
                                } else {
                                    builder =
                                        Self::add_text_delta_part(builder, event.index, text, None);
                                }
                            }
                            if let Some(thinking) = delta.thinking {
                                if let Some(idx) = event.index
                                    && self.get_content_block_type(idx).as_deref()
                                        == Some("thinking")
                                {
                                    self.append_thinking_content(idx, &thinking);
                                }
                                let provider_metadata = event
                                    .index
                                    .map(Self::anthropic_content_block_provider_metadata);
                                builder = Self::add_reasoning_delta_part(
                                    builder,
                                    event.index,
                                    thinking,
                                    provider_metadata,
                                );
                            }
                            if let Some(partial_json) = delta.partial_json
                                && !partial_json.is_empty()
                                && let Some(idx) = event.index
                            {
                                if self.get_content_block_type(idx).as_deref()
                                    == Some("json_tool_use")
                                {
                                    self.append_text_content(idx, &partial_json);
                                    let provider_metadata = self
                                        .tool_use_id_for_index(idx)
                                        .map(|tool_call_id| {
                                            Self::anthropic_json_tool_provider_metadata(
                                                idx,
                                                &tool_call_id,
                                            )
                                        })
                                        .unwrap_or_else(|| {
                                            Self::anthropic_content_block_provider_metadata(idx)
                                        });
                                    builder = builder.add_part(ChatStreamPart::TextDelta {
                                        id: idx.to_string(),
                                        delta: partial_json.clone(),
                                        provider_metadata: Some(provider_metadata),
                                    });
                                } else if let Ok(map) = self.tool_use_ids_by_index.lock()
                                    && let Some(tool_call_id) = map.get(&idx)
                                {
                                    self.append_tool_input_json(tool_call_id, &partial_json);
                                    builder = builder.add_part(ChatStreamPart::ToolInputDelta {
                                        id: tool_call_id.clone(),
                                        delta: partial_json,
                                        provider_metadata: Some(
                                            Self::anthropic_content_block_provider_metadata(idx),
                                        ),
                                    });
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
                    Some("compaction") => vec![ChatStreamEvent::Part {
                        part: ChatStreamPart::TextEnd {
                            id: idx.to_string(),
                            provider_metadata: Some(Self::anthropic_compaction_provider_metadata(
                                idx,
                            )),
                        },
                    }],
                    Some("tool_use") => {
                        let Some(tool_call_id) = self
                            .tool_use_ids_by_index
                            .lock()
                            .ok()
                            .and_then(|map| map.get(&idx).cloned())
                        else {
                            return vec![];
                        };

                        let tool_name = self
                            .tool_name_for_id(&tool_call_id)
                            .unwrap_or_else(|| "tool".to_string());
                        let input = self
                            .take_tool_input_json(&tool_call_id)
                            .filter(|value| !value.is_empty())
                            .unwrap_or_else(|| "{}".to_string());
                        let tool_caller = self.tool_caller_for_id(&tool_call_id);
                        let block_provider_metadata =
                            Some(Self::anthropic_content_block_provider_metadata(idx));
                        let tool_call_provider_metadata =
                            Self::anthropic_tool_call_provider_metadata(
                                Some(idx),
                                tool_caller.as_ref(),
                            );

                        vec![
                            ChatStreamEvent::Part {
                                part: ChatStreamPart::ToolInputEnd {
                                    id: tool_call_id.clone(),
                                    provider_metadata: block_provider_metadata,
                                },
                            },
                            ChatStreamEvent::Part {
                                part: ChatStreamPart::ToolCall(ChatStreamToolCall {
                                    tool_call_id,
                                    tool_name,
                                    input,
                                    provider_executed: None,
                                    dynamic: None,
                                    provider_metadata: tool_call_provider_metadata,
                                }),
                            },
                        ]
                    }
                    Some("server_tool_use") => {
                        let Some(tool_call_id) = self
                            .tool_use_ids_by_index
                            .lock()
                            .ok()
                            .and_then(|map| map.get(&idx).cloned())
                        else {
                            return vec![];
                        };

                        let tool_name = self
                            .tool_name_for_id(&tool_call_id)
                            .unwrap_or_else(|| "tool".to_string());
                        let input = self
                            .take_tool_input_json(&tool_call_id)
                            .filter(|value| !value.is_empty())
                            .unwrap_or_else(|| "{}".to_string());
                        let raw_server_tool_name = self.server_tool_name_for_id(&tool_call_id);
                        let dynamic = raw_server_tool_name
                            .as_deref()
                            .and_then(|name| self.dynamic_code_execution_flag(name));
                        let provider_metadata =
                            Self::anthropic_content_block_provider_metadata_with(
                                Some(idx),
                                |anthropic| {
                                    if let Some(raw_server_tool_name) =
                                        raw_server_tool_name.as_ref()
                                        && raw_server_tool_name != &tool_name
                                    {
                                        anthropic.insert(
                                            "serverToolName".to_string(),
                                            serde_json::json!(raw_server_tool_name),
                                        );
                                    }
                                },
                            );

                        vec![
                            ChatStreamEvent::Part {
                                part: ChatStreamPart::ToolInputEnd {
                                    id: tool_call_id.clone(),
                                    provider_metadata: provider_metadata.clone(),
                                },
                            },
                            ChatStreamEvent::Part {
                                part: ChatStreamPart::ToolCall(ChatStreamToolCall {
                                    tool_call_id,
                                    tool_name,
                                    input,
                                    provider_executed: Some(true),
                                    dynamic,
                                    provider_metadata,
                                }),
                            },
                        ]
                    }
                    Some("json_tool_use") => {
                        if let Some(tool_call_id) = self.tool_use_id_for_index(idx) {
                            vec![Self::vercel_json_tool_text_end_event(idx, &tool_call_id)]
                        } else {
                            vec![]
                        }
                    }
                    Some("thinking") | Some("redacted_thinking") => vec![ChatStreamEvent::Part {
                        part: ChatStreamPart::ReasoningEnd {
                            id: idx.to_string(),
                            provider_metadata: Some(
                                Self::anthropic_content_block_provider_metadata(idx),
                            ),
                        },
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
                    builder = Self::add_reasoning_delta_part(builder, None, thinking.clone(), None);
                }

                // Usage update
                if let Some(usage) = &event.usage {
                    self.merge_vercel_usage(usage);
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

                if let Ok(mut v) = self.vercel_container.lock() {
                    *v = event.delta.as_ref().and_then(|d| d.container.clone());
                }

                if let Some(delta) = &event.delta
                    && let Ok(mut v) = self.vercel_stop_sequence.lock()
                {
                    *v = delta.stop_sequence.clone();
                }

                // Finish reason -> StreamEnd
                if let Some(delta) = &event.delta
                    && let Some(stop_reason) = &delta.stop_reason
                {
                    let reason = if stop_reason == "tool_use"
                        && self.json_tool_seen.load(Ordering::Relaxed)
                    {
                        FinishReason::Stop
                    } else {
                        parse_finish_reason(Some(stop_reason.as_str()))
                            .unwrap_or(FinishReason::Unknown)
                    };

                    if self.state_tracker.needs_stream_end() {
                        if let ChatStreamEvent::Part { part } =
                            self.vercel_finish_event(Some(stop_reason.as_str()), &reason)
                        {
                            builder = builder.add_part(part);
                        }

                        let response =
                            self.build_stream_response(reason, Some(stop_reason.clone()));
                        builder = builder.add_stream_end(response);
                    }
                }

                builder.build()
            }
            "message_stop" => {
                let response = self.build_stream_response(FinishReason::Stop, None);
                if !self.state_tracker.needs_stream_end() {
                    return vec![];
                }

                let mut out = Vec::new();
                if let ChatStreamEvent::Part { part } =
                    self.vercel_finish_event(None, &FinishReason::Stop)
                {
                    out.push(ChatStreamEvent::Part { part });
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

            match serde_json::from_str::<serde_json::Value>(&event.data) {
                Ok(raw_json) => {
                    match serde_json::from_value::<AnthropicStreamEvent>(raw_json.clone()) {
                        Ok(anthropic_event) => self
                            .inject_raw_chunk(
                                self.convert_anthropic_event(anthropic_event),
                                raw_json,
                            )
                            .into_iter()
                            .map(Ok)
                            .collect(),
                        Err(e) => {
                            tracing::warn!("Failed to parse Anthropic SSE event: {}", e);
                            tracing::warn!("Raw event data: {}", event.data);
                            tracing::warn!("Event parsed as generic JSON: {:#}", raw_json);

                            if let Some(error_obj) = raw_json.get("error") {
                                let error_message = error_obj
                                    .get("message")
                                    .and_then(|m| m.as_str())
                                    .unwrap_or("Unknown error");

                                self.seen_error.store(true, Ordering::Relaxed);
                                let mut events = Vec::new();
                                self.append_stream_start_events(
                                    &mut events,
                                    self.fallback_stream_start_metadata(),
                                );
                                events.push(ChatStreamEvent::Error {
                                    error: format!("Anthropic API error: {}", error_message),
                                });
                                return self
                                    .inject_raw_chunk(events, raw_json)
                                    .into_iter()
                                    .map(Ok)
                                    .collect();
                            }

                            let mut events = Vec::new();
                            self.append_stream_start_events(
                                &mut events,
                                self.fallback_stream_start_metadata(),
                            );

                            let mut out: Vec<Result<ChatStreamEvent, LlmError>> = self
                                .inject_raw_chunk(events, raw_json)
                                .into_iter()
                                .map(Ok)
                                .collect();
                            out.push(Err(LlmError::ParseError(format!(
                                "Failed to parse Anthropic event: {}. Raw data: {}",
                                e, event.data
                            ))));
                            out
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to parse Anthropic SSE event: {}", e);
                    tracing::warn!("Raw event data: {}", event.data);

                    let mut events = Vec::new();
                    self.append_stream_start_events(
                        &mut events,
                        self.fallback_stream_start_metadata(),
                    );

                    let mut out: Vec<Result<ChatStreamEvent, LlmError>> = self
                        .inject_raw_chunk(events, serde_json::Value::String(event.data.clone()))
                        .into_iter()
                        .map(Ok)
                        .collect();
                    out.push(Err(LlmError::ParseError(format!(
                        "Failed to parse Anthropic event: {}. Raw data: {}",
                        e, event.data
                    ))));
                    out
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

        let response = self.build_stream_response(
            if self.seen_error.load(Ordering::Relaxed) {
                FinishReason::Error
            } else {
                FinishReason::Unknown
            },
            None,
        );

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
