//! OpenAI-compatible streaming implementation (protocol layer)
//!
//! Provides SSE event conversion for OpenAI-compatible providers.
//! The legacy OpenAiCompatibleStreaming client has been removed in favor of the unified HttpChatExecutor.

use crate::error::LlmError;
use crate::streaming::SseEventConverter;
use crate::streaming::{
    ChatStreamEvent, LanguageModelV3Source, LanguageModelV3StreamPart, StreamStateTracker,
    V3UnsupportedPartBehavior,
};
use crate::types::{
    ChatResponse, ChatStreamFinishInfo, ChatStreamPart, FinishReason, MessageContent,
    ProviderMetadataMap, ResponseMetadata, Usage,
};
use eventsource_stream::Event;
use serde::{Deserialize, Serialize};

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use super::adapter::ProviderAdapter;
use super::metadata::{
    NestedProviderMetadata, ensure_provider_metadata_namespace, merge_nested_provider_metadata,
    nested_provider_metadata_to_map, provider_options_key, resolve_provider_metadata_key,
};
use super::openai_config::OpenAiCompatibleConfig;

#[derive(Debug, Default, Clone)]
struct OpenAiCompatSerializeState {
    id: Option<String>,
    model: Option<String>,
    created: Option<u64>,
    system_fingerprint: Option<String>,
    service_tier: Option<String>,
    emitted_role: bool,
    tool_call_state_by_id: std::collections::HashMap<String, OpenAiCompatSerializeToolCallState>,
    next_tool_call_index: u32,
    finished: bool,
}

#[derive(Debug, Default, Clone)]
struct OpenAiCompatParseState {
    tool_call_state_by_index: std::collections::HashMap<(u32, u32), OpenAiCompatParseToolCallState>,
    response_metadata_emitted: bool,
    active_text_part_id: Option<String>,
    next_text_part_index: u32,
    active_reasoning_part_id: Option<String>,
    next_reasoning_part_index: u32,
    next_source_part_index: u32,
}

#[derive(Debug, Default, Clone)]
struct OpenAiCompatResponseState {
    id: Option<String>,
    model: Option<String>,
    created: Option<chrono::DateTime<chrono::Utc>>,
    system_fingerprint: Option<String>,
    service_tier: Option<String>,
}

#[derive(Debug, Default, Clone)]
struct OpenAiCompatParseToolCallState {
    id: String,
    name: String,
    arguments: String,
    thought_signature: Option<String>,
    stable_input_started: bool,
    stable_tool_call_emitted: bool,
}

#[derive(Debug, Default, Clone)]
struct OpenAiCompatSerializeToolCallState {
    index: u32,
    source: Option<OpenAiCompatToolStreamSource>,
    emitted_name: bool,
    emitted_arguments: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpenAiCompatToolStreamSource {
    LegacyDelta,
    StablePart,
}

fn ensure_serialize_tool_call_state<'a>(
    state: &'a mut OpenAiCompatSerializeState,
    id: &str,
    source: OpenAiCompatToolStreamSource,
) -> Option<&'a mut OpenAiCompatSerializeToolCallState> {
    use std::collections::hash_map::Entry;

    match state.tool_call_state_by_id.entry(id.to_string()) {
        Entry::Vacant(entry) => {
            let index = state.next_tool_call_index;
            state.next_tool_call_index = state.next_tool_call_index.saturating_add(1);
            Some(entry.insert(OpenAiCompatSerializeToolCallState {
                index,
                source: Some(source),
                ..OpenAiCompatSerializeToolCallState::default()
            }))
        }
        Entry::Occupied(entry) => {
            let slot = entry.into_mut();
            match slot.source {
                Some(existing) if existing != source => None,
                Some(_) => Some(slot),
                None => {
                    slot.source = Some(source);
                    Some(slot)
                }
            }
        }
    }
}

fn is_parsable_json(value: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(value).is_ok()
}

/// OpenAI-compatible stream event structure
#[derive(Debug, Deserialize, Serialize)]
pub struct OpenAiCompatibleStreamEvent {
    pub id: Option<String>,
    pub object: Option<String>,
    pub created: Option<u64>,
    pub model: Option<String>,
    pub choices: Option<Vec<StreamChoice>>,
    pub usage: Option<StreamUsage>,
}

/// Stream choice structure
#[derive(Debug, Deserialize, Serialize)]
pub struct StreamChoice {
    pub index: Option<u32>,
    pub delta: Option<StreamDelta>,
    pub finish_reason: Option<String>,
}

/// Stream delta structure with provider-specific fields
#[derive(Debug, Deserialize, Serialize)]
pub struct StreamDelta {
    pub role: Option<String>,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<serde_json::Value>>,
    pub annotations: Option<Vec<StreamAnnotation>>,

    // Provider-specific thinking/reasoning fields
    pub thinking: Option<String>,          // Standard thinking field
    pub reasoning_content: Option<String>, // DeepSeek reasoning field
    pub reasoning: Option<String>,         // Alternative reasoning field
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StreamAnnotation {
    #[serde(default, rename = "type")]
    pub annotation_type: Option<String>,
    pub url_citation: Option<StreamUrlCitation>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StreamUrlCitation {
    pub url: Option<String>,
    pub title: Option<String>,
}

/// Stream usage structure
#[derive(Debug, Deserialize, Serialize)]
pub struct StreamUsage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: Option<u32>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: Option<u32>,
}

/// Event converter for OpenAI-compatible providers
#[derive(Clone)]
pub struct OpenAiCompatibleEventConverter {
    config: OpenAiCompatibleConfig,
    adapter: Arc<dyn ProviderAdapter>,
    provider_metadata_key: String,
    include_raw_chunks: bool,
    state_tracker: StreamStateTracker,
    // Accumulate plain text content so StreamEnd can carry a fallback when no deltas were seen
    accumulated_content: Arc<tokio::sync::Mutex<String>>,
    // Track whether we have emitted any ContentDelta to avoid duplicate injection
    emitted_content: std::sync::Arc<std::sync::atomic::AtomicBool>,
    // Track the latest usage snapshot so StreamEnd can carry final usage.
    latest_usage: Arc<std::sync::Mutex<Option<Usage>>>,
    // Track provider-specific metadata gathered across chunks.
    latest_provider_metadata: Arc<std::sync::Mutex<Option<NestedProviderMetadata>>>,
    // Track top-level response fields so terminal StreamEnd can preserve them.
    latest_response_state: Arc<std::sync::Mutex<OpenAiCompatResponseState>>,

    parse_state: Arc<std::sync::Mutex<OpenAiCompatParseState>>,

    // Serialize state for reverse SSE encoding (ChatStreamEvent -> OpenAI-compatible SSE).
    serialize_state: Arc<std::sync::Mutex<OpenAiCompatSerializeState>>,
    v3_unsupported_part_behavior: V3UnsupportedPartBehavior,
}

impl OpenAiCompatibleEventConverter {
    fn created_datetime_from_unix_seconds(
        ts: Option<u64>,
    ) -> Option<chrono::DateTime<chrono::Utc>> {
        let ts = ts?;
        if ts == 0 {
            return None;
        }
        chrono::DateTime::from_timestamp(ts as i64, 0)
    }

    /// Create a new event converter
    pub fn new(config: OpenAiCompatibleConfig, adapter: Arc<dyn ProviderAdapter>) -> Self {
        Self {
            provider_metadata_key: resolve_provider_metadata_key(&config.provider_id, None),
            config,
            adapter,
            include_raw_chunks: false,
            state_tracker: StreamStateTracker::new(),
            accumulated_content: Arc::new(tokio::sync::Mutex::new(String::new())),
            emitted_content: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            latest_usage: Arc::new(std::sync::Mutex::new(None)),
            latest_provider_metadata: Arc::new(std::sync::Mutex::new(None)),
            latest_response_state: Arc::new(std::sync::Mutex::new(
                OpenAiCompatResponseState::default(),
            )),
            parse_state: Arc::new(std::sync::Mutex::new(OpenAiCompatParseState::default())),
            serialize_state: Arc::new(std::sync::Mutex::new(OpenAiCompatSerializeState::default())),
            v3_unsupported_part_behavior: V3UnsupportedPartBehavior::Drop,
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

    pub fn with_provider_metadata_key(mut self, key: impl Into<String>) -> Self {
        self.provider_metadata_key = key.into();
        self
    }

    fn current_provider_metadata(&self) -> NestedProviderMetadata {
        ensure_provider_metadata_namespace(
            self.latest_provider_metadata
                .lock()
                .ok()
                .and_then(|meta| meta.clone()),
            &self.provider_metadata_key,
            &provider_options_key(self.adapter.provider_id().as_ref()),
        )
    }

    /// Convert OpenAI-compatible stream event to multiple ChatStreamEvents
    #[allow(dead_code)]
    async fn convert_event_async(
        &self,
        event: OpenAiCompatibleStreamEvent,
    ) -> Vec<ChatStreamEvent> {
        use crate::streaming::EventBuilder;

        let mut builder = EventBuilder::new();

        // Check if we need to emit StreamStart first
        if self.needs_stream_start() {
            let metadata = self.create_stream_start_metadata(&event);
            builder = builder.add_stream_start(metadata);
        }

        // Process content delta
        if let Some(content) = self.extract_content(&event) {
            builder = builder.add_content_delta(
                content.clone(),
                Some(self.extract_choice_index(&event) as usize),
            );
        }

        // Process thinking/reasoning content using adapter
        if let Some(thinking) = self.extract_thinking(&event) {
            builder = builder.add_thinking_delta(thinking.clone());
        }

        // Process tool calls
        if let Some((id, name, args)) = self.extract_tool_call(&event) {
            builder = builder.add_tool_call_delta(
                id,
                Some(name),
                Some(args),
                Some(self.extract_choice_index(&event) as usize),
            );
        }

        // Process usage updates
        if let Some(usage) = self.extract_usage(&event) {
            builder = builder.add_usage_update(usage);
        }

        builder.build()
    }

    /// JSON-first conversion to avoid losing unknown fields with strict structs
    /// and to be compatible with different streaming shapes (e.g. Responses API).
    async fn convert_event_json_async(&self, json: &serde_json::Value) -> Vec<ChatStreamEvent> {
        use crate::streaming::EventBuilder;

        let mut builder = EventBuilder::new();

        // First event: emit StreamStart (extracted directly from JSON)
        if self.needs_stream_start() {
            let metadata = self.create_stream_start_metadata_from_json(json);
            builder = builder
                .add_stream_start(metadata)
                .add_part(ChatStreamPart::StreamStart { warnings: vec![] });
        }

        if self.include_raw_chunks {
            builder = builder.add_part(ChatStreamPart::Raw {
                raw_value: json.clone(),
            });
        }

        // If the event data itself is a JSON string (common when SSE named events
        // carry plain text as data, e.g., Responses "output_text.delta" proxied),
        // treat it directly as a content delta.
        if let Some(s) = json.as_str()
            && !s.trim().is_empty()
        {
            for part in self.open_text_lane() {
                builder = builder.add_part(part);
            }
            {
                let mut acc = self.accumulated_content.lock().await;
                acc.push_str(s);
            }
            self.emitted_content
                .store(true, std::sync::atomic::Ordering::Relaxed);
            return builder.add_content_delta(s.to_string(), None).build();
        }

        self.update_response_state_from_json(json);

        if let Some(part) = self.take_response_metadata_part() {
            builder = builder.add_part(part);
        }

        // Content (compatible with Chat Completions and Responses API)
        if let Some(content) = self.extract_content_from_json(json) {
            for part in self.open_text_lane() {
                builder = builder.add_part(part);
            }
            {
                let mut acc = self.accumulated_content.lock().await;
                acc.push_str(&content);
            }
            builder = builder.add_content_delta(content, self.extract_choice_index_from_json(json));
            self.emitted_content
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }

        // Thinking/Reasoning content (optional)
        if let Some(thinking) = self.extract_thinking_from_json(json) {
            for part in self.open_reasoning_lane() {
                builder = builder.add_part(part);
            }
            builder = builder.add_thinking_delta(thinking);
        }

        // Tool call deltas (optional) — support multiple tool calls in the same chunk
        let tool_call_events = self.extract_tool_call_events_from_json(json);
        if !tool_call_events.is_empty() {
            for event in tool_call_events {
                match event {
                    ChatStreamEvent::ToolCallDelta {
                        id,
                        function_name,
                        arguments_delta,
                        index,
                    } => {
                        builder =
                            builder.add_tool_call_delta(id, function_name, arguments_delta, index);
                    }
                    ChatStreamEvent::Part { part } => {
                        builder = builder.add_part(part);
                    }
                    ChatStreamEvent::PartWithReplay { part, replay } => {
                        builder = builder.add_part_with_replay(part, replay);
                    }
                    _ => {}
                }
            }
        }

        let source_events = self.extract_source_events_from_json(json);
        if !source_events.is_empty() {
            for event in source_events {
                match event {
                    ChatStreamEvent::Part { part } => {
                        builder = builder.add_part(part);
                    }
                    ChatStreamEvent::PartWithReplay { part, replay } => {
                        builder = builder.add_part_with_replay(part, replay);
                    }
                    _ => {}
                }
            }
        }

        // Usage updates (optional)
        if let Some(usage) = self.extract_usage_from_json(json) {
            *self.latest_usage.lock().unwrap() = Some(usage.clone());
            builder = builder.add_usage_update(usage);
        }

        {
            let provider_metadata = ensure_provider_metadata_namespace(
                self.adapter.extract_response_provider_metadata(json),
                &self.provider_metadata_key,
                &provider_options_key(self.adapter.provider_id().as_ref()),
            );
            let mut latest = self.latest_provider_metadata.lock().unwrap();
            if let Some(current) = latest.as_mut() {
                merge_nested_provider_metadata(current, provider_metadata);
            } else {
                *latest = Some(provider_metadata);
            }
        }

        // Detect finish_reason in choices to close stream even if server omits [DONE]
        // This mirrors real OpenAI Chat Completions behavior and Vercel AI SDK logic
        if let Some(reason) = json
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c0| c0.get("finish_reason"))
            .and_then(|v| v.as_str())
        {
            // Mark that StreamEnd is being emitted
            self.state_tracker.mark_stream_ended();

            // Build a StreamEnd using accumulated content snapshot
            let text = self
                .accumulated_content
                .try_lock()
                .map(|g| g.clone())
                .unwrap_or_default();
            // If we have not emitted any ContentDelta during the stream but did
            // accumulate text, emit a synthetic ContentDelta before StreamEnd so
            // downstream consumers that assert on deltas can pass.
            if !text.is_empty()
                && !self
                    .emitted_content
                    .load(std::sync::atomic::Ordering::Relaxed)
            {
                builder = builder
                    .add_content_delta(text.clone(), self.extract_choice_index_from_json(json));
                // Mark as emitted to prevent double insertion if multiple finish signals arrive
                self.emitted_content
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }
            let response_state = self
                .latest_response_state
                .lock()
                .ok()
                .map(|state| state.clone())
                .unwrap_or_default();
            let response = ChatResponse {
                id: response_state.id,
                model: response_state.model,
                content: MessageContent::Text(text),
                usage: self
                    .latest_usage
                    .lock()
                    .ok()
                    .and_then(|usage| usage.clone()),
                finish_reason: crate::standards::openai::utils::parse_finish_reason(Some(reason)),
                raw_finish_reason: Some(reason.to_string()),
                audio: None,
                system_fingerprint: response_state.system_fingerprint,
                service_tier: response_state.service_tier,
                warnings: None,
                provider_metadata: Some(Self::nested_provider_metadata_to_stream(
                    self.current_provider_metadata(),
                )),
            };
            for part in self.close_active_content_parts() {
                builder = builder.add_part(part);
            }
            builder = builder
                .add_part(self.build_finish_part(Some(reason)))
                .add_stream_end(response);
        }

        builder.build()
    }

    /// Check if StreamStart event needs to be emitted
    fn needs_stream_start(&self) -> bool {
        self.state_tracker.needs_stream_start()
    }

    /// Create stream start metadata
    #[allow(dead_code)]
    fn create_stream_start_metadata(
        &self,
        event: &OpenAiCompatibleStreamEvent,
    ) -> ResponseMetadata {
        ResponseMetadata {
            id: event.id.clone(),
            model: event.model.clone(),
            created: Self::created_datetime_from_unix_seconds(event.created),
            provider: self.config.provider_id.clone(),
            request_id: None,
            headers: None,
        }
    }

    /// Build StreamStart metadata directly from JSON
    fn create_stream_start_metadata_from_json(&self, json: &serde_json::Value) -> ResponseMetadata {
        let id = self.extract_non_empty_string(json, "id");
        let model = self.extract_model_from_json(json);
        let created =
            Self::created_datetime_from_unix_seconds(json.get("created").and_then(|v| v.as_u64()));
        ResponseMetadata {
            id,
            model,
            created,
            provider: self.config.provider_id.clone(),
            request_id: None,
            headers: None,
        }
    }

    fn create_stream_start_metadata_for_unparsed_event(&self) -> ResponseMetadata {
        ResponseMetadata {
            id: None,
            model: Some(self.config.model.trim())
                .filter(|model| !model.is_empty())
                .map(|model| model.to_string()),
            created: None,
            provider: self.config.provider_id.clone(),
            request_id: None,
            headers: None,
        }
    }

    fn update_response_state_from_json(&self, json: &serde_json::Value) {
        let mut state = self.latest_response_state.lock().unwrap();

        let id = self.extract_non_empty_string(json, "id");
        let created =
            Self::created_datetime_from_unix_seconds(json.get("created").and_then(|v| v.as_u64()));
        let system_fingerprint = self.extract_non_empty_string(json, "system_fingerprint");
        let service_tier = self.extract_non_empty_string(json, "service_tier");
        let payload_model = self.extract_non_empty_string(json, "model");

        if let Some(id) = id {
            state.id = Some(id);
        }

        let has_real_metadata = state.id.is_some()
            || created.is_some()
            || system_fingerprint.is_some()
            || service_tier.is_some();

        if let Some(model) = payload_model.or_else(|| {
            has_real_metadata
                .then(|| self.config.model.trim())
                .filter(|model| !model.is_empty())
                .map(|model| model.to_string())
        }) {
            state.model = Some(model);
        }

        if let Some(created) = created {
            state.created = Some(created);
        }

        if let Some(system_fingerprint) = system_fingerprint {
            state.system_fingerprint = Some(system_fingerprint);
        }

        if let Some(service_tier) = service_tier {
            state.service_tier = Some(service_tier);
        }
    }

    fn extract_non_empty_string(&self, json: &serde_json::Value, field: &str) -> Option<String> {
        json.get(field)
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
    }

    fn take_response_metadata_part(&self) -> Option<ChatStreamPart> {
        let response_state = self
            .latest_response_state
            .lock()
            .ok()
            .map(|state| state.clone())?;
        if response_state.id.is_none()
            && response_state.model.is_none()
            && response_state.created.is_none()
        {
            return None;
        }

        let mut parse_state = self
            .parse_state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if parse_state.response_metadata_emitted {
            return None;
        }
        parse_state.response_metadata_emitted = true;

        Some(ChatStreamPart::ResponseMetadata(ResponseMetadata {
            id: response_state.id,
            model: response_state.model,
            created: response_state.created,
            provider: self.config.provider_id.clone(),
            request_id: None,
            headers: None,
        }))
    }

    fn next_part_id(prefix: &str, response_id: Option<&str>, next_index: &mut u32) -> String {
        let index = *next_index;
        *next_index = (*next_index).saturating_add(1);

        match response_id.filter(|id| !id.trim().is_empty()) {
            Some(response_id) => format!("{prefix}_{response_id}_{index}"),
            None => format!("{prefix}_{index}"),
        }
    }

    fn open_text_lane(&self) -> Vec<ChatStreamPart> {
        let response_id = self
            .latest_response_state
            .lock()
            .ok()
            .and_then(|state| state.id.clone());
        let mut parse_state = self
            .parse_state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let mut parts = Vec::new();

        if let Some(id) = parse_state.active_reasoning_part_id.take() {
            parts.push(ChatStreamPart::ReasoningEnd {
                id,
                provider_metadata: None,
            });
        }

        if parse_state.active_text_part_id.is_none() {
            let id = Self::next_part_id(
                "text",
                response_id.as_deref(),
                &mut parse_state.next_text_part_index,
            );
            parse_state.active_text_part_id = Some(id.clone());
            parts.push(ChatStreamPart::TextStart {
                id,
                provider_metadata: None,
            });
        }

        parts
    }

    fn open_reasoning_lane(&self) -> Vec<ChatStreamPart> {
        let response_id = self
            .latest_response_state
            .lock()
            .ok()
            .and_then(|state| state.id.clone());
        let mut parse_state = self
            .parse_state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let mut parts = Vec::new();

        if let Some(id) = parse_state.active_text_part_id.take() {
            parts.push(ChatStreamPart::TextEnd {
                id,
                provider_metadata: None,
            });
        }

        if parse_state.active_reasoning_part_id.is_none() {
            let id = Self::next_part_id(
                "reasoning",
                response_id.as_deref(),
                &mut parse_state.next_reasoning_part_index,
            );
            parse_state.active_reasoning_part_id = Some(id.clone());
            parts.push(ChatStreamPart::ReasoningStart {
                id,
                provider_metadata: None,
            });
        }

        parts
    }

    fn close_active_content_parts(&self) -> Vec<ChatStreamPart> {
        let mut parse_state = self
            .parse_state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let mut parts = Vec::new();

        if let Some(id) = parse_state.active_text_part_id.take() {
            parts.push(ChatStreamPart::TextEnd {
                id,
                provider_metadata: None,
            });
        }

        if let Some(id) = parse_state.active_reasoning_part_id.take() {
            parts.push(ChatStreamPart::ReasoningEnd {
                id,
                provider_metadata: None,
            });
        }

        parts
    }

    fn build_finish_part(&self, reason: Option<&str>) -> ChatStreamPart {
        ChatStreamPart::Finish {
            usage: self
                .latest_usage
                .lock()
                .ok()
                .and_then(|usage| usage.clone())
                .unwrap_or_default(),
            finish_reason: ChatStreamFinishInfo {
                unified: match reason {
                    Some(reason) => {
                        crate::standards::openai::utils::parse_finish_reason(Some(reason))
                            .unwrap_or_else(|| FinishReason::Other(reason.to_string()))
                    }
                    None => FinishReason::Unknown,
                },
                raw: reason.map(ToString::to_string),
            },
            provider_metadata: Some(Self::nested_provider_metadata_to_stream(
                self.current_provider_metadata(),
            )),
        }
    }

    fn nested_provider_metadata_to_stream(
        metadata: NestedProviderMetadata,
    ) -> crate::types::StreamProviderMetadata {
        nested_provider_metadata_to_map(metadata)
    }

    fn extract_model_from_json(&self, json: &serde_json::Value) -> Option<String> {
        let model = json
            .get("model")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());

        if model.is_some() {
            return model;
        }

        Some(self.config.model.trim())
            .filter(|model| !model.is_empty())
            .map(|model| model.to_string())
    }

    /// Extract content from stream event using dynamic field accessor
    #[allow(dead_code)]
    fn extract_content(&self, event: &OpenAiCompatibleStreamEvent) -> Option<String> {
        let model = &self.config.model;
        let field_mappings = self.adapter.get_field_mappings(model);
        let field_accessor = self.adapter.get_field_accessor();

        // Convert event to JSON for dynamic field access
        if let Ok(json) = serde_json::to_value(event) {
            field_accessor.extract_content(&json, &field_mappings)
        } else {
            None
        }
    }

    /// Extract content from raw JSON, compatible with multiple field shapes
    fn extract_content_from_json(&self, json: &serde_json::Value) -> Option<String> {
        let model = &self.config.model;
        let field_mappings = self.adapter.get_field_mappings(model);
        let field_accessor = self.adapter.get_field_accessor();

        // First try standard mappings (e.g., choices.0.delta.content)
        if let Some(text) = field_accessor.extract_content(json, &field_mappings)
            && !text.trim().is_empty()
        {
            return Some(text);
        }

        // Responses API compatibility: delta.text and final aggregated message.content[0].text
        let compat_paths = [
            "delta.text",
            "choices.0.delta.text",
            "message.content.0.text",
            "choices.0.message.content.0.text",
        ];
        for p in compat_paths {
            if let Some(val) = json.pointer(&to_pointer(p))
                && let Some(s) = val.as_str()
                && !s.trim().is_empty()
            {
                return Some(s.to_string());
            }
        }

        // Additional Responses-style compatibility:
        // Some proxies or gateways forward OpenAI Responses SSE schema even when
        // hitting chat/completions. In that case, content can appear as a plain
        // string under `delta` with an accompanying type like
        // `response.output_text.delta`.
        if let Some(s) = json.get("delta").and_then(|d| d.as_str())
            && !s.trim().is_empty()
        {
            return Some(s.to_string());
        }

        // Also handle nested form {"delta": {"text": "..."}} occasionally seen in
        // some implementations of Responses-like streams.
        if let Some(s) = json
            .get("delta")
            .and_then(|d| d.get("text"))
            .and_then(|v| v.as_str())
            && !s.trim().is_empty()
        {
            return Some(s.to_string());
        }
        // Generic fallback: recursively search for first non-empty string under
        // commonly used keys for streaming text (content/text/output_text/outputText)
        fn find_first_text_like(v: &serde_json::Value) -> Option<&str> {
            const KEYS: [&str; 4] = ["content", "text", "output_text", "outputText"];
            match v {
                // Do not treat arbitrary strings (e.g. role/object/finish_reason) as content.
                // Only strings under known "text-like" keys are considered content.
                serde_json::Value::String(_s) => None,
                serde_json::Value::Object(map) => {
                    for k in KEYS {
                        if let Some(serde_json::Value::String(s)) = map.get(k)
                            && !s.trim().is_empty()
                        {
                            return Some(s);
                        }
                    }
                    for (_k, val) in map.iter() {
                        if let Some(s) = find_first_text_like(val) {
                            return Some(s);
                        }
                    }
                    None
                }
                serde_json::Value::Array(arr) => {
                    for item in arr {
                        if let Some(s) = find_first_text_like(item) {
                            return Some(s);
                        }
                    }
                    None
                }
                _ => None,
            }
        }
        if let Some(s) = find_first_text_like(json) {
            return Some(s.to_string());
        }
        None
    }

    /// Extract thinking/reasoning content using dynamic field accessor
    ///
    /// This uses the adapter's configurable field accessor to dynamically extract
    /// thinking content from any field structure, completely eliminating hardcoded field names.
    #[allow(dead_code)]
    fn extract_thinking(&self, event: &OpenAiCompatibleStreamEvent) -> Option<String> {
        let model = &self.config.model;
        let field_mappings = self.adapter.get_field_mappings(model);
        let field_accessor = self.adapter.get_field_accessor();

        // Convert event to JSON for dynamic field access
        if let Ok(json) = serde_json::to_value(event) {
            field_accessor.extract_thinking_content(&json, &field_mappings)
        } else {
            None
        }
    }

    /// Extract thinking/reasoning content from raw JSON
    fn extract_thinking_from_json(&self, json: &serde_json::Value) -> Option<String> {
        let model = &self.config.model;
        let field_mappings = self.adapter.get_field_mappings(model);
        let field_accessor = self.adapter.get_field_accessor();
        if let Some(t) = field_accessor.extract_thinking_content(json, &field_mappings)
            && !t.trim().is_empty()
        {
            return Some(t);
        }
        let compat_paths = ["delta.reasoning.text", "choices.0.delta.reasoning.text"];
        for p in compat_paths {
            if let Some(val) = json.pointer(&to_pointer(p))
                && let Some(s) = val.as_str()
                && !s.trim().is_empty()
            {
                return Some(s.to_string());
            }
        }
        None
    }

    /// Extract tool call information
    #[allow(dead_code)]
    fn extract_tool_call(
        &self,
        event: &OpenAiCompatibleStreamEvent,
    ) -> Option<(String, String, String)> {
        let delta = event.choices.as_ref()?.first()?.delta.as_ref()?;
        let tool_calls = delta.tool_calls.as_ref()?;
        let tool_call = tool_calls.first()?;

        let id = tool_call.get("id")?.as_str()?.to_string();
        let function = tool_call.get("function")?;
        let name = function.get("name")?.as_str()?.to_string();
        let arguments = function.get("arguments")?.as_str()?.to_string();

        Some((id, name, arguments))
    }

    /// Extract OpenAI-compatible tool-call chunks as stable parts plus legacy shadow deltas.
    fn extract_tool_call_events_from_json(&self, json: &serde_json::Value) -> Vec<ChatStreamEvent> {
        let mut out = Vec::new();
        if let Some(arr) = json
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c0| c0.get("delta"))
            .and_then(|d| d.get("tool_calls"))
            .and_then(|tc| tc.as_array())
        {
            let choice_index = json
                .get("choices")
                .and_then(|c| c.get(0))
                .and_then(|choice| choice.get("index"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            let idx = Some(choice_index as usize);

            let mut state = self
                .parse_state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());

            for (pos, tc) in arr.iter().enumerate() {
                let tool_call_index = tc
                    .get("index")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(pos as u64) as u32;
                let key = (choice_index, tool_call_index);
                let entry = state.tool_call_state_by_index.entry(key).or_default();

                let id_in_chunk = tc
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());
                if let Some(id) = id_in_chunk {
                    entry.id = id;
                } else if entry.id.is_empty() {
                    // Best-effort stability: some providers omit tool call ids in
                    // follow-up chunks; key by choice index + tool_call_index.
                    entry.id = format!("call_{choice_index}_{tool_call_index}");
                }

                let function = tc.get("function");
                let name_in_chunk = function
                    .and_then(|f| f.get("name"))
                    .and_then(|v| v.as_str())
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());
                let function_name_to_emit = match name_in_chunk {
                    Some(name) => {
                        if entry.name.is_empty() {
                            entry.name = name.clone();
                            Some(name)
                        } else if entry.name == name {
                            None
                        } else {
                            entry.name = name.clone();
                            Some(name)
                        }
                    }
                    None => None,
                };

                let args_in_chunk = function
                    .and_then(|f| f.get("arguments"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                if let Some(args) = args_in_chunk.as_ref() {
                    entry.arguments.push_str(args);
                }

                let thought_signature = tc
                    .get("extra_content")
                    .and_then(|value| value.get("google"))
                    .and_then(|value| value.get("thought_signature"))
                    .and_then(|value| value.as_str())
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(ToString::to_string);
                if thought_signature.is_some() {
                    entry.thought_signature = thought_signature;
                }

                let mut emitted_stable_start = false;
                if !entry.stable_input_started && !entry.name.is_empty() {
                    emitted_stable_start = true;
                    entry.stable_input_started = true;
                    out.push(ChatStreamEvent::Part {
                        part: ChatStreamPart::ToolInputStart {
                            id: entry.id.clone(),
                            tool_name: entry.name.clone(),
                            provider_metadata: None,
                            provider_executed: None,
                            dynamic: None,
                            title: None,
                        },
                    });
                }

                if entry.stable_input_started {
                    let stable_delta = if emitted_stable_start {
                        (!entry.arguments.is_empty()).then(|| entry.arguments.clone())
                    } else {
                        args_in_chunk.clone()
                    };

                    if let Some(delta) = stable_delta {
                        out.push(ChatStreamEvent::Part {
                            part: ChatStreamPart::ToolInputDelta {
                                id: entry.id.clone(),
                                delta,
                                provider_metadata: None,
                            },
                        });
                    }
                }

                if entry.stable_input_started
                    && !entry.stable_tool_call_emitted
                    && !entry.name.is_empty()
                    && is_parsable_json(&entry.arguments)
                {
                    out.push(ChatStreamEvent::Part {
                        part: ChatStreamPart::ToolInputEnd {
                            id: entry.id.clone(),
                            provider_metadata: None,
                        },
                    });
                    out.push(ChatStreamEvent::Part {
                        part: ChatStreamPart::ToolCall(crate::types::ChatStreamToolCall {
                            tool_call_id: entry.id.clone(),
                            tool_name: entry.name.clone(),
                            input: entry.arguments.clone(),
                            provider_executed: None,
                            dynamic: None,
                            provider_metadata: entry.thought_signature.as_ref().map(|signature| {
                                std::collections::HashMap::from([(
                                    self.provider_metadata_key.clone(),
                                    serde_json::json!({ "thoughtSignature": signature }),
                                )])
                            }),
                        }),
                    });
                    entry.stable_tool_call_emitted = true;
                }

                if function_name_to_emit.is_some() || args_in_chunk.is_some() {
                    out.push(ChatStreamEvent::ToolCallDelta {
                        id: entry.id.clone(),
                        function_name: function_name_to_emit,
                        arguments_delta: args_in_chunk,
                        index: idx,
                    });
                }
            }
        }
        out
    }

    fn extract_source_events_from_json(&self, json: &serde_json::Value) -> Vec<ChatStreamEvent> {
        let Some(annotations) = json
            .get("choices")
            .and_then(|choices| choices.get(0))
            .and_then(|choice| choice.get("delta"))
            .and_then(|delta| delta.get("annotations"))
            .and_then(|annotations| annotations.as_array())
        else {
            return Vec::new();
        };

        let response_id = self
            .latest_response_state
            .lock()
            .ok()
            .and_then(|state| state.id.clone());
        let mut parse_state = self
            .parse_state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let mut out = Vec::new();

        for annotation in annotations {
            let annotation_type = annotation
                .get("type")
                .and_then(|value| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty());
            if annotation_type.is_some_and(|value| !value.eq_ignore_ascii_case("url_citation")) {
                continue;
            }

            let Some(url_citation) = annotation.get("url_citation") else {
                continue;
            };
            let Some(url) = url_citation
                .get("url")
                .and_then(|value| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())
            else {
                continue;
            };
            let title = url_citation
                .get("title")
                .and_then(|value| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(|value| value.to_string());
            let id = Self::next_part_id(
                "source",
                response_id.as_deref(),
                &mut parse_state.next_source_part_index,
            );

            out.push(ChatStreamEvent::Part {
                part: ChatStreamPart::Source {
                    id,
                    source: crate::types::SourcePart::Url {
                        url: url.to_string(),
                        title,
                    },
                    provider_metadata: None,
                },
            });
        }

        out
    }

    /// Extract choice index
    #[allow(dead_code)]
    fn extract_choice_index(&self, event: &OpenAiCompatibleStreamEvent) -> u32 {
        event
            .choices
            .as_ref()
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.index)
            .unwrap_or(0)
    }

    /// Extract choice index from raw JSON
    fn extract_choice_index_from_json(&self, json: &serde_json::Value) -> Option<usize> {
        json.get("choices")
            .and_then(|c| c.get(0))
            .and_then(|choice| choice.get("index"))
            .and_then(|v| v.as_u64())
            .map(|i| i as usize)
    }

    /// Extract usage information
    #[allow(dead_code)]
    fn extract_usage(&self, event: &OpenAiCompatibleStreamEvent) -> Option<Usage> {
        event.usage.as_ref().and_then(|usage| {
            serde_json::to_value(usage).ok().and_then(|value| {
                crate::standards::openai::utils::parse_provider_openai_usage_value(
                    self.adapter.provider_id().as_ref(),
                    &value,
                )
            })
        })
    }

    fn extract_usage_from_json(&self, json: &serde_json::Value) -> Option<Usage> {
        let usage = json.get("usage")?;
        if usage.is_null() {
            return None;
        }

        crate::standards::openai::utils::parse_provider_openai_usage_value(
            self.adapter.provider_id().as_ref(),
            usage,
        )
    }
}

impl SseEventConverter for OpenAiCompatibleEventConverter {
    fn convert_event(
        &self,
        event: Event,
    ) -> Pin<Box<dyn Future<Output = Vec<Result<ChatStreamEvent, LlmError>>> + Send + Sync + '_>>
    {
        Box::pin(async move {
            match serde_json::from_str::<serde_json::Value>(&event.data) {
                Ok(value) => self
                    .convert_event_json_async(&value)
                    .await
                    .into_iter()
                    .map(Ok)
                    .collect(),
                Err(e) => {
                    let mut out = Vec::new();
                    if self.needs_stream_start() {
                        let metadata = self.create_stream_start_metadata_for_unparsed_event();
                        out.push(Ok(ChatStreamEvent::StreamStart { metadata }));
                        out.push(Ok(ChatStreamEvent::Part {
                            part: ChatStreamPart::StreamStart { warnings: vec![] },
                        }));
                    }
                    if self.include_raw_chunks {
                        out.push(Ok(ChatStreamEvent::Part {
                            part: ChatStreamPart::Raw {
                                raw_value: serde_json::Value::String(event.data.clone()),
                            },
                        }));
                    }
                    out.push(Err(LlmError::ParseError(format!(
                        "Failed to parse OpenAI-compatible event: {e}"
                    ))));
                    out
                }
            }
        })
    }

    fn handle_stream_end_events(&self) -> Vec<Result<ChatStreamEvent, LlmError>> {
        if !self.state_tracker.needs_stream_end() {
            return Vec::new();
        }
        self.state_tracker.mark_stream_ended();

        let mut out = Vec::new();

        for part in self.close_active_content_parts() {
            out.push(Ok(ChatStreamEvent::Part { part }));
        }

        {
            let mut parse_state = self
                .parse_state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());

            for entry in parse_state.tool_call_state_by_index.values_mut() {
                if entry.stable_tool_call_emitted || entry.name.is_empty() {
                    continue;
                }

                if !entry.stable_input_started {
                    entry.stable_input_started = true;
                    out.push(Ok(ChatStreamEvent::Part {
                        part: ChatStreamPart::ToolInputStart {
                            id: entry.id.clone(),
                            tool_name: entry.name.clone(),
                            provider_metadata: None,
                            provider_executed: None,
                            dynamic: None,
                            title: None,
                        },
                    }));

                    if !entry.arguments.is_empty() {
                        out.push(Ok(ChatStreamEvent::Part {
                            part: ChatStreamPart::ToolInputDelta {
                                id: entry.id.clone(),
                                delta: entry.arguments.clone(),
                                provider_metadata: None,
                            },
                        }));
                    }
                }

                out.push(Ok(ChatStreamEvent::Part {
                    part: ChatStreamPart::ToolInputEnd {
                        id: entry.id.clone(),
                        provider_metadata: None,
                    },
                }));
                out.push(Ok(ChatStreamEvent::Part {
                    part: ChatStreamPart::ToolCall(crate::types::ChatStreamToolCall {
                        tool_call_id: entry.id.clone(),
                        tool_name: entry.name.clone(),
                        input: entry.arguments.clone(),
                        provider_executed: None,
                        dynamic: None,
                        provider_metadata: entry.thought_signature.as_ref().map(|signature| {
                            std::collections::HashMap::from([(
                                self.provider_metadata_key.clone(),
                                serde_json::json!({ "thoughtSignature": signature }),
                            )])
                        }),
                    }),
                }));
                entry.stable_tool_call_emitted = true;
            }
        }

        out.push(Ok(ChatStreamEvent::Part {
            part: self.build_finish_part(None),
        }));

        let response_state = self
            .latest_response_state
            .lock()
            .ok()
            .map(|state| state.clone())
            .unwrap_or_default();
        out.push(Ok(ChatStreamEvent::StreamEnd {
            response: ChatResponse {
                id: response_state.id,
                model: response_state.model,
                content: MessageContent::Text(
                    self.accumulated_content
                        .try_lock()
                        .map(|g| g.clone())
                        .unwrap_or_default(),
                ),
                usage: self
                    .latest_usage
                    .lock()
                    .ok()
                    .and_then(|usage| usage.clone()),
                finish_reason: Some(FinishReason::Unknown),
                raw_finish_reason: None,
                audio: None,
                system_fingerprint: response_state.system_fingerprint,
                service_tier: response_state.service_tier,
                warnings: None,
                provider_metadata: Some(Self::nested_provider_metadata_to_stream(
                    self.current_provider_metadata(),
                )),
            },
        }));

        out
    }

    fn handle_stream_end(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
        // OpenAI-compatible providers normally emit finish_reason in the stream.
        // If we reach here without seeing finish_reason, the model has not transmitted
        // a finish reason (e.g., connection lost, server error, client cancelled).
        // Always emit StreamEnd with Unknown reason so users can detect this.
        //
        // Carry accumulated text into StreamEnd so the factory can inject a synthetic
        // ContentDelta when no deltas were observed during the stream.

        // Check if StreamEnd was already emitted
        if !self.state_tracker.needs_stream_end() {
            return None; // StreamEnd already emitted
        }

        let response_state = self
            .latest_response_state
            .lock()
            .ok()
            .map(|state| state.clone())
            .unwrap_or_default();
        let response = ChatResponse {
            id: response_state.id,
            model: response_state.model,
            content: MessageContent::Text(
                self.accumulated_content
                    .try_lock()
                    .map(|g| g.clone())
                    .unwrap_or_default(),
            ),
            usage: self
                .latest_usage
                .lock()
                .ok()
                .and_then(|usage| usage.clone()),
            finish_reason: Some(FinishReason::Unknown),
            raw_finish_reason: None,
            audio: None,
            system_fingerprint: response_state.system_fingerprint,
            service_tier: response_state.service_tier,
            warnings: None,
            provider_metadata: Some(Self::nested_provider_metadata_to_stream(
                self.current_provider_metadata(),
            )),
        };

        Some(Ok(ChatStreamEvent::StreamEnd { response }))
    }

    fn serialize_event(&self, event: &ChatStreamEvent) -> Result<Vec<u8>, LlmError> {
        fn sse_data_frame(value: &serde_json::Value) -> Result<Vec<u8>, LlmError> {
            let data = serde_json::to_vec(value).map_err(|e| {
                LlmError::JsonError(format!(
                    "Failed to serialize OpenAI-compatible SSE JSON: {e}"
                ))
            })?;
            let mut out = Vec::with_capacity(data.len() + 8);
            out.extend_from_slice(b"data: ");
            out.extend_from_slice(&data);
            out.extend_from_slice(b"\n\n");
            Ok(out)
        }

        fn done_frame() -> Vec<u8> {
            b"data: [DONE]\n\n".to_vec()
        }

        fn chunk_payload(
            state: &OpenAiCompatSerializeState,
            choices: serde_json::Value,
            usage: Option<serde_json::Value>,
        ) -> serde_json::Value {
            let mut payload = serde_json::Map::new();
            if let Some(id) = state.id.clone() {
                payload.insert("id".to_string(), serde_json::Value::String(id));
            }
            payload.insert(
                "object".to_string(),
                serde_json::Value::String("chat.completion.chunk".to_string()),
            );
            if let Some(created) = state.created {
                payload.insert("created".to_string(), serde_json::json!(created));
            }
            if let Some(model) = state.model.clone() {
                payload.insert("model".to_string(), serde_json::Value::String(model));
            }
            if let Some(system_fingerprint) = state.system_fingerprint.clone() {
                payload.insert(
                    "system_fingerprint".to_string(),
                    serde_json::Value::String(system_fingerprint),
                );
            }
            if let Some(service_tier) = state.service_tier.clone() {
                payload.insert(
                    "service_tier".to_string(),
                    serde_json::Value::String(service_tier),
                );
            }
            payload.insert("choices".to_string(), choices);
            if let Some(usage) = usage {
                payload.insert("usage".to_string(), usage);
            }
            serde_json::Value::Object(payload)
        }

        fn serialize_source_annotation_frame(
            state: &OpenAiCompatSerializeState,
            choice_index: u32,
            url: &str,
            title: Option<&str>,
        ) -> Result<Vec<u8>, LlmError> {
            let mut url_citation = serde_json::Map::new();
            url_citation.insert(
                "url".to_string(),
                serde_json::Value::String(url.to_string()),
            );
            if let Some(title) = title.map(str::trim).filter(|value| !value.is_empty()) {
                url_citation.insert(
                    "title".to_string(),
                    serde_json::Value::String(title.to_string()),
                );
            }

            let payload = chunk_payload(
                state,
                serde_json::json!([
                    {
                        "index": choice_index,
                        "delta": {
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url_citation": serde_json::Value::Object(url_citation),
                                }
                            ]
                        },
                        "finish_reason": serde_json::Value::Null
                    }
                ]),
                None,
            );
            sse_data_frame(&payload)
        }

        fn serialize_terminal_frame(
            state: &mut OpenAiCompatSerializeState,
            finish_reason: serde_json::Value,
            usage: Option<serde_json::Value>,
            logprobs: Option<serde_json::Value>,
        ) -> Result<Vec<u8>, LlmError> {
            let mut choice = serde_json::Map::new();
            choice.insert("index".to_string(), serde_json::json!(0));
            choice.insert("delta".to_string(), serde_json::json!({}));
            choice.insert("finish_reason".to_string(), finish_reason);
            if let Some(logprobs) = logprobs {
                choice.insert(
                    "logprobs".to_string(),
                    serde_json::json!({ "content": logprobs }),
                );
            }

            let payload = chunk_payload(
                state,
                serde_json::Value::Array(vec![serde_json::Value::Object(choice)]),
                usage,
            );

            let mut out = sse_data_frame(&payload)?;
            out.extend_from_slice(&done_frame());
            state.finished = true;
            Ok(out)
        }

        fn extract_logprobs_from_provider_value(
            provider_value: &serde_json::Value,
        ) -> Option<serde_json::Value> {
            provider_value
                .as_object()
                .and_then(|obj| obj.get("logprobs"))
                .filter(|value| !value.is_null())
                .cloned()
        }

        fn extract_logprobs_from_stream_provider_metadata(
            provider_id: &str,
            provider_metadata: &ProviderMetadataMap,
        ) -> Option<serde_json::Value> {
            let preferred_roots = [
                provider_id,
                "openai",
                "openrouter",
                "xai",
                "groq",
                "deepseek",
            ];

            for root in preferred_roots {
                if let Some(value) = provider_metadata.get(root)
                    && let Some(logprobs) = extract_logprobs_from_provider_value(value)
                {
                    return Some(logprobs);
                }
            }

            provider_metadata
                .values()
                .find_map(extract_logprobs_from_provider_value)
        }

        fn finish_reason_str(reason: &FinishReason) -> Option<&'static str> {
            match reason {
                FinishReason::Stop | FinishReason::StopSequence => Some("stop"),
                FinishReason::Length => Some("length"),
                FinishReason::ToolCalls => Some("tool_calls"),
                FinishReason::ContentFilter => Some("content_filter"),
                FinishReason::Error => Some("error"),
                FinishReason::Unknown => None,
                FinishReason::Other(_) => None,
            }
        }

        fn finish_reason_str_from_v3(
            reason: &crate::streaming::LanguageModelV3FinishReason,
        ) -> Option<&'static str> {
            fn map_candidate(candidate: &str) -> Option<&'static str> {
                let normalized = candidate.trim().to_ascii_lowercase();
                if normalized.is_empty() {
                    return None;
                }

                match normalized.as_str() {
                    "stop" | "end_turn" | "end-turn" | "stop_sequence" | "stop-sequence"
                    | "stop_symbol" | "stop-symbol" => Some("stop"),
                    "length"
                    | "max_tokens"
                    | "max-tokens"
                    | "max_completion_tokens"
                    | "max-completion-tokens" => Some("length"),
                    "tool_calls" | "tool-calls" | "tool_use" | "tool-use" | "function_call"
                    | "function-call" => Some("tool_calls"),
                    "content_filter" | "content-filter" | "refusal" => Some("content_filter"),
                    "error" => Some("error"),
                    _ => {
                        if normalized.contains("tool") {
                            Some("tool_calls")
                        } else if normalized.contains("length") || normalized.contains("max") {
                            Some("length")
                        } else if normalized.contains("content")
                            || normalized.contains("filter")
                            || normalized.contains("refusal")
                        {
                            Some("content_filter")
                        } else if normalized.contains("stop") || normalized.contains("end") {
                            Some("stop")
                        } else if normalized.contains("error") {
                            Some("error")
                        } else {
                            None
                        }
                    }
                }
            }

            map_candidate(&reason.unified).or_else(|| reason.raw.as_deref().and_then(map_candidate))
        }

        fn openai_chat_usage_payload(usage: &Usage) -> serde_json::Value {
            crate::standards::openai::utils::openai_chat_usage_value(usage)
        }

        fn openai_chat_usage_payload_from_v3(
            usage: &crate::streaming::LanguageModelV3Usage,
        ) -> serde_json::Value {
            let clamp = |value: Option<u64>| value.map(|value| value.min(u32::MAX as u64) as u32);

            let mut builder = Usage::builder().with_input_tokens(crate::types::UsageInputTokens {
                total: clamp(usage.input_tokens.total),
                no_cache: clamp(usage.input_tokens.no_cache),
                cache_read: clamp(usage.input_tokens.cache_read),
                cache_write: clamp(usage.input_tokens.cache_write),
            });
            builder = builder.with_output_tokens(crate::types::UsageOutputTokens {
                total: clamp(usage.output_tokens.total),
                text: clamp(usage.output_tokens.text),
                reasoning: clamp(usage.output_tokens.reasoning),
            });

            if let Some(cached_tokens) = clamp(usage.input_tokens.cache_read) {
                builder = builder.with_cached_tokens(cached_tokens);
            }
            if let Some(reasoning_tokens) = clamp(usage.output_tokens.reasoning) {
                builder = builder.with_reasoning_tokens(reasoning_tokens);
            }
            if let Some(raw) = usage.raw.clone() {
                builder = builder.with_raw_usage(raw);
            }

            openai_chat_usage_payload(&builder.build())
        }

        fn serialize_tool_delta_frame(
            state: &OpenAiCompatSerializeState,
            choice_index: u32,
            tool_call_index: u32,
            id: &str,
            function_name: Option<String>,
            arguments_delta: Option<String>,
        ) -> Result<Vec<u8>, LlmError> {
            let mut function = serde_json::Map::new();
            if let Some(name) = function_name {
                function.insert("name".to_string(), serde_json::Value::String(name));
            }
            if let Some(args) = arguments_delta {
                function.insert("arguments".to_string(), serde_json::Value::String(args));
            }

            let payload = chunk_payload(
                state,
                serde_json::json!([
                    {
                        "index": choice_index,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": tool_call_index,
                                    "id": id,
                                    "type": "function",
                                    "function": serde_json::Value::Object(function),
                                }
                            ]
                        },
                        "finish_reason": serde_json::Value::Null
                    }
                ]),
                None,
            );
            sse_data_frame(&payload)
        }

        let mut state = self
            .serialize_state
            .lock()
            .map_err(|_| LlmError::InternalError("serialize_state lock poisoned".to_string()))?;

        let mut serialize_inner = |event: &ChatStreamEvent| -> Result<Vec<u8>, LlmError> {
            match event {
                ChatStreamEvent::StreamStart { metadata } => {
                    *state = OpenAiCompatSerializeState::default();
                    state.id = metadata.id.clone();
                    state.model = metadata.model.clone();
                    state.created = metadata.created.map(|dt| dt.timestamp().max(0) as u64);

                    let payload = chunk_payload(
                        &state,
                        serde_json::json!([
                            {
                                "index": 0,
                                "delta": { "role": "assistant" },
                                "finish_reason": serde_json::Value::Null
                            }
                        ]),
                        None,
                    );
                    state.emitted_role = true;
                    sse_data_frame(&payload)
                }
                ChatStreamEvent::ContentDelta { delta, index } => {
                    let choice_index = index.unwrap_or(0) as u32;

                    // Some clients expect a role delta before the first content delta.
                    let mut out = Vec::new();
                    if !state.emitted_role {
                        let role_payload = chunk_payload(
                            &state,
                            serde_json::json!([
                                {
                                    "index": choice_index,
                                    "delta": { "role": "assistant" },
                                    "finish_reason": serde_json::Value::Null
                                }
                            ]),
                            None,
                        );
                        out.extend_from_slice(&sse_data_frame(&role_payload)?);
                        state.emitted_role = true;
                    }

                    let payload = chunk_payload(
                        &state,
                        serde_json::json!([
                            {
                                "index": choice_index,
                                "delta": { "content": delta },
                                "finish_reason": serde_json::Value::Null
                            }
                        ]),
                        None,
                    );
                    out.extend_from_slice(&sse_data_frame(&payload)?);
                    Ok(out)
                }
                ChatStreamEvent::ToolCallDelta {
                    id,
                    function_name,
                    arguments_delta,
                    index,
                } => {
                    let choice_index = index.unwrap_or(0) as u32;
                    let Some(tool_state) = ensure_serialize_tool_call_state(
                        &mut state,
                        id,
                        OpenAiCompatToolStreamSource::LegacyDelta,
                    ) else {
                        return Ok(Vec::new());
                    };
                    if function_name.is_some() {
                        tool_state.emitted_name = true;
                    }
                    if arguments_delta.is_some() {
                        tool_state.emitted_arguments = true;
                    }
                    let tool_call_index = tool_state.index;
                    serialize_tool_delta_frame(
                        &state,
                        choice_index,
                        tool_call_index,
                        id,
                        function_name.clone(),
                        arguments_delta.clone(),
                    )
                }
                ChatStreamEvent::UsageUpdate { usage } => {
                    let payload = chunk_payload(
                        &state,
                        serde_json::json!([]),
                        Some(openai_chat_usage_payload(usage)),
                    );
                    sse_data_frame(&payload)
                }
                ChatStreamEvent::StreamEnd { response } => {
                    if state.finished {
                        return Ok(Vec::new());
                    }
                    if state.id.is_none() {
                        state.id = response.id.clone();
                    }
                    if state.model.is_none() {
                        state.model = response.model.clone();
                    }
                    if state.system_fingerprint.is_none() {
                        state.system_fingerprint = response.system_fingerprint.clone();
                    }
                    if state.service_tier.is_none() {
                        state.service_tier = response.service_tier.clone();
                    }
                    let finish_reason = response
                        .finish_reason
                        .as_ref()
                        .and_then(finish_reason_str)
                        .map(|s| serde_json::Value::String(s.to_string()))
                        .unwrap_or(serde_json::Value::Null);

                    let usage = response.usage.as_ref().map(openai_chat_usage_payload);
                    let logprobs =
                        response
                            .provider_metadata
                            .as_ref()
                            .and_then(|provider_metadata| {
                                extract_logprobs_from_stream_provider_metadata(
                                    &self.config.provider_id,
                                    provider_metadata,
                                )
                            });
                    serialize_terminal_frame(&mut state, finish_reason, usage, logprobs)
                }
                ChatStreamEvent::Error { error } => {
                    let payload = serde_json::json!({
                        "error": { "message": error },
                    });
                    sse_data_frame(&payload)
                }
                ChatStreamEvent::ThinkingDelta { .. }
                | ChatStreamEvent::Custom { .. }
                | ChatStreamEvent::Part { .. }
                | ChatStreamEvent::PartWithReplay { .. } => Ok(Vec::new()),
            }
        };

        match event {
            ChatStreamEvent::Custom { .. }
            | ChatStreamEvent::Part { .. }
            | ChatStreamEvent::PartWithReplay { .. } => {
                let Some(part) = LanguageModelV3StreamPart::try_from_chat_event(event) else {
                    return Ok(Vec::new());
                };

                if let LanguageModelV3StreamPart::Finish {
                    usage,
                    finish_reason,
                    provider_metadata,
                } = &part
                {
                    if state.finished {
                        return Ok(Vec::new());
                    }

                    let finish_reason = finish_reason_str_from_v3(finish_reason)
                        .map(|reason| serde_json::Value::String(reason.to_string()))
                        .unwrap_or(serde_json::Value::Null);
                    let usage = openai_chat_usage_payload_from_v3(usage);
                    let logprobs = provider_metadata.as_ref().and_then(|provider_metadata| {
                        extract_logprobs_from_stream_provider_metadata(
                            &self.config.provider_id,
                            provider_metadata,
                        )
                    });
                    return serialize_terminal_frame(
                        &mut state,
                        finish_reason,
                        Some(usage),
                        logprobs,
                    );
                }

                if let LanguageModelV3StreamPart::ResponseMetadata(metadata) = &part {
                    if let Some(id) = metadata.id.clone() {
                        state.id = Some(id);
                    }
                    if let Some(model_id) = metadata.model_id.clone() {
                        state.model = Some(model_id);
                    }
                    if let Some(timestamp) = metadata.timestamp.as_ref() {
                        state.created = Some(timestamp.timestamp().max(0) as u64);
                    }
                    return Ok(Vec::new());
                }

                if let LanguageModelV3StreamPart::Source(LanguageModelV3Source::Url {
                    url,
                    title,
                    ..
                }) = &part
                {
                    return serialize_source_annotation_frame(&state, 0, url, title.as_deref());
                }

                match part {
                    LanguageModelV3StreamPart::ToolInputStart { id, tool_name, .. } => {
                        let Some(tool_state) = ensure_serialize_tool_call_state(
                            &mut state,
                            &id,
                            OpenAiCompatToolStreamSource::StablePart,
                        ) else {
                            return Ok(Vec::new());
                        };
                        if tool_state.emitted_name {
                            return Ok(Vec::new());
                        }
                        tool_state.emitted_name = true;
                        let tool_call_index = tool_state.index;
                        return serialize_tool_delta_frame(
                            &state,
                            0,
                            tool_call_index,
                            &id,
                            Some(tool_name),
                            None,
                        );
                    }
                    LanguageModelV3StreamPart::ToolInputDelta { id, delta, .. } => {
                        let Some(tool_state) = ensure_serialize_tool_call_state(
                            &mut state,
                            &id,
                            OpenAiCompatToolStreamSource::StablePart,
                        ) else {
                            return Ok(Vec::new());
                        };
                        tool_state.emitted_arguments = true;
                        let tool_call_index = tool_state.index;
                        return serialize_tool_delta_frame(
                            &state,
                            0,
                            tool_call_index,
                            &id,
                            None,
                            Some(delta),
                        );
                    }
                    LanguageModelV3StreamPart::ToolInputEnd { .. } => {
                        return Ok(Vec::new());
                    }
                    LanguageModelV3StreamPart::ToolCall(call) => {
                        let Some(tool_state) = ensure_serialize_tool_call_state(
                            &mut state,
                            &call.tool_call_id,
                            OpenAiCompatToolStreamSource::StablePart,
                        ) else {
                            return Ok(Vec::new());
                        };

                        let function_name = if tool_state.emitted_name {
                            None
                        } else {
                            tool_state.emitted_name = true;
                            Some(call.tool_name.clone())
                        };
                        let arguments_delta = if tool_state.emitted_arguments {
                            None
                        } else {
                            tool_state.emitted_arguments = true;
                            Some(call.input.clone())
                        };

                        if function_name.is_none() && arguments_delta.is_none() {
                            return Ok(Vec::new());
                        }

                        let tool_call_index = tool_state.index;
                        return serialize_tool_delta_frame(
                            &state,
                            0,
                            tool_call_index,
                            &call.tool_call_id,
                            function_name,
                            arguments_delta,
                        );
                    }
                    _ => {}
                }

                let mut out = Vec::new();
                let mut events = part.to_best_effort_chat_events();
                let only_runtime_parts = !events.is_empty()
                    && events.iter().all(|ev| {
                        matches!(
                            ev,
                            ChatStreamEvent::Part { .. } | ChatStreamEvent::PartWithReplay { .. }
                        )
                    });
                if (events.is_empty() || only_runtime_parts)
                    && self.v3_unsupported_part_behavior == V3UnsupportedPartBehavior::AsText
                    && let Some(text) = part.to_lossy_text()
                {
                    events = vec![ChatStreamEvent::ContentDelta {
                        delta: text,
                        index: None,
                    }];
                }
                for ev in events {
                    out.extend_from_slice(&serialize_inner(&ev)?);
                }
                Ok(out)
            }
            other => serialize_inner(other),
        }
    }
}

// Legacy OpenAiCompatibleStreaming client has been removed in favor of the unified HttpChatExecutor.
// The OpenAiCompatibleEventConverter is still used for SSE event conversion in tests.

// Convert a dotted path like "a.b.0.c" to a JSON Pointer "/a/b/0/c"
fn to_pointer(path: &str) -> String {
    let mut s = String::new();
    for part in path.split('.') {
        s.push('/');
        s.push_str(part);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::standards::openai::compat::provider_registry::{
        ConfigurableAdapter, ProviderConfig, ProviderFieldMappings,
    };
    use crate::streaming::SseEventConverter;
    use crate::types::{ChatResponse, FinishReason, MessageContent, ResponseMetadata, Usage};

    fn parse_sse_data_frames(bytes: &[u8]) -> Vec<serde_json::Value> {
        let text = String::from_utf8_lossy(bytes);
        text.split("\n\n")
            .filter_map(|chunk| {
                let line = chunk
                    .lines()
                    .find_map(|l| l.strip_prefix("data: "))
                    .map(str::trim)?;
                if line.is_empty() || line == "[DONE]" {
                    return None;
                }
                serde_json::from_str::<serde_json::Value>(line).ok()
            })
            .collect()
    }

    fn count_done_frames(bytes: &[u8]) -> usize {
        String::from_utf8_lossy(bytes)
            .matches("data: [DONE]")
            .count()
    }

    #[tokio::test]
    async fn openai_compatible_raw_chunks_follow_stream_start_before_response_metadata() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openai".to_string(),
            name: "OpenAI".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["chat".to_string(), "streaming".to_string()],
            default_model: Some("gpt-test".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "openai",
            "sk-test",
            "https://api.openai.com/v1",
            adapter.clone(),
        )
        .with_model("gpt-test");

        let converter =
            OpenAiCompatibleEventConverter::new(cfg, adapter).with_include_raw_chunks(true);
        let events = converter
            .convert_event(eventsource_stream::Event {
                event: "".to_string(),
                data: r#"{"id":"chat-id","model":"gpt-test","choices":[{"delta":{"content":"Hello"}}]}"#
                    .to_string(),
                id: "".to_string(),
                retry: None,
            })
            .await;

        let parts: Vec<_> = events
            .into_iter()
            .map(|event| event.expect("stream event"))
            .filter_map(|event| match event {
                ChatStreamEvent::Part { part } => Some(part),
                _ => None,
            })
            .collect();

        assert!(matches!(
            parts.first(),
            Some(ChatStreamPart::StreamStart { .. })
        ));
        match parts.get(1).expect("raw part") {
            ChatStreamPart::Raw { raw_value } => {
                assert_eq!(raw_value["id"], serde_json::json!("chat-id"));
            }
            other => panic!("expected raw part, got {other:?}"),
        }
        match parts.get(2).expect("response metadata part") {
            ChatStreamPart::ResponseMetadata(metadata) => {
                assert_eq!(metadata.id.as_deref(), Some("chat-id"));
                assert_eq!(metadata.model.as_deref(), Some("gpt-test"));
            }
            other => panic!("expected response metadata, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn openai_compatible_parse_error_with_raw_chunks_still_emits_stream_start_before_raw() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openai".to_string(),
            name: "OpenAI".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["chat".to_string(), "streaming".to_string()],
            default_model: Some("gpt-test".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "openai",
            "sk-test",
            "https://api.openai.com/v1",
            adapter.clone(),
        )
        .with_model("gpt-test");

        let converter =
            OpenAiCompatibleEventConverter::new(cfg, adapter).with_include_raw_chunks(true);
        let events = converter
            .convert_event(eventsource_stream::Event {
                event: "".to_string(),
                data: "not-json".to_string(),
                id: "".to_string(),
                retry: None,
            })
            .await;

        assert_eq!(events.len(), 4);
        match events.first().expect("stream-start event") {
            Ok(ChatStreamEvent::StreamStart { metadata }) => {
                assert_eq!(metadata.id, None);
                assert_eq!(metadata.model.as_deref(), Some("gpt-test"));
                assert_eq!(metadata.provider, "openai");
            }
            other => panic!("expected stream-start event, got {other:?}"),
        }
        match events.get(1).expect("stream-start part") {
            Ok(ChatStreamEvent::Part {
                part: ChatStreamPart::StreamStart { warnings },
            }) => {
                assert!(warnings.is_empty());
            }
            other => panic!("expected stream-start part, got {other:?}"),
        }
        match events.get(2).expect("raw part") {
            Ok(ChatStreamEvent::Part {
                part: ChatStreamPart::Raw { raw_value },
            }) => {
                assert_eq!(
                    raw_value,
                    &serde_json::Value::String("not-json".to_string())
                );
            }
            other => panic!("expected raw part, got {other:?}"),
        }
        match events.get(3).expect("parse error") {
            Err(LlmError::ParseError(message)) => {
                assert!(message.contains("Failed to parse OpenAI-compatible event"));
            }
            other => panic!("expected parse error, got {other:?}"),
        }
    }

    #[test]
    fn openai_compat_serializes_v3_custom_parts_best_effort() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openai".to_string(),
            name: "OpenAI".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string()],
            default_model: Some("gpt-4o-mini".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "openai",
            "sk-siumai-encoding-only",
            "https://api.openai.com/v1",
            adapter.clone(),
        )
        .with_model("gpt-4o-mini");

        let converter = OpenAiCompatibleEventConverter::new(cfg, adapter);

        let bytes = converter
            .serialize_event(&ChatStreamEvent::Custom {
                event_type: "openai:text-delta".to_string(),
                data: serde_json::json!({
                    "type": "text-delta",
                    "id": "0",
                    "delta": "Hello",
                }),
            })
            .expect("serialize custom text-delta");
        let frames = parse_sse_data_frames(&bytes);
        assert!(
            frames
                .iter()
                .any(|v| v["choices"][0]["delta"]["content"] == "Hello"),
            "expected content delta from custom text-delta: {frames:?}"
        );

        let bytes = converter
            .serialize_event(&ChatStreamEvent::Custom {
                event_type: "openai:tool-call".to_string(),
                data: serde_json::json!({
                    "type": "tool-call",
                    "toolCallId": "call_1",
                    "toolName": "get_weather",
                    "input": r#"{"city":"Tokyo"}"#,
                }),
            })
            .expect("serialize custom tool-call");
        let frames = parse_sse_data_frames(&bytes);
        assert!(
            frames
                .iter()
                .any(|v| v["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_1"),
            "expected tool_calls from custom tool-call: {frames:?}"
        );
    }

    #[test]
    fn openai_compat_accepts_tool_call_input_object_via_loose_parser() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openai".to_string(),
            name: "OpenAI".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string()],
            default_model: Some("gpt-4o-mini".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "openai",
            "sk-siumai-encoding-only",
            "https://api.openai.com/v1",
            adapter.clone(),
        )
        .with_model("gpt-4o-mini");

        let converter = OpenAiCompatibleEventConverter::new(cfg, adapter);

        let bytes = converter
            .serialize_event(&ChatStreamEvent::Custom {
                event_type: "gemini:tool".to_string(),
                data: serde_json::json!({
                    "type": "tool-call",
                    "toolCallId": "call_1",
                    "toolName": "get_weather",
                    "input": { "city": "Tokyo" }
                }),
            })
            .expect("serialize ok");

        let frames = parse_sse_data_frames(&bytes);
        assert!(
            frames
                .iter()
                .any(|v| v["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_1"),
            "expected tool_calls from loose tool-call: {frames:?}"
        );
    }

    #[tokio::test]
    async fn perplexity_streaming_emits_stream_end_with_usage_and_provider_metadata() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "perplexity".to_string(),
            name: "Perplexity".to_string(),
            base_url: "https://api.perplexity.ai".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["chat".to_string(), "streaming".to_string()],
            default_model: Some("sonar".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "perplexity",
            "sk-test",
            "https://api.perplexity.ai",
            adapter.clone(),
        )
        .with_model("sonar");

        let converter = OpenAiCompatibleEventConverter::new(cfg, adapter);

        let first = Event {
            event: "".to_string(),
            data: r#"{"id":"1","model":"sonar","created":1718345013,"citations":["https://example.com/rust"],"choices":[{"index":0,"delta":{"content":"Rust","role":"assistant"},"finish_reason":null}]}"#.to_string(),
            id: "".to_string(),
            retry: None,
        };
        let r1 = converter.convert_event(first).await;
        assert!(r1.iter().any(|event| matches!(
            event,
            Ok(ChatStreamEvent::ContentDelta { delta, .. }) if delta == "Rust"
        )));

        let final_chunk = Event {
            event: "".to_string(),
            data: r#"{"id":"1","model":"sonar","created":1718345013,"choices":[{"index":0,"delta":{"content":" ecosystem","role":null},"finish_reason":"stop"}],"images":[{"image_url":"https://images.example.com/rust.png","origin_url":"https://example.com/rust","height":900,"width":1600}],"usage":{"prompt_tokens":11,"completion_tokens":17,"total_tokens":28,"citation_tokens":7,"num_search_queries":2,"reasoning_tokens":3,"cost":{"input_tokens_cost":0.12,"output_tokens_cost":0.34,"request_cost":0.01,"total_cost":0.47}}}"#.to_string(),
            id: "".to_string(),
            retry: None,
        };
        let r2 = converter.convert_event(final_chunk).await;
        assert!(r2.iter().any(|event| matches!(
            event,
            Ok(ChatStreamEvent::UsageUpdate { usage })
                if usage.prompt_tokens() == Some(11)
                    && usage.completion_tokens() == Some(17)
                    && usage.total_tokens() == Some(28)
        )));

        let end = r2
            .iter()
            .find_map(|event| match event {
                Ok(ChatStreamEvent::StreamEnd { response }) => Some(response),
                _ => None,
            })
            .expect("stream end event");
        assert_eq!(
            end.usage.as_ref().and_then(|usage| usage.total_tokens()),
            Some(28)
        );
        let metadata = end.provider_metadata.as_ref().expect("provider metadata");
        let perplexity = metadata.get("perplexity").expect("perplexity metadata");
        assert_eq!(
            perplexity.get("citations"),
            Some(&serde_json::json!(["https://example.com/rust"]))
        );
        assert_eq!(perplexity["usage"]["citationTokens"], serde_json::json!(7));
        assert_eq!(
            perplexity["usage"]["numSearchQueries"],
            serde_json::json!(2)
        );
        assert_eq!(
            perplexity["images"][0]["imageUrl"],
            serde_json::json!("https://images.example.com/rust.png")
        );
        assert_eq!(perplexity["cost"]["requestCost"], serde_json::json!(0.01));
        assert!(converter.handle_stream_end().is_none());
    }

    #[tokio::test]
    async fn deepseek_streaming_emits_usage_then_single_stream_end_and_ignores_done() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.deepseek.com/v1".to_string(),
            field_mappings: ProviderFieldMappings {
                thinking_fields: vec!["reasoning_content".to_string(), "thinking".to_string()],
                content_field: "content".to_string(),
                tool_calls_field: "tool_calls".to_string(),
                role_field: "role".to_string(),
            },
            capabilities: vec![
                "tools".to_string(),
                "vision".to_string(),
                "reasoning".to_string(),
            ],
            default_model: Some("deepseek-chat".to_string()),
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "deepseek",
            "sk-test",
            "https://api.deepseek.com/v1",
            adapter.clone(),
        )
        .with_model("deepseek-chat");

        let converter = OpenAiCompatibleEventConverter::new(cfg, adapter);

        // First chunk: StreamStart + ContentDelta.
        let first = Event {
            event: "".to_string(),
            data: r#"{"id":"1","model":"deepseek-chat","created":1718345013,"choices":[{"index":0,"delta":{"content":"Hello","role":"assistant"},"finish_reason":null}]}"#.to_string(),
            id: "".to_string(),
            retry: None,
        };
        let r1 = converter.convert_event(first).await;
        assert!(
            r1.iter()
                .any(|e| matches!(e, Ok(ChatStreamEvent::StreamStart { .. })))
        );
        assert!(r1.iter().any(|e| matches!(
            e,
            Ok(ChatStreamEvent::ContentDelta { delta, .. }) if delta == "Hello"
        )));

        // Final chunk (DeepSeek docs): finish_reason + usage before data:[DONE].
        let final_chunk = Event {
            event: "".to_string(),
            data: r#"{"choices":[{"delta":{"content":"","role":null},"finish_reason":"stop","index":0,"logprobs":null}],"created":1718345013,"id":"1","model":"deepseek-chat","object":"chat.completion.chunk","usage":{"completion_tokens":9,"prompt_tokens":17,"total_tokens":26}}"#.to_string(),
            id: "".to_string(),
            retry: None,
        };
        let r2 = converter.convert_event(final_chunk).await;
        assert!(r2.iter().any(|e| matches!(
            e,
            Ok(ChatStreamEvent::UsageUpdate { usage })
                if usage.prompt_tokens() == Some(17)
                    && usage.completion_tokens() == Some(9)
                    && usage.total_tokens() == Some(26)
        )));
        assert!(
            r2.iter()
                .any(|e| matches!(e, Ok(ChatStreamEvent::StreamEnd { .. })))
        );

        // After StreamEnd is emitted from finish_reason, the [DONE] marker should not
        // trigger another StreamEnd in handle_stream_end().
        assert!(converter.handle_stream_end().is_none());
    }

    #[tokio::test]
    async fn openai_compatible_stream_end_preserves_terminal_chunk_fields() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openai".to_string(),
            name: "OpenAI".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["chat".to_string(), "streaming".to_string()],
            default_model: Some("gpt-4.1".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "openai",
            "sk-test",
            "https://api.openai.com/v1",
            adapter.clone(),
        )
        .with_model("gpt-4.1");

        let converter = OpenAiCompatibleEventConverter::new(cfg, adapter);

        let final_chunk = Event {
            event: "".to_string(),
            data: r#"{"id":"chatcmpl-123","model":"gpt-4.1","created":1718345013,"system_fingerprint":"fp_openai_123","service_tier":"scale","choices":[{"index":0,"delta":{"content":"Hello","role":"assistant"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}"#.to_string(),
            id: "".to_string(),
            retry: None,
        };

        let events = converter.convert_event(final_chunk).await;
        let stream_end = events
            .iter()
            .find_map(|event| match event {
                Ok(ChatStreamEvent::StreamEnd { response }) => Some(response),
                _ => None,
            })
            .expect("stream end event");

        assert_eq!(stream_end.id.as_deref(), Some("chatcmpl-123"));
        assert_eq!(stream_end.model.as_deref(), Some("gpt-4.1"));
        assert_eq!(
            stream_end.system_fingerprint.as_deref(),
            Some("fp_openai_123")
        );
        assert_eq!(stream_end.service_tier.as_deref(), Some("scale"));
        assert_eq!(stream_end.finish_reason, Some(FinishReason::Stop));
        assert_eq!(
            stream_end
                .usage
                .as_ref()
                .and_then(|usage| usage.total_tokens()),
            Some(8)
        );
    }

    #[tokio::test]
    async fn openai_compatible_handle_stream_end_preserves_cached_terminal_fields() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openai".to_string(),
            name: "OpenAI".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["chat".to_string(), "streaming".to_string()],
            default_model: Some("gpt-4.1".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "openai",
            "sk-test",
            "https://api.openai.com/v1",
            adapter.clone(),
        )
        .with_model("gpt-4.1");

        let converter = OpenAiCompatibleEventConverter::new(cfg, adapter);

        let chunk = Event {
            event: "".to_string(),
            data: r#"{"id":"chatcmpl-456","model":"gpt-4.1","created":1718345013,"system_fingerprint":"fp_openai_456","service_tier":"priority","choices":[{"index":0,"delta":{"content":"Partial","role":"assistant"},"finish_reason":null}]}"#.to_string(),
            id: "".to_string(),
            retry: None,
        };

        let events = converter.convert_event(chunk).await;
        assert!(events.iter().any(|event| matches!(
            event,
            Ok(ChatStreamEvent::ContentDelta { delta, .. }) if delta == "Partial"
        )));

        let stream_end = converter
            .handle_stream_end()
            .expect("stream end fallback")
            .expect("stream end ok");

        let ChatStreamEvent::StreamEnd { response } = stream_end else {
            panic!("expected stream end event");
        };

        assert_eq!(response.id.as_deref(), Some("chatcmpl-456"));
        assert_eq!(response.model.as_deref(), Some("gpt-4.1"));
        assert_eq!(
            response.system_fingerprint.as_deref(),
            Some("fp_openai_456")
        );
        assert_eq!(response.service_tier.as_deref(), Some("priority"));
        assert_eq!(response.finish_reason, Some(FinishReason::Unknown));
        assert_eq!(
            response.content,
            MessageContent::Text("Partial".to_string())
        );
    }

    #[tokio::test]
    async fn openai_compatible_handle_stream_end_events_close_text_and_emit_finish_part() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openai".to_string(),
            name: "OpenAI".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["chat".to_string(), "streaming".to_string()],
            default_model: Some("gpt-4.1".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "openai",
            "sk-test",
            "https://api.openai.com/v1",
            adapter.clone(),
        )
        .with_model("gpt-4.1");

        let converter = OpenAiCompatibleEventConverter::new(cfg, adapter);

        let chunk = Event {
            event: "".to_string(),
            data: r#"{"id":"chatcmpl-789","model":"gpt-4.1","created":1718345013,"choices":[{"index":0,"delta":{"content":"Partial","role":"assistant"},"finish_reason":null}]}"#.to_string(),
            id: "".to_string(),
            retry: None,
        };

        let _ = converter.convert_event(chunk).await;
        let end_events = converter.handle_stream_end_events();

        assert!(end_events.iter().any(|event| matches!(
            event,
            Ok(ChatStreamEvent::Part {
                part: ChatStreamPart::TextEnd { .. },
            })
        )));
        assert!(end_events.iter().any(|event| matches!(
            event,
            Ok(ChatStreamEvent::Part {
                part: ChatStreamPart::Finish {
                    finish_reason: ChatStreamFinishInfo {
                        unified: FinishReason::Unknown,
                        raw: None,
                    },
                    ..
                },
            })
        )));

        let stream_end = end_events
            .iter()
            .find_map(|event| match event {
                Ok(ChatStreamEvent::StreamEnd { response }) => Some(response),
                _ => None,
            })
            .expect("stream end event");
        assert_eq!(stream_end.id.as_deref(), Some("chatcmpl-789"));
        assert_eq!(stream_end.finish_reason, Some(FinishReason::Unknown));
        assert_eq!(
            stream_end.content,
            MessageContent::Text("Partial".to_string())
        );
    }

    #[tokio::test]
    async fn openai_compatible_handle_stream_end_events_finalize_unfinished_tool_calls() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openai".to_string(),
            name: "OpenAI".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["chat".to_string(), "streaming".to_string()],
            default_model: Some("gpt-4.1".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "openai",
            "sk-test",
            "https://api.openai.com/v1",
            adapter.clone(),
        )
        .with_model("gpt-4.1");

        let converter = OpenAiCompatibleEventConverter::new(cfg, adapter);

        let chunk = Event {
            event: "".to_string(),
            data: r#"{"id":"chatcmpl-tool","model":"gpt-4.1","created":1718345013,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Tok"}}]},"finish_reason":null}]}"#.to_string(),
            id: "".to_string(),
            retry: None,
        };

        let _ = converter.convert_event(chunk).await;
        let end_events = converter.handle_stream_end_events();

        assert!(end_events.iter().any(|event| matches!(
            event,
            Ok(ChatStreamEvent::Part {
                part: ChatStreamPart::ToolInputEnd { id, .. },
            }) if id == "call_1"
        )));
        assert!(end_events.iter().any(|event| matches!(
            event,
            Ok(ChatStreamEvent::Part {
                part: ChatStreamPart::ToolCall(tool_call),
            }) if tool_call.tool_call_id == "call_1"
                && tool_call.tool_name == "get_weather"
                && tool_call.input == "{\"city\":\"Tok"
        )));
        assert!(end_events.iter().any(|event| matches!(
            event,
            Ok(ChatStreamEvent::Part {
                part: ChatStreamPart::Finish {
                    finish_reason: ChatStreamFinishInfo {
                        unified: FinishReason::Unknown,
                        raw: None,
                    },
                    ..
                },
            })
        )));
    }

    #[test]
    fn openai_compatible_serializes_basic_text_deltas_as_chat_completion_chunks() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openai".to_string(),
            name: "OpenAI".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings {
                thinking_fields: vec![],
                content_field: "content".to_string(),
                tool_calls_field: "tool_calls".to_string(),
                role_field: "role".to_string(),
            },
            capabilities: vec!["tools".to_string()],
            default_model: Some("gpt-test".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "openai",
            "sk-test",
            "https://api.openai.com/v1",
            adapter.clone(),
        )
        .with_model("gpt-test");
        let conv = OpenAiCompatibleEventConverter::new(cfg, adapter);

        let start_bytes = conv
            .serialize_event(&ChatStreamEvent::StreamStart {
                metadata: ResponseMetadata {
                    id: Some("chatcmpl_test".to_string()),
                    model: Some("gpt-test".to_string()),
                    created: None,
                    provider: "openai-compatible".to_string(),
                    request_id: None,
                    headers: None,
                },
            })
            .expect("serialize start");
        let start_frames = parse_sse_data_frames(&start_bytes);
        assert_eq!(start_frames.len(), 1);
        assert_eq!(
            start_frames[0]["object"],
            serde_json::json!("chat.completion.chunk")
        );
        assert_eq!(
            start_frames[0]["choices"][0]["delta"]["role"],
            serde_json::json!("assistant")
        );

        let delta_bytes = conv
            .serialize_event(&ChatStreamEvent::ContentDelta {
                delta: "Hello".to_string(),
                index: None,
            })
            .expect("serialize delta");
        let delta_frames = parse_sse_data_frames(&delta_bytes);
        assert!(!delta_frames.is_empty());
        assert_eq!(
            delta_frames.last().unwrap()["choices"][0]["delta"]["content"],
            serde_json::json!("Hello")
        );

        let end_bytes = conv
            .serialize_event(&ChatStreamEvent::StreamEnd {
                response: ChatResponse {
                    id: None,
                    model: Some("gpt-test".to_string()),
                    content: MessageContent::Text("Hello".to_string()),
                    usage: Some(
                        Usage::builder()
                            .prompt_tokens(3)
                            .completion_tokens(5)
                            .total_tokens(8)
                            .build(),
                    ),
                    finish_reason: Some(FinishReason::Stop),
                    raw_finish_reason: None,
                    audio: None,
                    system_fingerprint: None,
                    service_tier: None,
                    warnings: None,
                    provider_metadata: None,
                },
            })
            .expect("serialize end");
        assert!(String::from_utf8_lossy(&end_bytes).contains("data: [DONE]"));
        let end_frames = parse_sse_data_frames(&end_bytes);
        assert!(!end_frames.is_empty());
        assert_eq!(
            end_frames[0]["choices"][0]["finish_reason"],
            serde_json::json!("stop")
        );
    }

    #[test]
    fn openai_compatible_serializes_usage_update_with_unknown_totals_as_null() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openai".to_string(),
            name: "OpenAI".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings {
                thinking_fields: vec![],
                content_field: "content".to_string(),
                tool_calls_field: "tool_calls".to_string(),
                role_field: "role".to_string(),
            },
            capabilities: vec!["tools".to_string()],
            default_model: Some("gpt-test".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "openai",
            "sk-test",
            "https://api.openai.com/v1",
            adapter.clone(),
        )
        .with_model("gpt-test");
        let conv = OpenAiCompatibleEventConverter::new(cfg, adapter);

        let bytes = conv
            .serialize_event(&ChatStreamEvent::UsageUpdate {
                usage: Usage::builder()
                    .with_raw_usage_value(serde_json::json!({
                        "vendor_tokens": 5
                    }))
                    .build(),
            })
            .expect("serialize usage update");
        let frames = parse_sse_data_frames(&bytes);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0]["usage"]["prompt_tokens"], serde_json::Value::Null);
        assert_eq!(
            frames[0]["usage"]["completion_tokens"],
            serde_json::Value::Null
        );
        assert_eq!(frames[0]["usage"]["total_tokens"], serde_json::Value::Null);
        assert_eq!(frames[0]["usage"]["vendor_tokens"], serde_json::json!(5));
    }

    #[test]
    fn openai_compatible_v3_finish_preserves_unknown_usage_totals_as_null() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openai".to_string(),
            name: "OpenAI".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings {
                thinking_fields: vec![],
                content_field: "content".to_string(),
                tool_calls_field: "tool_calls".to_string(),
                role_field: "role".to_string(),
            },
            capabilities: vec!["tools".to_string()],
            default_model: Some("gpt-test".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "openai",
            "sk-test",
            "https://api.openai.com/v1",
            adapter.clone(),
        )
        .with_model("gpt-test");
        let conv = OpenAiCompatibleEventConverter::new(cfg, adapter);

        let bytes = conv
            .serialize_event(&ChatStreamEvent::Custom {
                event_type: "openai:finish".to_string(),
                data: serde_json::json!({
                    "type": "finish",
                    "finishReason": {
                        "unified": "stop",
                    },
                    "usage": {
                        "inputTokens": {},
                        "outputTokens": {},
                        "raw": {
                            "vendor_tokens": 5
                        }
                    }
                }),
            })
            .expect("serialize finish");

        let frames = parse_sse_data_frames(&bytes);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0]["usage"]["prompt_tokens"], serde_json::Value::Null);
        assert_eq!(
            frames[0]["usage"]["completion_tokens"],
            serde_json::Value::Null
        );
        assert_eq!(frames[0]["usage"]["total_tokens"], serde_json::Value::Null);
        assert_eq!(frames[0]["usage"]["vendor_tokens"], serde_json::json!(5));
    }

    #[test]
    fn openai_compatible_v3_response_metadata_updates_terminal_chunk_state() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openai".to_string(),
            name: "OpenAI".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings {
                thinking_fields: vec![],
                content_field: "content".to_string(),
                tool_calls_field: "tool_calls".to_string(),
                role_field: "role".to_string(),
            },
            capabilities: vec!["tools".to_string()],
            default_model: Some("gpt-test".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "openai",
            "sk-test",
            "https://api.openai.com/v1",
            adapter.clone(),
        )
        .with_model("gpt-test");
        let conv = OpenAiCompatibleEventConverter::new(cfg, adapter);

        let metadata_bytes = conv
            .serialize_event(&ChatStreamEvent::Custom {
                event_type: "openai:response-metadata".to_string(),
                data: serde_json::json!({
                    "type": "response-metadata",
                    "id": "resp_123",
                    "modelId": "gpt-4.1-mini",
                    "timestamp": "2026-03-20T10:11:12Z",
                }),
            })
            .expect("serialize metadata");
        assert!(metadata_bytes.is_empty(), "metadata should not emit frames");

        let end_bytes = conv
            .serialize_event(&ChatStreamEvent::Custom {
                event_type: "openai:finish".to_string(),
                data: serde_json::json!({
                    "type": "finish",
                    "finishReason": {
                        "unified": "stop",
                        "raw": "end_turn",
                    },
                    "usage": {
                        "inputTokens": { "total": 3 },
                        "outputTokens": { "total": 5 },
                    }
                }),
            })
            .expect("serialize finish");

        assert_eq!(count_done_frames(&end_bytes), 1);
        let end_frames = parse_sse_data_frames(&end_bytes);
        assert_eq!(end_frames.len(), 1);
        assert_eq!(end_frames[0]["id"], serde_json::json!("resp_123"));
        assert_eq!(end_frames[0]["model"], serde_json::json!("gpt-4.1-mini"));
        assert_eq!(
            end_frames[0]["created"],
            serde_json::json!(
                chrono::DateTime::parse_from_rfc3339("2026-03-20T10:11:12Z")
                    .expect("parse timestamp")
                    .timestamp()
            )
        );
    }

    #[test]
    fn openai_compatible_stream_start_without_metadata_does_not_synthesize_chunk_metadata() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openai".to_string(),
            name: "OpenAI".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings {
                thinking_fields: vec![],
                content_field: "content".to_string(),
                tool_calls_field: "tool_calls".to_string(),
                role_field: "role".to_string(),
            },
            capabilities: vec!["tools".to_string()],
            default_model: Some("gpt-test".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "openai",
            "sk-test",
            "https://api.openai.com/v1",
            adapter.clone(),
        )
        .with_model("gpt-test");
        let conv = OpenAiCompatibleEventConverter::new(cfg, adapter);

        let start_bytes = conv
            .serialize_event(&ChatStreamEvent::StreamStart {
                metadata: ResponseMetadata {
                    id: None,
                    model: None,
                    created: None,
                    provider: "openai-compatible".to_string(),
                    request_id: None,
                    headers: None,
                },
            })
            .expect("serialize start without metadata");

        let frames = parse_sse_data_frames(&start_bytes);
        assert_eq!(frames.len(), 1);
        assert!(frames[0].get("id").is_none());
        assert!(frames[0].get("model").is_none());
        assert!(frames[0].get("created").is_none());
        assert_eq!(
            frames[0]["choices"][0]["delta"]["role"],
            serde_json::json!("assistant")
        );
    }

    #[test]
    fn openai_compatible_serializes_finish_logprobs_into_terminal_chunk() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openai".to_string(),
            name: "OpenAI".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings {
                thinking_fields: vec![],
                content_field: "content".to_string(),
                tool_calls_field: "tool_calls".to_string(),
                role_field: "role".to_string(),
            },
            capabilities: vec!["tools".to_string()],
            default_model: Some("gpt-test".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "openai",
            "sk-test",
            "https://api.openai.com/v1",
            adapter.clone(),
        )
        .with_model("gpt-test");
        let conv = OpenAiCompatibleEventConverter::new(cfg, adapter);

        let bytes = conv
            .serialize_event(&ChatStreamEvent::Part {
                part: ChatStreamPart::Finish {
                    usage: Usage::builder()
                        .prompt_tokens(4)
                        .completion_tokens(2)
                        .total_tokens(6)
                        .build(),
                    finish_reason: crate::types::ChatStreamFinishInfo {
                        unified: FinishReason::Stop,
                        raw: Some("stop".to_string()),
                    },
                    provider_metadata: Some(std::collections::HashMap::from([(
                        "openai".to_string(),
                        serde_json::json!({
                            "logprobs": [
                                {
                                    "token": "Hello",
                                    "logprob": -0.01,
                                    "bytes": [72, 101, 108, 108, 111],
                                    "top_logprobs": [
                                        { "token": "Hello", "logprob": -0.01 },
                                        { "token": "Hi", "logprob": -1.5 }
                                    ]
                                }
                            ]
                        }),
                    )])),
                },
            })
            .expect("serialize finish part");

        let frames = parse_sse_data_frames(&bytes);
        assert_eq!(frames.len(), 1);
        assert_eq!(
            frames[0]["choices"][0]["logprobs"]["content"][0]["token"],
            serde_json::json!("Hello")
        );
        assert_eq!(
            frames[0]["choices"][0]["logprobs"]["content"][0]["top_logprobs"][1]["token"],
            serde_json::json!("Hi")
        );
    }

    #[test]
    fn openai_compatible_serializes_finish_then_stream_end_with_single_terminal_chunk() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openai".to_string(),
            name: "OpenAI".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings {
                thinking_fields: vec![],
                content_field: "content".to_string(),
                tool_calls_field: "tool_calls".to_string(),
                role_field: "role".to_string(),
            },
            capabilities: vec!["tools".to_string()],
            default_model: Some("gpt-test".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "openai",
            "sk-test",
            "https://api.openai.com/v1",
            adapter.clone(),
        )
        .with_model("gpt-test");
        let conv = OpenAiCompatibleEventConverter::new(cfg, adapter);

        let finish_bytes = conv
            .serialize_event(&ChatStreamEvent::Custom {
                event_type: "openai:finish".to_string(),
                data: serde_json::json!({
                    "type": "finish",
                    "finishReason": {
                        "unified": "stop",
                        "raw": "end_turn",
                    },
                    "usage": {
                        "inputTokens": { "total": 2 },
                        "outputTokens": { "total": 4 },
                    }
                }),
            })
            .expect("serialize finish");

        let trailing_stream_end = conv
            .serialize_event(&ChatStreamEvent::StreamEnd {
                response: ChatResponse {
                    id: None,
                    model: Some("gpt-test".to_string()),
                    content: MessageContent::Text("Hello".to_string()),
                    usage: Some(
                        Usage::builder()
                            .prompt_tokens(2)
                            .completion_tokens(4)
                            .total_tokens(6)
                            .build(),
                    ),
                    finish_reason: Some(FinishReason::Stop),
                    raw_finish_reason: None,
                    audio: None,
                    system_fingerprint: None,
                    service_tier: None,
                    warnings: None,
                    provider_metadata: None,
                },
            })
            .expect("serialize trailing stream end");

        assert!(
            trailing_stream_end.is_empty(),
            "trailing StreamEnd should be suppressed after finish"
        );
        assert_eq!(count_done_frames(&finish_bytes), 1);
        let frames = parse_sse_data_frames(&finish_bytes);
        assert_eq!(frames.len(), 1);
        assert_eq!(
            frames[0]["choices"][0]["finish_reason"],
            serde_json::json!("stop")
        );
    }

    #[test]
    fn openai_compatible_serializes_tool_call_finish_consistently_across_v3_and_stream_end() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openai".to_string(),
            name: "OpenAI".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings {
                thinking_fields: vec![],
                content_field: "content".to_string(),
                tool_calls_field: "tool_calls".to_string(),
                role_field: "role".to_string(),
            },
            capabilities: vec!["tools".to_string()],
            default_model: Some("gpt-test".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "openai",
            "sk-test",
            "https://api.openai.com/v1",
            adapter.clone(),
        )
        .with_model("gpt-test");

        let finish_conv = OpenAiCompatibleEventConverter::new(cfg.clone(), adapter.clone());
        let stream_end_conv = OpenAiCompatibleEventConverter::new(cfg, adapter);

        let finish_bytes = finish_conv
            .serialize_event(&ChatStreamEvent::Custom {
                event_type: "openai:finish".to_string(),
                data: serde_json::json!({
                    "type": "finish",
                    "finishReason": {
                        "unified": "tool-calls",
                        "raw": "tool_use",
                    },
                    "usage": {
                        "inputTokens": { "total": 11 },
                        "outputTokens": { "total": 7 },
                    }
                }),
            })
            .expect("serialize v3 finish");

        let stream_end_bytes = stream_end_conv
            .serialize_event(&ChatStreamEvent::StreamEnd {
                response: ChatResponse {
                    id: None,
                    model: Some("gpt-test".to_string()),
                    content: MessageContent::Text(String::new()),
                    usage: Some(
                        Usage::builder()
                            .prompt_tokens(11)
                            .completion_tokens(7)
                            .total_tokens(18)
                            .build(),
                    ),
                    finish_reason: Some(FinishReason::ToolCalls),
                    raw_finish_reason: None,
                    audio: None,
                    system_fingerprint: None,
                    service_tier: None,
                    warnings: None,
                    provider_metadata: None,
                },
            })
            .expect("serialize stream end");

        assert_eq!(count_done_frames(&finish_bytes), 1);
        assert_eq!(count_done_frames(&stream_end_bytes), 1);

        let finish_frame = parse_sse_data_frames(&finish_bytes)
            .pop()
            .expect("v3 finish frame");
        let stream_end_frame = parse_sse_data_frames(&stream_end_bytes)
            .pop()
            .expect("stream end frame");

        assert_eq!(
            finish_frame["choices"][0], stream_end_frame["choices"][0],
            "terminal choice payload should be identical"
        );
        assert_eq!(
            finish_frame["usage"], stream_end_frame["usage"],
            "terminal usage payload should be identical"
        );
        assert_eq!(
            finish_frame["choices"][0]["finish_reason"],
            serde_json::json!("tool_calls")
        );
    }

    #[test]
    fn openai_compatible_serializes_tool_call_delta_as_tool_calls_delta() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openai".to_string(),
            name: "OpenAI".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings {
                thinking_fields: vec![],
                content_field: "content".to_string(),
                tool_calls_field: "tool_calls".to_string(),
                role_field: "role".to_string(),
            },
            capabilities: vec!["tools".to_string()],
            default_model: Some("gpt-test".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "openai",
            "sk-test",
            "https://api.openai.com/v1",
            adapter.clone(),
        )
        .with_model("gpt-test");
        let conv = OpenAiCompatibleEventConverter::new(cfg, adapter);

        let bytes = conv
            .serialize_event(&ChatStreamEvent::ToolCallDelta {
                id: "call_1".to_string(),
                function_name: Some("lookup".to_string()),
                arguments_delta: Some("{\"q\":\"rust\"}".to_string()),
                index: None,
            })
            .expect("serialize tool delta");
        let frames = parse_sse_data_frames(&bytes);
        assert_eq!(frames.len(), 1);
        assert_eq!(
            frames[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            serde_json::json!("lookup")
        );
    }

    #[test]
    fn openai_compatible_serializes_stable_tool_parts_without_duplicate_full_arguments() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openai".to_string(),
            name: "OpenAI".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings {
                thinking_fields: vec![],
                content_field: "content".to_string(),
                tool_calls_field: "tool_calls".to_string(),
                role_field: "role".to_string(),
            },
            capabilities: vec!["tools".to_string()],
            default_model: Some("gpt-test".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "openai",
            "sk-test",
            "https://api.openai.com/v1",
            adapter.clone(),
        )
        .with_model("gpt-test");
        let conv = OpenAiCompatibleEventConverter::new(cfg, adapter);

        let mut bytes = Vec::new();
        bytes.extend_from_slice(
            &conv
                .serialize_event(&ChatStreamEvent::Part {
                    part: ChatStreamPart::ToolInputStart {
                        id: "call_1".to_string(),
                        tool_name: "lookup".to_string(),
                        provider_metadata: None,
                        provider_executed: None,
                        dynamic: None,
                        title: None,
                    },
                })
                .expect("serialize tool-input-start"),
        );
        bytes.extend_from_slice(
            &conv
                .serialize_event(&ChatStreamEvent::Part {
                    part: ChatStreamPart::ToolInputDelta {
                        id: "call_1".to_string(),
                        delta: "{\"q\":\"rust\"}".to_string(),
                        provider_metadata: None,
                    },
                })
                .expect("serialize tool-input-delta"),
        );
        bytes.extend_from_slice(
            &conv
                .serialize_event(&ChatStreamEvent::Part {
                    part: ChatStreamPart::ToolInputEnd {
                        id: "call_1".to_string(),
                        provider_metadata: None,
                    },
                })
                .expect("serialize tool-input-end"),
        );
        bytes.extend_from_slice(
            &conv
                .serialize_event(&ChatStreamEvent::Part {
                    part: ChatStreamPart::ToolCall(crate::types::ChatStreamToolCall {
                        tool_call_id: "call_1".to_string(),
                        tool_name: "lookup".to_string(),
                        input: "{\"q\":\"rust\"}".to_string(),
                        provider_executed: None,
                        dynamic: None,
                        provider_metadata: None,
                    }),
                })
                .expect("serialize tool-call"),
        );

        let frames = parse_sse_data_frames(&bytes);
        assert_eq!(
            frames.len(),
            2,
            "stable tool parts should map to the minimal chat-completions deltas"
        );
        assert_eq!(
            frames[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            serde_json::json!("lookup")
        );
        assert_eq!(
            frames[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
            serde_json::json!("{\"q\":\"rust\"}")
        );
    }

    #[test]
    fn openai_compatible_stable_tool_parts_win_over_later_legacy_tool_deltas() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openai".to_string(),
            name: "OpenAI".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings {
                thinking_fields: vec![],
                content_field: "content".to_string(),
                tool_calls_field: "tool_calls".to_string(),
                role_field: "role".to_string(),
            },
            capabilities: vec!["tools".to_string()],
            default_model: Some("gpt-test".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "openai",
            "sk-test",
            "https://api.openai.com/v1",
            adapter.clone(),
        )
        .with_model("gpt-test");
        let conv = OpenAiCompatibleEventConverter::new(cfg, adapter);

        let mut bytes = Vec::new();
        bytes.extend_from_slice(
            &conv
                .serialize_event(&ChatStreamEvent::Part {
                    part: ChatStreamPart::ToolInputStart {
                        id: "call_1".to_string(),
                        tool_name: "lookup".to_string(),
                        provider_metadata: None,
                        provider_executed: None,
                        dynamic: None,
                        title: None,
                    },
                })
                .expect("serialize tool-input-start"),
        );
        bytes.extend_from_slice(
            &conv
                .serialize_event(&ChatStreamEvent::Part {
                    part: ChatStreamPart::ToolInputDelta {
                        id: "call_1".to_string(),
                        delta: "{\"q\":\"rust\"}".to_string(),
                        provider_metadata: None,
                    },
                })
                .expect("serialize tool-input-delta"),
        );
        bytes.extend_from_slice(
            &conv
                .serialize_event(&ChatStreamEvent::ToolCallDelta {
                    id: "call_1".to_string(),
                    function_name: Some("lookup".to_string()),
                    arguments_delta: Some("{\"q\":\"rust\"}".to_string()),
                    index: None,
                })
                .expect("serialize later legacy tool delta"),
        );

        let frames = parse_sse_data_frames(&bytes);
        assert_eq!(frames.len(), 2, "later legacy shadow should be suppressed");
        assert_eq!(
            frames[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            serde_json::json!("lookup")
        );
        assert_eq!(
            frames[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
            serde_json::json!("{\"q\":\"rust\"}")
        );
    }

    #[test]
    fn openai_compatible_serializes_stable_url_source_parts_as_delta_annotations() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openai".to_string(),
            name: "OpenAI".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings {
                thinking_fields: vec![],
                content_field: "content".to_string(),
                tool_calls_field: "tool_calls".to_string(),
                role_field: "role".to_string(),
            },
            capabilities: vec!["tools".to_string()],
            default_model: Some("gpt-test".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "openai",
            "sk-test",
            "https://api.openai.com/v1",
            adapter.clone(),
        )
        .with_model("gpt-test");
        let conv = OpenAiCompatibleEventConverter::new(cfg, adapter);

        let bytes = conv
            .serialize_event(&ChatStreamEvent::Part {
                part: ChatStreamPart::Source {
                    id: "source_1".to_string(),
                    source: crate::types::SourcePart::Url {
                        url: "https://example.com/rust".to_string(),
                        title: Some("Rust".to_string()),
                    },
                    provider_metadata: None,
                },
            })
            .expect("serialize source part");

        let frames = parse_sse_data_frames(&bytes);
        assert_eq!(frames.len(), 1);
        assert_eq!(
            frames[0]["choices"][0]["delta"]["annotations"][0]["type"],
            serde_json::json!("url_citation")
        );
        assert_eq!(
            frames[0]["choices"][0]["delta"]["annotations"][0]["url_citation"]["url"],
            serde_json::json!("https://example.com/rust")
        );
        assert_eq!(
            frames[0]["choices"][0]["delta"]["annotations"][0]["url_citation"]["title"],
            serde_json::json!("Rust")
        );
    }

    #[test]
    fn openai_compatible_serializes_document_source_as_text_when_configured() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openai".to_string(),
            name: "OpenAI".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings {
                thinking_fields: vec![],
                content_field: "content".to_string(),
                tool_calls_field: "tool_calls".to_string(),
                role_field: "role".to_string(),
            },
            capabilities: vec!["tools".to_string()],
            default_model: Some("gpt-test".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "openai",
            "sk-test",
            "https://api.openai.com/v1",
            adapter.clone(),
        )
        .with_model("gpt-test");
        let conv = OpenAiCompatibleEventConverter::new(cfg, adapter)
            .with_v3_unsupported_part_behavior(V3UnsupportedPartBehavior::AsText);

        let bytes = conv
            .serialize_event(&ChatStreamEvent::Part {
                part: ChatStreamPart::Source {
                    id: "source_doc_1".to_string(),
                    source: crate::types::SourcePart::Document {
                        media_type: "application/pdf".to_string(),
                        title: "Manual".to_string(),
                        filename: Some("manual.pdf".to_string()),
                    },
                    provider_metadata: None,
                },
            })
            .expect("serialize document source part");

        let frames = parse_sse_data_frames(&bytes);
        assert!(
            frames.iter().any(|frame| {
                frame
                    .pointer("/choices/0/delta/content")
                    .and_then(|value| value.as_str())
                    == Some("[source] Manual (application/pdf), manual.pdf")
            }),
            "document source should degrade to lossy text in AsText mode"
        );
    }

    #[test]
    fn openai_compatible_serializes_v4_custom_and_reasoning_file_as_text_when_configured() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openai".to_string(),
            name: "OpenAI".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings {
                thinking_fields: vec![],
                content_field: "content".to_string(),
                tool_calls_field: "tool_calls".to_string(),
                role_field: "role".to_string(),
            },
            capabilities: vec!["tools".to_string()],
            default_model: Some("gpt-test".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "openai",
            "sk-test",
            "https://api.openai.com/v1",
            adapter.clone(),
        )
        .with_model("gpt-test");
        let conv = OpenAiCompatibleEventConverter::new(cfg, adapter)
            .with_v3_unsupported_part_behavior(V3UnsupportedPartBehavior::AsText);

        let custom_bytes = conv
            .serialize_event(&ChatStreamEvent::Custom {
                event_type: "custom:any".to_string(),
                data: serde_json::json!({
                    "type": "custom",
                    "kind": "openai.compaction"
                }),
            })
            .expect("serialize custom part");
        let custom_frames = parse_sse_data_frames(&custom_bytes);
        assert!(
            custom_frames.iter().any(|frame| {
                frame
                    .pointer("/choices/0/delta/content")
                    .and_then(|value| value.as_str())
                    == Some("[custom] openai.compaction")
            }),
            "expected lossy custom content frame: {custom_frames:?}"
        );

        let reasoning_file_bytes = conv
            .serialize_event(&ChatStreamEvent::Custom {
                event_type: "custom:any".to_string(),
                data: serde_json::json!({
                    "type": "reasoning-file",
                    "mediaType": "image/png",
                    "data": "ZmFrZQ=="
                }),
            })
            .expect("serialize reasoning-file part");
        let reasoning_file_frames = parse_sse_data_frames(&reasoning_file_bytes);
        assert!(
            reasoning_file_frames.iter().any(|frame| {
                frame
                    .pointer("/choices/0/delta/content")
                    .and_then(|value| value.as_str())
                    == Some("[reasoning-file] mediaType=image/png base64_len=8")
            }),
            "expected lossy reasoning-file content frame: {reasoning_file_frames:?}"
        );
    }
}
