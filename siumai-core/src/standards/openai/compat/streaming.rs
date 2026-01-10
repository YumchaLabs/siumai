//! OpenAI-compatible streaming implementation (protocol layer)
//!
//! Provides SSE event conversion for OpenAI-compatible providers.
//! The legacy OpenAiCompatibleStreaming client has been removed in favor of the unified HttpChatExecutor.

use crate::error::LlmError;
use crate::streaming::SseEventConverter;
use crate::streaming::{ChatStreamEvent, LanguageModelV3StreamPart, StreamStateTracker};
use crate::types::{ChatResponse, FinishReason, MessageContent, ResponseMetadata, Usage};
use eventsource_stream::Event;
use serde::{Deserialize, Serialize};

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use super::adapter::ProviderAdapter;
use super::openai_config::OpenAiCompatibleConfig;

#[derive(Debug, Default, Clone)]
struct OpenAiCompatSerializeState {
    id: Option<String>,
    model: Option<String>,
    created: Option<u64>,
    emitted_role: bool,
    tool_call_index_by_id: std::collections::HashMap<String, u32>,
    next_tool_call_index: u32,
}

// Type alias for better readability and to reduce type complexity lints
type ToolCallDelta = (String, Option<String>, Option<String>, Option<usize>);

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

    // Provider-specific thinking/reasoning fields
    pub thinking: Option<String>,          // Standard thinking field
    pub reasoning_content: Option<String>, // DeepSeek reasoning field
    pub reasoning: Option<String>,         // Alternative reasoning field
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
    state_tracker: StreamStateTracker,
    // Accumulate plain text content so StreamEnd can carry a fallback when no deltas were seen
    accumulated_content: Arc<tokio::sync::Mutex<String>>,
    // Track whether we have emitted any ContentDelta to avoid duplicate injection
    emitted_content: std::sync::Arc<std::sync::atomic::AtomicBool>,

    // Serialize state for reverse SSE encoding (ChatStreamEvent -> OpenAI-compatible SSE).
    serialize_state: Arc<std::sync::Mutex<OpenAiCompatSerializeState>>,
}

impl OpenAiCompatibleEventConverter {
    /// Create a new event converter
    pub fn new(config: OpenAiCompatibleConfig, adapter: Arc<dyn ProviderAdapter>) -> Self {
        Self {
            config,
            adapter,
            state_tracker: StreamStateTracker::new(),
            accumulated_content: Arc::new(tokio::sync::Mutex::new(String::new())),
            emitted_content: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            serialize_state: Arc::new(std::sync::Mutex::new(OpenAiCompatSerializeState::default())),
        }
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
            builder = builder.add_stream_start(metadata);
        }

        // If the event data itself is a JSON string (common when SSE named events
        // carry plain text as data, e.g., Responses "output_text.delta" proxied),
        // treat it directly as a content delta.
        if let Some(s) = json.as_str()
            && !s.trim().is_empty()
        {
            {
                let mut acc = self.accumulated_content.lock().await;
                acc.push_str(s);
            }
            self.emitted_content
                .store(true, std::sync::atomic::Ordering::Relaxed);
            return builder.add_content_delta(s.to_string(), None).build();
        }

        // Content (compatible with Chat Completions and Responses API)
        if let Some(content) = self.extract_content_from_json(json) {
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
            builder = builder.add_thinking_delta(thinking);
        }

        // Tool call deltas (optional) — support multiple tool calls in the same chunk
        let tool_calls = self.extract_tool_calls_from_json(json);
        if !tool_calls.is_empty() {
            for (id, name, args, idx) in tool_calls {
                builder = builder.add_tool_call_delta(id, name, args, idx);
            }
        }

        // Usage updates (optional)
        if let Some(usage) = self.extract_usage_from_json(json) {
            builder = builder.add_usage_update(usage);
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
            let response = ChatResponse {
                id: None,
                model: None,
                content: MessageContent::Text(text),
                usage: None,
                finish_reason: crate::standards::openai::utils::parse_finish_reason(Some(reason)),
                audio: None,
                system_fingerprint: None,
                service_tier: None,
                warnings: None,
                provider_metadata: None,
            };
            builder = builder.add_stream_end(response);
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
            created: event.created.map(|ts| {
                chrono::DateTime::from_timestamp(ts as i64, 0).unwrap_or_else(chrono::Utc::now)
            }),
            provider: self.config.provider_id.clone(),
            request_id: None,
        }
    }

    /// Build StreamStart metadata directly from JSON
    fn create_stream_start_metadata_from_json(&self, json: &serde_json::Value) -> ResponseMetadata {
        let id = json
            .get("id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.to_string());
        let model = match json.get("model") {
            Some(v) => match v.as_str().map(str::trim) {
                Some(s) if !s.is_empty() => Some(s.to_string()),
                // Vercel parity: some model routers emit an initial chunk with an empty (or null)
                // `model`. Fall back to the configured request model to avoid surfacing empty model ids.
                _ => (!self.config.model.trim().is_empty()).then(|| self.config.model.clone()),
            },
            // If the field is entirely missing, do not invent a model id.
            None => None,
        };
        let created = json.get("created").and_then(|v| v.as_u64()).map(|ts| {
            chrono::DateTime::from_timestamp(ts as i64, 0).unwrap_or_else(chrono::Utc::now)
        });
        ResponseMetadata {
            id,
            model,
            created,
            provider: self.config.provider_id.clone(),
            request_id: None,
        }
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

    /// Extract tool-call deltas from raw JSON
    #[allow(dead_code)]
    fn extract_tool_call_from_json(&self, json: &serde_json::Value) -> Option<ToolCallDelta> {
        if let Some(calls) = json
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c0| c0.get("delta"))
            .and_then(|d| d.get("tool_calls"))
            .and_then(|tc| tc.as_array())
            && let Some(first) = calls.first()
        {
            let id = first.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let function = first.get("function");
            let name = function
                .and_then(|f| f.get("name"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let args = function
                .and_then(|f| f.get("arguments"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let idx = json
                .get("choices")
                .and_then(|c| c.get(0))
                .and_then(|choice| choice.get("index"))
                .and_then(|v| v.as_u64())
                .map(|i| i as usize);
            return Some((id.to_string(), name, args, idx));
        }
        None
    }

    /// Extract multiple tool calls (if present) from a single JSON chunk
    fn extract_tool_calls_from_json(&self, json: &serde_json::Value) -> Vec<ToolCallDelta> {
        let mut out = Vec::new();
        if let Some(arr) = json
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c0| c0.get("delta"))
            .and_then(|d| d.get("tool_calls"))
            .and_then(|tc| tc.as_array())
        {
            let idx = json
                .get("choices")
                .and_then(|c| c.get(0))
                .and_then(|choice| choice.get("index"))
                .and_then(|v| v.as_u64())
                .map(|i| i as usize);
            for first in arr {
                let id = first.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let function = first.get("function");
                let name = function
                    .and_then(|f| f.get("name"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let args = function
                    .and_then(|f| f.get("arguments"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                out.push((id.to_string(), name, args, idx));
            }
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
        event.usage.as_ref().map(|usage| {
            let mut builder = Usage::builder()
                .prompt_tokens(usage.prompt_tokens.unwrap_or(0))
                .completion_tokens(usage.completion_tokens.unwrap_or(0))
                .total_tokens(usage.total_tokens.unwrap_or(0));

            if let Some(cached) = usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|details| details.cached_tokens)
            {
                builder = builder.with_cached_tokens(cached);
            }

            if let Some(reasoning) = usage
                .completion_tokens_details
                .as_ref()
                .and_then(|details| details.reasoning_tokens)
            {
                builder = builder.with_reasoning_tokens(reasoning);
            }

            builder.build()
        })
    }

    /// Extract usage info from raw JSON
    fn extract_usage_from_json(&self, json: &serde_json::Value) -> Option<Usage> {
        let usage = json.get("usage")?;
        if usage.is_null() {
            return None;
        }

        let mut builder = Usage::builder()
            .prompt_tokens(
                usage
                    .get("prompt_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32,
            )
            .completion_tokens(
                usage
                    .get("completion_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32,
            )
            .total_tokens(
                usage
                    .get("total_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32,
            );

        if let Some(cached) = usage
            .get("prompt_tokens_details")
            .and_then(|d| d.get("cached_tokens"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
        {
            builder = builder.with_cached_tokens(cached);
        }

        if let Some(reasoning) = usage
            .get("completion_tokens_details")
            .and_then(|d| d.get("reasoning_tokens"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
        {
            builder = builder.with_reasoning_tokens(reasoning);
        }

        Some(builder.build())
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
                Err(e) => vec![Err(LlmError::ParseError(format!(
                    "Failed to parse OpenAI-compatible event: {e}"
                )))],
            }
        })
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

        let response = ChatResponse {
            id: None,
            model: None,
            content: MessageContent::Text(
                self.accumulated_content
                    .try_lock()
                    .map(|g| g.clone())
                    .unwrap_or_default(),
            ),
            usage: None,
            finish_reason: Some(FinishReason::Unknown),
            audio: None,
            system_fingerprint: None,
            service_tier: None,
            warnings: None,
            provider_metadata: None,
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

        fn now_epoch_secs() -> u64 {
            chrono::Utc::now().timestamp().max(0) as u64
        }

        let mut state = self
            .serialize_state
            .lock()
            .map_err(|_| LlmError::InternalError("serialize_state lock poisoned".to_string()))?;

        let ensure_state = |state: &mut OpenAiCompatSerializeState| {
            if state.id.is_none() {
                state.id = Some("chatcmpl-siumai-0".to_string());
            }
            if state.model.is_none() {
                state.model = Some(self.config.model.clone());
            }
            if state.created.is_none() {
                state.created = Some(now_epoch_secs());
            }
        };

        let mut serialize_inner = |event: &ChatStreamEvent| -> Result<Vec<u8>, LlmError> {
            match event {
                ChatStreamEvent::StreamStart { metadata } => {
                    *state = OpenAiCompatSerializeState::default();
                    state.id = metadata
                        .id
                        .clone()
                        .or_else(|| Some("chatcmpl-siumai-0".to_string()));
                    state.model = metadata
                        .model
                        .clone()
                        .or_else(|| Some(self.config.model.clone()));
                    state.created = metadata
                        .created
                        .map(|dt| dt.timestamp().max(0) as u64)
                        .or_else(|| Some(now_epoch_secs()));

                    let payload = serde_json::json!({
                        "id": state.id.clone(),
                        "object": "chat.completion.chunk",
                        "created": state.created,
                        "model": state.model.clone(),
                        "choices": [
                            {
                                "index": 0,
                                "delta": { "role": "assistant" },
                                "finish_reason": serde_json::Value::Null
                            }
                        ]
                    });
                    state.emitted_role = true;
                    sse_data_frame(&payload)
                }
                ChatStreamEvent::ContentDelta { delta, index } => {
                    ensure_state(&mut state);
                    let choice_index = index.unwrap_or(0) as u32;

                    // Some clients expect a role delta before the first content delta.
                    let mut out = Vec::new();
                    if !state.emitted_role {
                        let role_payload = serde_json::json!({
                            "id": state.id.clone(),
                            "object": "chat.completion.chunk",
                            "created": state.created,
                            "model": state.model.clone(),
                            "choices": [
                                {
                                    "index": choice_index,
                                    "delta": { "role": "assistant" },
                                    "finish_reason": serde_json::Value::Null
                                }
                            ]
                        });
                        out.extend_from_slice(&sse_data_frame(&role_payload)?);
                        state.emitted_role = true;
                    }

                    let payload = serde_json::json!({
                        "id": state.id.clone(),
                        "object": "chat.completion.chunk",
                        "created": state.created,
                        "model": state.model.clone(),
                        "choices": [
                            {
                                "index": choice_index,
                                "delta": { "content": delta },
                                "finish_reason": serde_json::Value::Null
                            }
                        ]
                    });
                    out.extend_from_slice(&sse_data_frame(&payload)?);
                    Ok(out)
                }
                ChatStreamEvent::ToolCallDelta {
                    id,
                    function_name,
                    arguments_delta,
                    index,
                } => {
                    ensure_state(&mut state);
                    let choice_index = index.unwrap_or(0) as u32;

                    let tool_call_index = match state.tool_call_index_by_id.get(id).copied() {
                        Some(i) => i,
                        None => {
                            let i = state.next_tool_call_index;
                            state.next_tool_call_index += 1;
                            state.tool_call_index_by_id.insert(id.clone(), i);
                            i
                        }
                    };

                    let mut function = serde_json::Map::new();
                    if let Some(name) = function_name.clone() {
                        function.insert("name".to_string(), serde_json::Value::String(name));
                    }
                    if let Some(args) = arguments_delta.clone() {
                        function.insert("arguments".to_string(), serde_json::Value::String(args));
                    }

                    let payload = serde_json::json!({
                        "id": state.id.clone(),
                        "object": "chat.completion.chunk",
                        "created": state.created,
                        "model": state.model.clone(),
                        "choices": [
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
                        ]
                    });
                    sse_data_frame(&payload)
                }
                ChatStreamEvent::UsageUpdate { usage } => {
                    ensure_state(&mut state);
                    let payload = serde_json::json!({
                        "id": state.id.clone(),
                        "object": "chat.completion.chunk",
                        "created": state.created,
                        "model": state.model.clone(),
                        "choices": [],
                        "usage": {
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": usage.completion_tokens,
                            "total_tokens": usage.total_tokens,
                        }
                    });
                    sse_data_frame(&payload)
                }
                ChatStreamEvent::StreamEnd { response } => {
                    ensure_state(&mut state);
                    if state.model.is_none() {
                        state.model = response.model.clone();
                    }
                    let finish_reason = response
                        .finish_reason
                        .as_ref()
                        .and_then(finish_reason_str)
                        .map(|s| serde_json::Value::String(s.to_string()))
                        .unwrap_or(serde_json::Value::Null);

                    let usage = response.usage.as_ref().map(|u| {
                        serde_json::json!({
                            "prompt_tokens": u.prompt_tokens,
                            "completion_tokens": u.completion_tokens,
                            "total_tokens": u.total_tokens,
                        })
                    });

                    let payload = serde_json::json!({
                        "id": state.id.clone(),
                        "object": "chat.completion.chunk",
                        "created": state.created,
                        "model": state.model.clone(),
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": finish_reason
                            }
                        ],
                        "usage": usage.unwrap_or(serde_json::Value::Null),
                    });

                    let mut out = sse_data_frame(&payload)?;
                    out.extend_from_slice(&done_frame());
                    Ok(out)
                }
                ChatStreamEvent::Error { error } => {
                    let payload = serde_json::json!({
                        "error": { "message": error },
                    });
                    sse_data_frame(&payload)
                }
                ChatStreamEvent::ThinkingDelta { .. } | ChatStreamEvent::Custom { .. } => {
                    Ok(Vec::new())
                }
            }
        };

        match event {
            ChatStreamEvent::Custom { data, .. } => {
                let Ok(part) = serde_json::from_value::<LanguageModelV3StreamPart>(data.clone())
                else {
                    return Ok(Vec::new());
                };

                let mut out = Vec::new();
                for ev in part.to_best_effort_chat_events() {
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
                if usage.prompt_tokens == 17
                    && usage.completion_tokens == 9
                    && usage.total_tokens == 26
        )));
        assert!(
            r2.iter()
                .any(|e| matches!(e, Ok(ChatStreamEvent::StreamEnd { .. })))
        );

        // After StreamEnd is emitted from finish_reason, the [DONE] marker should not
        // trigger another StreamEnd in handle_stream_end().
        assert!(converter.handle_stream_end().is_none());
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
}
