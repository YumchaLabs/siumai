//! OpenAI Compatible Streaming Implementation
//!
//! This module provides streaming functionality for OpenAI-compatible providers
//! like DeepSeek, OpenRouter, SiliconFlow, etc. It uses the same SSE format as
//! OpenAI but with provider-specific adaptations for thinking/reasoning content.

use crate::error::LlmError;
use crate::streaming::{ChatStream, ChatStreamEvent, StreamStateTracker};
use crate::streaming::{SseEventConverter, StreamFactory};
use crate::types::{
    ChatRequest, ChatResponse, FinishReason, MessageContent, ResponseMetadata, Usage,
};
use eventsource_stream::Event;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use super::adapter::ProviderAdapter;
use super::openai_config::OpenAiCompatibleConfig;
use crate::transformers::request::RequestTransformer;

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
        if self.needs_stream_start().await {
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
        if self.needs_stream_start().await {
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
                finish_reason: crate::providers::openai::utils::parse_finish_reason(Some(reason)),
                tool_calls: None,
                thinking: None,
                metadata: std::collections::HashMap::new(),
            };
            builder = builder.add_stream_end(response);
        }

        builder.build()
    }

    /// Check if StreamStart event needs to be emitted
    async fn needs_stream_start(&self) -> bool {
        self.state_tracker.needs_stream_start().await
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
            .map(|s| s.to_string());
        let model = json
            .get("model")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
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
                serde_json::Value::String(s) => {
                    if !s.trim().is_empty() {
                        Some(s.as_str())
                    } else {
                        None
                    }
                }
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
        event.usage.as_ref().map(|usage| Usage {
            prompt_tokens: usage.prompt_tokens.unwrap_or(0),
            completion_tokens: usage.completion_tokens.unwrap_or(0),
            total_tokens: usage.total_tokens.unwrap_or(0),
            cached_tokens: usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|details| details.cached_tokens),
            reasoning_tokens: usage
                .completion_tokens_details
                .as_ref()
                .and_then(|details| details.reasoning_tokens),
        })
    }

    /// Extract usage info from raw JSON
    fn extract_usage_from_json(&self, json: &serde_json::Value) -> Option<Usage> {
        let usage = json.get("usage")?;
        Some(Usage {
            prompt_tokens: usage
                .get("prompt_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            completion_tokens: usage
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            total_tokens: usage
                .get("total_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            cached_tokens: usage
                .get("prompt_tokens_details")
                .and_then(|d| d.get("cached_tokens"))
                .and_then(|v| v.as_u64())
                .map(|v| v as u32),
            reasoning_tokens: usage
                .get("completion_tokens_details")
                .and_then(|d| d.get("reasoning_tokens"))
                .and_then(|v| v.as_u64())
                .map(|v| v as u32),
        })
    }
}

impl SseEventConverter for OpenAiCompatibleEventConverter {
    fn convert_event(
        &self,
        event: Event,
    ) -> Pin<Box<dyn Future<Output = Vec<Result<ChatStreamEvent, LlmError>>> + Send + Sync + '_>>
    {
        Box::pin(async move {
            match crate::streaming::parse_json_with_repair::<serde_json::Value>(&event.data) {
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
        // Carry accumulated text into StreamEnd so the factory can inject a synthetic
        // ContentDelta when no deltas were observed during the stream
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
            finish_reason: Some(FinishReason::Stop),
            tool_calls: None,
            thinking: None,
            metadata: HashMap::new(),
        };

        Some(Ok(ChatStreamEvent::StreamEnd { response }))
    }
}

/// OpenAI-compatible streaming client
#[derive(Clone)]
pub struct OpenAiCompatibleStreaming {
    config: OpenAiCompatibleConfig,
    adapter: Arc<dyn ProviderAdapter>,
    http_client: reqwest::Client,
}

impl OpenAiCompatibleStreaming {
    /// Create a new OpenAI-compatible streaming client
    pub fn new(
        config: OpenAiCompatibleConfig,
        adapter: Arc<dyn ProviderAdapter>,
        http_client: reqwest::Client,
    ) -> Self {
        Self {
            config,
            adapter,
            http_client,
        }
    }

    /// Create a chat stream from ChatRequest
    pub async fn create_chat_stream(self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        let url = format!("{}/chat/completions", self.config.base_url);

        // Build request body using the same logic as non-streaming
        let mut request_body = self.build_request_body(&request)?;

        // Override with streaming-specific settings
        request_body["stream"] = serde_json::Value::Bool(true);

        // Build closure for one-shot 401 retry with header rebuild
        let http = self.http_client.clone();
        let url_for_retry = url.clone();
        let body_for_retry = request_body.clone();
        let api_key = self.config.api_key.clone();
        let headers_builder = move || {
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert(
                reqwest::header::CONTENT_TYPE,
                reqwest::header::HeaderValue::from_static("application/json"),
            );
            headers.insert(
                reqwest::header::AUTHORIZATION,
                reqwest::header::HeaderValue::from_str(&format!("Bearer {}", api_key))
                    .map_err(|e| LlmError::ConfigurationError(format!("Invalid API key: {e}")))?,
            );
            crate::utils::http_headers::inject_tracing_headers(&mut headers);
            Ok::<reqwest::header::HeaderMap, LlmError>(headers)
        };
        let build_request = move || {
            let headers = headers_builder()?;
            Ok(http
                .post(&url_for_retry)
                .headers(headers)
                .json(&body_for_retry))
        };

        let converter = OpenAiCompatibleEventConverter::new(self.config, self.adapter);
        StreamFactory::create_eventsource_stream_with_retry(
            "openai-compatible",
            build_request,
            converter,
        )
        .await
    }

    /// Build request body via unified transformer
    fn build_request_body(&self, request: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        let transformer = super::transformers::CompatRequestTransformer {
            config: self.config.clone(),
            adapter: self.adapter.clone(),
        };
        transformer.transform_chat(request)
    }

    /// Build HTTP headers
    #[allow(dead_code)]
    fn build_headers(&self) -> Result<reqwest::header::HeaderMap, LlmError> {
        let mut headers = reqwest::header::HeaderMap::new();

        headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/json"),
        );

        headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_str(&format!("Bearer {}", self.config.api_key))
                .map_err(|e| LlmError::ConfigurationError(format!("Invalid API key: {e}")))?,
        );

        Ok(headers)
    }
}

// Convert a dotted path like "a.b.0.c" to a JSON Pointer "/a/b/0/c"
fn to_pointer(path: &str) -> String {
    let mut s = String::new();
    for part in path.split('.') {
        s.push('/');
        s.push_str(part);
    }
    s
}
