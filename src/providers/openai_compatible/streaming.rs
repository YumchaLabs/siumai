//! OpenAI Compatible Streaming Implementation
//!
//! This module provides streaming functionality for OpenAI-compatible providers
//! like DeepSeek, OpenRouter, SiliconFlow, etc. It uses the same SSE format as
//! OpenAI but with provider-specific adaptations for thinking/reasoning content.

use crate::error::LlmError;
use crate::stream::{ChatStream, ChatStreamEvent};
use crate::types::{
    ChatRequest, ChatResponse, FinishReason, MessageContent, ResponseMetadata, Usage,
};
use crate::utils::streaming::{SseEventConverter, StreamFactory};
use eventsource_stream::Event;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;

use super::adapter::ProviderAdapter;
use super::openai_config::OpenAiCompatibleConfig;
use crate::transformers::request::RequestTransformer;

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
    stream_started: Arc<Mutex<bool>>,
}

impl OpenAiCompatibleEventConverter {
    /// Create a new event converter
    pub fn new(config: OpenAiCompatibleConfig, adapter: Arc<dyn ProviderAdapter>) -> Self {
        Self {
            config,
            adapter,
            stream_started: Arc::new(Mutex::new(false)),
        }
    }

    /// Convert OpenAI-compatible stream event to multiple ChatStreamEvents
    async fn convert_event_async(
        &self,
        event: OpenAiCompatibleStreamEvent,
    ) -> Vec<ChatStreamEvent> {
        use crate::utils::streaming::EventBuilder;

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

    /// Check if StreamStart event needs to be emitted
    async fn needs_stream_start(&self) -> bool {
        let mut started = self.stream_started.lock().await;
        if !*started {
            *started = true;
            true
        } else {
            false
        }
    }

    /// Create stream start metadata
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

    /// Extract content from stream event using dynamic field accessor
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

    /// Extract thinking/reasoning content using dynamic field accessor
    ///
    /// This uses the adapter's configurable field accessor to dynamically extract
    /// thinking content from any field structure, completely eliminating hardcoded field names.
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

    /// Extract tool call information
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

    /// Extract choice index
    fn extract_choice_index(&self, event: &OpenAiCompatibleStreamEvent) -> u32 {
        event
            .choices
            .as_ref()
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.index)
            .unwrap_or(0)
    }

    /// Extract usage information
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
}

impl SseEventConverter for OpenAiCompatibleEventConverter {
    fn convert_event(
        &self,
        event: Event,
    ) -> Pin<Box<dyn Future<Output = Vec<Result<ChatStreamEvent, LlmError>>> + Send + Sync + '_>>
    {
        Box::pin(async move {
            match serde_json::from_str::<OpenAiCompatibleStreamEvent>(&event.data) {
                Ok(compat_event) => {
                    let result: Vec<Result<ChatStreamEvent, LlmError>> = self
                        .convert_event_async(compat_event)
                        .await
                        .into_iter()
                        .map(Ok)
                        .collect();
                    result
                }
                Err(e) => {
                    vec![Err(LlmError::ParseError(format!(
                        "Failed to parse OpenAI-compatible event: {e}"
                    )))]
                }
            }
        })
    }

    fn handle_stream_end(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
        let response = ChatResponse {
            id: None,
            model: None,
            content: MessageContent::Text("".to_string()),
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

        // Create headers
        let headers = self.build_headers()?;

        // Create the stream using reqwest_eventsource for enhanced reliability
        let request_builder = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&request_body);

        let converter = OpenAiCompatibleEventConverter::new(self.config, self.adapter);
        StreamFactory::create_eventsource_stream(request_builder, converter).await
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
