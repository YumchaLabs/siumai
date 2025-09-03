//! OpenAI streaming implementation using eventsource-stream
//!
//! This module provides OpenAI streaming functionality using the unified
//! eventsource-stream infrastructure for reliable UTF-8 and SSE handling.

use crate::error::LlmError;
use crate::providers::openai::config::OpenAiConfig;
use crate::stream::{ChatStream, ChatStreamEvent};
use crate::types::{ChatResponse, FinishReason, MessageContent, ResponseMetadata, Usage};
use crate::utils::streaming::{SseEventConverter, StreamFactory};
use eventsource_stream::Event;
use serde::Deserialize;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;

/// OpenAI stream event structure
#[derive(Debug, Clone, Deserialize)]
struct OpenAiStreamEvent {
    id: Option<String>,
    model: Option<String>,
    choices: Option<Vec<OpenAiStreamChoice>>,
    usage: Option<OpenAiStreamUsage>,
}

/// OpenAI stream choice
#[derive(Debug, Clone, Deserialize)]
struct OpenAiStreamChoice {
    index: Option<usize>,
    delta: Option<OpenAiStreamDelta>,
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

/// OpenAI stream delta
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct OpenAiStreamDelta {
    role: Option<String>,
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiToolCallDelta>>,
    thinking: Option<String>,
}

impl<'de> serde::Deserialize<'de> for OpenAiStreamDelta {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value: serde_json::Value = serde_json::Value::deserialize(deserializer)?;

        // Extract thinking content using priority order: reasoning_content > thinking > reasoning
        let thinking = extract_thinking_from_multiple_fields(&value);

        // Extract other fields normally
        let role = value.get("role").and_then(|v| v.as_str()).map(String::from);

        let content = value
            .get("content")
            .and_then(|v| v.as_str())
            .map(String::from);

        let tool_calls = value
            .get("tool_calls")
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        Ok(OpenAiStreamDelta {
            role,
            content,
            tool_calls,
            thinking,
        })
    }
}

/// Extract thinking content from multiple possible field names with priority order
///
/// Priority order: reasoning_content > thinking > reasoning
/// This matches the OpenAI-compatible adapter's field priority logic
pub(crate) fn extract_thinking_from_multiple_fields(value: &serde_json::Value) -> Option<String> {
    // Field names in priority order (same as OpenAI-compatible adapter)
    let field_names = ["reasoning_content", "thinking", "reasoning"];

    for field_name in &field_names {
        if let Some(thinking_value) = value
            .get(field_name)
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty())
        {
            return Some(thinking_value.to_string());
        }
    }
    None
}

/// OpenAI tool call delta
#[derive(Debug, Clone, Deserialize)]
struct OpenAiToolCallDelta {
    #[allow(dead_code)]
    index: Option<usize>,
    id: Option<String>,
    function: Option<OpenAiFunctionCallDelta>,
}

/// OpenAI function call delta
#[derive(Debug, Clone, Deserialize)]
struct OpenAiFunctionCallDelta {
    name: Option<String>,
    arguments: Option<String>,
}

/// OpenAI usage information
#[derive(Debug, Clone, Deserialize)]
struct OpenAiStreamUsage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    total_tokens: Option<u32>,
    completion_tokens_details: Option<OpenAiCompletionTokensDetails>,
    prompt_tokens_details: Option<OpenAiPromptTokensDetails>,
}

/// OpenAI completion tokens details
#[derive(Debug, Clone, Deserialize)]
struct OpenAiCompletionTokensDetails {
    reasoning_tokens: Option<u32>,
}

/// OpenAI prompt tokens details
#[derive(Debug, Clone, Deserialize)]
struct OpenAiPromptTokensDetails {
    cached_tokens: Option<u32>,
}

/// OpenAI event converter
#[derive(Clone)]
pub struct OpenAiEventConverter {
    #[allow(dead_code)]
    config: OpenAiConfig,
    /// Track if StreamStart has been emitted
    stream_started: Arc<Mutex<bool>>,
}

impl OpenAiEventConverter {
    pub fn new(config: OpenAiConfig) -> Self {
        Self {
            config,
            stream_started: Arc::new(Mutex::new(false)),
        }
    }

    /// Convert OpenAI stream event to multiple ChatStreamEvents
    async fn convert_openai_event_async(&self, event: OpenAiStreamEvent) -> Vec<ChatStreamEvent> {
        use crate::utils::streaming::EventBuilder;

        let mut builder = EventBuilder::new();

        // Check if we need to emit StreamStart first
        if self.needs_stream_start().await {
            let metadata = self.create_stream_start_metadata(&event);
            builder = builder.add_stream_start(metadata);
        }

        // Process content delta - NO MORE CONTENT LOSS!
        if let Some(content) = self.extract_content(&event) {
            builder = builder.add_content_delta(content, self.extract_choice_index(&event));
        }

        // Process tool calls
        if let Some((id, name, args)) = self.extract_tool_call(&event) {
            builder =
                builder.add_tool_call_delta(id, name, args, self.extract_choice_index(&event));
        }

        // Process thinking content (for reasoning models)
        if let Some(thinking) = self.extract_thinking(&event) {
            builder = builder.add_thinking_delta(thinking);
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

    /// Extract content from OpenAI event
    fn extract_content(&self, event: &OpenAiStreamEvent) -> Option<String> {
        event
            .choices
            .as_ref()?
            .first()?
            .delta
            .as_ref()?
            .content
            .as_ref()
            .filter(|content| !content.is_empty())
            .cloned()
    }

    /// Extract tool call information
    fn extract_tool_call(
        &self,
        event: &OpenAiStreamEvent,
    ) -> Option<(String, Option<String>, Option<String>)> {
        let choice = event.choices.as_ref()?.first()?;
        let tool_call = choice.delta.as_ref()?.tool_calls.as_ref()?.first()?;

        let id = tool_call.id.clone()?;
        let function_name = tool_call.function.as_ref()?.name.clone();
        let arguments = tool_call.function.as_ref()?.arguments.clone();

        Some((id, function_name, arguments))
    }

    /// Extract thinking content
    fn extract_thinking(&self, event: &OpenAiStreamEvent) -> Option<String> {
        event
            .choices
            .as_ref()?
            .first()?
            .delta
            .as_ref()?
            .thinking
            .as_ref()
            .filter(|thinking| !thinking.is_empty())
            .cloned()
    }

    /// Extract usage information
    fn extract_usage(&self, event: &OpenAiStreamEvent) -> Option<Usage> {
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

    /// Extract choice index
    fn extract_choice_index(&self, event: &OpenAiStreamEvent) -> Option<usize> {
        event.choices.as_ref()?.first()?.index
    }

    /// Create StreamStart metadata from OpenAI event
    fn create_stream_start_metadata(&self, event: &OpenAiStreamEvent) -> ResponseMetadata {
        ResponseMetadata {
            id: event.id.clone(),
            model: event.model.clone(),
            created: Some(chrono::Utc::now()),
            provider: "openai".to_string(),
            request_id: None, // OpenAI doesn't provide request_id in stream events
        }
    }
}

impl SseEventConverter for OpenAiEventConverter {
    fn convert_event(
        &self,
        event: Event,
    ) -> Pin<Box<dyn Future<Output = Vec<Result<ChatStreamEvent, LlmError>>> + Send + Sync + '_>>
    {
        Box::pin(async move {
            match serde_json::from_str::<OpenAiStreamEvent>(&event.data) {
                Ok(openai_event) => self
                    .convert_openai_event_async(openai_event)
                    .await
                    .into_iter()
                    .map(Ok)
                    .collect(),
                Err(e) => {
                    vec![Err(LlmError::ParseError(format!(
                        "Failed to parse OpenAI event: {e}"
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

/// OpenAI streaming client
#[derive(Clone)]
pub struct OpenAiStreaming {
    config: OpenAiConfig,
    http_client: reqwest::Client,
}

impl OpenAiStreaming {
    /// Create a new OpenAI streaming client
    pub fn new(config: OpenAiConfig, http_client: reqwest::Client) -> Self {
        Self {
            config,
            http_client,
        }
    }

    /// Create a chat stream from ChatRequest
    pub async fn create_chat_stream(
        self,
        request: crate::types::ChatRequest,
    ) -> Result<ChatStream, LlmError> {
        let url = format!("{}/chat/completions", self.config.base_url);

        // Use the same request building logic as non-streaming
        let chat_capability = super::chat::OpenAiChatCapability::new(
            self.config.api_key.clone(),
            self.config.base_url.clone(),
            self.http_client.clone(),
            self.config.organization.clone(),
            self.config.project.clone(),
            self.config.http_config.clone(),
            self.config.common_params.clone(),
        );

        let mut request_body = chat_capability.build_chat_request_body(&request)?;

        // Override with streaming-specific settings
        request_body["stream"] = serde_json::Value::Bool(true);
        request_body["stream_options"] = serde_json::json!({
            "include_usage": true
        });

        // Create headers
        let mut headers = reqwest::header::HeaderMap::new();
        for (key, value) in self.config.get_headers() {
            let header_name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
                .map_err(|e| LlmError::HttpError(format!("Invalid header name: {e}")))?;
            let header_value = reqwest::header::HeaderValue::from_str(&value)
                .map_err(|e| LlmError::HttpError(format!("Invalid header value: {e}")))?;
            headers.insert(header_name, header_value);
        }

        // Create the stream using reqwest_eventsource for enhanced reliability
        let request_builder = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&request_body);

        let converter = OpenAiEventConverter::new(self.config);
        StreamFactory::create_eventsource_stream(request_builder, converter).await
    }
}
