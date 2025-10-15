//! `Groq` Streaming Implementation
//!
//! This module provides Groq-specific streaming functionality for chat completions.

use crate::error::LlmError;
use crate::stream::{ChatStream, ChatStreamEvent};
use crate::types::{ChatRequest, ResponseMetadata, Usage};
use crate::types::{ChatResponse, FinishReason, MessageContent};
use crate::utils::streaming::{SseEventConverter, StreamFactory};
use eventsource_stream::Event;

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;

use super::config::GroqConfig;
use super::types::*;
use super::utils::*;
use crate::transformers::request::RequestTransformer;

/// Groq event converter for SSE events
#[derive(Clone)]
pub struct GroqEventConverter {
    /// Track if StreamStart has been emitted
    stream_started: Arc<Mutex<bool>>,
}

impl GroqEventConverter {
    /// Create a new Groq event converter
    pub fn new() -> Self {
        Self {
            stream_started: Arc::new(Mutex::new(false)),
        }
    }

    /// Convert Groq stream response to multiple ChatStreamEvents
    async fn convert_groq_response_async(
        &self,
        response: GroqChatStreamChunk,
    ) -> Vec<ChatStreamEvent> {
        use crate::utils::streaming::EventBuilder;

        let mut builder = EventBuilder::new();

        // Check if we need to emit StreamStart first
        if self.needs_stream_start().await {
            let metadata = self.create_stream_start_metadata(&response);
            builder = builder.add_stream_start(metadata);
        }

        // Process content - NO MORE CONTENT LOSS!
        if let Some(content) = self.extract_content(&response) {
            builder = builder.add_content_delta(content, self.extract_choice_index(&response));
        }

        // Process usage updates
        if let Some(usage) = self.extract_usage(&response) {
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

    /// Extract content from Groq response
    fn extract_content(&self, response: &GroqChatStreamChunk) -> Option<String> {
        response
            .choices
            .first()?
            .delta
            .content
            .as_ref()
            .filter(|content| !content.is_empty())
            .cloned()
    }

    /// Extract choice index
    fn extract_choice_index(&self, response: &GroqChatStreamChunk) -> Option<usize> {
        Some(response.choices.first()?.index as usize)
    }

    /// Extract usage information
    fn extract_usage(&self, response: &GroqChatStreamChunk) -> Option<Usage> {
        response.usage.as_ref().map(|usage| Usage {
            prompt_tokens: usage.prompt_tokens.unwrap_or(0),
            completion_tokens: usage.completion_tokens.unwrap_or(0),
            total_tokens: usage.total_tokens.unwrap_or(0),
            cached_tokens: None,
            reasoning_tokens: None,
        })
    }

    /// Create StreamStart metadata
    fn create_stream_start_metadata(&self, response: &GroqChatStreamChunk) -> ResponseMetadata {
        ResponseMetadata {
            id: Some(response.id.clone()),
            model: Some(response.model.clone()),
            created: Some(chrono::Utc::now()),
            provider: "groq".to_string(),
            request_id: None,
        }
    }
}

impl Default for GroqEventConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl SseEventConverter for GroqEventConverter {
    fn convert_event(
        &self,
        event: Event,
    ) -> Pin<Box<dyn Future<Output = Vec<Result<ChatStreamEvent, LlmError>>> + Send + Sync + '_>>
    {
        Box::pin(async move {
            match serde_json::from_str::<GroqChatStreamChunk>(&event.data) {
                Ok(groq_response) => self
                    .convert_groq_response_async(groq_response)
                    .await
                    .into_iter()
                    .map(Ok)
                    .collect(),
                Err(e) => {
                    vec![Err(LlmError::ParseError(format!(
                        "Failed to parse Groq event: {e}"
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
            metadata: std::collections::HashMap::new(),
        };
        Some(Ok(ChatStreamEvent::StreamEnd { response }))
    }
}

/// `Groq` streaming client
#[derive(Clone)]
pub struct GroqStreaming {
    /// `Groq` configuration
    config: GroqConfig,
    /// HTTP client
    http_client: reqwest::Client,
}

impl GroqStreaming {
    /// Create a new `Groq` streaming client
    pub fn new(config: GroqConfig, http_client: reqwest::Client) -> Self {
        Self {
            config,
            http_client,
        }
    }

    /// Create a streaming chat completion request
    pub async fn create_chat_stream(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        let url = format!("{}/chat/completions", self.config.base_url);

        // Build request body via transformer
        let transformer = super::transformers::GroqRequestTransformer;
        let mut request_body = transformer.transform_chat(&request)?;

        // Override with streaming-specific settings
        request_body["stream"] = serde_json::Value::Bool(true);
        request_body["stream_options"] = serde_json::json!({
            "include_usage": true
        });

        // Validate parameters
        validate_groq_params(&request_body)?;

        // Build closure for one-shot 401 retry with header rebuild
        let http = self.http_client.clone();
        let api_key = self.config.api_key.clone();
        let extra_headers = self.config.http_config.headers.clone();
        let url_for_retry = url.clone();
        let body_for_retry = request_body.clone();
        let build_request = move || {
            let mut headers = build_headers(&api_key, &extra_headers)?;
            crate::utils::http_headers::inject_tracing_headers(&mut headers);
            Ok(http
                .post(&url_for_retry)
                .headers(headers)
                .json(&body_for_retry))
        };

        let converter = GroqEventConverter::new();
        StreamFactory::create_eventsource_stream_with_retry("groq", build_request, converter).await
    }
}
