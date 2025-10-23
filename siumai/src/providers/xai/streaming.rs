//! `xAI` Streaming Implementation
//!
//! Implements streaming chat completions for the `xAI` provider using eventsource-stream.

use crate::error::LlmError;
use crate::streaming::{ChatStream, ChatStreamEvent, StreamStateTracker};
use crate::streaming::{SseEventConverter, StreamFactory};
use crate::types::{ChatRequest, ChatResponse, FinishReason, MessageContent, ResponseMetadata};
use eventsource_stream::Event;

use std::future::Future;
use std::pin::Pin;

use super::config::XaiConfig;
use super::types::*;
use super::utils::*;
use crate::transformers::request::RequestTransformer;

/// `xAI` event converter
#[derive(Clone)]
pub struct XaiEventConverter {
    /// Track if StreamStart has been emitted
    state_tracker: StreamStateTracker,
}

impl XaiEventConverter {
    pub fn new() -> Self {
        Self {
            state_tracker: StreamStateTracker::new(),
        }
    }

    /// Convert xAI stream event to multiple ChatStreamEvents
    async fn convert_xai_event_async(&self, event: XaiStreamChunk) -> Vec<ChatStreamEvent> {
        use crate::streaming::EventBuilder;

        let mut builder = EventBuilder::new();

        // Check if we need to emit StreamStart first
        if self.needs_stream_start() {
            let metadata = self.create_stream_start_metadata(&event);
            builder = builder.add_stream_start(metadata);
        }

        // Process content - NO MORE CONTENT LOSS!
        if let Some(content) = self.extract_content(&event) {
            builder = builder.add_content_delta(content, self.extract_choice_index(&event));
        }

        // Process thinking content
        if let Some(thinking) = self.extract_thinking(&event) {
            builder = builder.add_thinking_delta(thinking);
        }

        // Process usage updates
        if let Some(usage) = self.extract_usage(&event) {
            builder = builder.add_usage_update(usage);
        }

        // Process stream end (when finish_reason is present)
        if let Some(finish_reason) = self.extract_finish_reason(&event) {
            // Mark that StreamEnd is being emitted
            self.state_tracker.mark_stream_ended();

            let chat_response = ChatResponse {
                id: Some(event.id.clone()),
                model: Some(event.model.clone()),
                content: MessageContent::Text(String::new()),
                usage: self.extract_usage(&event),
                finish_reason: Some(finish_reason),
                audio: None,
                system_fingerprint: None,
                service_tier: None,
                warnings: None,
                provider_metadata: None,
            };
            builder = builder.add_stream_end(chat_response);
        }

        builder.build()
    }

    /// Check if StreamStart event needs to be emitted
    fn needs_stream_start(&self) -> bool {
        self.state_tracker.needs_stream_start()
    }

    /// Extract content from xAI event
    fn extract_content(&self, event: &XaiStreamChunk) -> Option<String> {
        event
            .choices
            .first()?
            .delta
            .content
            .as_ref()
            .filter(|content| !content.is_empty())
            .cloned()
    }

    /// Extract thinking content
    fn extract_thinking(&self, event: &XaiStreamChunk) -> Option<String> {
        event
            .choices
            .first()?
            .delta
            .reasoning_content
            .as_ref()
            .filter(|thinking| !thinking.is_empty())
            .cloned()
    }

    /// Extract choice index
    fn extract_choice_index(&self, event: &XaiStreamChunk) -> Option<usize> {
        Some(event.choices.first()?.index as usize)
    }

    /// Extract usage information
    fn extract_usage(&self, event: &XaiStreamChunk) -> Option<crate::types::Usage> {
        let usage = event.usage.as_ref()?;
        let mut builder = crate::types::Usage::builder()
            .prompt_tokens(usage.prompt_tokens.unwrap_or(0))
            .completion_tokens(usage.completion_tokens.unwrap_or(0))
            .total_tokens(usage.total_tokens.unwrap_or(0));

        if let Some(reasoning) = usage.reasoning_tokens {
            builder = builder.with_reasoning_tokens(reasoning);
        }

        Some(builder.build())
    }

    /// Extract finish reason
    fn extract_finish_reason(&self, event: &XaiStreamChunk) -> Option<FinishReason> {
        let finish_reason_str = event.choices.first()?.finish_reason.as_ref()?;
        match finish_reason_str.as_str() {
            "stop" => Some(FinishReason::Stop),
            "length" => Some(FinishReason::Length),
            "tool_calls" => Some(FinishReason::ToolCalls),
            "content_filter" => Some(FinishReason::ContentFilter),
            _ => Some(FinishReason::Stop),
        }
    }

    /// Create StreamStart metadata
    fn create_stream_start_metadata(&self, event: &XaiStreamChunk) -> ResponseMetadata {
        ResponseMetadata {
            id: Some(event.id.clone()),
            model: Some(event.model.clone()),
            created: Some(
                chrono::DateTime::from_timestamp(event.created as i64, 0)
                    .unwrap_or_else(chrono::Utc::now),
            ),
            provider: "xai".to_string(),
            request_id: None,
        }
    }
}

impl Default for XaiEventConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl SseEventConverter for XaiEventConverter {
    fn convert_event(
        &self,
        event: Event,
    ) -> Pin<Box<dyn Future<Output = Vec<Result<ChatStreamEvent, LlmError>>> + Send + Sync + '_>>
    {
        Box::pin(async move {
            match crate::streaming::parse_json_with_repair::<XaiStreamChunk>(&event.data) {
                Ok(xai_event) => self
                    .convert_xai_event_async(xai_event)
                    .await
                    .into_iter()
                    .map(Ok)
                    .collect(),
                Err(e) => {
                    vec![Err(LlmError::ParseError(format!(
                        "Failed to parse xAI event: {e}"
                    )))]
                }
            }
        })
    }

    fn handle_stream_end(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
        // xAI normally emits finish_reason in the stream (handled in convert_xai_event_async).
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
            audio: None,
            system_fingerprint: None,
            service_tier: None,
            warnings: None,
            provider_metadata: None,
        };
        Some(Ok(ChatStreamEvent::StreamEnd { response }))
    }
}

/// `xAI` Streaming Client
#[derive(Clone)]
pub struct XaiStreaming {
    config: XaiConfig,
    http_client: reqwest::Client,
}

impl XaiStreaming {
    /// Create a new `xAI` streaming client
    pub const fn new(config: XaiConfig, http_client: reqwest::Client) -> Self {
        Self {
            config,
            http_client,
        }
    }

    /// Create a chat stream from ChatRequest
    pub async fn create_chat_stream(self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        let url = format!("{}/chat/completions", self.config.base_url);

        // Build request body via transformer
        let transformer = super::transformers::XaiRequestTransformer;
        let mut request_body = transformer.transform_chat(&request)?;

        // Override with streaming-specific settings
        request_body["stream"] = serde_json::Value::Bool(true);

        // Build closure for one-shot 401 retry with header rebuild
        let http = self.http_client.clone();
        let api_key = self.config.api_key.clone();
        let extra = self.config.http_config.headers.clone();
        let url_for_retry = url.clone();
        let body_for_retry = request_body.clone();
        let build_request = move || {
            use secrecy::ExposeSecret;
            let headers = build_headers(api_key.expose_secret(), &extra)?;
            Ok(http
                .post(&url_for_retry)
                .headers(headers)
                .json(&body_for_retry))
        };

        let converter = XaiEventConverter::new();
        StreamFactory::create_eventsource_stream_with_retry("xai", true, build_request, converter)
            .await
    }
}
