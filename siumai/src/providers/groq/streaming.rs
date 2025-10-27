//! `Groq` Streaming Implementation
//!
//! Provides SSE event conversion for Groq streaming responses.
//! The legacy GroqStreaming client has been removed in favor of the unified HttpChatExecutor.

use crate::error::LlmError;
use crate::streaming::SseEventConverter;
use crate::streaming::{ChatStreamEvent, StreamStateTracker};
use crate::types::{ChatResponse, FinishReason, MessageContent};
use crate::types::{ResponseMetadata, Usage};
use eventsource_stream::Event;

use std::future::Future;
use std::pin::Pin;

use super::types::*;

/// Groq event converter for SSE events
#[derive(Clone)]
pub struct GroqEventConverter {
    /// Track if StreamStart has been emitted
    state_tracker: StreamStateTracker,
}

impl GroqEventConverter {
    /// Create a new Groq event converter
    pub fn new() -> Self {
        Self {
            state_tracker: StreamStateTracker::new(),
        }
    }

    /// Convert Groq stream response to multiple ChatStreamEvents
    async fn convert_groq_response_async(
        &self,
        response: GroqChatStreamChunk,
    ) -> Vec<ChatStreamEvent> {
        use crate::streaming::EventBuilder;

        let mut builder = EventBuilder::new();

        // Check if we need to emit StreamStart first
        if self.needs_stream_start() {
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

        // Process finish_reason -> StreamEnd
        if let Some(finish_reason_str) = response
            .choices
            .first()
            .and_then(|choice| choice.finish_reason.as_ref())
        {
            let finish_reason = match finish_reason_str.as_str() {
                "stop" => FinishReason::Stop,
                "length" => FinishReason::Length,
                "tool_calls" => FinishReason::ToolCalls,
                "content_filter" => FinishReason::ContentFilter,
                _ => FinishReason::Stop,
            };

            // Mark that StreamEnd is being emitted
            self.state_tracker.mark_stream_ended();

            let response = ChatResponse {
                id: None,
                model: None,
                content: MessageContent::Text("".to_string()),
                usage: None,
                finish_reason: Some(finish_reason),
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
        response.usage.as_ref().map(|usage| {
            Usage::builder()
                .prompt_tokens(usage.prompt_tokens.unwrap_or(0))
                .completion_tokens(usage.completion_tokens.unwrap_or(0))
                .total_tokens(usage.total_tokens.unwrap_or(0))
                .build()
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
            match crate::streaming::parse_json_with_repair::<GroqChatStreamChunk>(&event.data) {
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
        // Groq normally emits finish_reason in the stream.
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

// Legacy GroqStreaming client has been removed in favor of the unified HttpChatExecutor.
// The GroqEventConverter is still used for SSE event conversion in tests.
