//! Anthropic streaming implementation using eventsource-stream
//!
//! This module provides Anthropic streaming functionality using the
//! eventsource-stream infrastructure for reliable UTF-8 and SSE handling.

use crate::error::LlmError;
use crate::params::AnthropicParams;
use crate::stream::{ChatStream, ChatStreamEvent};
use crate::transformers::request::RequestTransformer;
use crate::types::{ChatResponse, FinishReason, MessageContent, ResponseMetadata, Usage};
use crate::utils::streaming::{SseEventConverter, StreamFactory};
use eventsource_stream::Event;
use serde::Deserialize;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

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
    usage: Option<AnthropicUsage>,
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
    stop_reason: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    // Stop sequence token for deltas; usage is reflected via finish events elsewhere
    stop_sequence: Option<String>,
}

/// Anthropic usage structure
#[derive(Debug, Clone, Deserialize)]
struct AnthropicUsage {
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
}

/// Anthropic event converter
#[derive(Clone)]
pub struct AnthropicEventConverter {
    #[allow(dead_code)]
    // Retained for potential future behavior toggles; not read in the current converter
    config: AnthropicParams,
}

impl AnthropicEventConverter {
    pub fn new(config: AnthropicParams) -> Self {
        Self { config }
    }

    /// Convert Anthropic stream event to one or more ChatStreamEvents
    fn convert_anthropic_event(&self, event: AnthropicStreamEvent) -> Vec<ChatStreamEvent> {
        use crate::utils::streaming::EventBuilder;

        match event.r#type.as_str() {
            "message_start" => {
                if let Some(message) = event.message {
                    let metadata = ResponseMetadata {
                        id: message.id,
                        model: message.model,
                        created: Some(chrono::Utc::now()),
                        provider: "anthropic".to_string(),
                        request_id: None,
                    };
                    EventBuilder::new().add_stream_start(metadata).build()
                } else {
                    vec![]
                }
            }
            "content_block_delta" => {
                let mut builder = EventBuilder::new();
                if let Some(delta) = event.delta {
                    if let Some(text) = delta.text {
                        builder = builder.add_content_delta(text, None);
                    }
                    if let Some(thinking) = delta.thinking {
                        builder = builder.add_thinking_delta(thinking);
                    }
                }
                builder.build()
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
                    let usage_info = Usage {
                        prompt_tokens: usage.input_tokens.unwrap_or(0),
                        completion_tokens: usage.output_tokens.unwrap_or(0),
                        total_tokens: usage.input_tokens.unwrap_or(0)
                            + usage.output_tokens.unwrap_or(0),
                        cached_tokens: None,
                        reasoning_tokens: None,
                    };
                    builder = builder.add_usage_update(usage_info);
                }

                // Finish reason -> StreamEnd
                if let Some(delta) = &event.delta
                    && let Some(stop_reason) = &delta.stop_reason
                {
                    let reason = match stop_reason.as_str() {
                        "end_turn" => FinishReason::Stop,
                        "max_tokens" => FinishReason::Length,
                        "stop_sequence" => FinishReason::Stop,
                        "tool_use" => FinishReason::ToolCalls,
                        _ => FinishReason::Stop,
                    };

                    let response = ChatResponse {
                        id: None,
                        model: None,
                        content: MessageContent::Text("".to_string()),
                        usage: None, // usage already emitted as UsageUpdate above if present
                        finish_reason: Some(reason),
                        tool_calls: None,
                        thinking: None,
                        metadata: HashMap::new(),
                    };
                    builder = builder.add_stream_end(response);
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
                    tool_calls: None,
                    thinking: None,
                    metadata: HashMap::new(),
                };
                EventBuilder::new().add_stream_end(response).build()
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

                            return vec![Err(LlmError::ApiError {
                                code: 0, // Unknown status code from SSE
                                message: format!("Anthropic API error: {}", error_message),
                                details: Some(error_obj.clone()),
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

/// Anthropic streaming client
#[derive(Clone)]
pub struct AnthropicStreaming {
    config: AnthropicParams,
    http_client: reqwest::Client,
    api_key: String,
    base_url: String,
    http_config: crate::types::HttpConfig,
}

impl AnthropicStreaming {
    /// Create a new Anthropic streaming client
    pub fn new(
        config: AnthropicParams,
        http_client: reqwest::Client,
        api_key: String,
        base_url: String,
        http_config: crate::types::HttpConfig,
    ) -> Self {
        Self {
            config,
            http_client,
            api_key,
            base_url,
            http_config,
        }
    }

    // Legacy request merging helper removed; handled by Transformers

    /// Create a chat stream from ChatRequest
    pub async fn create_chat_stream(
        self,
        request: crate::types::ChatRequest,
    ) -> Result<ChatStream, LlmError> {
        // Build request body via transformer
        let transformer = super::transformers::AnthropicRequestTransformer::new(None);
        let mut request_body = transformer.transform_chat(&request)?;
        request_body["stream"] = serde_json::Value::Bool(true);

        // Create headers with authentication
        let headers = crate::utils::http_headers::ProviderHeaders::anthropic(
            &self.api_key,
            &self.http_config.headers,
        )?;

        // Build the API URL
        let url = crate::utils::url::join_url(&self.base_url, "/v1/messages");

        // Build closure for one-shot 401 retry with header rebuild
        let http = self.http_client.clone();
        let base_headers = self.http_config.headers.clone();
        let api_key = self.api_key.clone();
        let url_for_retry = url.clone();
        let body_for_retry = request_body.clone();
        let build_request = move || {
            let mut headers =
                crate::utils::http_headers::ProviderHeaders::anthropic(&api_key, &base_headers)?;
            crate::utils::http_headers::inject_tracing_headers(&mut headers);
            Ok(http
                .post(url_for_retry.clone())
                .headers(headers)
                .json(&body_for_retry))
        };

        let converter = AnthropicEventConverter::new(self.config);
        StreamFactory::create_eventsource_stream_with_retry("anthropic", build_request, converter)
            .await
    }

    // Legacy message conversion helper removed; handled by Transformers

    // Legacy tools conversion helper removed; handled by Transformers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::AnthropicParams;
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
}
