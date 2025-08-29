//! `xAI` Streaming Implementation
//!
//! Implements streaming chat completions for the `xAI` provider using eventsource-stream.

use crate::error::LlmError;
use crate::stream::{ChatStream, ChatStreamEvent};
use crate::types::{ChatRequest, ChatResponse, FinishReason, MessageContent, Usage};
use crate::utils::streaming::{SseEventConverter, StreamFactory};
use eventsource_stream::Event;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use super::config::XaiConfig;
use super::types::*;
use super::utils::*;

/// `xAI` event converter
#[derive(Clone)]
pub struct XaiEventConverter {
    #[allow(dead_code)]
    config: XaiConfig,
}

impl XaiEventConverter {
    pub fn new(config: XaiConfig) -> Self {
        Self { config }
    }

    /// Convert xAI stream event to ChatStreamEvent
    fn convert_xai_event(&self, event: XaiStreamChunk) -> ChatStreamEvent {
        // Handle choices first (prioritize content over usage)
        for choice in event.choices {
            let delta = choice.delta;

            // Handle content delta
            if let Some(content) = delta.content {
                return ChatStreamEvent::ContentDelta {
                    delta: content,
                    index: Some(choice.index as usize),
                };
            }

            // Handle reasoning content delta (xAI specific)
            if let Some(reasoning) = delta.reasoning_content {
                return ChatStreamEvent::ThinkingDelta { delta: reasoning };
            }

            // Handle tool calls
            if let Some(tool_calls) = delta.tool_calls {
                for tool_call in tool_calls {
                    if let Some(function) = tool_call.function {
                        if let Some(name) = function.name {
                            return ChatStreamEvent::ToolCallDelta {
                                id: tool_call.id.unwrap_or_default(),
                                function_name: Some(name),
                                arguments_delta: None,
                                index: Some(tool_call.index as usize),
                            };
                        }
                        if let Some(arguments) = function.arguments {
                            return ChatStreamEvent::ToolCallDelta {
                                id: tool_call.id.unwrap_or_default(),
                                function_name: None,
                                arguments_delta: Some(arguments),
                                index: Some(tool_call.index as usize),
                            };
                        }
                    }
                }
            }

            // Handle finish reason
            if let Some(finish_reason) = choice.finish_reason {
                let reason = parse_finish_reason(Some(&finish_reason));

                let response = ChatResponse {
                    id: Some(event.id),
                    model: Some(event.model),
                    content: MessageContent::Text("".to_string()),
                    usage: None,
                    finish_reason: Some(reason),
                    tool_calls: None,
                    thinking: None,
                    metadata: HashMap::new(),
                };

                return ChatStreamEvent::StreamEnd { response };
            }
        }

        // Handle usage information only if no content was found
        if let Some(usage) = event.usage {
            let usage_info = Usage {
                prompt_tokens: usage.prompt_tokens.unwrap_or(0),
                completion_tokens: usage.completion_tokens.unwrap_or(0),
                total_tokens: usage.total_tokens.unwrap_or(0),
                cached_tokens: None, // xAI doesn't provide cached tokens info
                reasoning_tokens: usage.reasoning_tokens,
            };
            return ChatStreamEvent::UsageUpdate { usage: usage_info };
        }

        // Default: empty content delta
        ChatStreamEvent::ContentDelta {
            delta: "".to_string(),
            index: None,
        }
    }
}

impl SseEventConverter for XaiEventConverter {
    fn convert_event(
        &self,
        event: Event,
    ) -> Pin<Box<dyn Future<Output = Option<Result<ChatStreamEvent, LlmError>>> + Send + Sync + '_>>
    {
        Box::pin(async move {
            match serde_json::from_str::<XaiStreamChunk>(&event.data) {
                Ok(xai_event) => Some(Ok(self.convert_xai_event(xai_event))),
                Err(e) => Some(Err(LlmError::ParseError(format!(
                    "Failed to parse xAI event: {e}"
                )))),
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

        // Use the same request building logic as non-streaming
        let chat_capability = super::chat::XaiChatCapability::new(
            self.config.api_key.clone(),
            self.config.base_url.clone(),
            self.http_client.clone(),
            self.config.http_config.clone(),
            self.config.common_params.clone(),
        );

        let mut request_body = chat_capability.build_chat_request_body(&request)?;

        // Override with streaming-specific settings
        request_body["stream"] = serde_json::Value::Bool(true);

        // Create headers
        let headers = build_headers(&self.config.api_key, &self.config.http_config.headers)?;

        // Create the stream using reqwest_eventsource for enhanced reliability
        let request_builder = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&request_body);

        let converter = XaiEventConverter::new(self.config);
        StreamFactory::create_eventsource_stream(request_builder, converter).await
    }
}
