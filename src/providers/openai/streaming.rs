//! OpenAI streaming implementation using eventsource-stream
//!
//! This module provides OpenAI streaming functionality using the unified
//! eventsource-stream infrastructure for reliable UTF-8 and SSE handling.

use crate::error::LlmError;
use crate::providers::openai::config::OpenAiConfig;
use crate::stream::{ChatStream, ChatStreamEvent};
use crate::types::{ChatResponse, FinishReason, MessageContent, Usage};
use crate::utils::streaming::{SseEventConverter, StreamFactory};
use eventsource_stream::Event;
use serde::Deserialize;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

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
    finish_reason: Option<String>,
}

/// OpenAI stream delta
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct OpenAiStreamDelta {
    role: Option<String>,
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiToolCallDelta>>,
    thinking: Option<String>,
}

/// OpenAI tool call delta
#[derive(Debug, Clone, Deserialize)]
struct OpenAiToolCallDelta {
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
}

impl OpenAiEventConverter {
    pub fn new(config: OpenAiConfig) -> Self {
        Self { config }
    }

    /// Convert OpenAI stream event to ChatStreamEvent
    fn convert_openai_event(&self, event: OpenAiStreamEvent) -> ChatStreamEvent {
        // Handle choices first (prioritize content over usage)
        if let Some(choices) = event.choices {
            for choice in choices {
                if let Some(delta) = choice.delta {
                    // Handle content delta
                    if let Some(content) = delta.content {
                        return ChatStreamEvent::ContentDelta {
                            delta: content,
                            index: choice.index,
                        };
                    }

                    // Handle thinking content (for reasoning models)
                    if let Some(thinking) = delta.thinking {
                        return ChatStreamEvent::ThinkingDelta { delta: thinking };
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
                                        index: tool_call.index,
                                    };
                                }
                                if let Some(arguments) = function.arguments {
                                    return ChatStreamEvent::ToolCallDelta {
                                        id: tool_call.id.unwrap_or_default(),
                                        function_name: None,
                                        arguments_delta: Some(arguments),
                                        index: tool_call.index,
                                    };
                                }
                            }
                        }
                    }
                }

                // Handle finish reason
                if let Some(finish_reason) = choice.finish_reason {
                    let reason = match finish_reason.as_str() {
                        "stop" => FinishReason::Stop,
                        "length" => FinishReason::Length,
                        "tool_calls" => FinishReason::ToolCalls,
                        "content_filter" => FinishReason::ContentFilter,
                        _ => FinishReason::Other(finish_reason),
                    };

                    let response = ChatResponse {
                        id: event.id,
                        model: event.model,
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
        }

        // Handle usage information only if no content was found
        if let Some(usage) = event.usage {
            let usage_info = Usage {
                prompt_tokens: usage.prompt_tokens.unwrap_or(0),
                completion_tokens: usage.completion_tokens.unwrap_or(0),
                total_tokens: usage.total_tokens.unwrap_or(0),
                cached_tokens: usage
                    .prompt_tokens_details
                    .and_then(|details| details.cached_tokens),
                reasoning_tokens: usage
                    .completion_tokens_details
                    .and_then(|details| details.reasoning_tokens),
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

impl SseEventConverter for OpenAiEventConverter {
    fn convert_event(
        &self,
        event: Event,
    ) -> Pin<Box<dyn Future<Output = Option<Result<ChatStreamEvent, LlmError>>> + Send + Sync + '_>>
    {
        Box::pin(async move {
            match serde_json::from_str::<OpenAiStreamEvent>(&event.data) {
                Ok(openai_event) => Some(Ok(self.convert_openai_event(openai_event))),
                Err(e) => Some(Err(LlmError::ParseError(format!(
                    "Failed to parse OpenAI event: {e}"
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
