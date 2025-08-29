//! Anthropic streaming implementation using eventsource-stream
//!
//! This module provides Anthropic streaming functionality using the
//! eventsource-stream infrastructure for reliable UTF-8 and SSE handling.

use crate::error::LlmError;
use crate::params::AnthropicParams;
use crate::stream::{ChatStream, ChatStreamEvent};
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
#[allow(dead_code)]
struct AnthropicStreamEvent {
    r#type: String,
    #[serde(default)]
    message: Option<AnthropicMessage>,
    #[serde(default)]
    delta: Option<AnthropicDelta>,
    #[serde(default)]
    usage: Option<AnthropicUsage>,
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    content_block: Option<serde_json::Value>,
}

/// Anthropic message structure
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct AnthropicMessage {
    id: Option<String>,
    model: Option<String>,
    role: Option<String>,
    content: Option<Vec<AnthropicContent>>,
    stop_reason: Option<String>,
}

/// Anthropic content structure
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct AnthropicContent {
    #[serde(rename = "type")]
    content_type: String,
    text: Option<String>,
}

/// Anthropic delta structure
/// Supports different delta types: text_delta, input_json_delta, thinking_delta, etc.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct AnthropicDelta {
    #[serde(rename = "type")]
    #[serde(default)]
    delta_type: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    partial_json: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
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
    config: AnthropicParams,
}

impl AnthropicEventConverter {
    pub fn new(config: AnthropicParams) -> Self {
        Self { config }
    }

    /// Convert Anthropic stream event to ChatStreamEvent
    fn convert_anthropic_event(&self, event: AnthropicStreamEvent) -> Option<ChatStreamEvent> {
        match event.r#type.as_str() {
            "message_start" => {
                // Extract metadata from message_start event
                if let Some(message) = event.message {
                    let metadata = ResponseMetadata {
                        id: message.id,
                        model: message.model,
                        created: Some(chrono::Utc::now()), // Anthropic doesn't provide created timestamp in stream
                        provider: "anthropic".to_string(),
                        request_id: None, // Anthropic doesn't provide request_id in stream events
                    };
                    Some(ChatStreamEvent::StreamStart { metadata })
                } else {
                    None
                }
            }
            "content_block_delta" => {
                if let Some(delta) = event.delta
                    && let Some(text) = delta.text
                {
                    return Some(ChatStreamEvent::ContentDelta {
                        delta: text,
                        index: None,
                    });
                }
                None
            }
            "message_delta" => {
                // Handle usage or finish reason from message_delta events
                let mut usage_info = None;
                let mut finish_reason = None;

                // Extract usage information
                if let Some(usage) = event.usage {
                    usage_info = Some(Usage {
                        prompt_tokens: usage.input_tokens.unwrap_or(0),
                        completion_tokens: usage.output_tokens.unwrap_or(0),
                        total_tokens: usage.input_tokens.unwrap_or(0)
                            + usage.output_tokens.unwrap_or(0),
                        cached_tokens: None,
                        reasoning_tokens: None,
                    });
                }

                // Extract finish reason from delta
                if let Some(delta) = event.delta
                    && let Some(stop_reason) = delta.stop_reason
                {
                    finish_reason = Some(match stop_reason.as_str() {
                        "end_turn" => FinishReason::Stop,
                        "max_tokens" => FinishReason::Length,
                        "stop_sequence" => FinishReason::Stop,
                        "tool_use" => FinishReason::ToolCalls,
                        _ => FinishReason::Stop,
                    });
                }

                // If we have a finish reason, emit StreamEnd event
                if let Some(reason) = finish_reason {
                    let response = ChatResponse {
                        id: None,
                        model: None,
                        content: MessageContent::Text("".to_string()),
                        usage: usage_info,
                        finish_reason: Some(reason),
                        tool_calls: None,
                        thinking: None,
                        metadata: HashMap::new(),
                    };
                    return Some(ChatStreamEvent::StreamEnd { response });
                }

                // Otherwise, just emit usage update if available
                if let Some(usage) = usage_info {
                    return Some(ChatStreamEvent::UsageUpdate { usage });
                }

                None
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
                Some(ChatStreamEvent::StreamEnd { response })
            }
            _ => None,
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
                Ok(anthropic_event) => {
                    if let Some(chat_event) = self.convert_anthropic_event(anthropic_event) {
                        vec![Ok(chat_event)]
                    } else {
                        vec![]
                    }
                }
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

    /// Merge provider-specific params into the request body, preserving core fields
    fn merge_provider_params_into_body(
        body: &mut serde_json::Value,
        request: &crate::types::ChatRequest,
    ) {
        if let Some(provider) = &request.provider_params
            && let serde_json::Value::Object(obj) = body
        {
            for (k, v) in &provider.params {
                if k == "stream" || k == "messages" || k == "model" {
                    continue;
                }
                obj.insert(k.clone(), v.clone());
            }
        }
    }

    /// Create a chat stream from ChatRequest
    pub async fn create_chat_stream(
        self,
        request: crate::types::ChatRequest,
    ) -> Result<ChatStream, LlmError> {
        // Build request body
        let (messages, system) = self.convert_messages(&request.messages)?;
        let mut request_body = serde_json::json!({
            "model": request.common_params.model,
            "messages": messages,
            "stream": true,
            "max_tokens": request.common_params.max_tokens.unwrap_or(1000)
        });

        // Add system message if present
        if let Some(system_content) = system {
            request_body["system"] = serde_json::Value::String(system_content);
        }

        // Add common parameters
        if let Some(temp) = request.common_params.temperature {
            request_body["temperature"] = temp.into();
        }
        if let Some(top_p) = request.common_params.top_p {
            request_body["top_p"] = top_p.into();
        }
        if let Some(stop) = &request.common_params.stop_sequences {
            request_body["stop_sequences"] = stop.clone().into();
        }

        // Add tools if provided
        if let Some(tools) = &request.tools {
            request_body["tools"] = self.convert_tools(tools)?;
        }

        // Merge provider-specific params if present (preserve core fields)
        if let Some(provider) = &request.provider_params
            && let serde_json::Value::Object(obj) = &mut request_body
        {
            for (k, v) in &provider.params {
                if k == "stream" || k == "messages" || k == "model" {
                    continue;
                }
                // Skip null values to prevent API errors
                if !v.is_null() {
                    obj.insert(k.clone(), v.clone());
                }
            }
        }

        // Merge provider-specific params before sending
        Self::merge_provider_params_into_body(&mut request_body, &request);

        // Create headers with authentication
        let headers = crate::utils::http_headers::ProviderHeaders::anthropic(
            &self.api_key,
            &self.http_config.headers,
        )?;

        // Build the API URL
        let url = crate::utils::url::join_url(&self.base_url, "/v1/messages");

        // Create the stream using reqwest_eventsource for enhanced reliability
        let request_builder = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&request_body);

        let converter = AnthropicEventConverter::new(self.config);
        StreamFactory::create_eventsource_stream(request_builder, converter).await
    }

    /// Convert messages to Anthropic format
    fn convert_messages(
        &self,
        messages: &[crate::types::ChatMessage],
    ) -> Result<(serde_json::Value, Option<String>), LlmError> {
        let mut anthropic_messages = Vec::new();
        let mut system_message = None;

        for msg in messages {
            match msg.role {
                crate::types::MessageRole::System => {
                    // Anthropic handles system messages separately
                    if let Some(text) = msg.content_text() {
                        system_message = Some(text.to_string());
                    }
                }
                crate::types::MessageRole::User => {
                    anthropic_messages.push(serde_json::json!({
                        "role": "user",
                        "content": msg.content_text().unwrap_or("")
                    }));
                }
                crate::types::MessageRole::Assistant => {
                    anthropic_messages.push(serde_json::json!({
                        "role": "assistant",
                        "content": msg.content_text().unwrap_or("")
                    }));
                }
                crate::types::MessageRole::Developer => {
                    // Developer messages are treated as system-level instructions
                    if let Some(text) = msg.content_text() {
                        let developer_text = format!("Developer instructions: {text}");
                        system_message = Some(match system_message {
                            Some(existing) => format!("{existing}\n\n{developer_text}"),
                            None => developer_text,
                        });
                    }
                }
                crate::types::MessageRole::Tool => {
                    // Tool results are handled as user messages in Anthropic
                    anthropic_messages.push(serde_json::json!({
                        "role": "user",
                        "content": msg.content_text().unwrap_or("")
                    }));
                }
            }
        }

        Ok((serde_json::Value::Array(anthropic_messages), system_message))
    }

    /// Convert tools to Anthropic format
    fn convert_tools(&self, tools: &[crate::types::Tool]) -> Result<serde_json::Value, LlmError> {
        let anthropic_tools: Vec<serde_json::Value> = tools
            .iter()
            .map(|tool| {
                serde_json::json!({
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "input_schema": tool.function.parameters
                })
            })
            .collect();

        Ok(serde_json::Value::Array(anthropic_tools))
    }
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

    #[test]
    fn test_merge_provider_params_into_body_preserves_core_fields_anthropic() {
        let request = crate::types::ChatRequest {
            messages: vec![],
            tools: None,
            common_params: crate::types::CommonParams {
                model: "claude-3-5-sonnet".to_string(),
                ..Default::default()
            },
            provider_params: Some(crate::types::ProviderParams {
                params: {
                    let mut m = std::collections::HashMap::new();
                    m.insert("tool_choice".to_string(), serde_json::json!("auto"));
                    m.insert("model".to_string(), serde_json::json!("override"));
                    m
                },
            }),
            http_config: None,
            web_search: None,
            stream: true,
        };

        let mut body = serde_json::json!({
            "model": request.common_params.model,
            "messages": [],
            "stream": true
        });

        super::AnthropicStreaming::merge_provider_params_into_body(&mut body, &request);

        assert_eq!(body["model"], serde_json::json!("claude-3-5-sonnet"));
        assert_eq!(body["messages"], serde_json::json!([]));
        assert_eq!(body["stream"], serde_json::json!(true));
        assert_eq!(body["tool_choice"], serde_json::json!("auto"));
    }

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
