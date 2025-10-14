//! Transformers for xAI Chat
//!
//! Centralizes request/response transformation for xAI to reduce duplication.

use crate::error::LlmError;
use crate::transformers::{
    request::RequestTransformer, response::ResponseTransformer, stream::StreamChunkTransformer,
};
use crate::types::{ChatRequest, ChatResponse, FunctionCall, MessageContent, ToolCall, Usage};
use crate::utils::streaming::SseEventConverter;
use eventsource_stream::Event;
use std::future::Future;
use std::pin::Pin;

/// Request transformer for xAI Chat
#[derive(Clone)]
pub struct XaiRequestTransformer;

impl RequestTransformer for XaiRequestTransformer {
    fn provider_id(&self) -> &str {
        "xai"
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        if req.common_params.model.is_empty() {
            return Err(LlmError::InvalidParameter(
                "Model must be specified".to_string(),
            ));
        }

        // Base body from common params
        let mut body = serde_json::json!({
            "model": req.common_params.model,
        });
        if let Some(t) = req.common_params.temperature {
            body["temperature"] = serde_json::json!(t);
        }
        if let Some(tp) = req.common_params.top_p {
            body["top_p"] = serde_json::json!(tp);
        }
        if let Some(max) = req.common_params.max_tokens {
            body["max_tokens"] = serde_json::json!(max);
        }
        if let Some(stops) = &req.common_params.stop_sequences {
            body["stop"] = serde_json::json!(stops);
        }

        // Messages via xAI utils
        let messages = super::utils::convert_messages(&req.messages)?;
        body["messages"] = serde_json::to_value(messages)?;

        // Tools if provided
        if let Some(tools) = &req.tools
            && !tools.is_empty()
        {
            body["tools"] = serde_json::to_value(tools)?;
        }

        // Stream flag
        body["stream"] = serde_json::json!(req.stream);

        // Merge provider_params if present (flat merge)
        if let Some(pp) = &req.provider_params
            && let Some(obj) = body.as_object_mut()
        {
            for (k, v) in &pp.params {
                obj.insert(k.clone(), v.clone());
            }
        }

        Ok(body)
    }
}

/// Response transformer for xAI Chat
#[derive(Clone)]
pub struct XaiResponseTransformer;

impl ResponseTransformer for XaiResponseTransformer {
    fn provider_id(&self) -> &str {
        "xai"
    }

    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResponse, LlmError> {
        let response: super::types::XaiChatResponse = serde_json::from_value(raw.clone())
            .map_err(|e| LlmError::ParseError(format!("Invalid xAI response: {e}")))?;

        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::ApiError {
                code: 500,
                message: "No choices in response".to_string(),
                details: None,
            })?;

        // Extract thinking content first
        let mut thinking_content: Option<String> = choice.message.reasoning_content.clone();

        let content = if let Some(content) = choice.message.content {
            match content {
                serde_json::Value::String(text) => {
                    if thinking_content.is_none() && super::utils::contains_thinking_tags(&text) {
                        thinking_content = super::utils::extract_thinking_content(&text);
                        MessageContent::Text(super::utils::filter_thinking_content(&text))
                    } else {
                        MessageContent::Text(text)
                    }
                }
                serde_json::Value::Array(parts) => {
                    let mut content_parts = Vec::new();
                    for part in parts {
                        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                            if thinking_content.is_none()
                                && super::utils::contains_thinking_tags(text)
                            {
                                thinking_content = super::utils::extract_thinking_content(text);
                                let filtered = super::utils::filter_thinking_content(text);
                                if !filtered.is_empty() {
                                    content_parts
                                        .push(crate::types::ContentPart::Text { text: filtered });
                                }
                            } else {
                                content_parts.push(crate::types::ContentPart::Text {
                                    text: text.to_string(),
                                });
                            }
                        }
                    }
                    MessageContent::MultiModal(content_parts)
                }
                _ => MessageContent::Text(String::new()),
            }
        } else {
            MessageContent::Text(String::new())
        };

        let tool_calls = choice.message.tool_calls.map(|calls| {
            calls
                .into_iter()
                .map(|call| ToolCall {
                    id: call.id,
                    r#type: call.r#type,
                    function: call.function.map(|f| FunctionCall {
                        name: f.name,
                        arguments: f.arguments,
                    }),
                })
                .collect()
        });

        let finish_reason = Some(super::utils::parse_finish_reason(
            choice.finish_reason.as_deref(),
        ));

        let usage = response.usage.map(|u| Usage {
            prompt_tokens: u.prompt_tokens.unwrap_or(0),
            completion_tokens: u.completion_tokens.unwrap_or(0),
            total_tokens: u.total_tokens.unwrap_or(0),
            reasoning_tokens: u.reasoning_tokens,
            cached_tokens: u.prompt_tokens_details.and_then(|d| d.cached_tokens),
        });

        Ok(ChatResponse {
            id: Some(response.id),
            content,
            model: Some(response.model),
            usage,
            finish_reason,
            tool_calls,
            thinking: thinking_content,
            metadata: std::collections::HashMap::new(),
        })
    }
}

/// Stream transformer wrapper for xAI
#[derive(Clone)]
pub struct XaiStreamChunkTransformer {
    pub provider_id: String,
    pub inner: super::streaming::XaiEventConverter,
}

impl StreamChunkTransformer for XaiStreamChunkTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }
    fn convert_event(
        &self,
        event: Event,
    ) -> Pin<
        Box<
            dyn Future<Output = Vec<Result<crate::stream::ChatStreamEvent, LlmError>>>
                + Send
                + Sync
                + '_,
        >,
    > {
        self.inner.convert_event(event)
    }
    fn handle_stream_end(&self) -> Option<Result<crate::stream::ChatStreamEvent, LlmError>> {
        self.inner.handle_stream_end()
    }
}
