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

        struct XaiChatHooks;
        impl crate::transformers::request::ProviderRequestHooks for XaiChatHooks {
            fn build_base_chat_body(
                &self,
                req: &ChatRequest,
            ) -> Result<serde_json::Value, LlmError> {
                let mut body = serde_json::json!({ "model": req.common_params.model });
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
                let messages = super::utils::convert_messages(&req.messages)?;
                body["messages"] = serde_json::to_value(messages)?;
                if let Some(tools) = &req.tools
                    && !tools.is_empty()
                {
                    body["tools"] = serde_json::to_value(tools)?;
                }
                body["stream"] = serde_json::json!(req.stream);
                Ok(body)
            }
            fn post_process_chat(
                &self,
                req: &ChatRequest,
                body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                if let Some(pp) = &req.provider_params {
                    if let Some(so) = pp
                        .params
                        .get("structured_output")
                        .and_then(|v| v.as_object())
                    {
                        let mut rf: Option<serde_json::Value> = None;
                        if let Some(schema_v) = so.get("schema") {
                            if let Some(n) = so.get("name").and_then(|v| v.as_str()) {
                                rf = Some(serde_json::json!({
                                    "type": "json_schema",
                                    "json_schema": {"name": n, "schema": schema_v, "strict": true}
                                }));
                            } else {
                                rf = Some(serde_json::json!({
                                    "type": "json_schema",
                                    "json_schema": {"schema": schema_v, "strict": true}
                                }));
                            }
                        } else if let Some(t) = so.get("type").and_then(|v| v.as_str()) {
                            if t.eq_ignore_ascii_case("json")
                                || t.eq_ignore_ascii_case("json_object")
                            {
                                rf = Some(serde_json::json!({"type": "json_object"}));
                            }
                        }
                        if let Some(val) = rf {
                            body["response_format"] = val;
                        }
                    }
                }
                Ok(())
            }
        }
        let hooks = XaiChatHooks;
        let profile = crate::transformers::request::MappingProfile {
            provider_id: "xai",
            rules: vec![
                crate::transformers::request::Rule::Range {
                    field: "temperature",
                    min: 0.0,
                    max: 2.0,
                    mode: crate::transformers::request::RangeMode::Error,
                    message: None,
                },
                crate::transformers::request::Rule::Range {
                    field: "top_p",
                    min: 0.0,
                    max: 1.0,
                    mode: crate::transformers::request::RangeMode::Error,
                    message: None,
                },
                crate::transformers::request::Rule::MergeProviderParams {
                    strategy: crate::transformers::request::ProviderParamsMergeStrategy::Flatten,
                },
            ],
            merge_strategy: crate::transformers::request::ProviderParamsMergeStrategy::Flatten,
        };
        let generic = crate::transformers::request::GenericRequestTransformer { profile, hooks };
        generic.transform_chat(req)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, ChatRequest, CommonParams, MessageContent, MessageRole, ProviderParams};

    #[test]
    fn maps_structured_output_to_response_format_schema() {
        let tx = XaiRequestTransformer;
        let request = ChatRequest {
            messages: vec![ChatMessage { role: MessageRole::User, content: MessageContent::Text("hello".into()), metadata: Default::default(), tool_calls: None, tool_call_id: None }],
            tools: None,
            common_params: CommonParams { model: "grok-beta".into(), ..Default::default() },
            provider_params: Some(ProviderParams::new().with_param(
                "structured_output",
                serde_json::json!({"schema": {"type":"object"}}),
            )),
            http_config: None,
            web_search: None,
            stream: false,
            telemetry: None,
        };
        let body = tx.transform_chat(&request).expect("ok");
        assert_eq!(body.get("response_format").and_then(|v| v.get("type")).and_then(|v| v.as_str()), Some("json_schema"));
    }

    #[test]
    fn maps_structured_output_to_response_format_json_object() {
        let tx = XaiRequestTransformer;
        let request = ChatRequest {
            messages: vec![ChatMessage { role: MessageRole::User, content: MessageContent::Text("hello".into()), metadata: Default::default(), tool_calls: None, tool_call_id: None }],
            tools: None,
            common_params: CommonParams { model: "grok-beta".into(), ..Default::default() },
            provider_params: Some(ProviderParams::new().with_param(
                "structured_output",
                serde_json::json!({"type": "json_object"}),
            )),
            http_config: None,
            web_search: None,
            stream: false,
            telemetry: None,
        };
        let body = tx.transform_chat(&request).expect("ok");
        assert_eq!(body.get("response_format").and_then(|v| v.get("type")).and_then(|v| v.as_str()), Some("json_object"));
    }
}
