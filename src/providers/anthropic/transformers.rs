//! Transformers for Anthropic Claude
//!
//! Centralizes request/response/stream chunk transformations to reduce duplication
//! across chat capability and streaming implementations.

use crate::error::LlmError;
use crate::stream::ChatStreamEvent;
use crate::transformers::{
    request::RequestTransformer, response::ResponseTransformer, stream::StreamChunkTransformer,
};
use crate::types::{ChatRequest, ChatResponse, FinishReason, MessageContent, ToolCall, Usage};
use crate::utils::streaming::SseEventConverter;
use eventsource_stream::Event;

use super::types::{AnthropicChatResponse, AnthropicSpecificParams};
use super::utils::{
    convert_messages as convert_messages_to_anthropic, convert_tools_to_anthropic_format,
    create_usage_from_response, extract_thinking_content, parse_finish_reason,
    parse_response_content_and_tools,
};
use crate::transformers::request::{
    GenericRequestTransformer, MappingProfile, ProviderRequestHooks, RangeMode, Rule,
};

/// Request transformer for Anthropic
#[derive(Clone, Default)]
pub struct AnthropicRequestTransformer {
    pub specific: Option<AnthropicSpecificParams>,
}

impl AnthropicRequestTransformer {
    pub fn new(specific: Option<AnthropicSpecificParams>) -> Self {
        Self { specific }
    }
}

impl RequestTransformer for AnthropicRequestTransformer {
    fn provider_id(&self) -> &str {
        "anthropic"
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        // Minimal stable validation
        if req.common_params.model.is_empty() {
            return Err(LlmError::InvalidParameter(
                "Model must be specified".to_string(),
            ));
        }

        // Hooks: build base body, then apply declarative rules
        struct AnthropicChatHooks<'a> {
            specific: Option<&'a AnthropicSpecificParams>,
        }
        impl<'a> ProviderRequestHooks for AnthropicChatHooks<'a> {
            fn build_base_chat_body(
                &self,
                req: &ChatRequest,
            ) -> Result<serde_json::Value, LlmError> {
                let (messages, system) = convert_messages_to_anthropic(&req.messages)?;
                let mut body = serde_json::json!({
                    "model": req.common_params.model,
                    "messages": messages,
                    // require max_tokens; default when not provided
                    "max_tokens": req.common_params.max_tokens.unwrap_or(4096),
                });
                if let Some(sys) = system {
                    body["system"] = serde_json::json!(sys);
                }
                if let Some(t) = req.common_params.temperature {
                    body["temperature"] = serde_json::json!(t);
                }
                if let Some(tp) = req.common_params.top_p {
                    body["top_p"] = serde_json::json!(tp);
                }
                if let Some(stops) = &req.common_params.stop_sequences {
                    body["stop_sequences"] = serde_json::json!(stops);
                }
                if let Some(tools) = &req.tools {
                    let arr = convert_tools_to_anthropic_format(tools)?;
                    if !arr.is_empty() {
                        body["tools"] = serde_json::Value::Array(arr);
                    }
                }
                if let Some(spec) = self.specific {
                    if let Some(thinking) = &spec.thinking_config {
                        body["thinking"] = thinking.to_request_params();
                    }
                    if let Some(meta) = &spec.metadata {
                        body["metadata"] = meta.clone();
                    }
                }
                if req.stream {
                    body["stream"] = serde_json::json!(true);
                }
                Ok(body)
            }

            fn post_process_chat(
                &self,
                req: &ChatRequest,
                body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                // Merge provider params with filtered keys to avoid overriding core fields
                if let Some(pp) = &req.provider_params {
                    if let Some(obj) = body.as_object_mut() {
                        for (k, v) in &pp.params {
                            if k == "model" || k == "messages" || k == "stream" {
                                continue;
                            }
                            if !v.is_null() {
                                obj.insert(k.clone(), v.clone());
                            }
                        }
                    }
                }
                Ok(())
            }
        }

        let hooks = AnthropicChatHooks {
            specific: self.specific.as_ref(),
        };
        let profile = MappingProfile {
            provider_id: "anthropic",
            rules: vec![
                // Stable ranges only
                Rule::Range {
                    field: "temperature",
                    min: 0.0,
                    max: 1.0,
                    mode: RangeMode::Error,
                    message: Some("Anthropic temperature must be between 0.0 and 1.0"),
                },
                Rule::Range {
                    field: "top_p",
                    min: 0.0,
                    max: 1.0,
                    mode: RangeMode::Error,
                    message: Some("Anthropic top_p must be between 0.0 and 1.0"),
                },
            ],
            // Not used directly; provider params merged via hooks with filtered keys
            merge_strategy: crate::transformers::request::ProviderParamsMergeStrategy::Flatten,
        };
        let generic = GenericRequestTransformer { profile, hooks };
        generic.transform_chat(req)
    }
}

/// Response transformer for Anthropic
#[derive(Clone, Default)]
pub struct AnthropicResponseTransformer;

impl ResponseTransformer for AnthropicResponseTransformer {
    fn provider_id(&self) -> &str {
        "anthropic"
    }

    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResponse, LlmError> {
        let response: AnthropicChatResponse = serde_json::from_value(raw.clone())
            .map_err(|e| LlmError::ParseError(format!("Invalid Anthropic response: {e}")))?;

        let (content, tool_calls): (MessageContent, Option<Vec<ToolCall>>) =
            parse_response_content_and_tools(&response.content);
        let usage: Option<Usage> = create_usage_from_response(response.usage.clone());
        let finish_reason: Option<FinishReason> =
            parse_finish_reason(response.stop_reason.as_deref());
        let thinking = extract_thinking_content(&response.content);

        Ok(ChatResponse {
            id: Some(response.id),
            model: Some(response.model),
            content,
            usage,
            finish_reason,
            tool_calls,
            thinking,
            metadata: std::collections::HashMap::new(),
        })
    }
}

/// Stream chunk transformer wrapping the existing AnthropicEventConverter
#[derive(Clone)]
pub struct AnthropicStreamChunkTransformer {
    pub provider_id: String,
    pub inner: super::streaming::AnthropicEventConverter,
}

impl StreamChunkTransformer for AnthropicStreamChunkTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn convert_event(
        &self,
        event: Event,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Vec<Result<ChatStreamEvent, LlmError>>>
                + Send
                + Sync
                + '_,
        >,
    > {
        self.inner.convert_event(event)
    }

    fn handle_stream_end(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
        self.inner.handle_stream_end()
    }
}
