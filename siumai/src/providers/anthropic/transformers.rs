//! Transformers for Anthropic Claude (legacy path)
//!
//! NOTE:
//! - The primary Anthropic integration is now implemented via the
//!   `siumai-std-anthropic` crate and the core execution pipeline
//!   (`AnthropicChatStandard` + `anthropic_like_chat_request_to_core_input`).
//! - This module is kept as a compatibility layer and fallback for builds
//!   where `std-anthropic-external` is not available. New features should
//!   be added to the std/core/provider path instead of extending these
//!   legacy transformers.

use crate::error::LlmError;
use crate::execution::transformers::{
    request::RequestTransformer, response::ResponseTransformer, stream::StreamChunkTransformer,
};
use crate::streaming::ChatStreamEvent;
use crate::streaming::SseEventConverter;
use crate::types::{ChatRequest, ChatResponse, FinishReason, Usage};
use eventsource_stream::Event;

use super::types::{AnthropicChatResponse, AnthropicSpecificParams};
use super::utils::{
    convert_messages as convert_messages_to_anthropic, convert_tools_to_anthropic_format,
    core_parsed_to_message_content, create_usage_from_response, parse_finish_reason,
};
use crate::execution::transformers::request::{
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

                        // Add tool_choice if specified
                        if let Some(choice) = &req.tool_choice
                            && let Some(anthropic_choice) =
                                crate::providers::anthropic::utils::convert_tool_choice(choice)
                        {
                            body["tool_choice"] = anthropic_choice;
                        }
                        // If None is returned, tools should be removed (handled by caller if needed)
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
                _req: &ChatRequest,
                _body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                // All provider-specific features are now handled via provider_options
                // in ProviderSpec::chat_before_send()
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
            merge_strategy:
                crate::execution::transformers::request::ProviderParamsMergeStrategy::Flatten,
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
        // Allow adapter to adjust the raw JSON first.
        let resp = raw.clone();
        // Legacy adapters (if any) may still be wired through the provider spec.
        // Currently there is no adapter for the in-crate path, so this is a no-op.
        // Kept for forward-compatibility with potential proxy variants.
        // (No adapter type exists here, unlike std-anthropic; this is a pure legacy path.)

        let response: AnthropicChatResponse = serde_json::from_value(resp.clone())
            .map_err(|e| LlmError::ParseError(format!("Invalid Anthropic response: {e}")))?;

        // Reuse std-anthropic core parser to obtain a unified content model
        // (text + tool_calls + thinking), then convert it into the aggregator's
        // MessageContent structure.
        let parsed_core = siumai_std_anthropic::anthropic::utils::parse_content_blocks_core(&resp);
        let content = core_parsed_to_message_content(&parsed_core);

        let usage: Option<Usage> = create_usage_from_response(response.usage.clone());
        let finish_reason: Option<FinishReason> =
            parse_finish_reason(response.stop_reason.as_deref());

        Ok(ChatResponse {
            id: Some(response.id),
            model: Some(response.model),
            content,
            usage,
            finish_reason,
            audio: None, // Anthropic doesn't support audio output
            system_fingerprint: None,
            service_tier: None,
            warnings: None,
            provider_metadata: None,
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

// Tests for structured_output via provider_params have been removed
// as this functionality is now handled via provider_options in ProviderSpec::chat_before_send()
