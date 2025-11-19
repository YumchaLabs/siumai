use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::http::headers::ProviderHeaders;
use crate::std_anthropic::anthropic::chat::AnthropicChatStandard;
use crate::traits::ProviderCapabilities;
use crate::types::ChatRequest;
use reqwest::header::HeaderMap;

/// Anthropic ProviderSpec implementation
///
/// This Spec uses the Anthropic standard from the standards layer,
/// with additional support for Anthropic-specific features like
/// Prompt Caching and Thinking Mode.
#[derive(Clone, Default)]
pub struct AnthropicSpec {
    /// Standard Anthropic Chat implementation
    chat_standard: AnthropicChatStandard,
}

impl AnthropicSpec {
    pub fn new() -> Self {
        Self {
            chat_standard: AnthropicChatStandard::new(),
        }
    }
}

impl ProviderSpec for AnthropicSpec {
    fn id(&self) -> &'static str {
        "anthropic"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_custom_feature("prompt_caching", true)
            .with_custom_feature("thinking_mode", true)
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        // External mode: delegate to provider crate core-spec.
        #[cfg(feature = "provider-anthropic-external")]
        {
            use siumai_core::provider_spec::CoreProviderSpec;

            let core_ctx = ctx.to_core_context();
            let core_spec = siumai_provider_anthropic::AnthropicCoreSpec::new();
            return core_spec.build_headers(&core_ctx);
        }

        // Default: use aggregator Anthropic header helper.
        #[cfg(not(feature = "provider-anthropic-external"))]
        {
            let api_key = ctx
                .api_key
                .as_ref()
                .ok_or_else(|| LlmError::MissingApiKey("Anthropic API key not provided".into()))?;
            return ProviderHeaders::anthropic(api_key, &ctx.http_extra_headers);
        }
    }

    fn chat_url(&self, _stream: bool, _req: &ChatRequest, ctx: &ProviderContext) -> String {
        // External mode: delegate to provider crate core-spec.
        #[cfg(feature = "provider-anthropic-external")]
        {
            use siumai_core::provider_spec::CoreProviderSpec;

            let core_ctx = ctx.to_core_context();
            let core_spec = siumai_provider_anthropic::AnthropicCoreSpec::new();
            return core_spec.chat_url(&core_ctx);
        }

        // Fallback: keep existing in-crate behavior.
        #[cfg(not(feature = "provider-anthropic-external"))]
        {
            format!("{}/v1/messages", ctx.base_url.trim_end_matches('/'))
        }
    }

    fn choose_chat_transformers(
        &self,
        _req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        #[cfg(feature = "provider-anthropic-external")]
        {
            use crate::core::provider_spec::{
                anthropic_like_chat_request_to_core_input, anthropic_like_map_core_stream_event,
                bridge_core_chat_transformers,
            };
            use siumai_core::provider_spec::{CoreChatTransformers, CoreProviderSpec};

            let core_ctx = ctx.to_core_context();
            let core_input = anthropic_like_chat_request_to_core_input(req);

            let core_spec = siumai_provider_anthropic::AnthropicCoreSpec::new();
            let core_txs: CoreChatTransformers =
                core_spec.choose_chat_transformers(&core_input, &core_ctx);

            bridge_core_chat_transformers(
                core_txs,
                anthropic_like_chat_request_to_core_input,
                |evt| anthropic_like_map_core_stream_event("anthropic", evt),
            )
        }

        #[cfg(all(
            not(feature = "provider-anthropic-external"),
            feature = "std-anthropic-external"
        ))]
        {
            use crate::core::provider_spec::{
                anthropic_like_chat_request_to_core_input, anthropic_like_map_core_stream_event,
                bridge_core_chat_transformers,
            };
            use siumai_core::provider_spec::CoreChatTransformers;

            let core_txs: CoreChatTransformers = CoreChatTransformers {
                request: self
                    .chat_standard
                    .create_request_transformer(&ctx.provider_id),
                response: self
                    .chat_standard
                    .create_response_transformer(&ctx.provider_id),
                stream: Some(self.chat_standard.create_stream_converter(&ctx.provider_id)),
            };

            bridge_core_chat_transformers(
                core_txs,
                anthropic_like_chat_request_to_core_input,
                |evt| anthropic_like_map_core_stream_event("anthropic", evt),
            )
        }

        // When neither external provider nor external standard is enabled,
        // Anthropic chat is not wired through the unified core pipeline.
        // This configuration is no longer supported, so we return a
        // transformer set that reports UnsupportedOperation for chat.
        #[cfg(all(
            not(feature = "provider-anthropic-external"),
            not(feature = "std-anthropic-external")
        ))]
        {
            use crate::core::provider_spec::ChatTransformers;
            use crate::execution::transformers::request::RequestTransformer;
            use crate::execution::transformers::response::ResponseTransformer;

            struct UnsupportedReq;
            impl RequestTransformer for UnsupportedReq {
                fn provider_id(&self) -> &str {
                    "anthropic"
                }
                fn transform_chat(
                    &self,
                    _req: &crate::types::ChatRequest,
                ) -> Result<serde_json::Value, LlmError> {
                    Err(LlmError::UnsupportedOperation(
                        "Anthropic chat is not available without std-anthropic-external or provider-anthropic-external"
                            .to_string(),
                    ))
                }
            }

            struct UnsupportedResp;
            impl ResponseTransformer for UnsupportedResp {
                fn provider_id(&self) -> &str {
                    "anthropic"
                }
            }

            ChatTransformers {
                request: Arc::new(UnsupportedReq),
                response: Arc::new(UnsupportedResp),
                stream: None,
                json: None,
            }
        }
    }

    fn chat_before_send(
        &self,
        req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        // 1. First check for CustomProviderOptions (using default implementation)
        if let Some(hook) = crate::core::default_custom_options_hook(self.id(), req) {
            return Some(hook);
        }

        // 2. Anthropic-specific typed options (thinking / response_format) have
        // already been mapped into `ChatInput::extra` by
        // `anthropic_like_chat_request_to_core_input` (keys
        // `anthropic_thinking` / `anthropic_response_format`) and are injected
        // into the final JSON by the `siumai-std-anthropic` default adapter.
        //
        // As a result, we no longer handle those fields here to avoid
        // overlapping responsibilities with the standards layer.
        None
    }
}
