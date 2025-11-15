//! MiniMaxi ProviderSpec implementation
//!
//! MiniMaxi exposes an Anthropic-compatible chat API. This Spec wires
//! MiniMaxi into the shared Anthropic standard so that:
//! - Non-external mode reuses the in-crate Anthropic transformers.
//! - External mode delegates to `siumai-provider-minimaxi` via the
//!   core-only `MinimaxiCoreSpec` and bridges back using
//!   `crate::core::bridge_core_chat_transformers`.

use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::http::headers::ProviderHeaders;
use crate::std_anthropic::anthropic::chat::AnthropicChatStandard;
use crate::traits::ProviderCapabilities;
use crate::types::ChatRequest;
use reqwest::header::HeaderMap;

/// MiniMaxi ProviderSpec implementation
///
/// MiniMaxi supports both OpenAI and Anthropic API formats. We use the
/// Anthropic Messages format (recommended by MiniMaxi) to get better
/// support for:
/// - Thinking content blocks (reasoning process)
/// - Tool use and interleaved thinking
/// - Extended thinking capabilities
#[derive(Clone)]
pub struct MinimaxiSpec {
    /// Anthropic Chat standard for request/response transformation
    chat_standard: AnthropicChatStandard,
}

impl Default for MinimaxiSpec {
    fn default() -> Self {
        Self::new()
    }
}

impl MinimaxiSpec {
    pub fn new() -> Self {
        Self {
            chat_standard: AnthropicChatStandard::new(),
        }
    }
}

impl ProviderSpec for MinimaxiSpec {
    fn id(&self) -> &'static str {
        "minimaxi"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_audio() // Enable audio capability
            .with_custom_feature("speech", true)
            .with_custom_feature("image_generation", true) // Enable image generation
        // Note: video and music capabilities are planned for future releases
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        // MiniMaxi Anthropic-compatible Chat API uses x-api-key header.
        // In external mode, delegate to the core-spec in the provider crate
        // to avoid duplicating header behavior.
        #[cfg(feature = "provider-minimaxi-external")]
        {
            use siumai_core::provider_spec::CoreProviderSpec;

            let core_ctx = ctx.to_core_context();
            let core_spec = siumai_provider_minimaxi::MinimaxiCoreSpec::new();
            return core_spec.build_headers(&core_ctx);
        }

        // Default: reuse aggregator Anthropic-style header helper.
        #[cfg(not(feature = "provider-minimaxi-external"))]
        {
            let api_key = ctx
                .api_key
                .as_ref()
                .ok_or_else(|| LlmError::MissingApiKey("MiniMaxi API key not provided".into()))?;
            ProviderHeaders::anthropic(api_key, &ctx.http_extra_headers)
        }
    }

    fn chat_url(
        &self,
        _stream: bool,
        _req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> String {
        #[cfg(feature = "provider-minimaxi-external")]
        {
            use siumai_core::provider_spec::CoreProviderSpec;

            let core_ctx = ctx.to_core_context();
            let core_spec = siumai_provider_minimaxi::MinimaxiCoreSpec::new();
            return core_spec.chat_url(&core_ctx);
        }

        #[cfg(not(feature = "provider-minimaxi-external"))]
        {
            // MiniMaxi uses Anthropic-compatible endpoint
            format!("{}/v1/messages", ctx.base_url.trim_end_matches('/'))
        }
    }

    fn choose_chat_transformers(
        &self,
        req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        #[cfg(feature = "provider-minimaxi-external")]
        {
            use crate::core::provider_spec::{
                anthropic_like_chat_request_to_core_input, anthropic_like_map_core_stream_event,
                bridge_core_chat_transformers,
            };
            use siumai_core::provider_spec::{CoreChatTransformers, CoreProviderSpec};

            let core_ctx = ctx.to_core_context();
            let core_spec = siumai_provider_minimaxi::MinimaxiCoreSpec::new();
            let core_txs: CoreChatTransformers = core_spec.choose_chat_transformers(
                &anthropic_like_chat_request_to_core_input(req),
                &core_ctx,
            );

            bridge_core_chat_transformers(
                core_txs,
                anthropic_like_chat_request_to_core_input,
                |evt| anthropic_like_map_core_stream_event("minimaxi", evt),
            )
        }

        #[cfg(all(
            not(feature = "provider-minimaxi-external"),
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
                |evt| anthropic_like_map_core_stream_event("minimaxi", evt),
            )
        }

        // When neither external provider nor external standard is enabled,
        // MiniMaxi chat is not wired through the unified core pipeline.
        // This configuration is no longer supported, so we return a
        // transformer set that reports UnsupportedOperation for chat.
        #[cfg(all(
            not(feature = "provider-minimaxi-external"),
            not(feature = "std-anthropic-external")
        ))]
        {
            use crate::core::provider_spec::ChatTransformers;
            use crate::execution::transformers::request::RequestTransformer;
            use crate::execution::transformers::response::ResponseTransformer;

            struct UnsupportedReq;
            impl RequestTransformer for UnsupportedReq {
                fn provider_id(&self) -> &str {
                    "minimaxi"
                }
                fn transform_chat(
                    &self,
                    _req: &crate::types::ChatRequest,
                ) -> Result<serde_json::Value, LlmError> {
                    Err(LlmError::UnsupportedOperation(
                        "MiniMaxi chat is not available without std-anthropic-external or provider-minimaxi-external"
                            .to_string(),
                    ))
                }
            }

            struct UnsupportedResp;
            impl ResponseTransformer for UnsupportedResp {
                fn provider_id(&self) -> &str {
                    "minimaxi"
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
}
