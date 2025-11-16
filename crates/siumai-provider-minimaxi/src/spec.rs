//! MiniMaxi core-level provider spec implementation.
//!
//! This module provides a `CoreProviderSpec` implementation based on
//! `siumai-core` and `siumai-std-anthropic`, which can be consumed by the
//! aggregator via feature gates.

use siumai_core::error::LlmError;
use siumai_core::execution::chat::ChatInput;
use siumai_core::execution::streaming::ChatStreamEventCore;
use siumai_core::provider_spec::{CoreChatTransformers, CoreProviderContext, CoreProviderSpec};
use siumai_core::traits::ProviderCapabilities;
use siumai_std_anthropic::anthropic::chat::AnthropicChatStandard;

/// MiniMaxi `CoreProviderSpec` implementation.
///
/// Chat capability is implemented via the Anthropic Messages standard;
/// other capabilities (audio/image, etc.) will be integrated gradually.
#[derive(Clone, Default)]
pub struct MinimaxiCoreSpec {
    chat_standard: AnthropicChatStandard,
}

impl MinimaxiCoreSpec {
    /// Construct with the default Anthropic standard.
    pub fn new() -> Self {
        Self {
            chat_standard: AnthropicChatStandard::new(),
        }
    }
}

impl CoreProviderSpec for MinimaxiCoreSpec {
    fn id(&self) -> &'static str {
        "minimaxi"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_audio()
            .with_custom_feature("speech", true)
            .with_custom_feature("image_generation", true)
    }

    fn build_headers(
        &self,
        ctx: &CoreProviderContext,
    ) -> Result<reqwest::header::HeaderMap, LlmError> {
        let api_key = ctx
            .api_key
            .as_ref()
            .ok_or_else(|| LlmError::MissingApiKey("MiniMaxi API key not provided".into()))?;

        // Chat APIs use Anthropic-compatible header strategy.
        crate::headers::build_anthropic_headers(api_key, &ctx.http_extra_headers)
    }

    fn chat_url(&self, ctx: &CoreProviderContext) -> String {
        // Default behavior: MiniMaxi Anthropic-compatible endpoint `/v1/messages`.
        format!("{}/v1/messages", ctx.base_url.trim_end_matches('/'))
    }

    fn choose_chat_transformers(
        &self,
        _input: &ChatInput,
        ctx: &CoreProviderContext,
    ) -> CoreChatTransformers {
        // Use Anthropic standard to construct core-level transformers.
        let req = self
            .chat_standard
            .create_request_transformer(&ctx.provider_id);
        let resp = self
            .chat_standard
            .create_response_transformer(&ctx.provider_id);
        let stream = self.chat_standard.create_stream_converter(&ctx.provider_id);

        CoreChatTransformers {
            request: req,
            response: resp,
            stream: Some(stream),
        }
    }

    fn map_core_stream_event(&self, event: ChatStreamEventCore) -> ChatStreamEventCore {
        // No additional processing for now; pass through.
        event
    }
}
