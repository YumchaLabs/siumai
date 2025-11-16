//! Anthropic core-level provider spec implementation.
//!
//! This module provides a `CoreProviderSpec` implementation based on
//! `siumai-core` and `siumai-std-anthropic`, which can be consumed by the
//! aggregator via feature gates.

use siumai_core::error::LlmError;
use siumai_core::execution::chat::ChatInput;
use siumai_core::execution::streaming::ChatStreamEventCore;
use siumai_core::provider_spec::{CoreChatTransformers, CoreProviderContext, CoreProviderSpec};
use siumai_core::traits::ProviderCapabilities;
use siumai_std_anthropic::anthropic::chat::{AnthropicChatStandard, AnthropicDefaultChatAdapter};
use std::sync::Arc;

/// Anthropic `CoreProviderSpec` implementation.
#[derive(Clone, Default)]
pub struct AnthropicCoreSpec {
    chat_standard: AnthropicChatStandard,
}

impl AnthropicCoreSpec {
    /// Construct with the default Anthropic standard.
    pub fn new() -> Self {
        Self {
            chat_standard: AnthropicChatStandard::with_adapter(Arc::new(
                AnthropicDefaultChatAdapter::default(),
            )),
        }
    }
}

impl CoreProviderSpec for AnthropicCoreSpec {
    fn id(&self) -> &'static str {
        "anthropic"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
    }

    fn build_headers(
        &self,
        ctx: &CoreProviderContext,
    ) -> Result<reqwest::header::HeaderMap, LlmError> {
        let api_key = ctx
            .api_key
            .as_ref()
            .ok_or_else(|| LlmError::MissingApiKey("Anthropic API key not provided".into()))?;

        // Reuse this crate's header helper to keep behavior consistent with the aggregator.
        crate::headers::build_anthropic_json_headers(api_key, &ctx.http_extra_headers)
    }

    fn chat_url(&self, ctx: &CoreProviderContext) -> String {
        // Default behavior: append Anthropic Messages API path to base_url.
        format!("{}/v1/messages", ctx.base_url.trim_end_matches('/'))
    }

    fn choose_chat_transformers(
        &self,
        _input: &ChatInput,
        ctx: &CoreProviderContext,
    ) -> CoreChatTransformers {
        // Use the standard Anthropic Chat implementation from the std crate.
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
