//! MiniMaxi ProviderSpec Implementation

use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::http::headers::ProviderHeaders;
use crate::standards::anthropic::chat::AnthropicChatStandard;
use crate::traits::ProviderCapabilities;
use reqwest::header::HeaderMap;

/// MiniMaxi ProviderSpec implementation
///
/// MiniMaxi supports both OpenAI and Anthropic API formats.
/// We use Anthropic format (recommended by MiniMaxi) for better support of:
/// - Thinking content blocks (reasoning process)
/// - Tool Use and Interleaved Thinking
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
            .with_custom_feature("video", true)
            .with_custom_feature("image_generation", true) // Enable image generation
            .with_custom_feature("music", true)
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        let api_key = ctx
            .api_key
            .as_ref()
            .ok_or_else(|| LlmError::MissingApiKey("MiniMaxi API key not provided".into()))?;

        // MiniMaxi Anthropic-compatible API uses x-api-key header (Anthropic standard)
        ProviderHeaders::anthropic(api_key, &ctx.http_extra_headers)
    }

    fn chat_url(
        &self,
        _stream: bool,
        _req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> String {
        // MiniMaxi uses Anthropic-compatible endpoint
        format!("{}/v1/messages", ctx.base_url.trim_end_matches('/'))
    }

    fn choose_chat_transformers(
        &self,
        _req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        // Use Anthropic standard transformers since MiniMaxi is Anthropic-compatible
        self.chat_standard.create_transformers(&ctx.provider_id)
    }

    fn audio_base_url(&self, ctx: &ProviderContext) -> String {
        // MiniMaxi TTS/STT endpoints use OpenAI-compatible format under /v1
        // If using Anthropic base_url, switch to OpenAI base_url for audio
        let base = ctx.base_url.trim_end_matches('/');
        if base.contains("/anthropic") {
            // Switch from Anthropic endpoint to OpenAI endpoint for audio
            "https://api.minimaxi.com/v1".to_string()
        } else if base.ends_with("/v1") {
            base.to_string()
        } else {
            format!("{}/v1", base)
        }
    }

    fn image_url(
        &self,
        _req: &crate::types::ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> String {
        // MiniMaxi image generation uses OpenAI-compatible format
        // If using Anthropic base_url, switch to OpenAI base_url for image
        let base = ctx.base_url.trim_end_matches('/');
        if base.contains("/anthropic") {
            "https://api.minimaxi.com/v1/image_generation".to_string()
        } else {
            format!("{}/image_generation", base)
        }
    }
}
