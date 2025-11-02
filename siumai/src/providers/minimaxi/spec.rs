//! MiniMaxi ProviderSpec Implementation

use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::http::headers::ProviderHeaders;
use crate::standards::openai::chat::OpenAiChatStandard;
use crate::traits::ProviderCapabilities;
use reqwest::header::HeaderMap;

/// MiniMaxi ProviderSpec implementation
///
/// MiniMaxi uses OpenAI-compatible API format, so we leverage the OpenAI standard.
#[derive(Clone)]
pub struct MinimaxiSpec {
    /// OpenAI Chat standard for request/response transformation
    chat_standard: OpenAiChatStandard,
}

impl Default for MinimaxiSpec {
    fn default() -> Self {
        Self::new()
    }
}

impl MinimaxiSpec {
    pub fn new() -> Self {
        Self {
            chat_standard: OpenAiChatStandard::new(),
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

        // MiniMaxi uses Bearer token authentication like OpenAI
        ProviderHeaders::openai(
            api_key,
            None, // organization
            None, // project
            &ctx.http_extra_headers,
        )
    }

    fn chat_url(
        &self,
        _stream: bool,
        _req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> String {
        // MiniMaxi uses OpenAI-compatible endpoint
        format!("{}/chat/completions", ctx.base_url.trim_end_matches('/'))
    }

    fn choose_chat_transformers(
        &self,
        _req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        // Use OpenAI standard transformers since MiniMaxi is OpenAI-compatible
        self.chat_standard.create_transformers(&ctx.provider_id)
    }

    fn audio_base_url(&self, ctx: &ProviderContext) -> String {
        // MiniMaxi uses the same base URL for audio endpoints
        ctx.base_url.trim_end_matches('/').to_string()
    }

    fn image_url(
        &self,
        _req: &crate::types::ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> String {
        // MiniMaxi image generation endpoint (base_url already includes /v1 by default)
        format!("{}/image_generation", ctx.base_url.trim_end_matches('/'))
    }
}
