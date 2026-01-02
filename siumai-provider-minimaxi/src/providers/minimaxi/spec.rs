//! MiniMaxi ProviderSpec Implementation

use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::http::headers::HttpHeaderBuilder;
use crate::standards::anthropic::chat::{AnthropicChatAdapter, AnthropicChatStandard};
use crate::traits::ProviderCapabilities;
use reqwest::header::HeaderMap;
use std::sync::Arc;

fn resolve_openai_base_url(base_url: &str) -> String {
    let base = base_url.trim_end_matches('/');
    if base.contains("/anthropic") {
        super::config::MinimaxiConfig::OPENAI_BASE_URL.to_string()
    } else if base.ends_with("/v1") {
        base.to_string()
    } else {
        format!("{}/v1", base)
    }
}

fn build_openai_like_headers(ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
    let api_key = ctx
        .api_key
        .as_ref()
        .ok_or_else(|| LlmError::MissingApiKey("MiniMaxi API key not provided".into()))?;

    let mut builder = HttpHeaderBuilder::new()
        .with_bearer_auth(api_key)?
        .with_json_content_type();

    if let Some(org) = ctx.organization.as_deref() {
        builder = builder.with_header("OpenAI-Organization", org)?;
    }
    if let Some(proj) = ctx.project.as_deref() {
        builder = builder.with_header("OpenAI-Project", proj)?;
    }

    builder = builder.with_custom_headers(&ctx.http_extra_headers)?;
    Ok(builder.build())
}

/// MiniMaxi ProviderSpec implementation
///
/// MiniMaxi supports both OpenAI and Anthropic API formats.
/// We use Anthropic format (recommended by MiniMaxi) for better support of:
/// - Thinking content blocks (reasoning process)
/// - Tool Use and Interleaved Thinking
/// - Extended thinking capabilities
#[derive(Clone)]
pub struct MinimaxiChatSpec {
    /// Anthropic Chat standard for request/response transformation
    chat_standard: AnthropicChatStandard,
}

#[derive(Debug, Default)]
struct MinimaxiAnthropicAdapter;

impl AnthropicChatAdapter for MinimaxiAnthropicAdapter {
    fn build_headers(
        &self,
        api_key: &str,
        _base_headers: &mut reqwest::header::HeaderMap,
    ) -> Result<(), LlmError> {
        if api_key.is_empty() {
            return Err(LlmError::MissingApiKey(
                "MiniMaxi API key not provided".into(),
            ));
        }
        Ok(())
    }
}

impl MinimaxiChatSpec {
    pub fn new() -> Self {
        Self {
            chat_standard: AnthropicChatStandard::with_adapter(Arc::new(MinimaxiAnthropicAdapter)),
        }
    }

    fn chat_spec(&self) -> crate::standards::anthropic::chat::AnthropicChatSpec {
        self.chat_standard.create_spec("minimaxi")
    }
}

impl Default for MinimaxiChatSpec {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderSpec for MinimaxiChatSpec {
    fn id(&self) -> &'static str {
        "minimaxi"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        self.chat_spec().build_headers(ctx)
    }

    fn chat_url(
        &self,
        stream: bool,
        req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> String {
        self.chat_spec().chat_url(stream, req, ctx)
    }

    fn choose_chat_transformers(
        &self,
        req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        self.chat_spec().choose_chat_transformers(req, ctx)
    }

    fn chat_before_send(
        &self,
        req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        self.chat_spec().chat_before_send(req, ctx)
    }
}

/// MiniMaxi audio spec (OpenAI-compatible endpoint).
///
/// Important: MiniMaxi uses Anthropic-compatible endpoints for chat, but OpenAI-compatible auth
/// (Bearer) for audio endpoints. Split specs keep each endpoint consistent and avoid mixing headers.
#[derive(Clone, Default)]
pub(crate) struct MinimaxiAudioSpec;

impl MinimaxiAudioSpec {
    pub fn new() -> Self {
        Self
    }
}

impl ProviderSpec for MinimaxiAudioSpec {
    fn id(&self) -> &'static str {
        "minimaxi"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_audio()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        build_openai_like_headers(ctx)
    }

    fn audio_base_url(&self, ctx: &ProviderContext) -> String {
        // MiniMaxi TTS/STT endpoints use OpenAI-compatible format under /v1.
        resolve_openai_base_url(&ctx.base_url)
    }

    fn choose_audio_transformer(&self, _ctx: &ProviderContext) -> crate::core::AudioTransformer {
        crate::core::AudioTransformer {
            transformer: Arc::new(super::transformers::audio::MinimaxiAudioTransformer),
        }
    }
}

/// MiniMaxi image generation spec (OpenAI-compatible endpoint).
#[derive(Clone, Default)]
pub(crate) struct MinimaxiImageSpec;

impl MinimaxiImageSpec {
    pub fn new() -> Self {
        Self
    }
}

impl ProviderSpec for MinimaxiImageSpec {
    fn id(&self) -> &'static str {
        "minimaxi"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_image_generation()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        build_openai_like_headers(ctx)
    }

    fn choose_image_transformers(
        &self,
        _req: &crate::types::ImageGenerationRequest,
        _ctx: &ProviderContext,
    ) -> crate::core::ImageTransformers {
        // Use OpenAI image protocol transformers with MiniMaxi response adapter.
        let standard =
            crate::providers::minimaxi::transformers::image::create_minimaxi_image_standard();
        let transformers = standard.create_transformers("minimaxi");
        crate::core::ImageTransformers {
            request: transformers.request,
            response: transformers.response,
        }
    }

    fn image_url(
        &self,
        _req: &crate::types::ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> String {
        format!(
            "{}/image_generation",
            resolve_openai_base_url(&ctx.base_url).trim_end_matches('/')
        )
    }
}

/// MiniMaxi video generation spec (OpenAI-compatible endpoint).
#[derive(Clone, Default)]
pub(crate) struct MinimaxiVideoSpec;

impl MinimaxiVideoSpec {
    pub fn new() -> Self {
        Self
    }

    pub fn video_generation_url(&self, ctx: &ProviderContext) -> String {
        format!(
            "{}/video_generation",
            resolve_openai_base_url(&ctx.base_url).trim_end_matches('/')
        )
    }

    pub fn video_query_url(&self, ctx: &ProviderContext, task_id: &str) -> String {
        let base_url = resolve_openai_base_url(&ctx.base_url);
        format!(
            "{}/query/video_generation?task_id={}",
            base_url.trim_end_matches('/'),
            task_id
        )
    }
}

impl ProviderSpec for MinimaxiVideoSpec {
    fn id(&self) -> &'static str {
        "minimaxi"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_custom_feature("video", true)
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        build_openai_like_headers(ctx)
    }
}

/// MiniMaxi music generation spec (OpenAI-compatible endpoint).
#[derive(Clone, Default)]
pub(crate) struct MinimaxiMusicSpec;

impl MinimaxiMusicSpec {
    pub fn new() -> Self {
        Self
    }

    pub fn music_generation_url(&self, ctx: &ProviderContext) -> String {
        format!(
            "{}/music_generation",
            resolve_openai_base_url(&ctx.base_url).trim_end_matches('/')
        )
    }
}

impl ProviderSpec for MinimaxiMusicSpec {
    fn id(&self) -> &'static str {
        "minimaxi"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_custom_feature("music", true)
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        build_openai_like_headers(ctx)
    }
}

/// Backward compatible name: historically referenced as `MinimaxiSpec`.
pub type MinimaxiSpec = MinimaxiChatSpec;
