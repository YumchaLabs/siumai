//! Gemini core-level provider spec implementation.
//!
//! 基于 `siumai-core` 与 `siumai-std-gemini`，提供 `CoreProviderSpec`
//! 实现，供聚合层通过 feature gate 进行桥接。

use siumai_core::error::LlmError;
use siumai_core::execution::chat::ChatInput;
use siumai_core::execution::streaming::ChatStreamEventCore;
use siumai_core::provider_spec::{CoreChatTransformers, CoreProviderContext, CoreProviderSpec};
use siumai_core::traits::ProviderCapabilities;
use siumai_std_gemini::gemini::chat::{GeminiChatStandard, GeminiDefaultChatAdapter};
use siumai_std_gemini::gemini::embedding::GeminiEmbeddingStandard;
use siumai_std_gemini::gemini::image::GeminiImageStandard;
use std::sync::Arc;

/// Gemini 在 core 层的 ProviderSpec 实现。
#[derive(Clone)]
pub struct GeminiCoreSpec {
    /// Chat standard implementation
    pub chat_standard: GeminiChatStandard,
    /// Embedding standard implementation
    pub embedding_standard: GeminiEmbeddingStandard,
    /// Image standard implementation
    pub image_standard: GeminiImageStandard,
}

impl Default for GeminiCoreSpec {
    fn default() -> Self {
        Self::new()
    }
}

impl GeminiCoreSpec {
    /// 使用带默认适配器的 Gemini Chat 标准实现构造。
    pub fn new() -> Self {
        let adapter = Arc::new(GeminiDefaultChatAdapter::default());
        Self {
            chat_standard: GeminiChatStandard::with_adapter(adapter),
            embedding_standard: GeminiEmbeddingStandard::new(),
            image_standard: GeminiImageStandard::new(),
        }
    }
}

impl CoreProviderSpec for GeminiCoreSpec {
    fn id(&self) -> &'static str {
        "gemini"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_embedding()
            .with_vision()
            .with_custom_feature("image_generation", true)
    }

    fn build_headers(
        &self,
        ctx: &CoreProviderContext,
    ) -> Result<reqwest::header::HeaderMap, LlmError> {
        let api_key = ctx.api_key.as_deref().unwrap_or("");
        crate::headers::build_gemini_json_headers(api_key, &ctx.http_extra_headers)
    }

    fn chat_url(&self, ctx: &CoreProviderContext) -> String {
        // 当前 Core 层仅返回 base_url，本身不负责拼接模型路径；
        // 具体路由规则仍由聚合层根据 ChatRequest / streaming 选择。
        ctx.base_url.trim_end_matches('/').to_string()
    }

    fn choose_chat_transformers(&self, _input: &ChatInput, ctx: &CoreProviderContext) -> CoreChatTransformers {
        let req = self
            .chat_standard
            .create_request_transformer(&ctx.provider_id);
        let resp = self
            .chat_standard
            .create_response_transformer(&ctx.provider_id);
        let stream = self
            .chat_standard
            .create_stream_converter(&ctx.provider_id);

        CoreChatTransformers {
            request: req,
            response: resp,
            stream: Some(stream),
        }
    }

    fn map_core_stream_event(&self, event: ChatStreamEventCore) -> ChatStreamEventCore {
        // 暂不做额外加工，直接透传。
        event
    }
}
