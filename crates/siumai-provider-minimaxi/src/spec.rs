//! MiniMaxi core-level provider spec 实现
//!
//! 该模块提供基于 `siumai-core` / `siumai-std-anthropic` 的
//! `CoreProviderSpec` 实现，供聚合层通过 feature gate 进行桥接。

use siumai_core::error::LlmError;
use siumai_core::execution::chat::ChatInput;
use siumai_core::execution::streaming::ChatStreamEventCore;
use siumai_core::provider_spec::{CoreChatTransformers, CoreProviderContext, CoreProviderSpec};
use siumai_core::traits::ProviderCapabilities;
use siumai_std_anthropic::anthropic::chat::AnthropicChatStandard;

/// MiniMaxi 在 core 层的 ProviderSpec 实现
///
/// Chat 能力通过 Anthropic Messages 标准实现；
/// 其他能力（audio/image 等）后续会逐步接入。
#[derive(Clone, Default)]
pub struct MinimaxiCoreSpec {
    chat_standard: AnthropicChatStandard,
}

impl MinimaxiCoreSpec {
    /// 使用默认标准实现构造
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

        // Chat API 使用 Anthropic 兼容 header 策略
        crate::headers::build_anthropic_headers(api_key, &ctx.http_extra_headers)
    }

    fn chat_url(&self, ctx: &CoreProviderContext) -> String {
        // 默认行为：MiniMaxi Anthropic 兼容 endpoint 的 /v1/messages
        format!("{}/v1/messages", ctx.base_url.trim_end_matches('/'))
    }

    fn choose_chat_transformers(
        &self,
        _input: &ChatInput,
        ctx: &CoreProviderContext,
    ) -> CoreChatTransformers {
        // 使用 Anthropic 标准实现构造 core 层 transformer
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
        // 暂不做额外加工
        event
    }
}
