//! Anthropic core-level provider spec 实现
//!
//! 该模块提供基于 `siumai-core` / `siumai-std-anthropic` 的
//! `CoreProviderSpec` 实现，供聚合层通过 feature gate 进行桥接。

use siumai_core::error::LlmError;
use siumai_core::execution::chat::ChatInput;
use siumai_core::execution::streaming::ChatStreamEventCore;
use siumai_core::provider_spec::{CoreChatTransformers, CoreProviderContext, CoreProviderSpec};
use siumai_core::traits::ProviderCapabilities;
use siumai_std_anthropic::anthropic::chat::{AnthropicChatStandard, AnthropicDefaultChatAdapter};
use std::sync::Arc;

/// Anthropic 在 core 层的 ProviderSpec 实现
#[derive(Clone, Default)]
pub struct AnthropicCoreSpec {
    chat_standard: AnthropicChatStandard,
}

impl AnthropicCoreSpec {
    /// 使用默认标准实现构造
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

        // 复用本 crate 的 header helper，保持与聚合层行为一致
        crate::headers::build_anthropic_json_headers(api_key, &ctx.http_extra_headers)
    }

    fn chat_url(&self, ctx: &CoreProviderContext) -> String {
        // 默认行为：使用 base_url 拼接 Anthropic Messages API 路径
        format!("{}/v1/messages", ctx.base_url.trim_end_matches('/'))
    }

    fn choose_chat_transformers(
        &self,
        _input: &ChatInput,
        ctx: &CoreProviderContext,
    ) -> CoreChatTransformers {
        // 使用标准 crate 提供的 Anthropic Chat 标准实现
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
        // 目前不做额外加工，直接透传
        event
    }
}
