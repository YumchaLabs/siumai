//! xAI core-level provider spec 实现
//!
//! 该模块提供基于 `siumai-core` / `siumai-std-openai` 的
//! `CoreProviderSpec` 实现，供聚合层通过 feature gate 进行桥接。

use siumai_core::error::LlmError;
use siumai_core::execution::chat::ChatInput;
use siumai_core::execution::streaming::ChatStreamEventCore;
use siumai_core::provider_spec::{CoreChatTransformers, CoreProviderContext, CoreProviderSpec};
use siumai_core::traits::ProviderCapabilities;
use siumai_std_openai::openai::chat::{OpenAiChatAdapter, OpenAiChatStandard};
use std::sync::Arc;

/// xAI 在 core 层的 ProviderSpec 实现
#[derive(Clone)]
pub struct XaiCoreSpec {
    chat_standard: OpenAiChatStandard,
}

impl Default for XaiCoreSpec {
    fn default() -> Self {
        Self::new()
    }
}

impl XaiCoreSpec {
    /// 使用带 xAI 适配器的 OpenAI Chat 标准实现构造
    pub fn new() -> Self {
        let adapter = Arc::new(XaiOpenAiChatAdapter::default());
        Self {
            chat_standard: OpenAiChatStandard::with_adapter(adapter),
        }
    }
}

/// xAI 专用 OpenAI Chat 适配器
///
/// 该适配器从 `ChatInput::extra` 中读取 xAI 相关参数：
/// - `xai_search_parameters`：预构造的搜索配置对象
/// - `xai_reasoning_effort`：推理强度字符串
/// 然后将其注入到最终请求 JSON 中。
#[derive(Clone, Default)]
struct XaiOpenAiChatAdapter;

impl OpenAiChatAdapter for XaiOpenAiChatAdapter {
    fn transform_request(
        &self,
        input: &ChatInput,
        body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        // 将预先编码到 ChatInput::extra 中的 search_parameters 合并到请求体。
        if let Some(params) = input.extra.get("xai_search_parameters") {
            body["search_parameters"] = params.clone();
        }

        // 注入 reasoning_effort（若存在）。
        if let Some(effort) = input.extra.get("xai_reasoning_effort") {
            body["reasoning_effort"] = effort.clone();
        }

        Ok(())
    }
}

impl CoreProviderSpec for XaiCoreSpec {
    fn id(&self) -> &'static str {
        "xai"
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
            .ok_or_else(|| LlmError::MissingApiKey("xAI API key not provided".into()))?;

        crate::headers::build_xai_json_headers(api_key, &ctx.http_extra_headers)
    }

    fn chat_url(&self, ctx: &CoreProviderContext) -> String {
        // Base URL already includes "/v1" (see aggregator config). Append only the
        // operation path to avoid duplicating the prefix.
        format!("{}/chat/completions", ctx.base_url.trim_end_matches('/'))
    }

    fn choose_chat_transformers(
        &self,
        _input: &ChatInput,
        ctx: &CoreProviderContext,
    ) -> CoreChatTransformers {
        // Reuse the OpenAI Chat standard for xAI. Provider-specific options
        // (search_parameters, reasoning_effort, etc.) are handled via
        // higher-level hooks in the aggregator.
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
        // 暂不做额外加工，直接透传
        event
    }
}
