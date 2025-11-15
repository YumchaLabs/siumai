//! Groq core-level provider spec 实现
//!
//!” 该模块提供基于 `siumai-core` / `siumai-std-openai` 的
//! `CoreProviderSpec` 实现，供聚合层通过 feature gate 进行桥接。

use siumai_core::error::LlmError;
use siumai_core::execution::chat::ChatInput;
use siumai_core::execution::streaming::ChatStreamEventCore;
use siumai_core::provider_spec::{CoreChatTransformers, CoreProviderContext, CoreProviderSpec};
use siumai_core::traits::ProviderCapabilities;
use siumai_std_openai::openai::chat::{OpenAiChatAdapter, OpenAiChatStandard};
use std::sync::Arc;

/// Groq 在 core 层的 ProviderSpec 实现
#[derive(Clone)]
pub struct GroqCoreSpec {
    chat_standard: OpenAiChatStandard,
}

impl Default for GroqCoreSpec {
    fn default() -> Self {
        Self::new()
    }
}

impl GroqCoreSpec {
    /// 使用带 Groq 适配器的 OpenAI Chat 标准实现构造
    pub fn new() -> Self {
        let adapter = Arc::new(GroqOpenAiChatAdapter::default());
        Self {
            chat_standard: OpenAiChatStandard::with_adapter(adapter),
        }
    }
}

/// Groq 专用 OpenAI Chat 适配器
///
/// 该适配器从 `ChatInput::extra` 中读取 `groq_extra_params` 对象，并将其
/// 所有键值对合并进最终请求 JSON 体中。
#[derive(Clone, Default)]
struct GroqOpenAiChatAdapter;

impl OpenAiChatAdapter for GroqOpenAiChatAdapter {
    fn transform_request(
        &self,
        input: &ChatInput,
        body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        if let Some(extra) = input.extra.get("groq_extra_params")
            && let Some(extra_obj) = extra.as_object()
            && let Some(body_obj) = body.as_object_mut()
        {
            for (k, v) in extra_obj {
                body_obj.insert(k.clone(), v.clone());
            }
        }
        Ok(())
    }
}

impl CoreProviderSpec for GroqCoreSpec {
    fn id(&self) -> &'static str {
        "groq"
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
            .ok_or_else(|| LlmError::MissingApiKey("Groq API key not provided".into()))?;

        crate::headers::build_groq_json_headers(api_key, &ctx.http_extra_headers)
    }

    fn chat_url(&self, ctx: &CoreProviderContext) -> String {
        // Base URL already includes "/openai/v1" (see GroqConfig default).
        // Append only the operation path to avoid duplicating the prefix.
        format!("{}/chat/completions", ctx.base_url.trim_end_matches('/'))
    }

    fn choose_chat_transformers(
        &self,
        _input: &ChatInput,
        ctx: &CoreProviderContext,
    ) -> CoreChatTransformers {
        // 初始版本：复用 OpenAI Chat 标准实现。Groq 的更丰富行为
        //（多模态、工具调用细节）仍由聚合层的 transformers 负责；
        //后续可将这些逻辑迁移到专门的 std-groq crate 中。
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
