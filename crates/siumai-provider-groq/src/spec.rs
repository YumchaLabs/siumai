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
        let adapter = Arc::new(GroqOpenAiChatAdapter);
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
        if let Some(body_obj) = body.as_object_mut() {
            // 先展开 escape hatch：extra_params。仅在目标字段不存在时写入，
            // 避免覆盖后续 typed options 映射产生的字段。
            if let Some(extra) = input.extra.get("groq_extra_params")
                && let Some(extra_obj) = extra.as_object()
            {
                for (k, v) in extra_obj {
                    body_obj.entry(k.clone()).or_insert_with(|| v.clone());
                }
            }

            // 再映射 typed GroqOptions 字段（优先级高于 extra_params）。
            if let Some(effort) = input.extra.get("groq_reasoning_effort") {
                body_obj.insert("reasoning_effort".to_string(), effort.clone());
            }
            if let Some(fmt) = input.extra.get("groq_reasoning_format") {
                body_obj.insert("reasoning_format".to_string(), fmt.clone());
            }
            if let Some(parallel) = input.extra.get("groq_parallel_tool_calls") {
                body_obj.insert("parallel_tool_calls".to_string(), parallel.clone());
            }
            if let Some(tier) = input.extra.get("groq_service_tier") {
                body_obj.insert("service_tier".to_string(), tier.clone());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn transform_request_merges_typed_and_extra_params() {
        let mut input = ChatInput::default();
        // Typed options mapped via ChatInput::extra
        input
            .extra
            .insert("groq_reasoning_effort".to_string(), json!("medium"));
        input
            .extra
            .insert("groq_reasoning_format".to_string(), json!("parsed"));
        input
            .extra
            .insert("groq_parallel_tool_calls".to_string(), json!(true));
        input
            .extra
            .insert("groq_service_tier".to_string(), json!("flex"));

        // Escape hatch params, including a conflicting service_tier
        input.extra.insert(
            "groq_extra_params".to_string(),
            json!({
                "service_tier": "on_demand",
                "custom_flag": true
            }),
        );

        let mut body = json!({
            "model": "qwen/qwen3-32b",
            "messages": [],
        });

        let adapter = GroqOpenAiChatAdapter::default();
        adapter.transform_request(&input, &mut body).unwrap();

        let obj = body.as_object().expect("body should remain a JSON object");

        // Typed fields should be present
        assert_eq!(
            obj.get("reasoning_effort").and_then(|v| v.as_str()),
            Some("medium")
        );
        assert_eq!(
            obj.get("reasoning_format").and_then(|v| v.as_str()),
            Some("parsed")
        );
        assert_eq!(
            obj.get("parallel_tool_calls").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(
            obj.get("service_tier").and_then(|v| v.as_str()),
            Some("flex")
        );

        // extra_params should also be merged
        assert_eq!(obj.get("custom_flag").and_then(|v| v.as_bool()), Some(true));

        // Typed service_tier should override extra_params["service_tier"]
        assert_eq!(
            obj.get("service_tier").and_then(|v| v.as_str()),
            Some("flex")
        );
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
