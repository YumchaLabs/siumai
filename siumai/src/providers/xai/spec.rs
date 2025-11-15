#[cfg(all(
    feature = "std-openai-external",
    not(feature = "provider-xai-external")
))]
use crate::core::provider_spec::{
    bridge_core_chat_transformers, map_core_stream_event_with_provider,
    openai_like_chat_request_to_core_input,
};
use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::http::headers::ProviderHeaders;
#[cfg(all(
    feature = "std-openai-external",
    not(feature = "provider-xai-external")
))]
use crate::std_openai::openai::chat::{OpenAiChatAdapter, OpenAiChatStandard};
use crate::traits::ProviderCapabilities;
use crate::types::{ChatRequest, ProviderOptions};
use reqwest::header::HeaderMap;
use std::sync::Arc;

/// xAI ProviderSpec implementation
#[derive(Clone, Copy, Default)]
pub struct XaiSpec;

impl ProviderSpec for XaiSpec {
    fn id(&self) -> &'static str {
        "xai"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        #[cfg(feature = "provider-xai-external")]
        {
            use siumai_core::provider_spec::CoreProviderSpec;

            let core_ctx = ctx.to_core_context();
            let core_spec = siumai_provider_xai::XaiCoreSpec::new();
            return core_spec.build_headers(&core_ctx);
        }

        #[cfg(not(feature = "provider-xai-external"))]
        {
            let api_key = ctx
                .api_key
                .as_ref()
                .ok_or_else(|| LlmError::MissingApiKey("xAI API key not provided".into()))?;
            return ProviderHeaders::xai(api_key, &ctx.http_extra_headers);
        }
    }

    fn chat_url(
        &self,
        _stream: bool,
        _req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> String {
        #[cfg(feature = "provider-xai-external")]
        {
            use siumai_core::provider_spec::CoreProviderSpec;

            let core_ctx = ctx.to_core_context();
            let core_spec = siumai_provider_xai::XaiCoreSpec::new();
            return core_spec.chat_url(&core_ctx);
        }

        #[cfg(not(feature = "provider-xai-external"))]
        {
            // Base URL already includes "/v1" (see XaiConfig::new default).
            // Append only the operation path to avoid duplicating the prefix.
            format!("{}/chat/completions", ctx.base_url.trim_end_matches('/'))
        }
    }

    fn choose_chat_transformers(
        &self,
        req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        // Shared helper: 将 ChatRequest 映射为带有 xAI ProviderOptions 的 ChatInput。
        fn xai_chat_request_to_core_input(
            req: &crate::types::ChatRequest,
        ) -> siumai_core::execution::chat::ChatInput {
            use serde_json::Value;

            let mut input = openai_like_chat_request_to_core_input(req);

            if let ProviderOptions::Xai(ref options) = req.provider_options {
                if let Some(ref sp) = options.search_parameters
                    && let Ok(v) = serde_json::to_value(sp)
                {
                    input.extra.insert("xai_search_parameters".to_string(), v);
                }

                if let Some(ref effort) = options.reasoning_effort {
                    input.extra.insert(
                        "xai_reasoning_effort".to_string(),
                        Value::String(effort.clone()),
                    );
                }
            }

            input
        }

        #[cfg(feature = "provider-xai-external")]
        {
            use crate::core::provider_spec::{
                bridge_core_chat_transformers, map_core_stream_event_with_provider,
            };
            use siumai_core::provider_spec::{CoreChatTransformers, CoreProviderSpec};

            let core_ctx = ctx.to_core_context();
            let core_spec = siumai_provider_xai::XaiCoreSpec::new();
            let core_txs: CoreChatTransformers =
                core_spec.choose_chat_transformers(&xai_chat_request_to_core_input(req), &core_ctx);

            return bridge_core_chat_transformers(
                core_txs,
                xai_chat_request_to_core_input,
                |evt| map_core_stream_event_with_provider("xai", evt),
            );
        }

        #[cfg(all(
            feature = "std-openai-external",
            not(feature = "provider-xai-external")
        ))]
        {
            use siumai_core::execution::chat::ChatInput;
            use siumai_core::provider_spec::CoreChatTransformers;

            // xAI uses an OpenAI-compatible Chat Completions surface. We reuse the
            // OpenAI standard and keep xAI-specific behavior in ProviderSpec
            // (headers, URL, provider_options-based hooks).
            struct XaiOpenAiChatAdapter;

            impl OpenAiChatAdapter for XaiOpenAiChatAdapter {
                fn transform_request(
                    &self,
                    input: &ChatInput,
                    body: &mut serde_json::Value,
                ) -> Result<(), LlmError> {
                    // 与 core 层适配器保持一致：从 `xai_*` extra 中读取配置。
                    if let Some(params) = input.extra.get("xai_search_parameters") {
                        body["search_parameters"] = params.clone();
                    }
                    if let Some(effort) = input.extra.get("xai_reasoning_effort") {
                        body["reasoning_effort"] = effort.clone();
                    }
                    Ok(())
                }

                fn transform_response(
                    &self,
                    _resp: &mut serde_json::Value,
                ) -> Result<(), LlmError> {
                    Ok(())
                }

                fn transform_sse_event(
                    &self,
                    _event: &mut serde_json::Value,
                ) -> Result<(), LlmError> {
                    Ok(())
                }
            }

            let std = OpenAiChatStandard::with_adapter(Arc::new(XaiOpenAiChatAdapter));
            let core_txs: CoreChatTransformers = CoreChatTransformers {
                request: std.create_request_transformer("xai"),
                response: std.create_response_transformer("xai"),
                stream: Some(std.create_stream_converter("xai")),
            };

            return bridge_core_chat_transformers(
                core_txs,
                xai_chat_request_to_core_input,
                |evt| map_core_stream_event_with_provider("xai", evt),
            );
        }

        #[cfg(not(feature = "std-openai-external"))]
        {
            let req_tx = crate::providers::xai::transformers::XaiRequestTransformer;
            let resp_tx = crate::providers::xai::transformers::XaiResponseTransformer;
            let inner = crate::providers::xai::streaming::XaiEventConverter::new();
            let stream_tx = crate::providers::xai::transformers::XaiStreamChunkTransformer {
                provider_id: "xai".to_string(),
                inner,
            };
            ChatTransformers {
                request: Arc::new(req_tx),
                response: Arc::new(resp_tx),
                stream: Some(Arc::new(stream_tx)),
                json: None,
            }
        }
    }

    fn chat_before_send(
        &self,
        req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        // 1. 仍然支持 CustomProviderOptions 注入
        crate::core::default_custom_options_hook(self.id(), req)
    }
}
