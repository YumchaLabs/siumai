use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::traits::ProviderCapabilities;
use crate::types::{ChatRequest, ProviderOptions};
use crate::utils::http_headers::{ProviderHeaders, inject_tracing_headers};
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
        let api_key = ctx
            .api_key
            .as_ref()
            .ok_or_else(|| LlmError::MissingApiKey("xAI API key not provided".into()))?;
        let mut headers = ProviderHeaders::xai(api_key, &ctx.http_extra_headers)?;
        inject_tracing_headers(&mut headers);
        Ok(headers)
    }

    fn chat_url(
        &self,
        _stream: bool,
        _req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> String {
        // xAI uses OpenAI-compatible API with /v1 prefix
        format!("{}/v1/chat/completions", ctx.base_url.trim_end_matches('/'))
    }

    fn choose_chat_transformers(
        &self,
        _req: &crate::types::ChatRequest,
        _ctx: &ProviderContext,
    ) -> ChatTransformers {
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

    fn chat_before_send(
        &self,
        req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> Option<crate::executors::BeforeSendHook> {
        // 1. First check for CustomProviderOptions (using default implementation)
        if let Some(hook) = crate::core::default_custom_options_hook(self.id(), req) {
            return Some(hook);
        }

        // 2. Handle xAI-specific options (search_parameters, reasoning_effort)
        // 🎯 Extract xAI-specific options from provider_options
        let (search_parameters, reasoning_effort) =
            if let ProviderOptions::Xai(ref options) = req.provider_options {
                (
                    options.search_parameters.clone(),
                    options.reasoning_effort.clone(),
                )
            } else {
                return None;
            };

        // Check if we have anything to inject
        if search_parameters.is_none() && reasoning_effort.is_none() {
            return None;
        }

        let hook = move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
            let mut out = body.clone();

            // 🎯 Inject search_parameters
            if let Some(ref params) = search_parameters {
                if let Ok(val) = serde_json::to_value(params) {
                    out["search_parameters"] = val;
                }
            }

            // 🎯 Inject reasoning_effort
            if let Some(ref effort) = reasoning_effort {
                out["reasoning_effort"] = serde_json::Value::String(effort.clone());
            }

            Ok(out)
        };
        Some(Arc::new(hook))
    }
}
