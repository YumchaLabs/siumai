use crate::error::LlmError;
use crate::provider_core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::traits::ProviderCapabilities;
use crate::types::{ChatRequest, ProviderOptions};
use crate::utils::http_headers::{ProviderHeaders, inject_tracing_headers};
use reqwest::header::HeaderMap;
use std::sync::Arc;

/// Anthropic ProviderSpec implementation
#[derive(Clone, Copy, Default)]
pub struct AnthropicSpec;

impl ProviderSpec for AnthropicSpec {
    fn id(&self) -> &'static str {
        "anthropic"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_custom_feature("prompt_caching", true)
            .with_custom_feature("thinking_mode", true)
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        let api_key = ctx
            .api_key
            .as_ref()
            .ok_or_else(|| LlmError::MissingApiKey("Anthropic API key not provided".into()))?;
        let mut headers = ProviderHeaders::anthropic(api_key, &ctx.http_extra_headers)?;
        inject_tracing_headers(&mut headers);
        Ok(headers)
    }

    fn chat_url(&self, _stream: bool, _req: &ChatRequest, ctx: &ProviderContext) -> String {
        format!("{}/v1/messages", ctx.base_url.trim_end_matches('/'))
    }

    fn choose_chat_transformers(
        &self,
        _req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> ChatTransformers {
        let req_tx =
            crate::providers::anthropic::transformers::AnthropicRequestTransformer::new(None);
        let resp_tx = crate::providers::anthropic::transformers::AnthropicResponseTransformer;
        let stream_converter = crate::providers::anthropic::streaming::AnthropicEventConverter::new(
            crate::params::AnthropicParams::default(),
        );
        let stream_tx =
            crate::providers::anthropic::transformers::AnthropicStreamChunkTransformer {
                provider_id: "anthropic".to_string(),
                inner: stream_converter,
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
        // Handle Custom variant for user extensions
        if let ProviderOptions::Custom {
            provider_id,
            options,
        } = &req.provider_options
        {
            if provider_id == "anthropic" {
                let custom_options = options.clone();
                let hook = move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
                    let mut out = body.clone();
                    if let Some(obj) = out.as_object_mut() {
                        for (k, v) in &custom_options {
                            obj.insert(k.clone(), v.clone());
                        }
                    }
                    Ok(out)
                };
                return Some(Arc::new(hook));
            }
        }

        // 🎯 Extract Anthropic-specific options from provider_options
        let thinking_mode = if let ProviderOptions::Anthropic(ref options) = req.provider_options {
            options.thinking_mode.clone()
        } else {
            return None;
        };

        // Check if we have thinking mode to inject
        if thinking_mode.is_none() {
            return None;
        }

        let hook = move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
            let mut out = body.clone();

            // 🎯 Inject thinking mode configuration
            // According to Anthropic API: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
            if let Some(ref thinking) = thinking_mode {
                if thinking.enabled {
                    let mut thinking_config = serde_json::json!({
                        "type": "enabled"
                    });

                    // Add budget_tokens if specified (minimum 1024)
                    if let Some(budget) = thinking.thinking_budget {
                        thinking_config["budget_tokens"] = serde_json::json!(budget);
                    }

                    out["thinking"] = thinking_config;
                }
            }

            Ok(out)
        };
        Some(Arc::new(hook))
    }
}
