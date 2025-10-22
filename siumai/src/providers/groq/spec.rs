use crate::error::LlmError;
use crate::provider_core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::traits::ProviderCapabilities;
use crate::utils::http_headers::{ProviderHeaders, inject_tracing_headers};
use reqwest::header::HeaderMap;
use std::sync::Arc;

/// Groq ProviderSpec implementation
#[derive(Clone, Copy, Default)]
pub struct GroqSpec;

impl ProviderSpec for GroqSpec {
    fn id(&self) -> &'static str {
        "groq"
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
            .ok_or_else(|| LlmError::MissingApiKey("Groq API key not provided".into()))?;
        let mut headers = ProviderHeaders::groq(api_key, &ctx.http_extra_headers)?;
        inject_tracing_headers(&mut headers);
        Ok(headers)
    }

    fn chat_url(
        &self,
        _stream: bool,
        _req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> String {
        // Groq uses OpenAI-compatible API with /openai/v1 prefix
        format!(
            "{}/openai/v1/chat/completions",
            ctx.base_url.trim_end_matches('/')
        )
    }

    fn choose_chat_transformers(
        &self,
        _req: &crate::types::ChatRequest,
        _ctx: &ProviderContext,
    ) -> ChatTransformers {
        let req_tx = crate::providers::groq::transformers::GroqRequestTransformer;
        let resp_tx = crate::providers::groq::transformers::GroqResponseTransformer;
        let inner = crate::providers::groq::streaming::GroqEventConverter::new();
        let stream_tx = crate::providers::groq::transformers::GroqStreamChunkTransformer {
            provider_id: "groq".to_string(),
            inner,
        };
        ChatTransformers {
            request: Arc::new(req_tx),
            response: Arc::new(resp_tx),
            stream: Some(Arc::new(stream_tx)),
            json: None,
        }
    }

    fn audio_base_url(&self, ctx: &ProviderContext) -> String {
        ctx.base_url.trim_end_matches('/').to_string()
    }

    fn choose_audio_transformer(
        &self,
        _ctx: &ProviderContext,
    ) -> crate::provider_core::AudioTransformer {
        crate::provider_core::AudioTransformer {
            transformer: Arc::new(crate::providers::groq::transformers::GroqAudioTransformer),
        }
    }
}
