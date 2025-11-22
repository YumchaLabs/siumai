use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::traits::ProviderCapabilities;
use crate::types::{ChatRequest, ProviderOptions};
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
        use siumai_core::provider_spec::CoreProviderSpec;

        // Delegate header construction to core-level Groq spec, which in turn
        // uses the OpenAI Chat standard implementation.
        let core_ctx = ctx.to_core_context();
        let core_spec = siumai_provider_groq::GroqCoreSpec::new();
        core_spec.build_headers(&core_ctx)
    }

    fn chat_url(
        &self,
        _stream: bool,
        _req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> String {
        use siumai_core::provider_spec::CoreProviderSpec;

        // Delegate route resolution to core-level Groq spec.
        let core_ctx = ctx.to_core_context();
        let core_spec = siumai_provider_groq::GroqCoreSpec::new();
        core_spec.chat_url(&core_ctx)
    }

    fn choose_chat_transformers(
        &self,
        req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        use crate::core::provider_spec::{
            bridge_core_chat_transformers, groq_chat_request_to_core_input,
            map_core_stream_event_with_provider,
        };
        use siumai_core::provider_spec::{CoreChatTransformers, CoreProviderSpec};

        let core_ctx = ctx.to_core_context();
        let core_input = groq_chat_request_to_core_input(req);

        let core_spec = siumai_provider_groq::GroqCoreSpec::new();
        let core_txs: CoreChatTransformers =
            core_spec.choose_chat_transformers(&core_input, &core_ctx);

        bridge_core_chat_transformers(core_txs, groq_chat_request_to_core_input, |evt| {
            map_core_stream_event_with_provider("groq", evt)
        })
    }

    fn audio_base_url(&self, ctx: &ProviderContext) -> String {
        ctx.base_url.trim_end_matches('/').to_string()
    }

    fn choose_audio_transformer(&self, _ctx: &ProviderContext) -> crate::core::AudioTransformer {
        crate::core::AudioTransformer {
            transformer: Arc::new(crate::providers::groq::transformers::GroqAudioTransformer),
        }
    }
}
