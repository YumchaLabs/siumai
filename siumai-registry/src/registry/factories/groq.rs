//! Provider factory implementations.

use super::*;
use crate::provider::ids;
use crate::text::LanguageModel as FamilyLanguageModel;
use siumai_core::speech::SpeechModel as FamilySpeechModel;
use siumai_core::transcription::TranscriptionModel as FamilyTranscriptionModel;
use siumai_provider_groq::providers::groq::GroqClient;

/// Groq provider factory
#[cfg(feature = "groq")]
pub struct GroqProviderFactory;

#[cfg(feature = "groq")]
impl GroqProviderFactory {
    async fn build_text_family_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<GroqClient, LlmError> {
        let http_config = ctx.http_config.clone().unwrap_or_default();
        let common_params = crate::utils::builder_helpers::resolve_common_params(
            ctx.common_params.clone(),
            model_id,
        );
        let mut builder = siumai_provider_groq::providers::groq::GroqBuilder::new(
            siumai_provider_groq::builder::BuilderBase::default(),
        )
        .model(common_params.model.clone())
        .with_http_config(http_config)
        .with_model_middlewares(ctx.model_middlewares.clone());

        if let Some(api_key) = ctx.api_key.clone() {
            builder = builder.api_key(api_key);
        }
        if let Some(base_url) = ctx.base_url.clone() {
            builder = builder.base_url(base_url);
        }
        if let Some(temperature) = common_params.temperature {
            builder = builder.temperature(temperature);
        }
        if let Some(max_tokens) = common_params.max_tokens {
            builder = builder.max_tokens(max_tokens);
        }
        if let Some(top_p) = common_params.top_p {
            builder = builder.top_p(top_p);
        }
        if let Some(stop_sequences) = common_params.stop_sequences.clone() {
            builder = builder.stop_sequences(stop_sequences);
        }
        if let Some(seed) = common_params.seed {
            builder = builder.seed(seed);
        }
        if let Some(http_client) = ctx.http_client.clone() {
            builder = builder.with_http_client(http_client);
        }
        if let Some(transport) = ctx.http_transport.clone() {
            builder = builder.fetch(transport);
        }
        if let Some(retry_options) = ctx.retry_options.clone() {
            builder = builder.with_retry(retry_options);
        }
        for interceptor in ctx.http_interceptors.clone() {
            builder = builder.with_http_interceptor(interceptor);
        }

        builder.build().await
    }
}

#[cfg(feature = "groq")]
#[async_trait::async_trait]
impl ProviderFactory for GroqProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        let meta = crate::native_provider_metadata::native_providers_metadata();
        meta.into_iter()
            .find(|m| m.id == ids::GROQ)
            .map(|m| m.capabilities)
            .unwrap_or_else(ProviderCapabilities::new)
    }

    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        let ctx = BuildContext::default();
        self.language_model_with_ctx(model_id, &ctx).await
    }

    async fn language_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = self.build_text_family_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn language_model_text_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyLanguageModel>, LlmError> {
        let client = self.build_text_family_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn embedding_model_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Groq does not currently expose a provider-owned embedding family path".to_string(),
        ))
    }

    async fn image_model_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Groq does not currently expose a provider-owned image family path".to_string(),
        ))
    }

    async fn speech_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.language_model_with_ctx(model_id, ctx).await
    }

    async fn speech_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilySpeechModel>, LlmError> {
        let client = self.build_text_family_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn transcription_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.language_model_with_ctx(model_id, ctx).await
    }

    async fn transcription_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyTranscriptionModel>, LlmError> {
        let client = self.build_text_family_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(ids::GROQ)
    }
}
