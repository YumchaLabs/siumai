//! Provider factory implementations.

use super::*;
use crate::embedding::EmbeddingModel as FamilyEmbeddingModel;
use crate::provider::ids;
use crate::text::LanguageModel as FamilyLanguageModel;
use siumai_provider_ollama::providers::ollama::{OllamaClient, OllamaConfig};

/// Ollama provider factory
#[cfg(feature = "ollama")]
pub struct OllamaProviderFactory;

#[cfg(feature = "ollama")]
impl OllamaProviderFactory {
    async fn build_text_family_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<OllamaClient, LlmError> {
        let http_config = ctx.http_config.clone().unwrap_or_default();
        let http_client = if let Some(client) = &ctx.http_client {
            client.clone()
        } else {
            build_http_client_from_config(&http_config)?
        };

        let base_url = crate::utils::builder_helpers::resolve_base_url(
            ctx.base_url.clone(),
            "http://localhost:11434",
        );
        let common_params = crate::utils::builder_helpers::resolve_common_params(
            ctx.common_params.clone(),
            model_id,
        );

        let mut auto_mws = crate::execution::middleware::build_auto_middlewares_vec(
            ids::OLLAMA,
            &common_params.model,
        );
        auto_mws.extend(ctx.model_middlewares.clone());

        let config = OllamaConfig::builder()
            .base_url(base_url)
            .model(common_params.model.clone())
            .common_params(common_params)
            .http_config(http_config)
            .http_transport_opt(ctx.http_transport.clone())
            .http_interceptors(ctx.http_interceptors.clone())
            .model_middlewares(auto_mws)
            .build()?;

        let mut client = OllamaClient::with_http_client(config, http_client)?;

        if let Some(tracing_config) = &ctx.tracing_config {
            client.set_tracing_config(Some(tracing_config.clone()));
        }
        if let Some(retry_options) = &ctx.retry_options {
            client.set_retry_options(Some(retry_options.clone()));
        }

        Ok(client)
    }
}

#[cfg(feature = "ollama")]
#[async_trait::async_trait]
impl ProviderFactory for OllamaProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        let meta = crate::native_provider_metadata::native_providers_metadata();
        meta.into_iter()
            .find(|m| m.id == ids::OLLAMA)
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
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.language_model_with_ctx(model_id, ctx).await
    }

    async fn embedding_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyEmbeddingModel>, LlmError> {
        let client = self.build_text_family_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn image_model_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Ollama does not currently expose a provider-owned image family path".to_string(),
        ))
    }

    async fn speech_model_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Ollama does not currently expose a provider-owned speech family path".to_string(),
        ))
    }

    async fn transcription_model_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Ollama does not currently expose a provider-owned transcription family path"
                .to_string(),
        ))
    }

    async fn reranking_model_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Ollama does not currently expose a provider-owned reranking family path".to_string(),
        ))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(ids::OLLAMA)
    }
}
