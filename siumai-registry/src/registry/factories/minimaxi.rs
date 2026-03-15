//! Provider factory implementations.

use super::*;
use crate::image::ImageModel as FamilyImageModel;
use crate::provider::ids;
use crate::text::LanguageModel as FamilyLanguageModel;
use siumai_core::speech::SpeechModel as FamilySpeechModel;

/// MiniMaxi provider factory
#[cfg(feature = "minimaxi")]
pub struct MiniMaxiProviderFactory;

#[cfg(feature = "minimaxi")]
impl MiniMaxiProviderFactory {
    async fn build_typed_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<siumai_provider_minimaxi::providers::minimaxi::client::MinimaxiClient, LlmError>
    {
        use siumai_provider_minimaxi::providers::minimaxi::client::MinimaxiClient;
        use siumai_provider_minimaxi::providers::minimaxi::config::MinimaxiConfig;

        let http_config = ctx.http_config.clone().unwrap_or_default();
        let http_client = if let Some(client) = &ctx.http_client {
            client.clone()
        } else {
            build_http_client_from_config(&http_config)?
        };

        let api_key = if let Some(key) = &ctx.api_key {
            key.clone()
        } else {
            std::env::var("MINIMAXI_API_KEY").map_err(|_| {
                LlmError::ConfigurationError(
                    "Missing MINIMAXI_API_KEY or explicit api_key in BuildContext".to_string(),
                )
            })?
        };

        let base_url = crate::utils::builder_helpers::resolve_base_url(
            ctx.base_url.clone(),
            MinimaxiConfig::DEFAULT_BASE_URL,
        );

        let common_params = crate::utils::builder_helpers::resolve_common_params(
            ctx.common_params.clone(),
            model_id,
        );

        let mut model_middlewares = crate::execution::middleware::build_auto_middlewares_vec(
            "minimaxi",
            &common_params.model,
        );
        model_middlewares.extend(ctx.model_middlewares.clone());

        let mut config = MinimaxiConfig::new(api_key)
            .with_base_url(base_url)
            .with_http_config(http_config)
            .with_http_interceptors(ctx.http_interceptors.clone())
            .with_model_middlewares(model_middlewares);
        if let Some(http_transport) = ctx.http_transport.clone() {
            config = config.with_http_transport(http_transport);
        }
        config.common_params = common_params;

        let mut client = MinimaxiClient::with_http_client(config, http_client)?;
        if let Some(tracing_config) = ctx.tracing_config.clone() {
            client = client.with_tracing(tracing_config);
        }
        if let Some(retry_options) = ctx.retry_options.clone() {
            client = client.with_retry(retry_options);
        }

        Ok(client)
    }
}

#[cfg(feature = "minimaxi")]
#[async_trait::async_trait]
impl ProviderFactory for MiniMaxiProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        let meta = crate::native_provider_metadata::native_providers_metadata();
        meta.into_iter()
            .find(|m| m.id == ids::MINIMAXI)
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
        let client = self.build_typed_client_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn language_model_text_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyLanguageModel>, LlmError> {
        let client = self.build_typed_client_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn embedding_model_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "MiniMaxi does not currently expose a provider-owned embedding family path".to_string(),
        ))
    }

    async fn image_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.language_model_with_ctx(model_id, ctx).await
    }

    async fn image_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyImageModel>, LlmError> {
        let client = self.build_typed_client_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn speech_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = self.build_typed_client_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn speech_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilySpeechModel>, LlmError> {
        let client = self.build_typed_client_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn transcription_model_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "MiniMaxi does not currently expose a provider-owned transcription family path"
                .to_string(),
        ))
    }

    async fn reranking_model_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "MiniMaxi does not currently expose a provider-owned reranking family path".to_string(),
        ))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(ids::MINIMAXI)
    }
}
