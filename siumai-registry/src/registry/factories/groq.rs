//! Provider factory implementations.

use super::*;
use crate::provider::ids;

/// Groq provider factory
#[cfg(feature = "groq")]
pub struct GroqProviderFactory;

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
        // Delegate to the context-aware implementation with default context.
        let ctx = BuildContext::default();
        self.language_model_with_ctx(model_id, &ctx).await
    }

    async fn language_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Resolve HTTP configuration and client.
        let http_config = ctx.http_config.clone().unwrap_or_default();
        let http_client = if let Some(client) = &ctx.http_client {
            client.clone()
        } else {
            build_http_client_from_config(&http_config)?
        };

        // Resolve API key for Groq (OpenAI-compatible).
        let api_key = if let Some(key) = &ctx.api_key {
            key.clone()
        } else {
            std::env::var("GROQ_API_KEY").map_err(|_| {
                LlmError::ConfigurationError(
                    "Missing GROQ_API_KEY or explicit api_key in BuildContext".to_string(),
                )
            })?
        };

        // Resolve common parameters.
        let common_params = crate::utils::builder_helpers::resolve_common_params(
            ctx.common_params.clone(),
            model_id,
        );

        // OpenAI-compatible Groq client (configuration-driven).
        let provider_config =
            siumai_provider_openai_compatible::providers::openai_compatible::get_provider_config(
                "groq",
            )
            .ok_or_else(|| {
                LlmError::ConfigurationError("Unknown OpenAI-compatible provider id: groq".into())
            })?;

        let adapter: Arc<
            dyn siumai_provider_openai_compatible::providers::openai_compatible::ProviderAdapter,
        > = Arc::new(
            siumai_provider_openai_compatible::providers::openai_compatible::ConfigurableAdapter::new(
                provider_config,
            ),
        );

        // Groq uses `/openai/v1`. If a custom base URL is provided without a path, append it.
        let base_url = if let Some(custom) = ctx.base_url.clone() {
            let path = custom.splitn(4, '/').nth(3).unwrap_or("");
            if path.is_empty() {
                format!("{}/openai/v1", custom.trim_end_matches('/'))
            } else {
                custom
            }
        } else {
            adapter.base_url().to_string()
        };

        let mut config =
            siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleConfig::new(
                "groq", &api_key, &base_url, adapter,
            );

        if !common_params.model.is_empty() {
            config = config.with_model(&common_params.model);
        }
        config = config.with_common_params(common_params.clone());
        config = config.with_http_config(http_config.clone());
        if let Some(transport) = ctx.http_transport.clone() {
            config = config.with_http_transport(transport);
        }

        let mut client =
            siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleClient::with_http_client(
                config,
                http_client,
            )
            .await?;

        // Apply retry options when present.
        if let Some(opts) = &ctx.retry_options {
            client.set_retry_options(Some(opts.clone()));
        }

        // Install HTTP interceptors.
        if !ctx.http_interceptors.is_empty() {
            client = client.with_http_interceptors(ctx.http_interceptors.clone());
        }

        // Auto + user middlewares.
        let mut auto_mws =
            crate::execution::middleware::build_auto_middlewares_vec("groq", &common_params.model);
        auto_mws.extend(ctx.model_middlewares.clone());
        if !auto_mws.is_empty() {
            client = client.with_model_middlewares(auto_mws);
        }

        Ok(Arc::new(
            siumai_provider_groq::providers::groq::GroqClient::new(client),
        ))
    }

    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Groq client is OpenAI-compatible and unified across capabilities.
        self.language_model_with_ctx(model_id, ctx).await
    }

    async fn image_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    async fn speech_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    async fn transcription_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(ids::GROQ)
    }
}
