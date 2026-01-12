//! Provider factory implementations.

use super::*;

/// Anthropic provider factory
#[cfg(feature = "anthropic")]
pub struct AnthropicProviderFactory;

#[cfg(feature = "anthropic")]
#[async_trait::async_trait]
impl ProviderFactory for AnthropicProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        let meta = crate::native_provider_metadata::native_providers_metadata();
        meta.into_iter()
            .find(|m| m.id == "anthropic")
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

        // Resolve API key: context override 鈫?environment variable.
        let api_key = if let Some(key) = &ctx.api_key {
            key.clone()
        } else {
            std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
                LlmError::ConfigurationError(
                    "Missing ANTHROPIC_API_KEY or explicit api_key in BuildContext".to_string(),
                )
            })?
        };

        // Resolve base URL (context override 鈫?default).
        let base_url = crate::utils::builder_helpers::resolve_base_url(
            ctx.base_url.clone(),
            "https://api.anthropic.com",
        );

        // Resolve common parameters (model, temperature, max_tokens, etc.).
        let common_params = crate::utils::builder_helpers::resolve_common_params(
            ctx.common_params.clone(),
            model_id,
        );

        crate::registry::factory::build_anthropic_client(
            api_key,
            base_url,
            http_client,
            common_params,
            http_config,
            None,
            ctx.tracing_config.clone(),
            ctx.retry_options.clone(),
            ctx.http_interceptors.clone(),
            ctx.model_middlewares.clone(),
            ctx.http_transport.clone(),
        )
        .await
    }

    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Anthropic uses a unified client for chat/embeddings/images.
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
        std::borrow::Cow::Borrowed("anthropic")
    }
}
