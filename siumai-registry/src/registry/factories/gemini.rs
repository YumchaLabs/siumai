//! Provider factory implementations.

use super::*;

/// Gemini provider factory
#[cfg(feature = "google")]
pub struct GeminiProviderFactory;

#[cfg(feature = "google")]
#[async_trait::async_trait]
impl ProviderFactory for GeminiProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        let meta = crate::native_provider_metadata::native_providers_metadata();
        meta.into_iter()
            .find(|m| m.id == "gemini")
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

        // Detect whether an explicit Authorization header or token provider is present.
        let has_auth_header = http_config
            .headers
            .keys()
            .any(|k| k.eq_ignore_ascii_case("authorization"));
        let has_token_provider = ctx
            .gemini_token_provider
            .as_ref()
            .map(|_| true)
            .unwrap_or(false);
        let requires_api_key = !(has_auth_header || has_token_provider);

        // Resolve API key: context override 鈫?GEMINI_API_KEY (when required) 鈫?empty for token-based auth.
        let api_key = if let Some(key) = &ctx.api_key {
            key.clone()
        } else if requires_api_key {
            std::env::var("GEMINI_API_KEY").map_err(|_| {
                LlmError::ConfigurationError(
                    "Missing GEMINI_API_KEY or explicit api_key in BuildContext (or provide Authorization header / token provider)"
                        .to_string(),
                )
            })?
        } else {
            String::new()
        };

        // Resolve base URL (context override 鈫?default).
        let base_url = crate::utils::builder_helpers::resolve_base_url(
            ctx.base_url.clone(),
            "https://generativelanguage.googleapis.com/v1beta",
        );

        // Resolve common parameters.
        let common_params = crate::utils::builder_helpers::resolve_common_params(
            ctx.common_params.clone(),
            model_id,
        );

        crate::registry::factory::build_gemini_client(
            api_key,
            base_url,
            http_client,
            common_params,
            http_config,
            None,
            ctx.gemini_token_provider.clone(),
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
        // Gemini uses a unified client for chat/embeddings/images.
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
        std::borrow::Cow::Borrowed("gemini")
    }
}
