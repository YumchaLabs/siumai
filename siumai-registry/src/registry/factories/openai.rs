//! Provider factory implementations.

use super::*;

/// OpenAI provider factory
#[cfg(feature = "openai")]
pub struct OpenAIProviderFactory;

#[cfg(feature = "openai")]
#[async_trait::async_trait]
impl ProviderFactory for OpenAIProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        let meta = crate::native_provider_metadata::native_providers_metadata();
        meta.into_iter()
            .find(|m| m.id == "openai")
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
        use crate::execution::http::client::build_http_client_from_config;

        // Resolve HTTP configuration and client (prefer provided client).
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
            std::env::var("OPENAI_API_KEY").map_err(|_| {
                LlmError::ConfigurationError(
                    "Missing OPENAI_API_KEY or explicit api_key in BuildContext".to_string(),
                )
            })?
        };

        // Resolve base URL (context override 鈫?default).
        let base_url = crate::utils::builder_helpers::resolve_base_url(
            ctx.base_url.clone(),
            "https://api.openai.com/v1",
        );

        // Resolve common parameters (model, temperature, max_tokens, etc.).
        let common_params = crate::utils::builder_helpers::resolve_common_params(
            ctx.common_params.clone(),
            model_id,
        );

        let mode = match ctx.provider_id.as_deref() {
            Some("openai-chat") => crate::registry::factory::OpenAiChatApiMode::ChatCompletions,
            // Default to Responses API for OpenAI (Vercel-aligned).
            _ => crate::registry::factory::OpenAiChatApiMode::Responses,
        };

        // Delegate to the shared OpenAI client builder used by SiumaiBuilder.
        match mode {
            crate::registry::factory::OpenAiChatApiMode::Responses => {
                crate::registry::factory::build_openai_client(
                    api_key,
                    base_url,
                    http_client,
                    common_params,
                    http_config,
                    None,
                    ctx.organization.clone(),
                    ctx.project.clone(),
                    ctx.tracing_config.clone(),
                    ctx.retry_options.clone(),
                    ctx.http_interceptors.clone(),
                    ctx.model_middlewares.clone(),
                    ctx.http_transport.clone(),
                )
                .await
            }
            crate::registry::factory::OpenAiChatApiMode::ChatCompletions => {
                crate::registry::factory::build_openai_chat_completions_client(
                    api_key,
                    base_url,
                    http_client,
                    common_params,
                    http_config,
                    None,
                    ctx.organization.clone(),
                    ctx.project.clone(),
                    ctx.tracing_config.clone(),
                    ctx.retry_options.clone(),
                    ctx.http_interceptors.clone(),
                    ctx.model_middlewares.clone(),
                    ctx.http_transport.clone(),
                )
                .await
            }
        }
    }

    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // For OpenAI, embeddings are served by the same client as chat.
        self.language_model_with_ctx(model_id, ctx).await
    }

    async fn image_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Image generation is also handled by the unified OpenAI client.
        self.language_model_with_ctx(model_id, ctx).await
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
        std::borrow::Cow::Borrowed("openai")
    }
}
