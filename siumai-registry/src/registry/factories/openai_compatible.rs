//! Provider factory implementations.

use super::*;

/// Generic OpenAI-compatible provider factory
#[cfg(feature = "openai")]
pub struct OpenAICompatibleProviderFactory {
    provider_id: String,
}

#[cfg(feature = "openai")]
impl OpenAICompatibleProviderFactory {
    pub fn new(provider_id: String) -> Self {
        Self { provider_id }
    }
}

#[cfg(feature = "openai")]
#[async_trait::async_trait]
impl ProviderFactory for OpenAICompatibleProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        let mut caps = ProviderCapabilities::new().with_chat().with_streaming();
        let Some(cfg) =
            siumai_provider_openai_compatible::providers::openai_compatible::get_provider_config(
                &self.provider_id,
            )
        else {
            return caps;
        };

        for c in cfg.capabilities {
            match c.as_str() {
                "tools" => {
                    caps = caps.with_tools();
                }
                "vision" => {
                    caps = caps.with_vision();
                }
                "embedding" => {
                    caps = caps.with_embedding();
                }
                "rerank" => {
                    caps = caps.with_rerank();
                }
                "reasoning" => {
                    caps = caps.with_custom_feature("thinking", true);
                }
                _ => {}
            }
        }
        caps
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

        // Resolve API key using shared helper (context override + configured envs + fallback).
        let provider_config =
            siumai_provider_openai_compatible::providers::openai_compatible::get_provider_config(
                &self.provider_id,
            );
        let api_key = if let Some(cfg) = &provider_config {
            crate::utils::builder_helpers::get_api_key_with_envs(
                ctx.api_key.clone(),
                &self.provider_id,
                cfg.api_key_env.as_deref(),
                &cfg.api_key_env_aliases,
            )?
        } else {
            crate::utils::builder_helpers::get_api_key_with_env(
                ctx.api_key.clone(),
                &self.provider_id,
            )?
        };

        // Resolve common parameters.
        let common_params = crate::utils::builder_helpers::resolve_common_params(
            ctx.common_params.clone(),
            model_id,
        );

        crate::registry::factory::build_openai_compatible_client(
            self.provider_id.clone(),
            api_key,
            ctx.base_url.clone(),
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
        // Generic OpenAI-compatible client is unified; reuse chat path.
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
        std::borrow::Cow::Borrowed("openai-compatible")
    }
}
