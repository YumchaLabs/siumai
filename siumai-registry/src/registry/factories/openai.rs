//! Provider factory implementations.

use super::*;
use crate::embedding::EmbeddingModel as FamilyEmbeddingModel;
use crate::image::ImageModel as FamilyImageModel;
use crate::text::LanguageModel as FamilyLanguageModel;
use siumai_core::speech::SpeechModel as FamilySpeechModel;
use siumai_core::transcription::TranscriptionModel as FamilyTranscriptionModel;

/// OpenAI provider factory
#[cfg(feature = "openai")]
pub struct OpenAIProviderFactory;

#[cfg(feature = "openai")]
impl OpenAIProviderFactory {
    async fn build_family_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<siumai_provider_openai::providers::openai::OpenAiClient, LlmError> {
        use crate::execution::http::client::build_http_client_from_config;

        let http_config = ctx.http_config.clone().unwrap_or_default();
        let http_client = if let Some(client) = &ctx.http_client {
            client.clone()
        } else {
            build_http_client_from_config(&http_config)?
        };

        let api_key = if let Some(key) = &ctx.api_key {
            key.clone()
        } else {
            std::env::var("OPENAI_API_KEY").map_err(|_| {
                LlmError::ConfigurationError(
                    "Missing OPENAI_API_KEY or explicit api_key in BuildContext".to_string(),
                )
            })?
        };

        let base_url = crate::utils::builder_helpers::resolve_base_url(
            ctx.base_url.clone(),
            "https://api.openai.com/v1",
        );

        let common_params = crate::utils::builder_helpers::resolve_common_params(
            ctx.common_params.clone(),
            model_id,
        );

        let mode = match ctx.provider_id.as_deref() {
            Some(crate::provider::ids::OPENAI_CHAT) => {
                crate::registry::factory::OpenAiChatApiMode::ChatCompletions
            }
            _ => crate::registry::factory::OpenAiChatApiMode::Responses,
        };

        let mut config = siumai_provider_openai::providers::openai::OpenAiConfig::new(api_key)
            .with_base_url(base_url)
            .with_model(common_params.model.clone());

        if let Some(temp) = common_params.temperature {
            config = config.with_temperature(temp);
        }
        if let Some(max_tokens) = common_params.max_tokens {
            config = config.with_max_tokens(max_tokens);
        }
        if let Some(org) = ctx.organization.clone() {
            config = config.with_organization(org);
        }
        if let Some(proj) = ctx.project.clone() {
            config = config.with_project(proj);
        }
        if let Some(transport) = ctx.http_transport.clone() {
            config = config.with_http_transport(transport);
        }

        if mode == crate::registry::factory::OpenAiChatApiMode::Responses {
            let mut overrides = crate::types::ProviderOptionsMap::new();
            overrides.insert(
                "openai",
                serde_json::json!({
                    "responsesApi": { "enabled": true }
                }),
            );
            config.provider_options_map.merge_overrides(overrides);
        }

        let mut client =
            siumai_provider_openai::providers::openai::OpenAiClient::new(config, http_client);
        if let Some(opts) = ctx.retry_options.clone() {
            client.set_retry_options(Some(opts));
        }
        if !ctx.http_interceptors.is_empty() {
            client = client.with_http_interceptors(ctx.http_interceptors.clone());
        }
        let mut auto_mws = crate::execution::middleware::build_auto_middlewares_vec(
            "openai",
            &common_params.model,
        );
        auto_mws.extend(ctx.model_middlewares.clone());
        if !auto_mws.is_empty() {
            client = client.with_model_middlewares(auto_mws);
        }

        Ok(client)
    }
}

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
            Some(crate::provider::ids::OPENAI_CHAT) => {
                crate::registry::factory::OpenAiChatApiMode::ChatCompletions
            }
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

    async fn language_model_text_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyLanguageModel>, LlmError> {
        let client = self.build_family_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = self.build_family_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn embedding_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyEmbeddingModel>, LlmError> {
        let client = self.build_family_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn image_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = self.build_family_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn image_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyImageModel>, LlmError> {
        let client = self.build_family_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn speech_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = self.build_family_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn speech_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilySpeechModel>, LlmError> {
        let client = self.build_family_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn transcription_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = self.build_family_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn transcription_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyTranscriptionModel>, LlmError> {
        let client = self.build_family_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("openai")
    }
}
