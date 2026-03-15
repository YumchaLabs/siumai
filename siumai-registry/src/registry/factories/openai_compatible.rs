//! Provider factory implementations.

use super::*;
use crate::embedding::EmbeddingModel as FamilyEmbeddingModel;
use crate::image::ImageModel as FamilyImageModel;
use crate::text::LanguageModel as FamilyLanguageModel;
use siumai_core::rerank::RerankingModel as FamilyRerankingModel;
use siumai_core::speech::SpeechModel as FamilySpeechModel;
use siumai_core::standards::openai::compat::provider_registry::provider_config_declares_chat_surface;
use siumai_core::transcription::TranscriptionModel as FamilyTranscriptionModel;

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

    fn ensure_capability(&self, capability: &str) -> Result<(), LlmError> {
        if self.capabilities().supports(capability) {
            return Ok(());
        }

        Err(LlmError::UnsupportedOperation(format!(
            "OpenAI-compatible provider '{}' does not expose the '{}' family path",
            self.provider_id, capability
        )))
    }

    async fn build_text_family_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<
        siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleClient,
        LlmError,
    > {
        let http_config = ctx.http_config.clone().unwrap_or_default();
        let provider_config =
            siumai_provider_openai_compatible::providers::openai_compatible::get_provider_config(
                &self.provider_id,
            );
        let common_params = crate::utils::builder_helpers::resolve_common_params(
            ctx.common_params.clone(),
            model_id,
        );

        if provider_config.is_some() {
            let mut builder = siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleBuilder::new(
                siumai_provider_openai_compatible::builder::BuilderBase::default(),
                &self.provider_id,
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
                builder = builder.stop(stop_sequences);
            }
            if let Some(seed) = common_params.seed {
                builder = builder.seed(seed);
            }
            if let Some(enabled) = ctx.reasoning_enabled {
                builder = builder.reasoning(enabled);
            }
            if let Some(budget) = ctx.reasoning_budget {
                builder = builder.reasoning_budget(budget);
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

            return builder.build().await;
        }

        let http_client = if let Some(client) = &ctx.http_client {
            client.clone()
        } else {
            build_http_client_from_config(&http_config)?
        };

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

        crate::registry::factory::build_openai_compatible_typed_client(
            self.provider_id.clone(),
            api_key,
            ctx.base_url.clone(),
            http_client,
            common_params,
            ctx.reasoning_enabled,
            ctx.reasoning_budget,
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
}

#[cfg(feature = "openai")]
#[async_trait::async_trait]
impl ProviderFactory for OpenAICompatibleProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        let mut caps = ProviderCapabilities::new();
        let Some(cfg) =
            siumai_provider_openai_compatible::providers::openai_compatible::get_provider_config(
                &self.provider_id,
            )
        else {
            return caps.with_chat().with_streaming();
        };

        let has_audio = cfg.capabilities.iter().any(|cap| cap == "audio");
        let has_speech = cfg
            .capabilities
            .iter()
            .any(|cap| matches!(cap.as_str(), "speech" | "tts"));
        let has_transcription = cfg
            .capabilities
            .iter()
            .any(|cap| matches!(cap.as_str(), "transcription" | "stt"));

        if provider_config_declares_chat_surface(&cfg) {
            caps = caps.with_chat().with_streaming();
        }

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
                "image_generation" => {
                    caps = caps.with_image_generation();
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

        if has_audio || (has_speech && has_transcription) {
            caps = caps.with_audio();
        } else {
            if has_speech {
                caps = caps.with_speech();
            }
            if has_transcription {
                caps = caps.with_transcription();
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
        self.ensure_capability("chat")?;
        let client = self.build_text_family_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn language_model_text_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyLanguageModel>, LlmError> {
        self.ensure_capability("chat")?;
        let client = self.build_text_family_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.ensure_capability("embedding")?;
        let client = self.build_text_family_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn embedding_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyEmbeddingModel>, LlmError> {
        self.ensure_capability("embedding")?;
        let client = self.build_text_family_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn image_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.ensure_capability("image_generation")?;
        let client = self.build_text_family_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn image_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyImageModel>, LlmError> {
        self.ensure_capability("image_generation")?;
        let client = self.build_text_family_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn reranking_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.ensure_capability("rerank")?;
        let client = self.build_text_family_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn reranking_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyRerankingModel>, LlmError> {
        self.ensure_capability("rerank")?;
        let client = self.build_text_family_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn speech_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.ensure_capability("speech")?;
        let client = self.build_text_family_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn speech_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilySpeechModel>, LlmError> {
        self.ensure_capability("speech")?;
        let client = self.build_text_family_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn transcription_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.ensure_capability("transcription")?;
        let client = self.build_text_family_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn transcription_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyTranscriptionModel>, LlmError> {
        self.ensure_capability("transcription")?;
        let client = self.build_text_family_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("openai-compatible")
    }
}
