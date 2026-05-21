//! TogetherAI provider factory.
//!
//! AI SDK exposes TogetherAI as a single provider surface:
//! - OpenAI-compatible chat/completion/embedding families
//! - provider-owned image + rerank
//!
//! Siumai additionally keeps TogetherAI speech/transcription as an OpenAI-compatible audio
//! extension because Together documents those endpoints, but those audio families are not part of
//! the audited `@ai-sdk/togetherai` package surface.
//!
//! Siumai keeps provider-owned image and rerank clients in `siumai-provider-togetherai` and reuses
//! the shared OpenAI-compatible runtime for text/audio under the canonical `togetherai` provider id.

use super::*;
use crate::provider::ids;
use crate::registry::factories::OpenAICompatibleProviderFactory;
use crate::text::LanguageModel as FamilyLanguageModel;
use crate::traits::{ImageExtras, ImageGenerationCapability};
use crate::traits::{ModelMetadata, ProviderCapabilities};
use crate::types::{
    ImageEditRequest, ImageGenerationRequest, ImageGenerationResponse, ImageVariationRequest,
};
use siumai_core::completion::CompletionModel as FamilyCompletionModel;
use siumai_core::embedding::EmbeddingModel as FamilyEmbeddingModel;
use siumai_core::image::ImageModel as FamilyImageModel;
use siumai_core::rerank::RerankingModel as FamilyRerankingModel;
use siumai_core::speech::SpeechModel as FamilySpeechModel;
use siumai_core::transcription::TranscriptionModel as FamilyTranscriptionModel;
use siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleClient;
use siumai_provider_togetherai::providers::togetherai::{
    TogetherAiClient, TogetherAiConfig, TogetherAiImageClient,
};
use std::borrow::Cow;
use std::sync::Arc;

const DEFAULT_BASE_URL: &str = "https://api.together.xyz/v1";
const DEFAULT_TEXT_MODEL: &str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo";

#[cfg(feature = "togetherai")]
fn togetherai_capabilities() -> ProviderCapabilities {
    OpenAICompatibleProviderFactory::new(ids::TOGETHERAI.to_string())
        .capabilities()
        .with_rerank()
}

#[cfg(feature = "togetherai")]
fn resolve_api_key(ctx: &BuildContext) -> Result<String, LlmError> {
    crate::provider_utils::builder_helpers::get_api_key_with_envs(
        ctx.api_key.clone(),
        ids::TOGETHERAI,
        Some("TOGETHER_API_KEY"),
        &["TOGETHER_AI_API_KEY".to_string()],
    )
    .map_err(|_| {
        LlmError::ConfigurationError(
            "Missing TOGETHER_API_KEY, TOGETHER_AI_API_KEY, or explicit api_key in BuildContext"
                .to_string(),
        )
    })
}

#[cfg(feature = "togetherai")]
fn resolve_root_base_url(ctx: &BuildContext) -> String {
    crate::provider_utils::builder_helpers::resolve_base_url(ctx.base_url.clone(), DEFAULT_BASE_URL)
}

#[cfg(feature = "togetherai")]
fn default_text_model() -> &'static str {
    siumai_provider_openai_compatible::providers::openai_compatible::default_models::get_default_chat_model(
        ids::TOGETHERAI,
    )
    .unwrap_or(DEFAULT_TEXT_MODEL)
}

#[cfg(feature = "togetherai")]
fn build_native_rerank_client_with_ctx(
    model_id: &str,
    ctx: &BuildContext,
) -> Result<TogetherAiClient, LlmError> {
    let http_config = ctx.http_config.clone().unwrap_or_default();
    let http_client = if let Some(client) = &ctx.http_client {
        client.clone()
    } else {
        build_http_client_from_config(&http_config)?
    };

    let mut cfg = siumai_provider_togetherai::providers::togetherai::TogetherAiConfig::new(
        resolve_api_key(ctx)?,
    )
    .with_base_url(crate::provider_utils::builder_helpers::resolve_base_url(
        ctx.base_url.clone(),
        siumai_provider_togetherai::providers::togetherai::TogetherAiConfig::DEFAULT_BASE_URL,
    ))
    .with_model(model_id)
    .with_http_config(http_config)
    .with_http_interceptors(ctx.http_interceptors.clone());

    if let Some(http_transport) = ctx.http_transport.clone() {
        cfg = cfg.with_http_transport(http_transport);
    }

    let mut client = TogetherAiClient::with_http_client(cfg, http_client)?;

    if let Some(retry_options) = ctx.retry_options.clone() {
        client = client.with_retry_options(retry_options);
    }

    Ok(client)
}

#[cfg(feature = "togetherai")]
fn build_rerank_client_arc(
    model_id: &str,
    ctx: &BuildContext,
) -> Result<Arc<TogetherAiClient>, LlmError> {
    let client = build_native_rerank_client_with_ctx(model_id, ctx)?;
    Ok(Arc::new(client))
}

#[cfg(feature = "togetherai")]
async fn build_text_client_with_ctx(
    model_id: &str,
    ctx: &BuildContext,
) -> Result<OpenAiCompatibleClient, LlmError> {
    let http_config = ctx.http_config.clone().unwrap_or_default();
    let http_client = if let Some(client) = &ctx.http_client {
        client.clone()
    } else {
        build_http_client_from_config(&http_config)?
    };

    let common_params = crate::provider_utils::builder_helpers::resolve_common_params(
        ctx.common_params.clone(),
        model_id,
    );

    crate::registry::factory::build_openai_compatible_typed_client(
        ids::TOGETHERAI.to_string(),
        resolve_api_key(ctx)?,
        Some(resolve_root_base_url(ctx)),
        http_client,
        common_params,
        ctx.reasoning_enabled,
        ctx.reasoning_budget,
        http_config,
        None,
        None,
        ctx.tracing_config.clone(),
        ctx.retry_options.clone(),
        ctx.http_interceptors.clone(),
        ctx.model_middlewares.clone(),
        ctx.http_transport.clone(),
    )
    .await
}

#[cfg(feature = "togetherai")]
async fn build_text_client_arc(
    model_id: &str,
    ctx: &BuildContext,
) -> Result<Arc<OpenAiCompatibleClient>, LlmError> {
    let client = build_text_client_with_ctx(model_id, ctx).await?;
    Ok(Arc::new(client))
}

#[cfg(feature = "togetherai")]
fn build_image_client_arc(
    model_id: &str,
    ctx: &BuildContext,
) -> Result<Arc<TogetherAiImageClient>, LlmError> {
    let mut config = TogetherAiConfig::new(resolve_api_key(ctx)?)
        .with_base_url(resolve_root_base_url(ctx))
        .with_model(model_id)
        .with_http_config(ctx.http_config.clone().unwrap_or_default())
        .with_http_interceptors(ctx.http_interceptors.clone());

    if config.common_params.model.is_empty() || config.common_params.model == default_text_model() {
        config = config.with_model(TogetherAiConfig::DEFAULT_IMAGE_MODEL);
    }

    if let Some(transport) = ctx.http_transport.clone() {
        config = config.with_http_transport(transport);
    }

    let mut client = if let Some(http_client) = ctx.http_client.clone() {
        TogetherAiImageClient::with_http_client(config, http_client)?
    } else {
        TogetherAiImageClient::from_config(config)?
    };

    if let Some(retry_options) = ctx.retry_options.clone() {
        client = client.with_retry_options(retry_options);
    }

    Ok(Arc::new(client))
}

#[cfg(feature = "togetherai")]
#[derive(Clone)]
struct TogetherAiCompatCompositeClient {
    text_client: OpenAiCompatibleClient,
    image_client: TogetherAiImageClient,
    rerank_client: TogetherAiClient,
}

#[cfg(feature = "togetherai")]
impl std::fmt::Debug for TogetherAiCompatCompositeClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TogetherAiCompatCompositeClient")
            .field("provider_id", &ids::TOGETHERAI)
            .field("text_model", &self.text_client.model_id())
            .field("image_model", &self.image_client.model_id())
            .field("rerank_model", &self.rerank_client.model_id())
            .finish()
    }
}

#[cfg(feature = "togetherai")]
impl LlmClient for TogetherAiCompatCompositeClient {
    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed(ids::TOGETHERAI)
    }

    fn supported_models(&self) -> Vec<String> {
        let mut models = self.text_client.supported_models();
        for model in self.image_client.supported_models() {
            if !models.iter().any(|existing| existing == &model) {
                models.push(model);
            }
        }
        for model in self.rerank_client.supported_models() {
            if !models.iter().any(|existing| existing == &model) {
                models.push(model);
            }
        }
        models
    }

    fn capabilities(&self) -> ProviderCapabilities {
        togetherai_capabilities()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_chat_capability(&self) -> Option<&dyn crate::traits::ChatCapability> {
        self.text_client.as_chat_capability()
    }

    fn as_embedding_capability(&self) -> Option<&dyn crate::traits::EmbeddingCapability> {
        self.text_client.as_embedding_capability()
    }

    fn as_completion_capability(&self) -> Option<&dyn crate::traits::CompletionCapability> {
        self.text_client.as_completion_capability()
    }

    fn as_embedding_extensions(&self) -> Option<&dyn crate::traits::EmbeddingExtensions> {
        self.text_client.as_embedding_extensions()
    }

    fn as_audio_capability(&self) -> Option<&dyn crate::traits::AudioCapability> {
        self.text_client.as_audio_capability()
    }

    fn as_speech_capability(&self) -> Option<&dyn crate::traits::SpeechCapability> {
        self.text_client.as_speech_capability()
    }

    fn as_speech_extras(&self) -> Option<&dyn crate::traits::SpeechExtras> {
        self.text_client.as_speech_extras()
    }

    fn as_transcription_capability(&self) -> Option<&dyn crate::traits::TranscriptionCapability> {
        self.text_client.as_transcription_capability()
    }

    fn as_transcription_extras(&self) -> Option<&dyn crate::traits::TranscriptionExtras> {
        self.text_client.as_transcription_extras()
    }

    fn as_image_generation_capability(
        &self,
    ) -> Option<&dyn crate::traits::ImageGenerationCapability> {
        Some(self)
    }

    fn as_image_extras(&self) -> Option<&dyn crate::traits::ImageExtras> {
        Some(self)
    }

    fn as_file_management_capability(
        &self,
    ) -> Option<&dyn crate::traits::FileManagementCapability> {
        self.text_client.as_file_management_capability()
    }

    fn as_model_listing_capability(&self) -> Option<&dyn crate::traits::ModelListingCapability> {
        self.text_client.as_model_listing_capability()
    }

    fn as_rerank_capability(&self) -> Option<&dyn crate::traits::RerankCapability> {
        self.rerank_client.as_rerank_capability()
    }
}

#[cfg(feature = "togetherai")]
#[async_trait::async_trait]
impl ImageGenerationCapability for TogetherAiCompatCompositeClient {
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        self.image_client.generate_images(request).await
    }

    fn max_images_per_call(&self) -> Option<u32> {
        ImageGenerationCapability::max_images_per_call(&self.image_client)
    }
}

#[cfg(feature = "togetherai")]
#[async_trait::async_trait]
impl ImageExtras for TogetherAiCompatCompositeClient {
    async fn edit_image(
        &self,
        request: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        self.image_client.edit_image(request).await
    }

    async fn create_variation(
        &self,
        request: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        self.image_client.create_variation(request).await
    }

    fn get_supported_formats(&self) -> Vec<String> {
        self.image_client.get_supported_formats()
    }

    fn supports_image_editing(&self) -> bool {
        self.image_client.supports_image_editing()
    }
}

/// TogetherAI provider factory.
#[cfg(feature = "togetherai")]
pub struct TogetherAiProviderFactory;

#[cfg(feature = "togetherai")]
#[async_trait::async_trait]
impl ProviderFactory for TogetherAiProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        togetherai_capabilities()
    }

    async fn compat_language_client(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        let ctx = BuildContext::default();
        self.compat_language_client_with_ctx(model_id, &ctx).await
    }

    async fn compat_language_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let text_client = build_text_client_with_ctx(model_id, ctx).await?;
        let image_client = (*build_image_client_arc(model_id, ctx)?).clone();
        let rerank_client = build_native_rerank_client_with_ctx(model_id, ctx)?;
        Ok(Arc::new(TogetherAiCompatCompositeClient {
            text_client,
            image_client,
            rerank_client,
        }))
    }

    async fn language_model_text_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyLanguageModel>, LlmError> {
        let client = build_text_client_arc(model_id, ctx).await?;
        Ok(client)
    }

    async fn compat_completion_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = build_text_client_arc(model_id, ctx).await?;
        Ok(client)
    }

    async fn completion_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyCompletionModel>, LlmError> {
        let client = build_text_client_arc(model_id, ctx).await?;
        Ok(client)
    }

    async fn compat_embedding_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = build_text_client_arc(model_id, ctx).await?;
        Ok(client)
    }

    async fn embedding_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyEmbeddingModel>, LlmError> {
        let client = build_text_client_arc(model_id, ctx).await?;
        Ok(client)
    }

    async fn compat_image_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = build_image_client_arc(model_id, ctx)?;
        Ok(client)
    }

    async fn image_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyImageModel>, LlmError> {
        let client = build_image_client_arc(model_id, ctx)?;
        Ok(client)
    }

    async fn compat_speech_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = build_text_client_arc(model_id, ctx).await?;
        Ok(client)
    }

    async fn speech_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilySpeechModel>, LlmError> {
        let client = build_text_client_arc(model_id, ctx).await?;
        Ok(client)
    }

    async fn compat_transcription_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = build_text_client_arc(model_id, ctx).await?;
        Ok(client)
    }

    async fn transcription_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyTranscriptionModel>, LlmError> {
        let client = build_text_client_arc(model_id, ctx).await?;
        Ok(client)
    }

    async fn compat_reranking_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = build_rerank_client_arc(model_id, ctx)?;
        Ok(client)
    }

    async fn reranking_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyRerankingModel>, LlmError> {
        let client = build_rerank_client_arc(model_id, ctx)?;
        Ok(client)
    }

    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed(ids::TOGETHERAI)
    }
}
