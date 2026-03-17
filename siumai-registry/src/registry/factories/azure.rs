//! Provider factory implementations.

use super::*;
use crate::embedding::EmbeddingModel as FamilyEmbeddingModel;
use crate::image::ImageModel as FamilyImageModel;
use crate::text::LanguageModel as FamilyLanguageModel;
use siumai_core::speech::SpeechModel as FamilySpeechModel;
use siumai_core::transcription::TranscriptionModel as FamilyTranscriptionModel;

/// Azure OpenAI provider factory (Responses API by default).
///
/// Mirrors Vercel AI SDK's `@ai-sdk/azure` provider granularity.
#[cfg(feature = "azure")]
#[derive(Debug, Clone)]
pub struct AzureOpenAiProviderFactory {
    chat_mode: siumai_provider_azure::providers::azure_openai::AzureChatMode,
    url_config: siumai_provider_azure::providers::azure_openai::AzureUrlConfig,
    provider_metadata_key: &'static str,
}

#[cfg(feature = "azure")]
impl Default for AzureOpenAiProviderFactory {
    fn default() -> Self {
        Self::new(siumai_provider_azure::providers::azure_openai::AzureChatMode::Responses)
    }
}

#[cfg(feature = "azure")]
impl AzureOpenAiProviderFactory {
    pub fn new(chat_mode: siumai_provider_azure::providers::azure_openai::AzureChatMode) -> Self {
        Self {
            chat_mode,
            url_config: siumai_provider_azure::providers::azure_openai::AzureUrlConfig::default(),
            provider_metadata_key: "azure",
        }
    }

    pub fn with_url_config(
        mut self,
        url_config: siumai_provider_azure::providers::azure_openai::AzureUrlConfig,
    ) -> Self {
        self.url_config = url_config;
        self
    }

    pub fn with_api_version(mut self, api_version: impl Into<String>) -> Self {
        self.url_config.api_version = api_version.into();
        self
    }

    pub fn with_deployment_based_urls(mut self, enabled: bool) -> Self {
        self.url_config.use_deployment_based_urls = enabled;
        self
    }

    pub fn with_provider_metadata_key(mut self, key: &'static str) -> Self {
        self.provider_metadata_key = key;
        self
    }

    async fn build_family_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<siumai_provider_azure::providers::azure_openai::AzureOpenAiClient, LlmError> {
        let http_config = ctx.http_config.clone().unwrap_or_default();
        let common_params = crate::utils::builder_helpers::resolve_common_params(
            ctx.common_params.clone(),
            model_id,
        );
        if common_params.model.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Azure OpenAI requires a model (deployment id)".to_string(),
            ));
        }

        let mut builder = siumai_provider_azure::providers::azure_openai::AzureOpenAiBuilder::new(
            siumai_provider_azure::builder::BuilderBase::default(),
        )
        .chat_mode(self.chat_mode)
        .url_config(self.url_config.clone())
        .provider_metadata_key(self.provider_metadata_key)
        .model(common_params.model.clone())
        .with_http_config(http_config)
        .with_model_middlewares(ctx.model_middlewares.clone());

        if let Some(api_key) = ctx.api_key.clone() {
            builder = builder.api_key(api_key);
        }
        if let Some(base_url) = ctx.base_url.clone() {
            builder = builder.base_url(base_url);
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

        builder.build()
    }
}

#[cfg(feature = "azure")]
#[async_trait::async_trait]
impl ProviderFactory for AzureOpenAiProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        let meta = crate::native_provider_metadata::native_providers_metadata();
        meta.into_iter()
            .find(|m| m.id == "azure")
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
        let client = self.build_family_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
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
        std::borrow::Cow::Borrowed("azure")
    }
}
