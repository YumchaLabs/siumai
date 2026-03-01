//! `Groq` hybrid client implementation.
//!
//! - Chat/streaming/tools reuse the OpenAI-compatible vendor client.
//! - Audio (TTS/STT) uses Groq's OpenAI-like audio endpoints via `GroqSpec`.

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::streaming::ChatStream;
use crate::traits::{
    AudioCapability, ChatCapability, EmbeddingCapability, ImageExtras, ImageGenerationCapability,
    ModelListingCapability, RerankCapability,
};
use crate::types::{ChatMessage, ChatRequest, ChatResponse, ModelInfo, Tool};
use async_trait::async_trait;
use siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleClient;
use siumai_provider_openai_compatible::providers::openai_compatible::{
    ConfigurableAdapter, OpenAiCompatibleConfig, get_provider_config,
};
use std::sync::Arc;

mod audio;

#[derive(Clone, Debug)]
pub struct GroqClient {
    inner: OpenAiCompatibleClient,
}

impl GroqClient {
    pub fn new(inner: OpenAiCompatibleClient) -> Self {
        Self { inner }
    }

    /// Construct a `GroqClient` from a `GroqConfig` (config-first construction).
    pub async fn from_config(config: super::GroqConfig) -> Result<Self, LlmError> {
        config.validate()?;

        let provider = get_provider_config("groq").ok_or_else(|| {
            LlmError::ConfigurationError("OpenAI-compatible provider config not found: groq".into())
        })?;
        let adapter = Arc::new(ConfigurableAdapter::new(provider.clone()));

        use secrecy::ExposeSecret;
        let openai_cfg = OpenAiCompatibleConfig::new(
            "groq",
            config.api_key.expose_secret(),
            &config.base_url,
            adapter,
        )
        .with_http_config(config.http_config.clone())
        .with_common_params(config.common_params.clone())
        .with_model(&config.common_params.model);

        let inner = OpenAiCompatibleClient::from_config(openai_cfg).await?;
        Ok(Self::new(inner))
    }

    pub fn inner(&self) -> &OpenAiCompatibleClient {
        &self.inner
    }

    fn provider_context(&self) -> crate::core::ProviderContext {
        self.inner.provider_context()
    }

    fn http_client(&self) -> reqwest::Client {
        self.inner.http_client()
    }

    fn retry_options(&self) -> Option<crate::retry_api::RetryOptions> {
        self.inner.retry_options()
    }

    fn http_interceptors(
        &self,
    ) -> Vec<std::sync::Arc<dyn crate::execution::http::interceptor::HttpInterceptor>> {
        self.inner.http_interceptors()
    }

    fn http_transport(
        &self,
    ) -> Option<std::sync::Arc<dyn crate::execution::http::transport::HttpTransport>> {
        self.inner.http_transport()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::groq::GroqConfig;

    #[tokio::test]
    async fn groq_client_from_config_builds_inner_client() {
        let cfg = GroqConfig::new("test-key").with_model("llama-3.3-70b-versatile");
        let client = GroqClient::from_config(cfg).await.expect("from_config ok");
        assert_eq!(client.provider_id(), std::borrow::Cow::Borrowed("groq"));
    }
}

#[async_trait]
impl ChatCapability for GroqClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        self.inner.chat_with_tools(messages, tools).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        self.inner.chat_stream(messages, tools).await
    }

    async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.inner.chat_request(request).await
    }

    async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        self.inner.chat_stream_request(request).await
    }
}

impl LlmClient for GroqClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        crate::client::LlmClient::provider_id(&self.inner)
    }

    fn supported_models(&self) -> Vec<String> {
        crate::client::LlmClient::supported_models(&self.inner)
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        self.inner.capabilities().with_audio()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_chat_capability(&self) -> Option<&dyn ChatCapability> {
        Some(self)
    }

    fn as_embedding_capability(&self) -> Option<&dyn EmbeddingCapability> {
        self.inner.as_embedding_capability()
    }

    fn as_audio_capability(&self) -> Option<&dyn AudioCapability> {
        Some(self)
    }

    fn as_speech_capability(&self) -> Option<&dyn crate::traits::SpeechCapability> {
        Some(self)
    }

    fn as_transcription_capability(&self) -> Option<&dyn crate::traits::TranscriptionCapability> {
        Some(self)
    }

    fn as_image_generation_capability(&self) -> Option<&dyn ImageGenerationCapability> {
        self.inner.as_image_generation_capability()
    }

    fn as_image_extras(&self) -> Option<&dyn ImageExtras> {
        self.inner.as_image_extras()
    }

    fn as_rerank_capability(&self) -> Option<&dyn RerankCapability> {
        self.inner.as_rerank_capability()
    }

    fn as_model_listing_capability(&self) -> Option<&dyn ModelListingCapability> {
        Some(self)
    }
}

#[async_trait]
impl ModelListingCapability for GroqClient {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.inner.list_models().await
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        self.inner.get_model(model_id).await
    }
}
