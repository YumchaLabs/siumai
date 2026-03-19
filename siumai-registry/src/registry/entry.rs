//! Provider Registry - Vercel AI SDK Aligned Architecture
//!
//! This module provides a provider registry system that aligns with Vercel AI SDK's design:
//! - ProviderFactory trait for creating provider clients
//! - Registry stores factory instances (not hardcoded logic)
//! - Handles delegate to factories for client creation
//! - Easy to extend with new providers

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::client::LlmClient;
use crate::embedding::EmbeddingModel as FamilyEmbeddingModel;
use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::image::ImageModel as FamilyImageModel;
use crate::retry_api::RetryOptions;
use crate::streaming::{ChatStream, ChatStreamHandle};
use crate::text::{LanguageModel as FamilyLanguageModel, TextModelV3};
use crate::traits::{
    AudioCapability, ChatCapability, EmbeddingCapability, EmbeddingExtensions,
    FileManagementCapability, ImageExtras, ImageGenerationCapability, MusicGenerationCapability,
    ProviderCapabilities, RerankCapability, VideoGenerationCapability,
};
use crate::types::{
    AudioFeature, AudioStream, AudioTranslationRequest, BatchEmbeddingRequest,
    BatchEmbeddingResponse, ChatMessage, ChatRequest, ChatResponse, EmbeddingRequest,
    EmbeddingResponse, FileDeleteResponse, FileListQuery, FileListResponse, FileObject,
    FileUploadRequest, ImageEditRequest, ImageGenerationRequest, ImageGenerationResponse,
    ImageVariationRequest, LanguageInfo, MusicGenerationRequest, MusicGenerationResponse,
    RerankRequest, RerankResponse, SttRequest, SttResponse, Tool, TtsRequest, TtsResponse,
    VideoGenerationRequest, VideoGenerationResponse, VideoTaskStatusResponse, VoiceInfo,
};
use siumai_core::rerank::RerankingModel as FamilyRerankingModel;
use siumai_core::speech::SpeechModel as FamilySpeechModel;
use siumai_core::transcription::TranscriptionModel as FamilyTranscriptionModel;

use lru::LruCache;
use std::num::NonZeroUsize;
use tokio::sync::Mutex as TokioMutex;

/// Provider factory trait - similar to Vercel AI SDK's ProviderV3
///
/// Each provider implements this trait to create clients for different model types.
/// This allows the registry to be provider-agnostic and easily extensible.
///
/// Note: Middlewares are applied by the Handle after client creation, not by the factory.
/// This keeps the factory simple and aligns with Vercel AI SDK's design where
/// middleware wrapping happens at the registry level.
#[async_trait::async_trait]
pub trait ProviderFactory: Send + Sync {
    /// Create a language model client for the given model ID
    ///
    /// The returned client should NOT have middlewares applied - the Handle will apply them.
    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError>;

    /// Create a language model client with build context (interceptors, retry, etc.)
    /// Default implementation falls back to `language_model` for backward compatibility.
    async fn language_model_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.language_model(model_id).await
    }

    /// Create a text-family language model with build context.
    ///
    /// This is a parallel family-model-centered bridge introduced during the V4 refactor.
    /// The default implementation adapts the existing generic client path so providers can
    /// migrate incrementally.
    async fn language_model_text_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyLanguageModel>, LlmError> {
        let client = self.language_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(ClientBackedLanguageModel::new(
            client,
            self.provider_id().into_owned(),
            model_id.to_string(),
        )))
    }

    /// Create a text-family language model without an explicit build context.
    async fn language_model_text(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn FamilyLanguageModel>, LlmError> {
        self.language_model_text_with_ctx(model_id, &BuildContext::default())
            .await
    }

    /// Create an embedding model client for the given model ID
    async fn embedding_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Default: delegate to language_model (many providers use same client)
        self.language_model(model_id).await
    }

    /// Create an embedding model client with build context (default delegates to language_model_with_ctx)
    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Default: keep behavior of embedding_model() to allow custom embedding clients
        self.embedding_model(model_id).await
    }

    /// Create an embedding-family model with build context.
    async fn embedding_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyEmbeddingModel>, LlmError> {
        let client = self.embedding_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(ClientBackedEmbeddingModel::new(
            client,
            self.provider_id().into_owned(),
            model_id.to_string(),
        )))
    }

    /// Create an embedding-family model without an explicit build context.
    async fn embedding_model_family(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn FamilyEmbeddingModel>, LlmError> {
        self.embedding_model_family_with_ctx(model_id, &BuildContext::default())
            .await
    }

    /// Create an image model client for the given model ID
    async fn image_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.language_model(model_id).await
    }

    /// Create an image model client with build context (default delegates to language_model_with_ctx)
    async fn image_model_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.image_model(model_id).await
    }

    /// Create an image-family model with build context.
    async fn image_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyImageModel>, LlmError> {
        let client = self.image_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(ClientBackedImageModel::new(
            client,
            self.provider_id().into_owned(),
            model_id.to_string(),
        )))
    }

    /// Create an image-family model without an explicit build context.
    async fn image_model_family(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn FamilyImageModel>, LlmError> {
        self.image_model_family_with_ctx(model_id, &BuildContext::default())
            .await
    }

    /// Create a speech model client for the given model ID
    async fn speech_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.language_model(model_id).await
    }

    /// Create a speech model client with build context (default delegates to language_model_with_ctx)
    async fn speech_model_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.speech_model(model_id).await
    }

    /// Create a speech-family model with build context.
    async fn speech_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilySpeechModel>, LlmError> {
        let client = self.speech_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(ClientBackedSpeechModel::new(
            client,
            self.provider_id().into_owned(),
            model_id.to_string(),
        )))
    }

    /// Create a speech-family model without an explicit build context.
    async fn speech_model_family(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn FamilySpeechModel>, LlmError> {
        self.speech_model_family_with_ctx(model_id, &BuildContext::default())
            .await
    }

    /// Create a transcription model client for the given model ID
    async fn transcription_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.speech_model(model_id).await
    }

    /// Create a transcription model client with build context (default delegates to speech_model_with_ctx)
    async fn transcription_model_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.transcription_model(model_id).await
    }

    /// Create a transcription-family model with build context.
    async fn transcription_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyTranscriptionModel>, LlmError> {
        let client = self.transcription_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(ClientBackedTranscriptionModel::new(
            client,
            self.provider_id().into_owned(),
            model_id.to_string(),
        )))
    }

    /// Create a transcription-family model without an explicit build context.
    async fn transcription_model_family(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn FamilyTranscriptionModel>, LlmError> {
        self.transcription_model_family_with_ctx(model_id, &BuildContext::default())
            .await
    }

    /// Create a reranking model client for the given model ID
    async fn reranking_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.language_model(model_id).await
    }

    /// Create a reranking model client with build context (default delegates to reranking_model)
    async fn reranking_model_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.reranking_model(model_id).await
    }

    /// Create a reranking-family model with build context.
    async fn reranking_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyRerankingModel>, LlmError> {
        let client = self.reranking_model_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(ClientBackedRerankingModel::new(
            client,
            self.provider_id().into_owned(),
            model_id.to_string(),
        )))
    }

    /// Create a reranking-family model without an explicit build context.
    async fn reranking_model_family(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn FamilyRerankingModel>, LlmError> {
        self.reranking_model_family_with_ctx(model_id, &BuildContext::default())
            .await
    }

    /// Get the provider name
    fn provider_id(&self) -> std::borrow::Cow<'static, str>;

    /// Declared provider-level capabilities (metadata only).
    ///
    /// This is used by registry handles to expose capability hints without
    /// requiring runtime lookups into the global provider registry.
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
    }
}

/// Cache entry with TTL support
struct CacheEntry {
    model: Arc<dyn FamilyLanguageModel>,
    created_at: Instant,
}

/// Speech-family cache entry with TTL support
struct SpeechCacheEntry {
    model: Arc<dyn FamilySpeechModel>,
    created_at: Instant,
}

/// Transcription-family cache entry with TTL support
struct TranscriptionCacheEntry {
    model: Arc<dyn FamilyTranscriptionModel>,
    created_at: Instant,
}

/// Default adapter used to bridge the legacy generic-client path into the new
/// text-family-centered provider factory interface.
struct ClientBackedEmbeddingModel {
    client: Arc<dyn LlmClient>,
    provider_id: String,
    model_id: String,
}

impl ClientBackedEmbeddingModel {
    fn new(client: Arc<dyn LlmClient>, provider_id: String, model_id: String) -> Self {
        Self {
            client,
            provider_id,
            model_id,
        }
    }
}

impl crate::traits::ModelMetadata for ClientBackedEmbeddingModel {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[async_trait::async_trait]
impl EmbeddingCapability for ClientBackedEmbeddingModel {
    async fn embed(&self, input: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        let embedding = self.client.as_embedding_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation("Provider does not support embeddings".to_string())
        })?;
        embedding.embed(input).await
    }

    fn as_embedding_extensions(&self) -> Option<&dyn EmbeddingExtensions> {
        Some(self)
    }

    fn embedding_dimension(&self) -> usize {
        self.client
            .as_embedding_capability()
            .map(|embedding| embedding.embedding_dimension())
            .unwrap_or(0)
    }
}

#[async_trait::async_trait]
impl EmbeddingExtensions for ClientBackedEmbeddingModel {
    async fn embed_with_config(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        if let Some(extensions) = self.client.as_embedding_extensions() {
            return extensions.embed_with_config(request).await;
        }

        self.embed(request.input).await
    }

    async fn embed_batch(
        &self,
        requests: BatchEmbeddingRequest,
    ) -> Result<BatchEmbeddingResponse, LlmError> {
        if let Some(extensions) = self.client.as_embedding_extensions() {
            return extensions.embed_batch(requests).await;
        }

        let mut responses = Vec::new();
        for request in requests.requests {
            let result = self
                .embed(request.input)
                .await
                .map_err(|error| error.to_string());
            responses.push(result);
            if requests.batch_options.fail_fast && responses.last().is_some_and(|r| r.is_err()) {
                break;
            }
        }

        Ok(BatchEmbeddingResponse {
            responses,
            metadata: std::collections::HashMap::new(),
        })
    }
}

struct ClientBackedImageModel {
    client: Arc<dyn LlmClient>,
    provider_id: String,
    model_id: String,
}

impl ClientBackedImageModel {
    fn new(client: Arc<dyn LlmClient>, provider_id: String, model_id: String) -> Self {
        Self {
            client,
            provider_id,
            model_id,
        }
    }
}

impl crate::traits::ModelMetadata for ClientBackedImageModel {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[async_trait::async_trait]
impl crate::image::ImageModelV3 for ClientBackedImageModel {
    async fn generate(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let image = self
            .client
            .as_image_generation_capability()
            .ok_or_else(|| {
                LlmError::UnsupportedOperation(
                    "Provider does not support image generation".to_string(),
                )
            })?;
        image.generate_images(request).await
    }
}

struct ClientBackedSpeechModel {
    client: Arc<dyn LlmClient>,
    provider_id: String,
    model_id: String,
}

impl ClientBackedSpeechModel {
    fn new(client: Arc<dyn LlmClient>, provider_id: String, model_id: String) -> Self {
        Self {
            client,
            provider_id,
            model_id,
        }
    }
}

impl crate::traits::ModelMetadata for ClientBackedSpeechModel {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[async_trait::async_trait]
impl siumai_core::speech::SpeechModelV3 for ClientBackedSpeechModel {
    async fn synthesize(&self, request: TtsRequest) -> Result<TtsResponse, LlmError> {
        let speech = self.client.as_speech_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation("Provider does not support text-to-speech".to_string())
        })?;
        speech.tts(request).await
    }
}

struct ClientBackedTranscriptionModel {
    client: Arc<dyn LlmClient>,
    provider_id: String,
    model_id: String,
}

impl ClientBackedTranscriptionModel {
    fn new(client: Arc<dyn LlmClient>, provider_id: String, model_id: String) -> Self {
        Self {
            client,
            provider_id,
            model_id,
        }
    }
}

impl crate::traits::ModelMetadata for ClientBackedTranscriptionModel {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[async_trait::async_trait]
impl siumai_core::transcription::TranscriptionModelV3 for ClientBackedTranscriptionModel {
    async fn transcribe(&self, request: SttRequest) -> Result<SttResponse, LlmError> {
        let transcription = self.client.as_transcription_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation("Provider does not support speech-to-text".to_string())
        })?;
        transcription.stt(request).await
    }
}

struct ClientBackedRerankingModel {
    client: Arc<dyn LlmClient>,
    provider_id: String,
    model_id: String,
}

impl ClientBackedRerankingModel {
    fn new(client: Arc<dyn LlmClient>, provider_id: String, model_id: String) -> Self {
        Self {
            client,
            provider_id,
            model_id,
        }
    }
}

impl crate::traits::ModelMetadata for ClientBackedRerankingModel {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[async_trait::async_trait]
impl siumai_core::rerank::RerankModelV3 for ClientBackedRerankingModel {
    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, LlmError> {
        let rerank = self.client.as_rerank_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation("Provider does not support reranking".to_string())
        })?;
        rerank.rerank(request).await
    }
}

struct ClientBackedLanguageModel {
    client: Arc<dyn LlmClient>,
    provider_id: String,
    model_id: String,
}

impl ClientBackedLanguageModel {
    fn new(client: Arc<dyn LlmClient>, provider_id: String, model_id: String) -> Self {
        Self {
            client,
            provider_id,
            model_id,
        }
    }
}

impl crate::traits::ModelMetadata for ClientBackedLanguageModel {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[async_trait::async_trait]
impl TextModelV3 for ClientBackedLanguageModel {
    async fn generate(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        let chat = self.client.as_chat_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation("Provider does not support chat".to_string())
        })?;
        chat.chat_request(request).await
    }

    async fn stream(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        let chat = self.client.as_chat_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation("Provider does not support chat".to_string())
        })?;
        chat.chat_stream_request(request).await
    }

    async fn stream_with_cancel(&self, request: ChatRequest) -> Result<ChatStreamHandle, LlmError> {
        let chat = self.client.as_chat_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation("Provider does not support chat".to_string())
        })?;
        chat.chat_stream_request_with_cancel(request).await
    }
}

impl CacheEntry {
    fn new(model: Arc<dyn FamilyLanguageModel>) -> Self {
        Self {
            model,
            created_at: Instant::now(),
        }
    }

    fn is_expired(&self, ttl: Option<Duration>) -> bool {
        if let Some(ttl) = ttl {
            self.created_at.elapsed() > ttl
        } else {
            false
        }
    }
}

impl SpeechCacheEntry {
    fn new(model: Arc<dyn FamilySpeechModel>) -> Self {
        Self {
            model,
            created_at: Instant::now(),
        }
    }

    fn is_expired(&self, ttl: Option<Duration>) -> bool {
        if let Some(ttl) = ttl {
            self.created_at.elapsed() > ttl
        } else {
            false
        }
    }
}

impl TranscriptionCacheEntry {
    fn new(model: Arc<dyn FamilyTranscriptionModel>) -> Self {
        Self {
            model,
            created_at: Instant::now(),
        }
    }

    fn is_expired(&self, ttl: Option<Duration>) -> bool {
        if let Some(ttl) = ttl {
            self.created_at.elapsed() > ttl
        } else {
            false
        }
    }
}

/// Options for creating a provider registry handle.
#[derive(Default, Clone)]
pub struct ProviderBuildOverrides {
    /// Optional pre-built HTTP client override.
    pub http_client: Option<reqwest::Client>,
    /// Optional custom HTTP transport override.
    pub http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    /// Optional HTTP configuration override.
    pub http_config: Option<crate::types::HttpConfig>,
    /// Optional API key override.
    pub api_key: Option<String>,
    /// Optional base URL override.
    pub base_url: Option<String>,
    /// Optional unified reasoning enable flag for registry-built language models.
    pub reasoning_enabled: Option<bool>,
    /// Optional unified reasoning budget for registry-built language models.
    pub reasoning_budget: Option<i32>,
}

impl ProviderBuildOverrides {
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = Some(client);
        self
    }

    pub fn with_http_transport(
        mut self,
        transport: Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.http_transport = Some(transport);
        self
    }

    pub fn with_http_config(mut self, config: crate::types::HttpConfig) -> Self {
        self.http_config = Some(config);
        self
    }

    pub fn fetch(
        self,
        transport: Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.with_http_transport(transport)
    }

    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn with_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub fn with_reasoning(mut self, enabled: bool) -> Self {
        self.reasoning_enabled = Some(enabled);
        self
    }

    pub fn with_reasoning_budget(mut self, budget: i32) -> Self {
        self.reasoning_budget = Some(budget);
        self
    }

    fn merged_with(&self, provider_override: Option<&ProviderBuildOverrides>) -> Self {
        if let Some(provider_override) = provider_override {
            Self {
                http_client: provider_override
                    .http_client
                    .clone()
                    .or_else(|| self.http_client.clone()),
                http_transport: provider_override
                    .http_transport
                    .clone()
                    .or_else(|| self.http_transport.clone()),
                http_config: merge_http_config(
                    self.http_config.as_ref(),
                    provider_override.http_config.as_ref(),
                ),
                api_key: provider_override
                    .api_key
                    .clone()
                    .or_else(|| self.api_key.clone()),
                base_url: provider_override
                    .base_url
                    .clone()
                    .or_else(|| self.base_url.clone()),
                reasoning_enabled: provider_override
                    .reasoning_enabled
                    .or(self.reasoning_enabled),
                reasoning_budget: provider_override.reasoning_budget.or(self.reasoning_budget),
            }
        } else {
            self.clone()
        }
    }
}

fn merge_http_config(
    base: Option<&crate::types::HttpConfig>,
    provider_override: Option<&crate::types::HttpConfig>,
) -> Option<crate::types::HttpConfig> {
    match (base, provider_override) {
        (None, None) => None,
        (Some(base), None) => Some(base.clone()),
        (None, Some(provider_override)) => Some(provider_override.clone()),
        (Some(base), Some(provider_override)) => {
            let mut merged = base.clone();
            if provider_override.timeout.is_some() {
                merged.timeout = provider_override.timeout;
            }
            if provider_override.connect_timeout.is_some() {
                merged.connect_timeout = provider_override.connect_timeout;
            }
            merged.headers.extend(provider_override.headers.clone());
            if provider_override.proxy.is_some() {
                merged.proxy = provider_override.proxy.clone();
            }
            if provider_override.user_agent.is_some() {
                merged.user_agent = provider_override.user_agent.clone();
            }
            merged.stream_disable_compression = provider_override.stream_disable_compression;
            Some(merged)
        }
    }
}

#[cfg(test)]
mod provider_build_override_tests {
    use super::ProviderBuildOverrides;
    use crate::types::HttpConfig;
    use std::time::Duration;

    #[test]
    fn merged_with_merges_http_config_over_global_defaults() {
        let mut base_http_config = HttpConfig::default();
        base_http_config.timeout = Some(Duration::from_secs(30));
        base_http_config.connect_timeout = Some(Duration::from_secs(5));
        base_http_config
            .headers
            .insert("authorization".to_string(), "Bearer global".to_string());
        base_http_config
            .headers
            .insert("x-global-header".to_string(), "keep-me".to_string());
        base_http_config.proxy = Some("http://global-proxy".to_string());
        base_http_config.user_agent = Some("global-agent".to_string());
        base_http_config.stream_disable_compression = true;

        let mut provider_http_config = HttpConfig::empty();
        provider_http_config.timeout = Some(Duration::from_secs(90));
        provider_http_config
            .headers
            .insert("authorization".to_string(), "Bearer provider".to_string());
        provider_http_config
            .headers
            .insert("x-provider-header".to_string(), "set-me".to_string());
        provider_http_config.user_agent = Some("provider-agent".to_string());
        provider_http_config.stream_disable_compression = false;

        let merged = ProviderBuildOverrides::default()
            .with_api_key("global-key")
            .with_base_url("https://example.com/global")
            .with_http_config(base_http_config)
            .merged_with(Some(
                &ProviderBuildOverrides::default()
                    .with_api_key("provider-key")
                    .with_http_config(provider_http_config),
            ));

        assert_eq!(merged.api_key.as_deref(), Some("provider-key"));
        assert_eq!(
            merged.base_url.as_deref(),
            Some("https://example.com/global")
        );

        let merged_http_config = merged.http_config.expect("merged http config");
        assert_eq!(merged_http_config.timeout, Some(Duration::from_secs(90)));
        assert_eq!(
            merged_http_config.connect_timeout,
            Some(Duration::from_secs(5))
        );
        assert_eq!(
            merged_http_config
                .headers
                .get("authorization")
                .map(String::as_str),
            Some("Bearer provider")
        );
        assert_eq!(
            merged_http_config
                .headers
                .get("x-global-header")
                .map(String::as_str),
            Some("keep-me")
        );
        assert_eq!(
            merged_http_config
                .headers
                .get("x-provider-header")
                .map(String::as_str),
            Some("set-me")
        );
        assert_eq!(
            merged_http_config.proxy.as_deref(),
            Some("http://global-proxy")
        );
        assert_eq!(
            merged_http_config.user_agent.as_deref(),
            Some("provider-agent")
        );
        assert!(!merged_http_config.stream_disable_compression);
    }
}

pub struct RegistryOptions {
    pub separator: char,
    pub language_model_middleware: Vec<Arc<dyn LanguageModelMiddleware>>,
    /// HTTP interceptors applied to all clients created via the registry
    pub http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional pre-built HTTP client applied to all clients created via the registry.
    pub http_client: Option<reqwest::Client>,
    /// Optional custom HTTP transport applied to all clients created via the registry.
    pub http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    /// Optional HTTP configuration applied to all clients created via the registry.
    /// When set, this configuration is passed through BuildContext to provider
    /// factories so they can build HTTP clients consistently with SiumaiBuilder.
    pub http_config: Option<crate::types::HttpConfig>,
    /// Optional API key override applied to registry-built provider clients.
    pub api_key: Option<String>,
    /// Optional base URL override applied to registry-built provider clients.
    pub base_url: Option<String>,
    /// Optional unified reasoning enable flag applied to registry-built language models.
    pub reasoning_enabled: Option<bool>,
    /// Optional unified reasoning budget applied to registry-built language models.
    pub reasoning_budget: Option<i32>,
    /// Optional per-provider build overrides applied after global registry overrides.
    pub provider_build_overrides: HashMap<String, ProviderBuildOverrides>,
    /// Unified retry options applied to clients created via the registry (optional)
    pub retry_options: Option<RetryOptions>,
    /// Maximum number of cached clients (LRU eviction when exceeded)
    pub max_cache_entries: Option<usize>,
    /// Time-to-live for cached clients (None = no expiration)
    pub client_ttl: Option<Duration>,
    /// Whether to automatically add model-specific middlewares (e.g., ExtractReasoningMiddleware)
    /// based on provider and model ID. Default: true
    pub auto_middleware: bool,
}

/// Provider registry handle - aligned with Vercel AI SDK design
///
/// Stores provider factories and delegates client creation to them.
/// This makes the registry extensible and provider-agnostic.
///
/// Features LRU cache with optional TTL to prevent unbounded growth.
pub struct ProviderRegistryHandle {
    /// Registered provider factories (provider_id -> factory)
    providers: HashMap<String, Arc<dyn ProviderFactory>>,
    /// Separator for parsing "provider:model" identifiers
    separator: char,
    /// Middlewares to apply to all language models
    pub(crate) middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    /// HTTP interceptors to apply to clients created via this registry
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Registry-level pre-built HTTP client propagated via BuildContext.
    http_client: Option<reqwest::Client>,
    /// Registry-level custom HTTP transport propagated via BuildContext.
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    /// Registry-level HTTP configuration propagated via BuildContext.
    http_config: Option<crate::types::HttpConfig>,
    /// Registry-level API key propagated via BuildContext.
    api_key: Option<String>,
    /// Registry-level base URL propagated via BuildContext.
    base_url: Option<String>,
    /// Registry-level unified reasoning flag propagated via BuildContext.
    reasoning_enabled: Option<bool>,
    /// Registry-level unified reasoning budget propagated via BuildContext.
    reasoning_budget: Option<i32>,
    /// Registry-level per-provider build overrides applied after global overrides.
    provider_build_overrides: HashMap<String, ProviderBuildOverrides>,
    /// LRU cache for language model clients (key: "provider:model")
    /// Uses async Mutex for concurrent access and per-key build de-duplication
    language_model_cache: Arc<TokioMutex<LruCache<String, CacheEntry>>>,
    /// LRU cache for speech-family models (key: "provider:model")
    speech_model_cache: Arc<TokioMutex<LruCache<String, SpeechCacheEntry>>>,
    /// LRU cache for transcription-family models (key: "provider:model")
    transcription_model_cache: Arc<TokioMutex<LruCache<String, TranscriptionCacheEntry>>>,
    /// TTL for cached clients
    client_ttl: Option<Duration>,
    /// Whether to automatically add model-specific middlewares
    auto_middleware: bool,
    /// Registry-level retry options applied during client build (optional)
    retry_options: Option<RetryOptions>,
}

/// Build-time context for ProviderFactory client construction.
///
/// This struct carries all cross-cutting configuration needed to build
/// concrete provider clients in a unified way (HTTP config, auth, tracing,
/// interceptors, retry options, etc.).
#[derive(Default, Clone)]
pub struct BuildContext {
    /// HTTP interceptors applied at the registry / builder level.
    pub http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Unified retry options (optional).
    pub retry_options: Option<RetryOptions>,
    /// Optional model-level middlewares (applied before provider mapping).
    pub model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    /// Optional pre-built HTTP client. When present, factories should prefer
    /// this client over constructing a new one from `http_config`.
    pub http_client: Option<reqwest::Client>,
    /// Optional custom HTTP transport (Vercel-style "custom fetch" parity).
    pub http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    /// Optional HTTP configuration (timeouts, headers, proxy, user-agent, etc.).
    /// When no custom client is supplied, factories may use this to build one.
    pub http_config: Option<crate::types::HttpConfig>,
    /// Optional API key override. When `None`, factories may fall back to
    /// environment variables or other defaults.
    pub api_key: Option<String>,
    /// Optional base URL override for the provider.
    pub base_url: Option<String>,
    /// Optional organization identifier (e.g., OpenAI `organization`).
    pub organization: Option<String>,
    /// Optional project identifier (e.g., OpenAI `project`).
    pub project: Option<String>,
    /// Optional tracing configuration for providers that support it.
    pub tracing_config: Option<crate::observability::tracing::TracingConfig>,
    /// Optional Google-family token provider (Gemini / Vertex / Anthropic-on-Vertex).
    #[cfg(any(feature = "google", feature = "google-vertex"))]
    pub google_token_provider: Option<std::sync::Arc<dyn crate::auth::TokenProvider>>,
    /// Backward-compatible alias for older Gemini-focused call sites.
    #[cfg(any(feature = "google", feature = "google-vertex"))]
    pub gemini_token_provider: Option<std::sync::Arc<dyn crate::auth::TokenProvider>>,
    /// Optional common parameters (model id, temperature, max_tokens, etc.).
    /// When `None`, factories may construct minimal defaults based on `model_id`.
    pub common_params: Option<crate::types::CommonParams>,
    /// Optional unified reasoning enable flag propagated from top-level builders.
    pub reasoning_enabled: Option<bool>,
    /// Optional unified reasoning budget propagated from top-level builders.
    pub reasoning_budget: Option<i32>,
    /// Optional canonical provider id override (for adapter-style providers).
    pub provider_id: Option<String>,
}

impl BuildContext {
    #[cfg(any(feature = "google", feature = "google-vertex"))]
    pub fn resolved_google_token_provider(
        &self,
    ) -> Option<std::sync::Arc<dyn crate::auth::TokenProvider>> {
        self.google_token_provider
            .clone()
            .or_else(|| self.gemini_token_provider.clone())
    }
}

fn build_registry_context(
    provider_id: &str,
    http_interceptors: &[Arc<dyn HttpInterceptor>],
    retry_options: &Option<RetryOptions>,
    http_client: &Option<reqwest::Client>,
    http_transport: &Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    http_config: &Option<crate::types::HttpConfig>,
    api_key: &Option<String>,
    base_url: &Option<String>,
    reasoning_enabled: Option<bool>,
    reasoning_budget: Option<i32>,
) -> BuildContext {
    BuildContext {
        http_interceptors: http_interceptors.to_vec(),
        retry_options: retry_options.clone(),
        http_client: http_client.clone(),
        http_transport: http_transport.clone(),
        http_config: http_config.clone(),
        api_key: api_key.clone(),
        base_url: base_url.clone(),
        reasoning_enabled,
        reasoning_budget,
        provider_id: Some(provider_id.to_string()),
        ..Default::default()
    }
}

fn request_model_missing(slot: Option<&str>) -> bool {
    match slot {
        Some(value) => value.trim().is_empty(),
        None => true,
    }
}

fn apply_speech_handle_default_model(mut request: TtsRequest, model_id: &str) -> TtsRequest {
    if request_model_missing(request.model.as_deref()) && !model_id.trim().is_empty() {
        request.model = Some(model_id.to_string());
    }
    request
}

fn apply_transcription_handle_default_model(mut request: SttRequest, model_id: &str) -> SttRequest {
    if request_model_missing(request.model.as_deref()) && !model_id.trim().is_empty() {
        request.model = Some(model_id.to_string());
    }
    request
}

fn apply_translation_handle_default_model(
    mut request: AudioTranslationRequest,
    model_id: &str,
) -> AudioTranslationRequest {
    if request_model_missing(request.model.as_deref()) && !model_id.trim().is_empty() {
        request.model = Some(model_id.to_string());
    }
    request
}

impl ProviderRegistryHandle {
    /// Split a registry model id like "provider:model" into (provider, model).
    fn split_id(&self, id: &str) -> Result<(String, String), LlmError> {
        if let Some((p, m)) = id.split_once(self.separator) {
            if p.is_empty() || m.is_empty() {
                return Err(LlmError::InvalidParameter(format!(
                    "Invalid model id for registry: {} (must be 'provider{}model')",
                    id, self.separator
                )));
            }
            Ok((p.to_string(), m.to_string()))
        } else {
            Err(LlmError::InvalidParameter(format!(
                "Invalid model id for registry: {} (must be 'provider{}model')",
                id, self.separator
            )))
        }
    }

    /// Get a provider factory by ID
    fn get_provider(&self, provider_id: &str) -> Result<&Arc<dyn ProviderFactory>, LlmError> {
        self.providers.get(provider_id).ok_or_else(|| {
            LlmError::ConfigurationError(format!(
                "No such provider: {}. Available providers: {:?}",
                provider_id,
                self.providers.keys().collect::<Vec<_>>()
            ))
        })
    }

    fn resolve_provider_build_overrides(&self, provider_id: &str) -> ProviderBuildOverrides {
        ProviderBuildOverrides {
            http_client: self.http_client.clone(),
            http_transport: self.http_transport.clone(),
            http_config: self.http_config.clone(),
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            reasoning_enabled: self.reasoning_enabled,
            reasoning_budget: self.reasoning_budget,
        }
        .merged_with(self.provider_build_overrides.get(provider_id))
    }

    /// Resolve language model - returns a handle that delegates to the factory
    ///
    /// Uses LRU cache with optional TTL to avoid rebuilding clients repeatedly.
    /// Cache key is the full "provider:model" identifier.
    ///
    /// Applies middleware provider_id override if configured.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use siumai_registry::registry::entry::create_provider_registry;
    /// # use std::collections::HashMap;
    /// let registry = create_provider_registry(HashMap::new(), None);
    /// let handle = registry.language_model("openai:gpt-4")?;
    /// # Ok::<(), siumai_registry::error::LlmError>(())
    /// ```
    pub fn language_model(&self, id: &str) -> Result<LanguageModelHandle, LlmError> {
        let (mut provider_id, model_id) = self.split_id(id)?;

        // Normalize common provider id aliases (e.g. "google" -> "gemini") when possible.
        // Only apply normalization when the canonical id is registered and the raw id is not,
        // to avoid surprising overrides for custom registries.
        let normalized = crate::provider::resolver::normalize_provider_id(&provider_id);
        if normalized != provider_id
            && !self.providers.contains_key(&provider_id)
            && self.providers.contains_key(&normalized)
        {
            provider_id = normalized;
        }

        // Combine global middlewares with auto middlewares
        let mut middlewares = self.middlewares.clone();
        if self.auto_middleware {
            let auto_middlewares =
                crate::execution::middleware::build_auto_middlewares_vec(&provider_id, &model_id);
            middlewares.extend(auto_middlewares);
        }

        // Apply middleware provider_id override (aligned with Vercel AI SDK)
        if !middlewares.is_empty() {
            provider_id = crate::execution::middleware::language_model::apply_provider_id_override(
                &middlewares,
                &provider_id,
            );
        }

        let factory = self.get_provider(&provider_id)?;
        let capabilities = factory.capabilities();
        if !capabilities.supports("chat") {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not expose registry language_model/chat handles; use family-specific entries instead",
                provider_id
            )));
        }
        let build_overrides = self.resolve_provider_build_overrides(&provider_id);

        Ok(LanguageModelHandle {
            factory: factory.clone(),
            provider_id,
            model_id,
            middlewares,
            cache: self.language_model_cache.clone(),
            client_ttl: self.client_ttl,
            http_interceptors: self.http_interceptors.clone(),
            http_client: build_overrides.http_client,
            http_transport: build_overrides.http_transport,
            http_config: build_overrides.http_config,
            api_key: build_overrides.api_key,
            base_url: build_overrides.base_url,
            reasoning_enabled: build_overrides.reasoning_enabled,
            reasoning_budget: build_overrides.reasoning_budget,
            retry_options: self.retry_options.clone(),
            capabilities,
        })
    }

    /// Resolve embedding model - returns a handle that delegates to the factory
    pub fn embedding_model(&self, id: &str) -> Result<EmbeddingModelHandle, LlmError> {
        let (provider_id, model_id) = self.split_id(id)?;
        let factory = self.get_provider(&provider_id)?;
        let capabilities = factory.capabilities();
        if !capabilities.supports("embedding") {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not expose embedding on the embedding_model handle",
                provider_id
            )));
        }
        let build_overrides = self.resolve_provider_build_overrides(&provider_id);

        Ok(EmbeddingModelHandle {
            factory: factory.clone(),
            provider_id,
            model_id,
            http_interceptors: self.http_interceptors.clone(),
            http_client: build_overrides.http_client,
            http_transport: build_overrides.http_transport,
            http_config: build_overrides.http_config,
            api_key: build_overrides.api_key,
            base_url: build_overrides.base_url,
            retry_options: self.retry_options.clone(),
        })
    }

    /// Resolve image model - returns a handle that delegates to the factory
    pub fn image_model(&self, id: &str) -> Result<ImageModelHandle, LlmError> {
        let (provider_id, model_id) = self.split_id(id)?;
        let factory = self.get_provider(&provider_id)?;
        let capabilities = factory.capabilities();
        if !capabilities.supports("image_generation") {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not expose image_generation on the image_model handle",
                provider_id
            )));
        }
        let build_overrides = self.resolve_provider_build_overrides(&provider_id);

        Ok(ImageModelHandle {
            factory: factory.clone(),
            provider_id,
            model_id,
            http_interceptors: self.http_interceptors.clone(),
            http_client: build_overrides.http_client,
            http_transport: build_overrides.http_transport,
            http_config: build_overrides.http_config,
            api_key: build_overrides.api_key,
            base_url: build_overrides.base_url,
            retry_options: self.retry_options.clone(),
        })
    }

    /// Resolve speech model - returns a handle that delegates to the factory
    pub fn speech_model(&self, id: &str) -> Result<SpeechModelHandle, LlmError> {
        let (provider_id, model_id) = self.split_id(id)?;
        let factory = self.get_provider(&provider_id)?;
        let capabilities = factory.capabilities();
        if !capabilities.supports("speech") {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not expose speech on the speech_model handle",
                provider_id
            )));
        }
        let build_overrides = self.resolve_provider_build_overrides(&provider_id);

        Ok(SpeechModelHandle {
            factory: factory.clone(),
            provider_id,
            model_id,
            http_interceptors: self.http_interceptors.clone(),
            http_client: build_overrides.http_client,
            http_transport: build_overrides.http_transport,
            http_config: build_overrides.http_config,
            api_key: build_overrides.api_key,
            base_url: build_overrides.base_url,
            cache: self.speech_model_cache.clone(),
            client_ttl: self.client_ttl,
            retry_options: self.retry_options.clone(),
        })
    }

    /// Resolve transcription model - returns a handle that delegates to the factory
    pub fn transcription_model(&self, id: &str) -> Result<TranscriptionModelHandle, LlmError> {
        let (provider_id, model_id) = self.split_id(id)?;
        let factory = self.get_provider(&provider_id)?;
        let capabilities = factory.capabilities();
        if !capabilities.supports("transcription") {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not expose transcription on the transcription_model handle",
                provider_id
            )));
        }
        let build_overrides = self.resolve_provider_build_overrides(&provider_id);

        Ok(TranscriptionModelHandle {
            factory: factory.clone(),
            provider_id,
            model_id,
            http_interceptors: self.http_interceptors.clone(),
            http_client: build_overrides.http_client,
            http_transport: build_overrides.http_transport,
            http_config: build_overrides.http_config,
            api_key: build_overrides.api_key,
            base_url: build_overrides.base_url,
            cache: self.transcription_model_cache.clone(),
            client_ttl: self.client_ttl,
            retry_options: self.retry_options.clone(),
        })
    }

    /// Resolve reranking model - returns a handle that delegates to the factory
    pub fn reranking_model(&self, id: &str) -> Result<RerankingModelHandle, LlmError> {
        let (provider_id, model_id) = self.split_id(id)?;
        let factory = self.get_provider(&provider_id)?;
        let capabilities = factory.capabilities();
        if !capabilities.supports("rerank") {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not expose rerank on the reranking_model handle",
                provider_id
            )));
        }
        let build_overrides = self.resolve_provider_build_overrides(&provider_id);

        Ok(RerankingModelHandle {
            factory: factory.clone(),
            provider_id,
            model_id,
            http_interceptors: self.http_interceptors.clone(),
            http_client: build_overrides.http_client,
            http_transport: build_overrides.http_transport,
            http_config: build_overrides.http_config,
            api_key: build_overrides.api_key,
            base_url: build_overrides.base_url,
            retry_options: self.retry_options.clone(),
        })
    }
}

/// Create a provider registry handle - aligned with Vercel AI SDK
///
/// # Arguments
/// * `providers` - Map of provider_id -> ProviderFactory instances
/// * `opts` - Optional registry configuration (separator, middlewares)
///
/// # Example
/// ```rust,no_run
/// use std::collections::HashMap;
/// use std::sync::Arc;
/// use siumai_registry::registry::entry::{create_provider_registry, ProviderFactory};
///
/// let mut providers = HashMap::new();
/// // providers.insert("openai".to_string(), Arc::new(OpenAIProviderFactory) as Arc<dyn ProviderFactory>);
///
/// let registry = create_provider_registry(providers, None);
/// ```
pub fn create_provider_registry(
    providers: HashMap<String, Arc<dyn ProviderFactory>>,
    opts: Option<RegistryOptions>,
) -> ProviderRegistryHandle {
    let (
        separator,
        middlewares,
        http_interceptors,
        http_client,
        http_transport,
        http_config,
        api_key,
        base_url,
        reasoning_enabled,
        reasoning_budget,
        provider_build_overrides,
        retry_options,
        max_cache_entries,
        client_ttl,
        auto_middleware,
    ) = if let Some(o) = opts {
        (
            o.separator,
            o.language_model_middleware,
            o.http_interceptors,
            o.http_client,
            o.http_transport,
            o.http_config,
            o.api_key,
            o.base_url,
            o.reasoning_enabled,
            o.reasoning_budget,
            o.provider_build_overrides,
            o.retry_options,
            o.max_cache_entries,
            o.client_ttl,
            o.auto_middleware,
        )
    } else {
        // Defaults: no middlewares, no interceptors, auto middleware enabled
        (
            ':',
            Vec::new(),
            Vec::new(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            HashMap::new(),
            None,
            None,
            None,
            true,
        )
    };

    // Create LRU cache with specified capacity (default: 100 entries)
    let cache_capacity = max_cache_entries.unwrap_or(100);
    let cache =
        LruCache::new(NonZeroUsize::new(cache_capacity).expect("Cache capacity must be > 0"));
    let speech_cache =
        LruCache::new(NonZeroUsize::new(cache_capacity).expect("Cache capacity must be > 0"));
    let transcription_cache =
        LruCache::new(NonZeroUsize::new(cache_capacity).expect("Cache capacity must be > 0"));

    ProviderRegistryHandle {
        providers,
        separator,
        middlewares,
        http_interceptors,
        http_client,
        http_transport,
        language_model_cache: Arc::new(TokioMutex::new(cache)),
        speech_model_cache: Arc::new(TokioMutex::new(speech_cache)),
        transcription_model_cache: Arc::new(TokioMutex::new(transcription_cache)),
        client_ttl,
        auto_middleware,
        http_config,
        api_key,
        base_url,
        reasoning_enabled,
        reasoning_budget,
        provider_build_overrides,
        retry_options,
    }
}

/// Language model handle - delegates to provider factory
///
/// This handle stores a reference to the provider factory and delegates
/// client creation to it. This aligns with Vercel AI SDK's design where
/// the registry returns model instances that know how to create themselves.
///
/// Features LRU cache with TTL to avoid rebuilding clients on every call.
#[derive(Clone)]
pub struct LanguageModelHandle {
    /// Provider factory for creating clients
    factory: Arc<dyn ProviderFactory>,
    /// Provider ID (e.g., "openai")
    pub provider_id: String,
    /// Model ID to pass to the factory (e.g., "gpt-4")
    pub model_id: String,
    /// Middlewares to apply to the client
    pub middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    /// Registry-level HTTP interceptors to attempt injecting into clients
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Registry-level pre-built HTTP client copied into the handle
    http_client: Option<reqwest::Client>,
    /// Registry-level custom HTTP transport copied into the handle
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    /// Registry-level HTTP configuration copied into the handle
    http_config: Option<crate::types::HttpConfig>,
    /// Registry-level API key copied into the handle
    api_key: Option<String>,
    /// Registry-level base URL copied into the handle
    base_url: Option<String>,
    /// Registry-level unified reasoning flag copied into the handle
    reasoning_enabled: Option<bool>,
    /// Registry-level unified reasoning budget copied into the handle
    reasoning_budget: Option<i32>,
    /// Shared LRU cache for clients
    cache: Arc<TokioMutex<LruCache<String, CacheEntry>>>,
    /// TTL for cached clients
    client_ttl: Option<Duration>,
    /// Registry-level retry options copied into the handle
    retry_options: Option<RetryOptions>,
    /// Provider-level capability hints captured at construction time
    capabilities: ProviderCapabilities,
}

impl LanguageModelHandle {
    fn ensure_chat_capability(&self, stream: bool) -> Result<(), LlmError> {
        if !self.capabilities.supports("chat") {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not expose chat on the language_model handle",
                self.provider_id
            )));
        }

        if stream && !self.capabilities.supports("streaming") {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not expose chat streaming on the language_model handle",
                self.provider_id
            )));
        }

        Ok(())
    }

    /// Get or create a cached client
    ///
    /// This method implements LRU cache with TTL:
    /// 1. Check cache for existing client
    /// 2. If found and not expired, return it
    /// 3. If not found or expired, build new client and cache it
    /// 4. LRU eviction happens automatically when cache is full
    ///
    /// Note: Cache key includes the potentially overridden model_id to ensure
    /// correct caching when middleware overrides the model.
    async fn get_or_create_language_model(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn FamilyLanguageModel>, LlmError> {
        let cache_key = format!("{}:{}", self.provider_id, model_id);

        let mut cache = self.cache.lock().await;
        if let Some(entry) = cache.get(&cache_key) {
            if !entry.is_expired(self.client_ttl) {
                return Ok(entry.model.clone());
            }
            cache.pop(&cache_key);
        }

        drop(cache);
        let ctx = build_registry_context(
            &self.provider_id,
            &self.http_interceptors,
            &self.retry_options,
            &self.http_client,
            &self.http_transport,
            &self.http_config,
            &self.api_key,
            &self.base_url,
            self.reasoning_enabled,
            self.reasoning_budget,
        );
        let model = self
            .factory
            .language_model_text_with_ctx(model_id, &ctx)
            .await?;

        let mut cache = self.cache.lock().await;
        cache.put(cache_key, CacheEntry::new(model.clone()));

        Ok(model)
    }

    async fn build_language_client(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        let ctx = build_registry_context(
            &self.provider_id,
            &self.http_interceptors,
            &self.retry_options,
            &self.http_client,
            &self.http_transport,
            &self.http_config,
            &self.api_key,
            &self.base_url,
            self.reasoning_enabled,
            self.reasoning_budget,
        );

        self.factory.language_model_with_ctx(model_id, &ctx).await
    }
}

impl crate::traits::ModelMetadata for LanguageModelHandle {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

/// Implement unified client metadata trait for LanguageModelHandle.
///
/// This allows using a registry language model handle anywhere an `LlmClient`
/// is expected (e.g., inside the unified `Siumai` wrapper), while keeping
/// execution logic delegated to the underlying provider clients.
impl LlmClient for LanguageModelHandle {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Owned(self.provider_id.clone())
    }

    fn supported_models(&self) -> Vec<String> {
        // We only know the configured model id for this handle; return that.
        vec![self.model_id.clone()]
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.capabilities.clone()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_chat_capability(&self) -> Option<&dyn ChatCapability> {
        self.capabilities.supports("chat").then_some(self)
    }

    fn as_file_management_capability(&self) -> Option<&dyn FileManagementCapability> {
        self.capabilities
            .supports("file_management")
            .then_some(self)
    }

    fn as_video_generation_capability(&self) -> Option<&dyn VideoGenerationCapability> {
        self.capabilities.supports("video").then_some(self)
    }

    fn as_music_generation_capability(&self) -> Option<&dyn MusicGenerationCapability> {
        self.capabilities.supports("music").then_some(self)
    }
}

/// Implementation of ChatCapability for LanguageModelHandle
///
/// This allows the handle to be used directly as a chat client, aligning with
/// Vercel AI SDK's design where registry.languageModel() returns a callable model.
#[async_trait::async_trait]
impl ChatCapability for LanguageModelHandle {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        self.ensure_chat_capability(false)?;
        // Apply middleware overrides (aligned with Vercel AI SDK)
        let model_id = if !self.middlewares.is_empty() {
            crate::execution::middleware::language_model::apply_model_id_override(
                &self.middlewares,
                &self.model_id,
            )
        } else {
            self.model_id.clone()
        };

        // Get or create cached family model with potentially overridden model_id
        let model = self.get_or_create_language_model(&model_id).await?;

        // Apply middlewares if any
        if !self.middlewares.is_empty() {
            let mut req = ChatRequest::new(messages);
            if let Some(t) = tools {
                req = req.with_tools(t);
            }
            req = crate::execution::middleware::language_model::apply_transform_chain(
                &self.middlewares,
                req,
            );
            model.generate(req).await
        } else {
            let mut req = ChatRequest::new(messages);
            if let Some(t) = tools {
                req = req.with_tools(t);
            }
            model.generate(req).await
        }
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        self.ensure_chat_capability(true)?;
        // Apply middleware overrides (aligned with Vercel AI SDK)
        let model_id = if !self.middlewares.is_empty() {
            crate::execution::middleware::language_model::apply_model_id_override(
                &self.middlewares,
                &self.model_id,
            )
        } else {
            self.model_id.clone()
        };

        // Get or create cached family model with potentially overridden model_id
        let model = self.get_or_create_language_model(&model_id).await?;

        // Apply middlewares if any
        if !self.middlewares.is_empty() {
            let mut req = ChatRequest::new(messages);
            if let Some(t) = tools {
                req = req.with_tools(t);
            }
            req = crate::execution::middleware::language_model::apply_transform_chain(
                &self.middlewares,
                req,
            );
            model.stream(req.with_streaming(true)).await
        } else {
            let mut req = ChatRequest::new(messages).with_streaming(true);
            if let Some(t) = tools {
                req = req.with_tools(t);
            }
            model.stream(req).await
        }
    }

    async fn chat_stream_with_cancel(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStreamHandle, LlmError> {
        self.ensure_chat_capability(true)?;
        let this = self.clone();
        Ok(
            crate::utils::cancel::make_cancellable_stream_handle_from_handle_future(async move {
                // Align with chat_stream(...) middleware behavior, but preserve provider-specific cancellation.
                let model_id = if !this.middlewares.is_empty() {
                    crate::execution::middleware::language_model::apply_model_id_override(
                        &this.middlewares,
                        &this.model_id,
                    )
                } else {
                    this.model_id.clone()
                };

                let model = this.get_or_create_language_model(&model_id).await?;

                let mut req = ChatRequest::new(messages).with_streaming(true);
                if let Some(t) = tools {
                    req = req.with_tools(t);
                }
                req.common_params.model = model_id.clone();

                if !this.middlewares.is_empty() {
                    req = crate::execution::middleware::language_model::apply_transform_chain(
                        &this.middlewares,
                        req,
                    );
                }

                model.stream_with_cancel(req).await
            }),
        )
    }

    async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.ensure_chat_capability(false)?;
        let model_id = if !self.middlewares.is_empty() {
            crate::execution::middleware::language_model::apply_model_id_override(
                &self.middlewares,
                &self.model_id,
            )
        } else {
            self.model_id.clone()
        };

        let model = self.get_or_create_language_model(&model_id).await?;

        let mut req = request.with_streaming(false);
        if req.common_params.model.trim().is_empty() {
            req.common_params.model = model_id.clone();
        }
        if !self.middlewares.is_empty() {
            req = crate::execution::middleware::language_model::apply_transform_chain(
                &self.middlewares,
                req,
            );
        }

        model.generate(req).await
    }

    async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        self.ensure_chat_capability(true)?;
        let model_id = if !self.middlewares.is_empty() {
            crate::execution::middleware::language_model::apply_model_id_override(
                &self.middlewares,
                &self.model_id,
            )
        } else {
            self.model_id.clone()
        };

        let model = self.get_or_create_language_model(&model_id).await?;

        let mut req = request.with_streaming(true);
        if req.common_params.model.trim().is_empty() {
            req.common_params.model = model_id.clone();
        }
        if !self.middlewares.is_empty() {
            req = crate::execution::middleware::language_model::apply_transform_chain(
                &self.middlewares,
                req,
            );
        }

        model.stream(req).await
    }

    async fn chat_stream_request_with_cancel(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStreamHandle, LlmError> {
        self.ensure_chat_capability(true)?;
        let this = self.clone();
        Ok(
            crate::utils::cancel::make_cancellable_stream_handle_from_handle_future(async move {
                let model_id = if !this.middlewares.is_empty() {
                    crate::execution::middleware::language_model::apply_model_id_override(
                        &this.middlewares,
                        &this.model_id,
                    )
                } else {
                    this.model_id.clone()
                };

                let model = this.get_or_create_language_model(&model_id).await?;

                let mut req = request.with_streaming(true);
                if req.common_params.model.trim().is_empty() {
                    req.common_params.model = model_id.clone();
                }
                if !this.middlewares.is_empty() {
                    req = crate::execution::middleware::language_model::apply_transform_chain(
                        &this.middlewares,
                        req,
                    );
                }

                model.stream_with_cancel(req).await
            }),
        )
    }
}

#[async_trait::async_trait]
impl FileManagementCapability for LanguageModelHandle {
    async fn upload_file(&self, request: FileUploadRequest) -> Result<FileObject, LlmError> {
        let client = self.build_language_client(&self.model_id).await?;
        let files = client.as_file_management_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation(format!(
                "Provider {} does not support file management.",
                self.provider_id
            ))
        })?;
        files.upload_file(request).await
    }

    async fn list_files(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError> {
        let client = self.build_language_client(&self.model_id).await?;
        let files = client.as_file_management_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation(format!(
                "Provider {} does not support file management.",
                self.provider_id
            ))
        })?;
        files.list_files(query).await
    }

    async fn retrieve_file(&self, file_id: String) -> Result<FileObject, LlmError> {
        let client = self.build_language_client(&self.model_id).await?;
        let files = client.as_file_management_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation(format!(
                "Provider {} does not support file management.",
                self.provider_id
            ))
        })?;
        files.retrieve_file(file_id).await
    }

    async fn delete_file(&self, file_id: String) -> Result<FileDeleteResponse, LlmError> {
        let client = self.build_language_client(&self.model_id).await?;
        let files = client.as_file_management_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation(format!(
                "Provider {} does not support file management.",
                self.provider_id
            ))
        })?;
        files.delete_file(file_id).await
    }

    async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        let client = self.build_language_client(&self.model_id).await?;
        let files = client.as_file_management_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation(format!(
                "Provider {} does not support file management.",
                self.provider_id
            ))
        })?;
        files.get_file_content(file_id).await
    }
}

#[async_trait::async_trait]
impl VideoGenerationCapability for LanguageModelHandle {
    async fn create_video_task(
        &self,
        request: VideoGenerationRequest,
    ) -> Result<VideoGenerationResponse, LlmError> {
        let client = self.build_language_client(&self.model_id).await?;
        let video = client.as_video_generation_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation(format!(
                "Provider {} does not support video generation.",
                self.provider_id
            ))
        })?;
        video.create_video_task(request).await
    }

    async fn query_video_task(&self, task_id: &str) -> Result<VideoTaskStatusResponse, LlmError> {
        let client = self.build_language_client(&self.model_id).await?;
        let video = client.as_video_generation_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation(format!(
                "Provider {} does not support video generation.",
                self.provider_id
            ))
        })?;
        video.query_video_task(task_id).await
    }

    fn get_supported_models(&self) -> Vec<String> {
        if self.capabilities.supports("video") {
            vec![self.model_id.clone()]
        } else {
            vec![]
        }
    }

    fn get_supported_resolutions(&self, _model: &str) -> Vec<String> {
        vec![]
    }

    fn get_supported_durations(&self, _model: &str) -> Vec<u32> {
        vec![]
    }
}

#[async_trait::async_trait]
impl MusicGenerationCapability for LanguageModelHandle {
    async fn generate_music(
        &self,
        request: MusicGenerationRequest,
    ) -> Result<MusicGenerationResponse, LlmError> {
        let client = self.build_language_client(&self.model_id).await?;
        let music = client.as_music_generation_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation(format!(
                "Provider {} does not support music generation.",
                self.provider_id
            ))
        })?;
        music.generate_music(request).await
    }

    fn get_supported_music_models(&self) -> Vec<String> {
        if self.capabilities.supports("music") {
            vec![self.model_id.clone()]
        } else {
            vec![]
        }
    }

    fn get_supported_audio_formats(&self) -> Vec<String> {
        vec![]
    }

    fn supports_lyrics(&self) -> bool {
        self.capabilities.supports("music")
    }
}

/// Embedding model handle - delegates to factory for client creation
#[derive(Clone)]
pub struct EmbeddingModelHandle {
    factory: Arc<dyn ProviderFactory>,
    provider_id: String,
    pub model_id: String,
    /// Registry-level HTTP interceptors to attempt injecting into clients
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Registry-level pre-built HTTP client copied into the handle
    http_client: Option<reqwest::Client>,
    /// Registry-level custom HTTP transport copied into the handle
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    /// Registry-level retry options copied into the handle
    retry_options: Option<RetryOptions>,
    /// Registry-level HTTP configuration copied into the handle
    http_config: Option<crate::types::HttpConfig>,
    /// Registry-level API key copied into the handle
    api_key: Option<String>,
    /// Registry-level base URL copied into the handle
    base_url: Option<String>,
}

/// Implementation of EmbeddingCapability for EmbeddingModelHandle
///
/// This allows the handle to be used directly as an embedding client, aligning with
/// Vercel AI SDK's design where registry.textEmbeddingModel() returns a callable model.
#[async_trait::async_trait]
impl EmbeddingCapability for EmbeddingModelHandle {
    async fn embed(&self, input: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        // Build client from factory with context
        let ctx = build_registry_context(
            &self.provider_id,
            &self.http_interceptors,
            &self.retry_options,
            &self.http_client,
            &self.http_transport,
            &self.http_config,
            &self.api_key,
            &self.base_url,
            None,
            None,
        );
        let model = self
            .factory
            .embedding_model_family_with_ctx(&self.model_id, &ctx)
            .await?;

        model
            .embed(crate::types::EmbeddingRequest {
                input,
                ..Default::default()
            })
            .await
    }

    fn as_embedding_extensions(&self) -> Option<&dyn EmbeddingExtensions> {
        Some(self)
    }

    fn embedding_dimension(&self) -> usize {
        // Default dimension - providers should override this
        // We can't get this without building a client, so we return a default
        // In practice, users should check the provider's documentation
        1536 // OpenAI's default
    }
}

impl crate::traits::ModelMetadata for EmbeddingModelHandle {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[async_trait::async_trait]
impl EmbeddingExtensions for EmbeddingModelHandle {
    async fn embed_with_config(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        let ctx = build_registry_context(
            &self.provider_id,
            &self.http_interceptors,
            &self.retry_options,
            &self.http_client,
            &self.http_transport,
            &self.http_config,
            &self.api_key,
            &self.base_url,
            None,
            None,
        );
        let model = self
            .factory
            .embedding_model_family_with_ctx(&self.model_id, &ctx)
            .await?;
        model.embed(request).await
    }

    async fn embed_batch(
        &self,
        requests: BatchEmbeddingRequest,
    ) -> Result<BatchEmbeddingResponse, LlmError> {
        let ctx = build_registry_context(
            &self.provider_id,
            &self.http_interceptors,
            &self.retry_options,
            &self.http_client,
            &self.http_transport,
            &self.http_config,
            &self.api_key,
            &self.base_url,
            None,
            None,
        );
        let model = self
            .factory
            .embedding_model_family_with_ctx(&self.model_id, &ctx)
            .await?;
        model.embed_many(requests).await
    }
}

impl EmbeddingModelHandle {}

/// Image model handle - delegates to factory for client creation
#[derive(Clone)]
pub struct ImageModelHandle {
    factory: Arc<dyn ProviderFactory>,
    provider_id: String,
    pub model_id: String,
    /// Registry-level HTTP interceptors to attempt injecting into clients
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Registry-level pre-built HTTP client copied into the handle
    http_client: Option<reqwest::Client>,
    /// Registry-level custom HTTP transport copied into the handle
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    /// Registry-level retry options copied into the handle
    retry_options: Option<RetryOptions>,
    /// Registry-level HTTP configuration copied into the handle
    http_config: Option<crate::types::HttpConfig>,
    /// Registry-level API key copied into the handle
    api_key: Option<String>,
    /// Registry-level base URL copied into the handle
    base_url: Option<String>,
}

/// Implementation of ImageGenerationCapability for ImageModelHandle
///
/// This allows the handle to be used directly as an image generation client, aligning with
/// Vercel AI SDK's design where registry.imageModel() returns a callable model.
#[async_trait::async_trait]
impl ImageGenerationCapability for ImageModelHandle {
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        // Build client from factory with context
        let ctx = build_registry_context(
            &self.provider_id,
            &self.http_interceptors,
            &self.retry_options,
            &self.http_client,
            &self.http_transport,
            &self.http_config,
            &self.api_key,
            &self.base_url,
            None,
            None,
        );
        let model = self
            .factory
            .image_model_family_with_ctx(&self.model_id, &ctx)
            .await?;

        model.generate(request).await
    }
}

#[async_trait::async_trait]
impl ImageExtras for ImageModelHandle {
    async fn edit_image(
        &self,
        request: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let ctx = build_registry_context(
            &self.provider_id,
            &self.http_interceptors,
            &self.retry_options,
            &self.http_client,
            &self.http_transport,
            &self.http_config,
            &self.api_key,
            &self.base_url,
            None,
            None,
        );
        let client_raw = self
            .factory
            .image_model_with_ctx(&self.model_id, &ctx)
            .await?;
        let client = client_raw;

        let image_client = client.as_image_extras().ok_or_else(|| {
            LlmError::UnsupportedOperation("Provider does not support image extras".to_string())
        })?;

        image_client.edit_image(request).await
    }

    async fn create_variation(
        &self,
        request: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let ctx = build_registry_context(
            &self.provider_id,
            &self.http_interceptors,
            &self.retry_options,
            &self.http_client,
            &self.http_transport,
            &self.http_config,
            &self.api_key,
            &self.base_url,
            None,
            None,
        );
        let client_raw = self
            .factory
            .image_model_with_ctx(&self.model_id, &ctx)
            .await?;
        let client = client_raw;

        let image_client = client.as_image_extras().ok_or_else(|| {
            LlmError::UnsupportedOperation("Provider does not support image extras".to_string())
        })?;

        image_client.create_variation(request).await
    }

    fn get_supported_sizes(&self) -> Vec<String> {
        // Best-effort defaults (provider-specific metadata is optional).
        vec![
            "1024x1024".to_string(),
            "512x512".to_string(),
            "256x256".to_string(),
        ]
    }

    fn get_supported_formats(&self) -> Vec<String> {
        vec!["url".to_string(), "b64_json".to_string()]
    }

    fn supports_image_editing(&self) -> bool {
        false
    }

    fn supports_image_variations(&self) -> bool {
        false
    }
}

impl crate::traits::ModelMetadata for ImageModelHandle {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

impl ImageModelHandle {}

/// Speech model handle (TTS) - delegates to factory for client creation
#[derive(Clone)]
pub struct SpeechModelHandle {
    factory: Arc<dyn ProviderFactory>,
    provider_id: String,
    pub model_id: String,
    /// Registry-level HTTP interceptors to attempt injecting into clients
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Registry-level pre-built HTTP client copied into the handle
    http_client: Option<reqwest::Client>,
    /// Registry-level custom HTTP transport copied into the handle
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    /// Registry-level retry options copied into the handle
    retry_options: Option<RetryOptions>,
    /// Registry-level HTTP configuration copied into the handle
    http_config: Option<crate::types::HttpConfig>,
    /// Registry-level API key copied into the handle
    api_key: Option<String>,
    /// Registry-level base URL copied into the handle
    base_url: Option<String>,
    /// Shared LRU cache for speech-family models
    cache: Arc<TokioMutex<LruCache<String, SpeechCacheEntry>>>,
    /// TTL for cached speech-family models
    client_ttl: Option<Duration>,
}

/// Implementation of AudioCapability for SpeechModelHandle
///
/// This allows the handle to be used directly as a TTS client, aligning with
/// Vercel AI SDK's design where registry.speechModel() returns a callable model.
#[async_trait::async_trait]
impl AudioCapability for SpeechModelHandle {
    fn supported_features(&self) -> &[AudioFeature] {
        // Default features - providers should override this
        &[AudioFeature::TextToSpeech]
    }

    async fn text_to_speech(&self, request: TtsRequest) -> Result<TtsResponse, LlmError> {
        let model = self.get_or_create_speech_model(&self.model_id).await?;
        model
            .synthesize(apply_speech_handle_default_model(request, &self.model_id))
            .await
    }

    async fn text_to_speech_stream(&self, request: TtsRequest) -> Result<AudioStream, LlmError> {
        let client = self.build_speech_client(&self.model_id).await?;
        let extras = client.as_speech_extras().ok_or_else(|| {
            LlmError::UnsupportedOperation(format!(
                "Provider {} does not support streaming text-to-speech.",
                self.provider_id
            ))
        })?;

        extras
            .tts_stream(apply_speech_handle_default_model(request, &self.model_id))
            .await
    }

    async fn get_voices(&self) -> Result<Vec<VoiceInfo>, LlmError> {
        let client = self.build_speech_client(&self.model_id).await?;
        let extras = client.as_speech_extras().ok_or_else(|| {
            LlmError::UnsupportedOperation(format!(
                "Provider {} does not support voice listing.",
                self.provider_id
            ))
        })?;

        extras.get_voices().await
    }
}

impl crate::traits::ModelMetadata for SpeechModelHandle {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

impl SpeechModelHandle {
    async fn build_speech_client(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        let ctx = build_registry_context(
            &self.provider_id,
            &self.http_interceptors,
            &self.retry_options,
            &self.http_client,
            &self.http_transport,
            &self.http_config,
            &self.api_key,
            &self.base_url,
            None,
            None,
        );
        self.factory.speech_model_with_ctx(model_id, &ctx).await
    }

    async fn get_or_create_speech_model(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn FamilySpeechModel>, LlmError> {
        let cache_key = format!("{}:{}", self.provider_id, model_id);

        let mut cache = self.cache.lock().await;
        if let Some(entry) = cache.get(&cache_key) {
            if !entry.is_expired(self.client_ttl) {
                return Ok(entry.model.clone());
            }
            cache.pop(&cache_key);
        }

        drop(cache);
        let ctx = build_registry_context(
            &self.provider_id,
            &self.http_interceptors,
            &self.retry_options,
            &self.http_client,
            &self.http_transport,
            &self.http_config,
            &self.api_key,
            &self.base_url,
            None,
            None,
        );
        let model = self
            .factory
            .speech_model_family_with_ctx(model_id, &ctx)
            .await?;

        let mut cache = self.cache.lock().await;
        cache.put(cache_key, SpeechCacheEntry::new(model.clone()));

        Ok(model)
    }

    /// Text to speech (deprecated - use trait method directly)
    #[deprecated(
        since = "0.10.3",
        note = "Use the AudioCapability trait method directly"
    )]
    pub async fn text_to_speech(
        &self,
        req: crate::types::TtsRequest,
    ) -> Result<crate::types::TtsResponse, LlmError> {
        // Delegate to trait implementation
        AudioCapability::text_to_speech(self, req).await
    }
}

/// Transcription model handle (STT) - delegates to factory for client creation
#[derive(Clone)]
pub struct TranscriptionModelHandle {
    factory: Arc<dyn ProviderFactory>,
    provider_id: String,
    pub model_id: String,
    /// Registry-level HTTP interceptors to attempt injecting into clients
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Registry-level pre-built HTTP client copied into the handle
    http_client: Option<reqwest::Client>,
    /// Registry-level custom HTTP transport copied into the handle
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    /// Registry-level retry options copied into the handle
    retry_options: Option<RetryOptions>,
    /// Registry-level HTTP configuration copied into the handle
    http_config: Option<crate::types::HttpConfig>,
    /// Registry-level API key copied into the handle
    api_key: Option<String>,
    /// Registry-level base URL copied into the handle
    base_url: Option<String>,
    /// Shared LRU cache for transcription-family models
    cache: Arc<TokioMutex<LruCache<String, TranscriptionCacheEntry>>>,
    /// TTL for cached transcription-family models
    client_ttl: Option<Duration>,
}

/// Implementation of AudioCapability for TranscriptionModelHandle
///
/// This allows the handle to be used directly as an STT client, aligning with
/// Vercel AI SDK's design where registry.transcriptionModel() returns a callable model.
#[async_trait::async_trait]
impl AudioCapability for TranscriptionModelHandle {
    fn supported_features(&self) -> &[AudioFeature] {
        // Default features - providers should override this
        &[AudioFeature::SpeechToText]
    }

    async fn speech_to_text(&self, request: SttRequest) -> Result<SttResponse, LlmError> {
        let model = self
            .get_or_create_transcription_model(&self.model_id)
            .await?;
        model
            .transcribe(apply_transcription_handle_default_model(
                request,
                &self.model_id,
            ))
            .await
    }

    async fn speech_to_text_stream(&self, request: SttRequest) -> Result<AudioStream, LlmError> {
        let client = self.build_transcription_client(&self.model_id).await?;
        let extras = client.as_transcription_extras().ok_or_else(|| {
            LlmError::UnsupportedOperation(format!(
                "Provider {} does not support streaming speech-to-text.",
                self.provider_id
            ))
        })?;

        extras
            .stt_stream(apply_transcription_handle_default_model(
                request,
                &self.model_id,
            ))
            .await
    }

    async fn translate_audio(
        &self,
        request: AudioTranslationRequest,
    ) -> Result<SttResponse, LlmError> {
        let client = self.build_transcription_client(&self.model_id).await?;
        let extras = client.as_transcription_extras().ok_or_else(|| {
            LlmError::UnsupportedOperation(format!(
                "Provider {} does not support audio translation.",
                self.provider_id
            ))
        })?;

        extras
            .audio_translate(apply_translation_handle_default_model(
                request,
                &self.model_id,
            ))
            .await
    }

    async fn get_supported_languages(&self) -> Result<Vec<LanguageInfo>, LlmError> {
        let client = self.build_transcription_client(&self.model_id).await?;
        let extras = client.as_transcription_extras().ok_or_else(|| {
            LlmError::UnsupportedOperation(format!(
                "Provider {} does not support language listing.",
                self.provider_id
            ))
        })?;

        extras.get_supported_languages().await
    }
}

impl crate::traits::ModelMetadata for TranscriptionModelHandle {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

impl TranscriptionModelHandle {
    async fn build_transcription_client(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let ctx = build_registry_context(
            &self.provider_id,
            &self.http_interceptors,
            &self.retry_options,
            &self.http_client,
            &self.http_transport,
            &self.http_config,
            &self.api_key,
            &self.base_url,
            None,
            None,
        );
        self.factory
            .transcription_model_with_ctx(model_id, &ctx)
            .await
    }

    async fn get_or_create_transcription_model(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn FamilyTranscriptionModel>, LlmError> {
        let cache_key = format!("{}:{}", self.provider_id, model_id);

        let mut cache = self.cache.lock().await;
        if let Some(entry) = cache.get(&cache_key) {
            if !entry.is_expired(self.client_ttl) {
                return Ok(entry.model.clone());
            }
            cache.pop(&cache_key);
        }

        drop(cache);
        let ctx = build_registry_context(
            &self.provider_id,
            &self.http_interceptors,
            &self.retry_options,
            &self.http_client,
            &self.http_transport,
            &self.http_config,
            &self.api_key,
            &self.base_url,
            None,
            None,
        );
        let model = self
            .factory
            .transcription_model_family_with_ctx(model_id, &ctx)
            .await?;

        let mut cache = self.cache.lock().await;
        cache.put(cache_key, TranscriptionCacheEntry::new(model.clone()));

        Ok(model)
    }
}

/// Reranking model handle - delegates to factory for client creation
#[derive(Clone)]
pub struct RerankingModelHandle {
    factory: Arc<dyn ProviderFactory>,
    provider_id: String,
    pub model_id: String,
    /// Registry-level HTTP interceptors to attempt injecting into clients
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Registry-level pre-built HTTP client copied into the handle
    http_client: Option<reqwest::Client>,
    /// Registry-level custom HTTP transport copied into the handle
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    /// Registry-level retry options copied into the handle
    retry_options: Option<RetryOptions>,
    /// Registry-level HTTP configuration copied into the handle
    http_config: Option<crate::types::HttpConfig>,
    /// Registry-level API key copied into the handle
    api_key: Option<String>,
    /// Registry-level base URL copied into the handle
    base_url: Option<String>,
}

/// Implementation of RerankCapability for RerankingModelHandle
///
/// This allows the handle to be used directly as a reranking client, aligning with
/// Vercel AI SDK's design where registry.rerankingModel() returns a callable model.
#[async_trait::async_trait]
impl RerankCapability for RerankingModelHandle {
    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, LlmError> {
        // Build client from factory with context
        let ctx = build_registry_context(
            &self.provider_id,
            &self.http_interceptors,
            &self.retry_options,
            &self.http_client,
            &self.http_transport,
            &self.http_config,
            &self.api_key,
            &self.base_url,
            None,
            None,
        );
        let model = self
            .factory
            .reranking_model_family_with_ctx(&self.model_id, &ctx)
            .await?;

        model.rerank(request).await
    }

    fn max_documents(&self) -> Option<u32> {
        None
    }

    fn supported_models(&self) -> Vec<String> {
        vec![self.model_id.clone()]
    }
}

impl crate::traits::ModelMetadata for RerankingModelHandle {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[cfg(test)]
use std::sync::atomic::AtomicUsize;
#[cfg(test)]
use std::sync::atomic::Ordering;
#[cfg(test)]
pub static TEST_BUILD_COUNT: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
pub struct TestProvClient;
#[cfg(test)]
#[async_trait::async_trait]
impl ChatCapability for TestProvClient {
    async fn chat_with_tools(
        &self,
        _messages: Vec<crate::types::ChatMessage>,
        _tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<crate::types::ChatResponse, LlmError> {
        Ok(crate::types::ChatResponse::new(
            crate::types::MessageContent::Text("ok".to_string()),
        ))
    }

    async fn chat_stream(
        &self,
        _messages: Vec<crate::types::ChatMessage>,
        _tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<crate::streaming::ChatStream, LlmError> {
        Err(LlmError::UnsupportedOperation("mock stream".into()))
    }
}
#[cfg(test)]
impl LlmClient for TestProvClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov")
    }
    fn supported_models(&self) -> Vec<String> {
        vec!["model".into()]
    }
    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new().with_chat()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(TestProvClient)
    }

    fn as_chat_capability(&self) -> Option<&dyn ChatCapability> {
        Some(self)
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub struct TestProvEmbedClient;
#[cfg(test)]
#[async_trait::async_trait]
impl crate::traits::EmbeddingCapability for TestProvEmbedClient {
    async fn embed(&self, input: Vec<String>) -> Result<crate::types::EmbeddingResponse, LlmError> {
        Ok(crate::types::EmbeddingResponse::new(
            vec![vec![input.len() as f32]],
            "test-embed-model".to_string(),
        ))
    }
    fn embedding_dimension(&self) -> usize {
        1
    }
}
#[cfg(test)]
impl LlmClient for TestProvEmbedClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov_embed")
    }
    fn supported_models(&self) -> Vec<String> {
        vec!["model".into()]
    }
    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(TestProvEmbedClient)
    }
    fn as_embedding_capability(&self) -> Option<&dyn crate::traits::EmbeddingCapability> {
        Some(self)
    }
}

#[cfg(test)]
pub struct TestProvImageClient;
#[cfg(test)]
#[async_trait::async_trait]
impl crate::traits::ImageGenerationCapability for TestProvImageClient {
    async fn generate_images(
        &self,
        request: crate::types::ImageGenerationRequest,
    ) -> Result<crate::types::ImageGenerationResponse, LlmError> {
        Ok(crate::types::ImageGenerationResponse {
            images: vec![crate::types::GeneratedImage {
                url: Some(format!("https://example.com/{}.png", request.prompt)),
                b64_json: None,
                format: None,
                width: None,
                height: None,
                revised_prompt: None,
                metadata: std::collections::HashMap::new(),
            }],
            metadata: std::collections::HashMap::new(),
            warnings: None,
            response: None,
        })
    }
}
#[cfg(test)]
impl LlmClient for TestProvImageClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov_image")
    }
    fn supported_models(&self) -> Vec<String> {
        vec!["model".into()]
    }
    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new().with_image_generation()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(TestProvImageClient)
    }
    fn as_image_generation_capability(
        &self,
    ) -> Option<&dyn crate::traits::ImageGenerationCapability> {
        Some(self)
    }
}

#[cfg(test)]
pub struct TestImageProviderFactory;

#[cfg(test)]
#[async_trait::async_trait]
impl ProviderFactory for TestImageProviderFactory {
    async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        Ok(Arc::new(TestProvImageClient))
    }

    async fn image_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        Ok(Arc::new(TestProvImageClient))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov_image")
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_image_generation()
    }
}

#[cfg(test)]
pub struct TestProviderFactory {
    id: &'static str,
}

#[cfg(test)]
impl TestProviderFactory {
    pub const fn new(id: &'static str) -> Self {
        Self { id }
    }
}

#[cfg(test)]
#[async_trait::async_trait]
impl ProviderFactory for TestProviderFactory {
    async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        TEST_BUILD_COUNT.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(TestProvClient))
    }

    async fn embedding_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        Ok(Arc::new(TestProvEmbedClient))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(self.id)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        match self.id {
            "testprov_embed" => ProviderCapabilities::new().with_embedding(),
            _ => ProviderCapabilities::new().with_chat(),
        }
    }
}

#[cfg(test)]
mod embedding_tests;
#[cfg(test)]
mod file_tests;
#[cfg(test)]
mod image_tests;
#[cfg(test)]
mod music_tests;
#[cfg(test)]
mod rerank_tests;
#[cfg(test)]
mod speech_tests;
#[cfg(test)]
mod transcription_tests;
#[cfg(test)]
mod video_tests;

#[cfg(test)]
#[async_trait::async_trait]
impl crate::traits::ChatCapability for TestProvEmbedClient {
    async fn chat_with_tools(
        &self,
        _messages: Vec<crate::types::ChatMessage>,
        _tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<crate::types::ChatResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "chat not supported in TestProvEmbedClient".into(),
        ))
    }

    async fn chat_stream(
        &self,
        _messages: Vec<crate::types::ChatMessage>,
        _tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<crate::streaming::ChatStream, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "chat stream not supported in TestProvEmbedClient".into(),
        ))
    }
}
