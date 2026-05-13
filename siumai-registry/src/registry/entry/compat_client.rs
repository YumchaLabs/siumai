use std::sync::Arc;

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::streaming::{ChatStream, ChatStreamHandle};
use crate::text::TextModel;
use crate::traits::{EmbeddingCapability, EmbeddingExtensions, RerankCapability};
use crate::types::{
    BatchEmbeddingRequest, BatchEmbeddingResponse, ChatRequest, ChatResponse, CompletionRequest,
    CompletionResponse, EmbeddingRequest, EmbeddingResponse, ImageGenerationRequest,
    ImageGenerationResponse, RerankRequest, RerankResponse, SttRequest, SttResponse, TtsRequest,
    TtsResponse, VideoGenerationRequest, VideoGenerationResponse, VideoTaskStatusResponse,
};

/// Compatibility adapters that bridge legacy generic clients into family models.
pub(super) struct ClientBackedEmbeddingModel {
    client: Arc<dyn LlmClient>,
    provider_id: String,
    model_id: String,
}

impl ClientBackedEmbeddingModel {
    pub(super) fn new(client: Arc<dyn LlmClient>, provider_id: String, model_id: String) -> Self {
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

        EmbeddingCapability::embed(self, request.input).await
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
            let result = EmbeddingCapability::embed(self, request.input)
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

pub(super) struct ClientBackedImageModel {
    client: Arc<dyn LlmClient>,
    provider_id: String,
    model_id: String,
}

impl ClientBackedImageModel {
    pub(super) fn new(client: Arc<dyn LlmClient>, provider_id: String, model_id: String) -> Self {
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
impl crate::image::ImageModel for ClientBackedImageModel {
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

pub(super) struct ClientBackedSpeechModel {
    client: Arc<dyn LlmClient>,
    provider_id: String,
    model_id: String,
}

impl ClientBackedSpeechModel {
    pub(super) fn new(client: Arc<dyn LlmClient>, provider_id: String, model_id: String) -> Self {
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
impl siumai_core::speech::SpeechModel for ClientBackedSpeechModel {
    async fn synthesize(&self, request: TtsRequest) -> Result<TtsResponse, LlmError> {
        let speech = self.client.as_speech_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation("Provider does not support text-to-speech".to_string())
        })?;
        speech.tts(request).await
    }
}

pub(super) struct ClientBackedTranscriptionModel {
    client: Arc<dyn LlmClient>,
    provider_id: String,
    model_id: String,
}

impl ClientBackedTranscriptionModel {
    pub(super) fn new(client: Arc<dyn LlmClient>, provider_id: String, model_id: String) -> Self {
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
impl siumai_core::transcription::TranscriptionModel for ClientBackedTranscriptionModel {
    async fn transcribe(&self, request: SttRequest) -> Result<SttResponse, LlmError> {
        let transcription = self.client.as_transcription_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation("Provider does not support speech-to-text".to_string())
        })?;
        transcription.stt(request).await
    }
}

pub(super) struct ClientBackedVideoModel {
    client: Arc<dyn LlmClient>,
    provider_id: String,
    model_id: String,
}

impl ClientBackedVideoModel {
    pub(super) fn new(client: Arc<dyn LlmClient>, provider_id: String, model_id: String) -> Self {
        Self {
            client,
            provider_id,
            model_id,
        }
    }
}

impl crate::traits::ModelMetadata for ClientBackedVideoModel {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[async_trait::async_trait]
impl siumai_core::video::VideoModel for ClientBackedVideoModel {
    async fn create_task(
        &self,
        request: VideoGenerationRequest,
    ) -> Result<VideoGenerationResponse, LlmError> {
        let video = self
            .client
            .as_video_generation_capability()
            .ok_or_else(|| {
                LlmError::UnsupportedOperation(
                    "Provider does not support video generation".to_string(),
                )
            })?;
        video.create_video_task(request).await
    }

    async fn query_task(&self, task_id: &str) -> Result<VideoTaskStatusResponse, LlmError> {
        let video = self
            .client
            .as_video_generation_capability()
            .ok_or_else(|| {
                LlmError::UnsupportedOperation(
                    "Provider does not support video generation".to_string(),
                )
            })?;
        video.query_video_task(task_id).await
    }

    async fn materialize_video_reference(
        &self,
        provider_reference: &crate::types::ProviderReference,
    ) -> Result<crate::types::MaterializedVideoAsset, LlmError> {
        let video = self
            .client
            .as_video_generation_capability()
            .ok_or_else(|| {
                LlmError::UnsupportedOperation(
                    "Provider does not support video generation".to_string(),
                )
            })?;
        video.materialize_video_reference(provider_reference).await
    }

    fn supported_models(&self) -> Vec<String> {
        self.client
            .as_video_generation_capability()
            .map(|video| video.get_supported_models())
            .unwrap_or_else(|| vec![self.model_id.clone()])
    }

    fn supported_resolutions(&self, model: &str) -> Vec<String> {
        self.client
            .as_video_generation_capability()
            .map(|video| video.get_supported_resolutions(model))
            .unwrap_or_default()
    }

    fn supported_durations(&self, model: &str) -> Vec<u32> {
        self.client
            .as_video_generation_capability()
            .map(|video| video.get_supported_durations(model))
            .unwrap_or_default()
    }
}

pub(super) struct ClientBackedRerankingModel {
    client: Arc<dyn LlmClient>,
    provider_id: String,
    model_id: String,
}

impl ClientBackedRerankingModel {
    pub(super) fn new(client: Arc<dyn LlmClient>, provider_id: String, model_id: String) -> Self {
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
impl RerankCapability for ClientBackedRerankingModel {
    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, LlmError> {
        let rerank = self.client.as_rerank_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation("Provider does not support reranking".to_string())
        })?;
        rerank.rerank(request).await
    }
}

pub(super) struct ClientBackedLanguageModel {
    client: Arc<dyn LlmClient>,
    provider_id: String,
    model_id: String,
}

impl ClientBackedLanguageModel {
    pub(super) fn new(client: Arc<dyn LlmClient>, provider_id: String, model_id: String) -> Self {
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
impl TextModel for ClientBackedLanguageModel {
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

pub(super) struct ClientBackedCompletionModel {
    client: Arc<dyn LlmClient>,
    provider_id: String,
    model_id: String,
}

impl ClientBackedCompletionModel {
    pub(super) fn new(client: Arc<dyn LlmClient>, provider_id: String, model_id: String) -> Self {
        Self {
            client,
            provider_id,
            model_id,
        }
    }
}

impl crate::traits::ModelMetadata for ClientBackedCompletionModel {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[async_trait::async_trait]
impl siumai_core::completion::CompletionModel for ClientBackedCompletionModel {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let completion = self.client.as_completion_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation("Provider does not support completions".to_string())
        })?;
        completion.complete(request).await
    }

    async fn stream(&self, request: CompletionRequest) -> Result<ChatStream, LlmError> {
        let completion = self.client.as_completion_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation("Provider does not support completions".to_string())
        })?;
        completion.complete_stream(request).await
    }

    async fn stream_with_cancel(
        &self,
        request: CompletionRequest,
    ) -> Result<ChatStreamHandle, LlmError> {
        let completion = self.client.as_completion_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation("Provider does not support completions".to_string())
        })?;
        completion.complete_stream_with_cancel(request).await
    }
}
