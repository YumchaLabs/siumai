use std::sync::Arc;

use crate::client::LlmClient;
use crate::embedding::EmbeddingModel as FamilyEmbeddingModel;
use crate::error::LlmError;
use crate::image::ImageModel as FamilyImageModel;
use crate::text::LanguageModel as FamilyLanguageModel;
use crate::traits::ProviderCapabilities;
use siumai_core::completion::CompletionModel as FamilyCompletionModel;
use siumai_core::rerank::RerankingModel as FamilyRerankingModel;
use siumai_core::speech::SpeechModel as FamilySpeechModel;
use siumai_core::transcription::TranscriptionModel as FamilyTranscriptionModel;
use siumai_core::video::VideoModel as FamilyVideoModel;

use super::build_context::BuildContext;
use super::compat_client::{
    ClientBackedCompletionModel, ClientBackedEmbeddingModel, ClientBackedImageModel,
    ClientBackedLanguageModel, ClientBackedRerankingModel, ClientBackedSpeechModel,
    ClientBackedTranscriptionModel, ClientBackedVideoModel,
};

/// Provider factory trait - similar to Vercel AI SDK's ProviderV3.
///
/// The primary contract is to create family model objects for the registry.
/// Generic `LlmClient` construction remains available only through explicit
/// `compat_*_client*` methods for historical entry points and
/// extension-only surfaces that have not become first-class families yet.
///
/// Note: Middlewares are applied by the Handle after client creation, not by the factory.
/// This keeps the factory simple and aligns with Vercel AI SDK's design where
/// middleware wrapping happens at the registry level.
#[async_trait::async_trait]
pub trait ProviderFactory: Send + Sync {
    /// Create a text-family language model with build context.
    ///
    /// This is the primary registry execution path for language models. The
    /// default implementation adapts the explicit compatibility client path so
    /// external providers can migrate incrementally.
    async fn language_model_text_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyLanguageModel>, LlmError> {
        let client = self.compat_language_client_with_ctx(model_id, ctx).await?;
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

    /// Legacy generic-client entry point for language models.
    ///
    /// Prefer `language_model_text_with_ctx(...)` for new registry execution.
    ///
    /// The returned client should NOT have middlewares applied - the Handle will apply them.
    #[deprecated(
        since = "0.11.0-beta.7",
        note = "Use compat_language_client(...) for legacy LlmClient construction or language_model_text_with_ctx(...) for registry execution."
    )]
    async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "Provider '{}' does not expose a legacy generic language LlmClient compatibility path",
            self.provider_id()
        )))
    }

    /// Compatibility alias for creating a generic `LlmClient` language client.
    ///
    /// New registry execution should prefer `language_model_text_with_ctx(...)`.
    /// This method makes remaining generic-client bridges explicit without
    /// breaking existing `ProviderFactory` implementations that still override
    /// `language_model(...)`.
    async fn compat_language_client(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        #[allow(deprecated)]
        self.language_model(model_id).await
    }

    /// Create a language model client with build context (interceptors, retry, etc.)
    /// Default implementation falls back to `language_model` for backward compatibility.
    #[deprecated(
        since = "0.11.0-beta.7",
        note = "Use compat_language_client_with_ctx(...) for legacy LlmClient construction or language_model_text_with_ctx(...) for registry execution."
    )]
    async fn language_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.compat_language_client_with_ctx(model_id, ctx).await
    }

    /// Compatibility alias for creating a generic `LlmClient` language client with context.
    async fn compat_language_client_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        #[allow(deprecated)]
        self.language_model(model_id).await
    }

    /// Create a completion-family model with build context.
    async fn completion_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyCompletionModel>, LlmError> {
        let client = self
            .compat_completion_client_with_ctx(model_id, ctx)
            .await?;
        Ok(Arc::new(ClientBackedCompletionModel::new(
            client,
            self.provider_id().into_owned(),
            model_id.to_string(),
        )))
    }

    /// Create a completion-family model without an explicit build context.
    async fn completion_model_family(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn FamilyCompletionModel>, LlmError> {
        self.completion_model_family_with_ctx(model_id, &BuildContext::default())
            .await
    }

    /// Create a completion model client for the given model ID.
    #[deprecated(
        since = "0.11.0-beta.7",
        note = "Use compat_completion_client(...) for legacy LlmClient construction or completion_model_family_with_ctx(...) for registry execution."
    )]
    async fn completion_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        #[allow(deprecated)]
        self.language_model(model_id).await
    }

    /// Compatibility alias for creating a generic `LlmClient` completion client.
    async fn compat_completion_client(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        #[allow(deprecated)]
        self.completion_model(model_id).await
    }

    /// Create a completion model client with build context.
    #[deprecated(
        since = "0.11.0-beta.7",
        note = "Use compat_completion_client_with_ctx(...) for legacy LlmClient construction or completion_model_family_with_ctx(...) for registry execution."
    )]
    async fn completion_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.compat_completion_client_with_ctx(model_id, ctx).await
    }

    /// Compatibility alias for creating a generic `LlmClient` completion client with context.
    async fn compat_completion_client_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        #[allow(deprecated)]
        self.completion_model(model_id).await
    }

    /// Create an embedding-family model with build context.
    async fn embedding_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyEmbeddingModel>, LlmError> {
        let client = self.compat_embedding_client_with_ctx(model_id, ctx).await?;
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

    /// Create an embedding model client for the given model ID
    #[deprecated(
        since = "0.11.0-beta.7",
        note = "Use compat_embedding_client(...) for legacy LlmClient construction or embedding_model_family_with_ctx(...) for registry execution."
    )]
    async fn embedding_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        #[allow(deprecated)]
        self.language_model(model_id).await
    }

    /// Compatibility alias for creating a generic `LlmClient` embedding client.
    async fn compat_embedding_client(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        #[allow(deprecated)]
        self.embedding_model(model_id).await
    }

    /// Create an embedding model client with build context.
    #[deprecated(
        since = "0.11.0-beta.7",
        note = "Use compat_embedding_client_with_ctx(...) for legacy LlmClient construction or embedding_model_family_with_ctx(...) for registry execution."
    )]
    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.compat_embedding_client_with_ctx(model_id, ctx).await
    }

    /// Compatibility alias for creating a generic `LlmClient` embedding client with context.
    async fn compat_embedding_client_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        #[allow(deprecated)]
        self.embedding_model(model_id).await
    }

    /// Create an image-family model with build context.
    async fn image_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyImageModel>, LlmError> {
        let client = self.compat_image_client_with_ctx(model_id, ctx).await?;
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

    /// Create an image model client for the given model ID
    #[deprecated(
        since = "0.11.0-beta.7",
        note = "Use compat_image_client(...) for legacy LlmClient construction or image_model_family_with_ctx(...) for registry execution."
    )]
    async fn image_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        #[allow(deprecated)]
        self.language_model(model_id).await
    }

    /// Compatibility alias for creating a generic `LlmClient` image client.
    async fn compat_image_client(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        #[allow(deprecated)]
        self.image_model(model_id).await
    }

    /// Create an image model client with build context.
    #[deprecated(
        since = "0.11.0-beta.7",
        note = "Use compat_image_client_with_ctx(...) for legacy LlmClient construction or image_model_family_with_ctx(...) for registry execution."
    )]
    async fn image_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.compat_image_client_with_ctx(model_id, ctx).await
    }

    /// Compatibility alias for creating a generic `LlmClient` image client with context.
    async fn compat_image_client_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        #[allow(deprecated)]
        self.image_model(model_id).await
    }

    /// Create a speech-family model with build context.
    async fn speech_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilySpeechModel>, LlmError> {
        let client = self.compat_speech_client_with_ctx(model_id, ctx).await?;
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

    /// Create a speech model client for the given model ID
    #[deprecated(
        since = "0.11.0-beta.7",
        note = "Use compat_speech_client(...) for legacy LlmClient construction or speech_model_family_with_ctx(...) for registry execution."
    )]
    async fn speech_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        #[allow(deprecated)]
        self.language_model(model_id).await
    }

    /// Compatibility alias for creating a generic `LlmClient` speech client.
    async fn compat_speech_client(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        #[allow(deprecated)]
        self.speech_model(model_id).await
    }

    /// Create a speech model client with build context.
    #[deprecated(
        since = "0.11.0-beta.7",
        note = "Use compat_speech_client_with_ctx(...) for legacy LlmClient construction or speech_model_family_with_ctx(...) for registry execution."
    )]
    async fn speech_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.compat_speech_client_with_ctx(model_id, ctx).await
    }

    /// Compatibility alias for creating a generic `LlmClient` speech client with context.
    async fn compat_speech_client_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        #[allow(deprecated)]
        self.speech_model(model_id).await
    }

    /// Create a transcription-family model with build context.
    async fn transcription_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyTranscriptionModel>, LlmError> {
        let client = self
            .compat_transcription_client_with_ctx(model_id, ctx)
            .await?;
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

    /// Create a transcription model client for the given model ID
    #[deprecated(
        since = "0.11.0-beta.7",
        note = "Use compat_transcription_client(...) for legacy LlmClient construction or transcription_model_family_with_ctx(...) for registry execution."
    )]
    async fn transcription_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        #[allow(deprecated)]
        self.speech_model(model_id).await
    }

    /// Compatibility alias for creating a generic `LlmClient` transcription client.
    async fn compat_transcription_client(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        #[allow(deprecated)]
        self.transcription_model(model_id).await
    }

    /// Create a transcription model client with build context.
    #[deprecated(
        since = "0.11.0-beta.7",
        note = "Use compat_transcription_client_with_ctx(...) for legacy LlmClient construction or transcription_model_family_with_ctx(...) for registry execution."
    )]
    async fn transcription_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.compat_transcription_client_with_ctx(model_id, ctx)
            .await
    }

    /// Compatibility alias for creating a generic `LlmClient` transcription client with context.
    async fn compat_transcription_client_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        #[allow(deprecated)]
        self.transcription_model(model_id).await
    }

    /// Create a video-family model with build context.
    async fn video_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyVideoModel>, LlmError> {
        let client = self.compat_video_client_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(ClientBackedVideoModel::new(
            client,
            self.provider_id().into_owned(),
            model_id.to_string(),
        )))
    }

    /// Create a video-family model without an explicit build context.
    async fn video_model_family(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn FamilyVideoModel>, LlmError> {
        self.video_model_family_with_ctx(model_id, &BuildContext::default())
            .await
    }

    /// Create a video model client for the given model ID.
    #[deprecated(
        since = "0.11.0-beta.7",
        note = "Use compat_video_client(...) for legacy LlmClient construction or video_model_family_with_ctx(...) for registry execution."
    )]
    async fn video_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        #[allow(deprecated)]
        self.language_model(model_id).await
    }

    /// Compatibility alias for creating a generic `LlmClient` video client.
    async fn compat_video_client(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        #[allow(deprecated)]
        self.video_model(model_id).await
    }

    /// Create a video model client with build context.
    #[deprecated(
        since = "0.11.0-beta.7",
        note = "Use compat_video_client_with_ctx(...) for legacy LlmClient construction or video_model_family_with_ctx(...) for registry execution."
    )]
    async fn video_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.compat_video_client_with_ctx(model_id, ctx).await
    }

    /// Compatibility alias for creating a generic `LlmClient` video client with context.
    async fn compat_video_client_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        #[allow(deprecated)]
        self.video_model(model_id).await
    }

    /// Create a reranking-family model with build context.
    async fn reranking_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyRerankingModel>, LlmError> {
        let client = self.compat_reranking_client_with_ctx(model_id, ctx).await?;
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

    /// Create a reranking model client for the given model ID
    #[deprecated(
        since = "0.11.0-beta.7",
        note = "Use compat_reranking_client(...) for legacy LlmClient construction or reranking_model_family_with_ctx(...) for registry execution."
    )]
    async fn reranking_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        #[allow(deprecated)]
        self.language_model(model_id).await
    }

    /// Compatibility alias for creating a generic `LlmClient` reranking client.
    async fn compat_reranking_client(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        #[allow(deprecated)]
        self.reranking_model(model_id).await
    }

    /// Create a reranking model client with build context.
    #[deprecated(
        since = "0.11.0-beta.7",
        note = "Use compat_reranking_client_with_ctx(...) for legacy LlmClient construction or reranking_model_family_with_ctx(...) for registry execution."
    )]
    async fn reranking_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.compat_reranking_client_with_ctx(model_id, ctx).await
    }

    /// Compatibility alias for creating a generic `LlmClient` reranking client with context.
    async fn compat_reranking_client_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        #[allow(deprecated)]
        self.reranking_model(model_id).await
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
