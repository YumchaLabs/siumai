use std::sync::Arc;

use crate::compat::client::LlmClient;
use crate::embedding::EmbeddingModel as FamilyEmbeddingModel;
use crate::error::LlmError;
use crate::image::ImageModel as FamilyImageModel;
use crate::text::LanguageModel as FamilyLanguageModel;
use crate::traits::{
    FileManagementCapability, ImageExtras, MusicGenerationCapability, ProviderCapabilities,
    SkillsCapability, SpeechExtras, TranscriptionExtras,
};
use siumai_core::completion::CompletionModel as FamilyCompletionModel;
use siumai_core::rerank::RerankingModel as FamilyRerankingModel;
use siumai_core::speech::SpeechModel as FamilySpeechModel;
use siumai_core::transcription::TranscriptionModel as FamilyTranscriptionModel;
use siumai_core::video::VideoModel as FamilyVideoModel;

use super::build_context::BuildContext;
use super::extension_adapters::{
    ClientBackedFileManagementCapability, ClientBackedImageExtras,
    ClientBackedMusicGenerationCapability, ClientBackedSkillsCapability, ClientBackedSpeechExtras,
    ClientBackedTranscriptionExtras,
};

fn unsupported_native_family_model(provider_id: &str, family: &str) -> LlmError {
    LlmError::UnsupportedOperation(format!(
        "Provider '{provider_id}' does not expose a native {family} family model path"
    ))
}

fn unsupported_extension(provider_id: &str, extension: &str) -> LlmError {
    LlmError::UnsupportedOperation(format!(
        "Provider '{provider_id}' does not expose a {extension} extension path"
    ))
}

/// Family-first provider factory facet used by stable registry handles.
///
/// This facet is intentionally free of generic `LlmClient` construction. `ProviderFactory` remains
/// the source-compatible implementation trait for custom providers, while registry execution should
/// depend on this family facet whenever it builds stable model-family objects.
#[async_trait::async_trait]
pub trait ProviderFamilyFactory: Send + Sync {
    async fn build_language_model_text_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyLanguageModel>, LlmError>;

    async fn build_completion_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyCompletionModel>, LlmError>;

    async fn build_embedding_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyEmbeddingModel>, LlmError>;

    async fn build_image_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyImageModel>, LlmError>;

    async fn build_speech_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilySpeechModel>, LlmError>;

    async fn build_transcription_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyTranscriptionModel>, LlmError>;

    async fn build_video_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyVideoModel>, LlmError>;

    async fn build_reranking_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyRerankingModel>, LlmError>;

    fn family_provider_id(&self) -> std::borrow::Cow<'static, str>;

    fn family_capabilities(&self) -> ProviderCapabilities;
}

/// Compatibility factory facet used only by historical generic-client entry points.
#[async_trait::async_trait]
pub trait ProviderCompatibilityFactory: Send + Sync {
    async fn build_compat_language_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError>;

    async fn build_compat_completion_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError>;

    async fn build_compat_embedding_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError>;

    async fn build_compat_image_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError>;

    async fn build_compat_speech_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError>;

    async fn build_compat_transcription_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError>;

    async fn build_compat_video_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError>;

    async fn build_compat_reranking_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError>;
}

/// Extension-capability facet for non-family surfaces that still need registry access.
#[async_trait::async_trait]
pub trait ProviderExtensionFactory: Send + Sync {
    async fn build_file_management_capability_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FileManagementCapability>, LlmError>;

    async fn build_skills_capability_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn SkillsCapability>, LlmError>;

    async fn build_music_generation_capability_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn MusicGenerationCapability>, LlmError>;

    async fn build_image_extras_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn ImageExtras>, LlmError>;

    async fn build_speech_extras_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn SpeechExtras>, LlmError>;

    async fn build_transcription_extras_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn TranscriptionExtras>, LlmError>;
}

/// Provider factory trait - similar to Vercel AI SDK's ProviderV3.
///
/// The primary contract is to create family model objects for the registry.
/// Generic `LlmClient` construction remains available only through explicit
/// `compat_*_client*` methods for historical entry points and
/// extension-only surfaces that have not become first-class families yet.
/// Registry production code should use the narrower `ProviderFamilyFactory`,
/// `ProviderExtensionFactory`, or `ProviderCompatibilityFactory` facets instead
/// of treating this implementation trait as a single execution interface.
///
/// Note: Middlewares are applied by the Handle after client creation, not by the factory.
/// This keeps the factory simple and aligns with Vercel AI SDK's design where
/// middleware wrapping happens at the registry level.
#[async_trait::async_trait]
pub trait ProviderFactory: Send + Sync {
    /// Create a text-family language model with build context.
    ///
    /// This is the primary registry execution path for language models.
    async fn language_model_text_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyLanguageModel>, LlmError> {
        Err(unsupported_native_family_model(
            &self.provider_id(),
            "language",
        ))
    }

    /// Create a text-family language model without an explicit build context.
    async fn language_model_text(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn FamilyLanguageModel>, LlmError> {
        self.language_model_text_with_ctx(model_id, &BuildContext::default())
            .await
    }

    /// Compatibility alias for creating a generic `LlmClient` language client.
    ///
    /// New registry execution should prefer `language_model_text_with_ctx(...)`.
    async fn compat_language_client(
        &self,
        _model_id: &str,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "Provider '{}' does not expose a legacy generic language LlmClient compatibility path",
            self.provider_id()
        )))
    }

    /// Compatibility alias for creating a generic `LlmClient` language client with context.
    async fn compat_language_client_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.compat_language_client(model_id).await
    }

    /// Create a file-management extension capability with build context.
    ///
    /// File management remains an extension surface rather than a stable model family. The default
    /// implementation adapts a legacy generic client, while provider factories can override this
    /// method with native extension objects.
    async fn file_management_capability_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FileManagementCapability>, LlmError> {
        let client = self.compat_language_client_with_ctx(model_id, ctx).await?;
        if client.as_file_management_capability().is_none() {
            return Err(unsupported_extension(
                self.provider_id().as_ref(),
                "file-management",
            ));
        }

        Ok(Arc::new(ClientBackedFileManagementCapability::new(
            client,
            self.provider_id().into_owned(),
        )))
    }

    /// Create a file-management extension capability without an explicit build context.
    async fn file_management_capability(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn FileManagementCapability>, LlmError> {
        self.file_management_capability_with_ctx(model_id, &BuildContext::default())
            .await
    }

    /// Create a skill-upload extension capability with build context.
    async fn skills_capability_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn SkillsCapability>, LlmError> {
        let client = self.compat_language_client_with_ctx(model_id, ctx).await?;
        if client.as_skills_capability().is_none() {
            return Err(unsupported_extension(self.provider_id().as_ref(), "skills"));
        }

        Ok(Arc::new(ClientBackedSkillsCapability::new(
            client,
            self.provider_id().into_owned(),
        )))
    }

    /// Create a skill-upload extension capability without an explicit build context.
    async fn skills_capability(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn SkillsCapability>, LlmError> {
        self.skills_capability_with_ctx(model_id, &BuildContext::default())
            .await
    }

    /// Create a music-generation extension capability with build context.
    async fn music_generation_capability_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn MusicGenerationCapability>, LlmError> {
        let client = self.compat_language_client_with_ctx(model_id, ctx).await?;
        if client.as_music_generation_capability().is_none() {
            return Err(unsupported_extension(
                self.provider_id().as_ref(),
                "music-generation",
            ));
        }

        Ok(Arc::new(ClientBackedMusicGenerationCapability::new(
            client,
            self.provider_id().into_owned(),
        )))
    }

    /// Create a music-generation extension capability without an explicit build context.
    async fn music_generation_capability(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn MusicGenerationCapability>, LlmError> {
        self.music_generation_capability_with_ctx(model_id, &BuildContext::default())
            .await
    }

    /// Create an image extras extension with build context.
    async fn image_extras_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn ImageExtras>, LlmError> {
        let client = self.compat_image_client_with_ctx(model_id, ctx).await?;
        if client.as_image_extras().is_none() {
            return Err(unsupported_extension(
                self.provider_id().as_ref(),
                "image extras",
            ));
        }

        Ok(Arc::new(ClientBackedImageExtras::new(
            client,
            self.provider_id().into_owned(),
        )))
    }

    /// Create an image extras extension without an explicit build context.
    async fn image_extras(&self, model_id: &str) -> Result<Arc<dyn ImageExtras>, LlmError> {
        self.image_extras_with_ctx(model_id, &BuildContext::default())
            .await
    }

    /// Create a speech extras extension with build context.
    async fn speech_extras_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn SpeechExtras>, LlmError> {
        let client = self.compat_speech_client_with_ctx(model_id, ctx).await?;
        if client.as_speech_extras().is_none() {
            return Err(unsupported_extension(
                self.provider_id().as_ref(),
                "speech extras",
            ));
        }

        Ok(Arc::new(ClientBackedSpeechExtras::new(
            client,
            self.provider_id().into_owned(),
        )))
    }

    /// Create a speech extras extension without an explicit build context.
    async fn speech_extras(&self, model_id: &str) -> Result<Arc<dyn SpeechExtras>, LlmError> {
        self.speech_extras_with_ctx(model_id, &BuildContext::default())
            .await
    }

    /// Create a transcription extras extension with build context.
    async fn transcription_extras_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn TranscriptionExtras>, LlmError> {
        let client = self
            .compat_transcription_client_with_ctx(model_id, ctx)
            .await?;
        if client.as_transcription_extras().is_none() {
            return Err(unsupported_extension(
                self.provider_id().as_ref(),
                "transcription extras",
            ));
        }

        Ok(Arc::new(ClientBackedTranscriptionExtras::new(
            client,
            self.provider_id().into_owned(),
        )))
    }

    /// Create a transcription extras extension without an explicit build context.
    async fn transcription_extras(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn TranscriptionExtras>, LlmError> {
        self.transcription_extras_with_ctx(model_id, &BuildContext::default())
            .await
    }

    /// Create a completion-family model with build context.
    async fn completion_model_family_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyCompletionModel>, LlmError> {
        Err(unsupported_native_family_model(
            &self.provider_id(),
            "completion",
        ))
    }

    /// Create a completion-family model without an explicit build context.
    async fn completion_model_family(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn FamilyCompletionModel>, LlmError> {
        self.completion_model_family_with_ctx(model_id, &BuildContext::default())
            .await
    }

    /// Compatibility alias for creating a generic `LlmClient` completion client.
    async fn compat_completion_client(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.compat_language_client(model_id).await
    }

    /// Compatibility alias for creating a generic `LlmClient` completion client with context.
    async fn compat_completion_client_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.compat_completion_client(model_id).await
    }

    /// Create an embedding-family model with build context.
    async fn embedding_model_family_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyEmbeddingModel>, LlmError> {
        Err(unsupported_native_family_model(
            &self.provider_id(),
            "embedding",
        ))
    }

    /// Create an embedding-family model without an explicit build context.
    async fn embedding_model_family(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn FamilyEmbeddingModel>, LlmError> {
        self.embedding_model_family_with_ctx(model_id, &BuildContext::default())
            .await
    }

    /// Compatibility alias for creating a generic `LlmClient` embedding client.
    async fn compat_embedding_client(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.compat_language_client(model_id).await
    }

    /// Compatibility alias for creating a generic `LlmClient` embedding client with context.
    async fn compat_embedding_client_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.compat_embedding_client(model_id).await
    }

    /// Create an image-family model with build context.
    async fn image_model_family_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyImageModel>, LlmError> {
        Err(unsupported_native_family_model(
            &self.provider_id(),
            "image",
        ))
    }

    /// Create an image-family model without an explicit build context.
    async fn image_model_family(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn FamilyImageModel>, LlmError> {
        self.image_model_family_with_ctx(model_id, &BuildContext::default())
            .await
    }

    /// Compatibility alias for creating a generic `LlmClient` image client.
    async fn compat_image_client(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.compat_language_client(model_id).await
    }

    /// Compatibility alias for creating a generic `LlmClient` image client with context.
    async fn compat_image_client_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.compat_image_client(model_id).await
    }

    /// Create a speech-family model with build context.
    async fn speech_model_family_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilySpeechModel>, LlmError> {
        Err(unsupported_native_family_model(
            &self.provider_id(),
            "speech",
        ))
    }

    /// Create a speech-family model without an explicit build context.
    async fn speech_model_family(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn FamilySpeechModel>, LlmError> {
        self.speech_model_family_with_ctx(model_id, &BuildContext::default())
            .await
    }

    /// Compatibility alias for creating a generic `LlmClient` speech client.
    async fn compat_speech_client(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.compat_language_client(model_id).await
    }

    /// Compatibility alias for creating a generic `LlmClient` speech client with context.
    async fn compat_speech_client_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.compat_speech_client(model_id).await
    }

    /// Create a transcription-family model with build context.
    async fn transcription_model_family_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyTranscriptionModel>, LlmError> {
        Err(unsupported_native_family_model(
            &self.provider_id(),
            "transcription",
        ))
    }

    /// Create a transcription-family model without an explicit build context.
    async fn transcription_model_family(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn FamilyTranscriptionModel>, LlmError> {
        self.transcription_model_family_with_ctx(model_id, &BuildContext::default())
            .await
    }

    /// Compatibility alias for creating a generic `LlmClient` transcription client.
    async fn compat_transcription_client(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.compat_speech_client(model_id).await
    }

    /// Compatibility alias for creating a generic `LlmClient` transcription client with context.
    async fn compat_transcription_client_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.compat_transcription_client(model_id).await
    }

    /// Create a video-family model with build context.
    async fn video_model_family_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyVideoModel>, LlmError> {
        Err(unsupported_native_family_model(
            &self.provider_id(),
            "video",
        ))
    }

    /// Create a video-family model without an explicit build context.
    async fn video_model_family(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn FamilyVideoModel>, LlmError> {
        self.video_model_family_with_ctx(model_id, &BuildContext::default())
            .await
    }

    /// Compatibility alias for creating a generic `LlmClient` video client.
    async fn compat_video_client(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.compat_language_client(model_id).await
    }

    /// Compatibility alias for creating a generic `LlmClient` video client with context.
    async fn compat_video_client_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.compat_video_client(model_id).await
    }

    /// Create a reranking-family model with build context.
    async fn reranking_model_family_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyRerankingModel>, LlmError> {
        Err(unsupported_native_family_model(
            &self.provider_id(),
            "reranking",
        ))
    }

    /// Create a reranking-family model without an explicit build context.
    async fn reranking_model_family(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn FamilyRerankingModel>, LlmError> {
        self.reranking_model_family_with_ctx(model_id, &BuildContext::default())
            .await
    }

    /// Compatibility alias for creating a generic `LlmClient` reranking client.
    async fn compat_reranking_client(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.compat_language_client(model_id).await
    }

    /// Compatibility alias for creating a generic `LlmClient` reranking client with context.
    async fn compat_reranking_client_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.compat_reranking_client(model_id).await
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

/// Internal registry view of a provider implementation split by execution facet.
///
/// Public custom providers still implement `ProviderFactory`. The registry immediately adapts that
/// broad implementation trait into narrower facet objects so handles only receive the construction
/// surface they actually need.
#[derive(Clone)]
pub(in crate::registry::entry) struct ProviderFactoryFacets {
    family_factory: Arc<dyn ProviderFamilyFactory>,
    extension_factory: Arc<dyn ProviderExtensionFactory>,
}

impl ProviderFactoryFacets {
    pub(in crate::registry::entry) fn from_provider_factory(
        factory: Arc<dyn ProviderFactory>,
    ) -> Self {
        let adapter = Arc::new(ProviderFactoryFacetAdapter {
            factory: factory.clone(),
        });

        Self {
            family_factory: adapter.clone(),
            extension_factory: adapter,
        }
    }

    pub(in crate::registry::entry) fn capabilities(&self) -> ProviderCapabilities {
        self.family_factory.family_capabilities()
    }

    pub(in crate::registry::entry) fn family_factory(&self) -> Arc<dyn ProviderFamilyFactory> {
        self.family_factory.clone()
    }

    pub(in crate::registry::entry) fn extension_factory(
        &self,
    ) -> Arc<dyn ProviderExtensionFactory> {
        self.extension_factory.clone()
    }
}

#[cfg(feature = "builtins")]
pub(crate) fn compatibility_facet_from_provider_factory(
    factory: Arc<dyn ProviderFactory>,
) -> Arc<dyn ProviderCompatibilityFactory> {
    Arc::new(ProviderFactoryFacetAdapter { factory })
}

struct ProviderFactoryFacetAdapter {
    factory: Arc<dyn ProviderFactory>,
}

#[async_trait::async_trait]
impl ProviderFamilyFactory for ProviderFactoryFacetAdapter {
    async fn build_language_model_text_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyLanguageModel>, LlmError> {
        ProviderFactory::language_model_text_with_ctx(self.factory.as_ref(), model_id, ctx).await
    }

    async fn build_completion_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyCompletionModel>, LlmError> {
        ProviderFactory::completion_model_family_with_ctx(self.factory.as_ref(), model_id, ctx)
            .await
    }

    async fn build_embedding_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyEmbeddingModel>, LlmError> {
        ProviderFactory::embedding_model_family_with_ctx(self.factory.as_ref(), model_id, ctx).await
    }

    async fn build_image_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyImageModel>, LlmError> {
        ProviderFactory::image_model_family_with_ctx(self.factory.as_ref(), model_id, ctx).await
    }

    async fn build_speech_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilySpeechModel>, LlmError> {
        ProviderFactory::speech_model_family_with_ctx(self.factory.as_ref(), model_id, ctx).await
    }

    async fn build_transcription_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyTranscriptionModel>, LlmError> {
        ProviderFactory::transcription_model_family_with_ctx(self.factory.as_ref(), model_id, ctx)
            .await
    }

    async fn build_video_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyVideoModel>, LlmError> {
        ProviderFactory::video_model_family_with_ctx(self.factory.as_ref(), model_id, ctx).await
    }

    async fn build_reranking_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyRerankingModel>, LlmError> {
        ProviderFactory::reranking_model_family_with_ctx(self.factory.as_ref(), model_id, ctx).await
    }

    fn family_provider_id(&self) -> std::borrow::Cow<'static, str> {
        self.factory.provider_id()
    }

    fn family_capabilities(&self) -> ProviderCapabilities {
        self.factory.capabilities()
    }
}

#[async_trait::async_trait]
impl ProviderCompatibilityFactory for ProviderFactoryFacetAdapter {
    async fn build_compat_language_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        ProviderFactory::compat_language_client_with_ctx(self.factory.as_ref(), model_id, ctx).await
    }

    async fn build_compat_completion_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        ProviderFactory::compat_completion_client_with_ctx(self.factory.as_ref(), model_id, ctx)
            .await
    }

    async fn build_compat_embedding_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        ProviderFactory::compat_embedding_client_with_ctx(self.factory.as_ref(), model_id, ctx)
            .await
    }

    async fn build_compat_image_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        ProviderFactory::compat_image_client_with_ctx(self.factory.as_ref(), model_id, ctx).await
    }

    async fn build_compat_speech_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        ProviderFactory::compat_speech_client_with_ctx(self.factory.as_ref(), model_id, ctx).await
    }

    async fn build_compat_transcription_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        ProviderFactory::compat_transcription_client_with_ctx(self.factory.as_ref(), model_id, ctx)
            .await
    }

    async fn build_compat_video_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        ProviderFactory::compat_video_client_with_ctx(self.factory.as_ref(), model_id, ctx).await
    }

    async fn build_compat_reranking_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        ProviderFactory::compat_reranking_client_with_ctx(self.factory.as_ref(), model_id, ctx)
            .await
    }
}

#[async_trait::async_trait]
impl ProviderExtensionFactory for ProviderFactoryFacetAdapter {
    async fn build_file_management_capability_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FileManagementCapability>, LlmError> {
        ProviderFactory::file_management_capability_with_ctx(self.factory.as_ref(), model_id, ctx)
            .await
    }

    async fn build_skills_capability_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn SkillsCapability>, LlmError> {
        ProviderFactory::skills_capability_with_ctx(self.factory.as_ref(), model_id, ctx).await
    }

    async fn build_music_generation_capability_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn MusicGenerationCapability>, LlmError> {
        ProviderFactory::music_generation_capability_with_ctx(self.factory.as_ref(), model_id, ctx)
            .await
    }

    async fn build_image_extras_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn ImageExtras>, LlmError> {
        ProviderFactory::image_extras_with_ctx(self.factory.as_ref(), model_id, ctx).await
    }

    async fn build_speech_extras_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn SpeechExtras>, LlmError> {
        ProviderFactory::speech_extras_with_ctx(self.factory.as_ref(), model_id, ctx).await
    }

    async fn build_transcription_extras_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn TranscriptionExtras>, LlmError> {
        ProviderFactory::transcription_extras_with_ctx(self.factory.as_ref(), model_id, ctx).await
    }
}

#[async_trait::async_trait]
impl<T> ProviderFamilyFactory for T
where
    T: ProviderFactory + ?Sized,
{
    async fn build_language_model_text_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyLanguageModel>, LlmError> {
        ProviderFactory::language_model_text_with_ctx(self, model_id, ctx).await
    }

    async fn build_completion_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyCompletionModel>, LlmError> {
        ProviderFactory::completion_model_family_with_ctx(self, model_id, ctx).await
    }

    async fn build_embedding_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyEmbeddingModel>, LlmError> {
        ProviderFactory::embedding_model_family_with_ctx(self, model_id, ctx).await
    }

    async fn build_image_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyImageModel>, LlmError> {
        ProviderFactory::image_model_family_with_ctx(self, model_id, ctx).await
    }

    async fn build_speech_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilySpeechModel>, LlmError> {
        ProviderFactory::speech_model_family_with_ctx(self, model_id, ctx).await
    }

    async fn build_transcription_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyTranscriptionModel>, LlmError> {
        ProviderFactory::transcription_model_family_with_ctx(self, model_id, ctx).await
    }

    async fn build_video_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyVideoModel>, LlmError> {
        ProviderFactory::video_model_family_with_ctx(self, model_id, ctx).await
    }

    async fn build_reranking_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyRerankingModel>, LlmError> {
        ProviderFactory::reranking_model_family_with_ctx(self, model_id, ctx).await
    }

    fn family_provider_id(&self) -> std::borrow::Cow<'static, str> {
        ProviderFactory::provider_id(self)
    }

    fn family_capabilities(&self) -> ProviderCapabilities {
        ProviderFactory::capabilities(self)
    }
}

#[async_trait::async_trait]
impl<T> ProviderCompatibilityFactory for T
where
    T: ProviderFactory + ?Sized,
{
    async fn build_compat_language_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        ProviderFactory::compat_language_client_with_ctx(self, model_id, ctx).await
    }

    async fn build_compat_completion_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        ProviderFactory::compat_completion_client_with_ctx(self, model_id, ctx).await
    }

    async fn build_compat_embedding_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        ProviderFactory::compat_embedding_client_with_ctx(self, model_id, ctx).await
    }

    async fn build_compat_image_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        ProviderFactory::compat_image_client_with_ctx(self, model_id, ctx).await
    }

    async fn build_compat_speech_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        ProviderFactory::compat_speech_client_with_ctx(self, model_id, ctx).await
    }

    async fn build_compat_transcription_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        ProviderFactory::compat_transcription_client_with_ctx(self, model_id, ctx).await
    }

    async fn build_compat_video_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        ProviderFactory::compat_video_client_with_ctx(self, model_id, ctx).await
    }

    async fn build_compat_reranking_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        ProviderFactory::compat_reranking_client_with_ctx(self, model_id, ctx).await
    }
}

#[async_trait::async_trait]
impl<T> ProviderExtensionFactory for T
where
    T: ProviderFactory + ?Sized,
{
    async fn build_file_management_capability_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FileManagementCapability>, LlmError> {
        ProviderFactory::file_management_capability_with_ctx(self, model_id, ctx).await
    }

    async fn build_skills_capability_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn SkillsCapability>, LlmError> {
        ProviderFactory::skills_capability_with_ctx(self, model_id, ctx).await
    }

    async fn build_music_generation_capability_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn MusicGenerationCapability>, LlmError> {
        ProviderFactory::music_generation_capability_with_ctx(self, model_id, ctx).await
    }

    async fn build_image_extras_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn ImageExtras>, LlmError> {
        ProviderFactory::image_extras_with_ctx(self, model_id, ctx).await
    }

    async fn build_speech_extras_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn SpeechExtras>, LlmError> {
        ProviderFactory::speech_extras_with_ctx(self, model_id, ctx).await
    }

    async fn build_transcription_extras_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn TranscriptionExtras>, LlmError> {
        ProviderFactory::transcription_extras_with_ctx(self, model_id, ctx).await
    }
}
