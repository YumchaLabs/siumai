use super::Siumai;
use crate::client::LlmClient;
use crate::traits::*;
use std::borrow::Cow;

impl LlmClient for Siumai {
    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Owned(self.metadata.provider_id.clone())
    }

    fn supported_models(&self) -> Vec<String> {
        self.metadata.supported_models.clone()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.metadata.capabilities.clone()
    }

    fn as_embedding_capability(&self) -> Option<&dyn EmbeddingCapability> {
        self.client.as_embedding_capability()
    }

    fn as_audio_capability(&self) -> Option<&dyn AudioCapability> {
        self.client.as_audio_capability()
    }

    fn as_speech_capability(&self) -> Option<&dyn SpeechCapability> {
        self.client.as_speech_capability()
    }

    fn as_transcription_capability(&self) -> Option<&dyn TranscriptionCapability> {
        self.client.as_transcription_capability()
    }

    #[allow(deprecated)]
    fn as_vision_capability(&self) -> Option<&dyn VisionCapability> {
        self.client.as_vision_capability()
    }

    fn as_image_generation_capability(&self) -> Option<&dyn ImageGenerationCapability> {
        self.client.as_image_generation_capability()
    }

    fn as_image_extras(&self) -> Option<&dyn ImageExtras> {
        self.client.as_image_extras()
    }

    fn as_file_management_capability(&self) -> Option<&dyn FileManagementCapability> {
        self.client.as_file_management_capability()
    }

    fn as_moderation_capability(&self) -> Option<&dyn ModerationCapability> {
        self.client.as_moderation_capability()
    }

    fn as_model_listing_capability(&self) -> Option<&dyn ModelListingCapability> {
        self.client.as_model_listing_capability()
    }

    fn as_rerank_capability(&self) -> Option<&dyn RerankCapability> {
        self.client.as_rerank_capability()
    }

    fn as_video_generation_capability(&self) -> Option<&dyn VideoGenerationCapability> {
        self.client.as_video_generation_capability()
    }

    fn as_music_generation_capability(&self) -> Option<&dyn MusicGenerationCapability> {
        self.client.as_music_generation_capability()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }
}
