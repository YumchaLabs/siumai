use super::OpenAiCompatibleClient;
use crate::client::LlmClient;
use crate::traits::{
    AudioCapability, ChatCapability, CompletionCapability, EmbeddingCapability,
    ImageGenerationCapability, RerankCapability, SpeechCapability, SpeechExtras,
    TranscriptionCapability, TranscriptionExtras,
};
use siumai_core::traits::ModelMetadata;

impl ModelMetadata for OpenAiCompatibleClient {
    fn provider_id(&self) -> &str {
        self.config.provider_id.as_str()
    }

    fn model_id(&self) -> &str {
        self.config.model.as_str()
    }
}

impl LlmClient for OpenAiCompatibleClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        self.config.adapter.provider_id()
    }

    fn supported_models(&self) -> Vec<String> {
        // Return a basic list - could be enhanced with adapter-specific models
        vec![self.config.model.clone()]
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        let adapter_caps = self.config.adapter.capabilities();
        let has_full_audio =
            adapter_caps.audio || (adapter_caps.speech && adapter_caps.transcription);

        // Convert adapter capabilities to library capabilities
        let mut caps = crate::traits::ProviderCapabilities::new();

        if adapter_caps.chat {
            caps = caps.with_chat();
        }
        if adapter_caps.completion {
            caps = caps.with_completion();
        }
        if adapter_caps.streaming {
            caps = caps.with_streaming();
        }
        if has_full_audio {
            caps = caps.with_audio();
        } else {
            if adapter_caps.speech {
                caps = caps.with_speech();
            }
            if adapter_caps.transcription {
                caps = caps.with_transcription();
            }
        }
        if adapter_caps.embedding {
            caps = caps.with_embedding();
        }
        if adapter_caps.supports("rerank") {
            caps = caps.with_rerank();
        }
        if adapter_caps.tools {
            caps = caps.with_tools();
        }
        if adapter_caps.vision {
            caps = caps.with_vision();
        }
        if self.config.adapter.supports_image_generation() {
            caps = caps.with_image_generation();
        }
        for (name, enabled) in &adapter_caps.custom_features {
            caps = caps.with_custom_feature(name, *enabled);
        }

        caps
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new((*self).clone())
    }

    fn as_chat_capability(&self) -> Option<&dyn ChatCapability> {
        if self.config.adapter.capabilities().chat {
            Some(self)
        } else {
            None
        }
    }

    fn as_embedding_capability(&self) -> Option<&dyn EmbeddingCapability> {
        if self.config.adapter.capabilities().embedding {
            Some(self)
        } else {
            None
        }
    }

    fn as_completion_capability(&self) -> Option<&dyn CompletionCapability> {
        if self.capabilities().supports("completion") {
            Some(self)
        } else {
            None
        }
    }

    fn as_embedding_extensions(&self) -> Option<&dyn crate::traits::EmbeddingExtensions> {
        if self.config.adapter.capabilities().embedding {
            Some(self)
        } else {
            None
        }
    }

    fn as_audio_capability(&self) -> Option<&dyn AudioCapability> {
        if self.capabilities().supports("audio") {
            Some(self)
        } else {
            None
        }
    }

    fn as_speech_capability(&self) -> Option<&dyn SpeechCapability> {
        if self.capabilities().supports("speech") {
            Some(self)
        } else {
            None
        }
    }

    fn as_speech_extras(&self) -> Option<&dyn SpeechExtras> {
        if self.capabilities().supports("speech") {
            Some(self)
        } else {
            None
        }
    }

    fn as_transcription_capability(&self) -> Option<&dyn TranscriptionCapability> {
        if self.capabilities().supports("transcription") {
            Some(self)
        } else {
            None
        }
    }

    fn as_transcription_extras(&self) -> Option<&dyn TranscriptionExtras> {
        if self.capabilities().supports("transcription") {
            Some(self)
        } else {
            None
        }
    }

    fn as_image_generation_capability(&self) -> Option<&dyn ImageGenerationCapability> {
        if self.config.adapter.supports_image_generation() {
            Some(self)
        } else {
            None
        }
    }

    fn as_image_extras(&self) -> Option<&dyn crate::traits::ImageExtras> {
        if self.config.adapter.supports_image_generation() {
            Some(self)
        } else {
            None
        }
    }

    fn as_rerank_capability(&self) -> Option<&dyn RerankCapability> {
        // Keep capability gating consistent with executor-level guards:
        // rerank must be explicitly declared by the adapter/spec.
        if self.config.adapter.capabilities().supports("rerank") {
            Some(self)
        } else {
            None
        }
    }

    fn as_model_listing_capability(&self) -> Option<&dyn crate::traits::ModelListingCapability> {
        Some(self)
    }
}
