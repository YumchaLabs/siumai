//! ProviderCapabilities structure

use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct ProviderCapabilities {
    pub chat: bool,
    pub audio: bool,
    /// Text-to-speech (Vercel-aligned `SpeechModel`)
    pub speech: bool,
    /// Speech-to-text (Vercel-aligned `TranscriptionModel`)
    pub transcription: bool,
    pub vision: bool,
    pub tools: bool,
    pub embedding: bool,
    /// Image generation (Vercel-aligned `ImageModel`)
    pub image_generation: bool,
    pub rerank: bool,
    pub streaming: bool,
    pub file_management: bool,
    pub custom_features: HashMap<String, bool>,
}

impl ProviderCapabilities {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_chat(mut self) -> Self {
        self.chat = true;
        self
    }
    pub fn with_audio(mut self) -> Self {
        self.audio = true;
        // Backward compatible: historically `audio` implied both TTS and STT.
        self.speech = true;
        self.transcription = true;
        self
    }
    pub fn with_speech(mut self) -> Self {
        self.speech = true;
        self
    }
    pub fn with_transcription(mut self) -> Self {
        self.transcription = true;
        self
    }
    pub fn with_vision(mut self) -> Self {
        self.vision = true;
        self
    }
    pub fn with_tools(mut self) -> Self {
        self.tools = true;
        self
    }
    pub fn with_embedding(mut self) -> Self {
        self.embedding = true;
        self
    }
    pub fn with_image_generation(mut self) -> Self {
        self.image_generation = true;
        self
    }
    pub fn with_rerank(mut self) -> Self {
        self.rerank = true;
        self
    }
    pub fn with_streaming(mut self) -> Self {
        self.streaming = true;
        self
    }
    pub fn with_file_management(mut self) -> Self {
        self.file_management = true;
        self
    }
    pub fn with_custom_feature(mut self, name: &str, enabled: bool) -> Self {
        self.custom_features.insert(name.to_string(), enabled);
        self
    }

    pub fn supports(&self, feature: &str) -> bool {
        match feature {
            "chat" => self.chat,
            // Backward compatible: treat `audio` as aggregate.
            "audio" => {
                self.audio
                    || self.speech
                    || self.transcription
                    || self.custom_features.get(feature).copied().unwrap_or(false)
            }
            "speech" | "tts" => {
                self.speech
                    || self.audio
                    || self.custom_features.get("speech").copied().unwrap_or(false)
                    || self.custom_features.get("tts").copied().unwrap_or(false)
            }
            "transcription" | "stt" => {
                self.transcription
                    || self.audio
                    || self
                        .custom_features
                        .get("transcription")
                        .copied()
                        .unwrap_or(false)
                    || self.custom_features.get("stt").copied().unwrap_or(false)
            }
            "vision" => self.vision,
            "tools" => self.tools,
            "embedding" => self.embedding,
            // Backward compatible: also honor custom feature flags.
            "image_generation" => {
                self.image_generation || self.custom_features.get(feature).copied().unwrap_or(false)
            }
            // Backward compatible: also honor custom feature flags.
            "rerank" => self.rerank || self.custom_features.get(feature).copied().unwrap_or(false),
            "streaming" => self.streaming,
            "file_management" => self.file_management,
            _ => self.custom_features.get(feature).copied().unwrap_or(false),
        }
    }
}
