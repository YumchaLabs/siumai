//! Transcription (speech-to-text) capability trait
//!
//! This trait intentionally mirrors Vercel AI SDK's `TranscriptionModel` family.
//! It provides a minimal, provider-agnostic STT interface.
//!
//! Provider-specific extras (streaming, translation, language listing, etc.) are
//! intentionally *not* part of this unified surface. Use `TranscriptionExtras`
//! or provider extension APIs instead.

use crate::error::LlmError;
use crate::traits::AudioCapability;
use crate::types::{AudioStream, AudioTranslationRequest, LanguageInfo, SttRequest, SttResponse};
use async_trait::async_trait;

#[async_trait]
pub trait TranscriptionCapability: Send + Sync {
    async fn stt(&self, request: SttRequest) -> Result<SttResponse, LlmError>;
}

#[async_trait]
impl<T> TranscriptionCapability for T
where
    T: AudioCapability + Send + Sync,
{
    async fn stt(&self, request: SttRequest) -> Result<SttResponse, LlmError> {
        AudioCapability::speech_to_text(self, request).await
    }
}

/// Provider-specific transcription extras (non-unified surface).
#[async_trait]
pub trait TranscriptionExtras: Send + Sync {
    async fn stt_stream(&self, request: SttRequest) -> Result<AudioStream, LlmError>;

    async fn audio_translate(
        &self,
        request: AudioTranslationRequest,
    ) -> Result<SttResponse, LlmError> {
        let _ = request;
        Err(LlmError::UnsupportedOperation(
            "Audio translation not supported by this provider".to_string(),
        ))
    }

    async fn get_supported_languages(&self) -> Result<Vec<LanguageInfo>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Language listing not supported by this provider".to_string(),
        ))
    }
}

#[async_trait]
impl<T> TranscriptionExtras for T
where
    T: AudioCapability + Send + Sync,
{
    async fn stt_stream(&self, request: SttRequest) -> Result<AudioStream, LlmError> {
        AudioCapability::speech_to_text_stream(self, request).await
    }

    async fn audio_translate(
        &self,
        request: AudioTranslationRequest,
    ) -> Result<SttResponse, LlmError> {
        AudioCapability::translate_audio(self, request).await
    }

    async fn get_supported_languages(&self) -> Result<Vec<LanguageInfo>, LlmError> {
        AudioCapability::get_supported_languages(self).await
    }
}
