//! Transcription (speech-to-text) capability trait
//!
//! This trait intentionally mirrors Vercel AI SDK's `TranscriptionModel` family.
//! In Siumai, most providers implement `AudioCapability`; this trait is a
//! focused view for STT-only call sites (e.g. registry `transcription_model`).

use crate::error::LlmError;
use crate::traits::AudioCapability;
use crate::types::{AudioStream, AudioTranslationRequest, LanguageInfo, SttRequest, SttResponse};
use async_trait::async_trait;

#[async_trait]
pub trait TranscriptionCapability: Send + Sync {
    async fn stt(&self, request: SttRequest) -> Result<SttResponse, LlmError>;

    async fn stt_stream(&self, request: SttRequest) -> Result<AudioStream, LlmError> {
        let _ = request;
        Err(LlmError::UnsupportedOperation(
            "Streaming speech-to-text not supported by this provider".to_string(),
        ))
    }

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
impl<T> TranscriptionCapability for T
where
    T: AudioCapability + Send + Sync,
{
    async fn stt(&self, request: SttRequest) -> Result<SttResponse, LlmError> {
        AudioCapability::speech_to_text(self, request).await
    }

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
