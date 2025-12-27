//! Speech (text-to-speech) capability trait
//!
//! This trait intentionally mirrors Vercel AI SDK's `SpeechModel` family.
//! In Siumai, most providers implement `AudioCapability`; this trait is a
//! focused view for TTS-only call sites (e.g. registry `speech_model`).

use crate::error::LlmError;
use crate::traits::AudioCapability;
use crate::types::{AudioStream, TtsRequest, TtsResponse, VoiceInfo};
use async_trait::async_trait;

#[async_trait]
pub trait SpeechCapability: Send + Sync {
    async fn tts(&self, request: TtsRequest) -> Result<TtsResponse, LlmError>;

    async fn tts_stream(&self, request: TtsRequest) -> Result<AudioStream, LlmError> {
        let _ = request;
        Err(LlmError::UnsupportedOperation(
            "Streaming text-to-speech not supported by this provider".to_string(),
        ))
    }

    async fn get_voices(&self) -> Result<Vec<VoiceInfo>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Voice listing not supported by this provider".to_string(),
        ))
    }
}

#[async_trait]
impl<T> SpeechCapability for T
where
    T: AudioCapability + Send + Sync,
{
    async fn tts(&self, request: TtsRequest) -> Result<TtsResponse, LlmError> {
        AudioCapability::text_to_speech(self, request).await
    }

    async fn tts_stream(&self, request: TtsRequest) -> Result<AudioStream, LlmError> {
        AudioCapability::text_to_speech_stream(self, request).await
    }

    async fn get_voices(&self) -> Result<Vec<VoiceInfo>, LlmError> {
        AudioCapability::get_voices(self).await
    }
}
