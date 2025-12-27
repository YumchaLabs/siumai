//! Speech (text-to-speech) capability trait
//!
//! This trait intentionally mirrors Vercel AI SDK's `SpeechModel` family:
//! it provides a minimal, provider-agnostic TTS interface.
//!
//! Provider-specific extras (streaming, voice listing, etc.) are intentionally
//! *not* part of this unified surface. Use `SpeechExtras` or provider extension
//! APIs instead.

use crate::error::LlmError;
use crate::traits::AudioCapability;
use crate::types::{AudioStream, TtsRequest, TtsResponse, VoiceInfo};
use async_trait::async_trait;

#[async_trait]
pub trait SpeechCapability: Send + Sync {
    async fn tts(&self, request: TtsRequest) -> Result<TtsResponse, LlmError>;
}

#[async_trait]
impl<T> SpeechCapability for T
where
    T: AudioCapability + Send + Sync,
{
    async fn tts(&self, request: TtsRequest) -> Result<TtsResponse, LlmError> {
        AudioCapability::text_to_speech(self, request).await
    }
}

/// Provider-specific speech extras (non-unified surface).
#[async_trait]
pub trait SpeechExtras: Send + Sync {
    async fn tts_stream(&self, request: TtsRequest) -> Result<AudioStream, LlmError>;

    async fn get_voices(&self) -> Result<Vec<VoiceInfo>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Voice listing not supported by this provider".to_string(),
        ))
    }
}

#[async_trait]
impl<T> SpeechExtras for T
where
    T: AudioCapability + Send + Sync,
{
    async fn tts_stream(&self, request: TtsRequest) -> Result<AudioStream, LlmError> {
        AudioCapability::text_to_speech_stream(self, request).await
    }

    async fn get_voices(&self) -> Result<Vec<VoiceInfo>, LlmError> {
        AudioCapability::get_voices(self).await
    }
}
