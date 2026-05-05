//! Speech (text-to-speech) model family.
//!
//! This module provides a Rust-first, family-oriented abstraction for TTS.
//! It is intentionally implemented as an adapter over the existing
//! `SpeechCapability` while provider construction continues moving toward
//! model-family traits.

use async_trait::async_trait;

use crate::error::LlmError;
use crate::traits::{ModelMetadata, SpeechCapability};
use crate::types::{TtsRequest, TtsResponse};

/// Stable Rust interface for speech synthesis models.
#[async_trait]
pub trait SpeechModel: ModelMetadata + Send + Sync {
    /// Synthesize audio from text.
    async fn synthesize(&self, request: TtsRequest) -> Result<TtsResponse, LlmError>;
}

/// Adapter: any `SpeechCapability` with metadata can be used as a `SpeechModel`.
#[async_trait]
impl<T> SpeechModel for T
where
    T: SpeechCapability + ModelMetadata + Send + Sync + ?Sized,
{
    async fn synthesize(&self, request: TtsRequest) -> Result<TtsResponse, LlmError> {
        SpeechCapability::tts(self, request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ModelSpecVersion;
    use std::collections::HashMap;

    struct FakeSpeech;

    impl crate::traits::ModelMetadata for FakeSpeech {
        fn provider_id(&self) -> &str {
            "fake"
        }

        fn model_id(&self) -> &str {
            "fake-speech"
        }
    }

    #[async_trait]
    impl SpeechCapability for FakeSpeech {
        async fn tts(&self, request: TtsRequest) -> Result<TtsResponse, LlmError> {
            Ok(TtsResponse {
                audio_data: request.text.into_bytes(),
                format: "pcm".to_string(),
                duration: None,
                sample_rate: None,
                metadata: HashMap::new(),
                request: None,
                warnings: None,
                provider_metadata: None,
                response: None,
            })
        }
    }

    #[tokio::test]
    async fn adapter_synthesize_uses_capability() {
        let model = FakeSpeech;
        let resp = SpeechModel::synthesize(&model, TtsRequest::new("hello".to_string()))
            .await
            .unwrap();
        assert_eq!(resp.audio_data, b"hello");
    }

    #[test]
    fn speech_model_trait_includes_metadata() {
        let model = FakeSpeech;

        fn assert_speech_model<M>(model: &M)
        where
            M: SpeechModel + ?Sized,
        {
            assert_eq!(crate::traits::ModelMetadata::provider_id(model), "fake");
            assert_eq!(crate::traits::ModelMetadata::model_id(model), "fake-speech");
            assert_eq!(
                crate::traits::ModelMetadata::specification_version(model),
                ModelSpecVersion::V1
            );
        }

        assert_speech_model(&model);
    }
}
