//! Transcription (speech-to-text) model family.
//!
//! This module provides a Rust-first, family-oriented abstraction for STT.
//! It is intentionally implemented as an adapter over the existing
//! `TranscriptionCapability` while provider construction continues moving toward
//! model-family traits.

use async_trait::async_trait;

use crate::error::LlmError;
use crate::traits::{ModelMetadata, TranscriptionCapability};
use crate::types::{SttRequest, SttResponse};

/// Stable Rust interface for transcription models.
#[async_trait]
pub trait TranscriptionModel: ModelMetadata + Send + Sync {
    /// Transcribe audio into text.
    async fn transcribe(&self, request: SttRequest) -> Result<SttResponse, LlmError>;
}

/// Adapter: any `TranscriptionCapability` with metadata can be used as a `TranscriptionModel`.
#[async_trait]
impl<T> TranscriptionModel for T
where
    T: TranscriptionCapability + ModelMetadata + Send + Sync + ?Sized,
{
    async fn transcribe(&self, request: SttRequest) -> Result<SttResponse, LlmError> {
        TranscriptionCapability::stt(self, request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ModelSpecVersion;
    use std::collections::HashMap;

    struct FakeTranscription;

    impl crate::traits::ModelMetadata for FakeTranscription {
        fn provider_id(&self) -> &str {
            "fake"
        }

        fn model_id(&self) -> &str {
            "fake-transcription"
        }
    }

    #[async_trait]
    impl TranscriptionCapability for FakeTranscription {
        async fn stt(&self, _request: SttRequest) -> Result<SttResponse, LlmError> {
            Ok(SttResponse {
                text: "ok".to_string(),
                language: None,
                confidence: None,
                words: None,
                duration: None,
                metadata: HashMap::new(),
                request: None,
                warnings: None,
                provider_metadata: None,
                response: None,
            })
        }
    }

    #[tokio::test]
    async fn adapter_transcribe_uses_capability() {
        let model = FakeTranscription;
        let resp =
            TranscriptionModel::transcribe(&model, SttRequest::from_audio(Vec::new(), "audio/wav"))
                .await
                .unwrap();
        assert_eq!(resp.text, "ok");
    }

    #[test]
    fn transcription_model_trait_includes_metadata() {
        let model = FakeTranscription;

        fn assert_transcription_model<M>(model: &M)
        where
            M: TranscriptionModel + ?Sized,
        {
            assert_eq!(crate::traits::ModelMetadata::provider_id(model), "fake");
            assert_eq!(
                crate::traits::ModelMetadata::model_id(model),
                "fake-transcription"
            );
            assert_eq!(
                crate::traits::ModelMetadata::specification_version(model),
                ModelSpecVersion::V1
            );
        }

        assert_transcription_model(&model);
    }
}
