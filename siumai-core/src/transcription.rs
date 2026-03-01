//! Transcription (speech-to-text) model family (V3).
//!
//! This module provides a Rust-first, family-oriented abstraction for STT.
//! In V3-M2 it is implemented as an adapter over `TranscriptionCapability`.

use async_trait::async_trait;

use crate::error::LlmError;
use crate::traits::TranscriptionCapability;
use crate::types::{SttRequest, SttResponse};

/// V3 interface for transcription models.
#[async_trait]
pub trait TranscriptionModelV3: Send + Sync {
    /// Transcribe audio into text.
    async fn transcribe(&self, request: SttRequest) -> Result<SttResponse, LlmError>;
}

/// Adapter: any `TranscriptionCapability` can be used as a `TranscriptionModelV3`.
#[async_trait]
impl<T> TranscriptionModelV3 for T
where
    T: TranscriptionCapability + Send + Sync + ?Sized,
{
    async fn transcribe(&self, request: SttRequest) -> Result<SttResponse, LlmError> {
        TranscriptionCapability::stt(self, request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct FakeTranscription;

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
            })
        }
    }

    #[tokio::test]
    async fn adapter_transcribe_uses_capability() {
        let model = FakeTranscription;
        let resp = TranscriptionModelV3::transcribe(&model, SttRequest::from_audio(Vec::new()))
            .await
            .unwrap();
        assert_eq!(resp.text, "ok");
    }
}
