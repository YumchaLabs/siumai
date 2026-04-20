//! Transcription (speech-to-text) model family APIs.
//!
//! This is the recommended Rust-first surface for STT:
//! - `transcribe`
//!
//! `transcribe` accepts the metadata-bearing `TranscriptionModel` family trait rather
//! than the legacy `TranscriptionModelV3` compatibility layer.

use crate::retry_api::{RetryOptions, retry_with};
use siumai_core::error::LlmError;
use siumai_core::types::HttpConfig;
use std::collections::HashMap;
use std::time::Duration;

pub use siumai_core::transcription::{TranscriptionModel, TranscriptionModelV3};
pub use siumai_core::types::{SttRequest, SttResponse};

/// Options for `transcription::transcribe`.
#[derive(Debug, Clone, Default)]
pub struct TranscribeOptions {
    /// Optional retry policy applied around the model call.
    pub retry: Option<RetryOptions>,
    /// Optional per-call request timeout.
    ///
    /// This is applied via `SttRequest.http_config.timeout`.
    pub timeout: Option<Duration>,
    /// Optional per-call extra headers.
    ///
    /// These are merged into `SttRequest.http_config.headers`.
    pub headers: HashMap<String, String>,
}

fn apply_stt_call_options(
    mut request: SttRequest,
    timeout: Option<Duration>,
    headers: HashMap<String, String>,
) -> SttRequest {
    if timeout.is_some() || !headers.is_empty() {
        let mut http = request.http_config.take().unwrap_or_else(HttpConfig::empty);
        if let Some(t) = timeout {
            http.timeout = Some(t);
        }
        if !headers.is_empty() {
            http.headers.extend(headers);
        }
        request.http_config = Some(http);
    }
    request
}

/// Transcribe audio into text.
pub async fn transcribe<M: TranscriptionModel + ?Sized>(
    model: &M,
    request: SttRequest,
    options: TranscribeOptions,
) -> Result<SttResponse, LlmError> {
    let request = apply_stt_call_options(request, options.timeout, options.headers);
    let response = if let Some(retry) = options.retry {
        retry_with(
            || {
                let req = request.clone();
                async move { model.transcribe(req).await }
            },
            retry,
        )
        .await
    } else {
        model.transcribe(request).await
    }?;

    if response.text.trim().is_empty() {
        return Err(LlmError::NoTranscriptGenerated {
            responses: response.response.clone().into_iter().collect(),
        });
    }

    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use siumai_core::traits::ModelMetadata;
    use siumai_core::types::HttpResponseInfo;

    struct EmptyTranscriptionModel;

    impl ModelMetadata for EmptyTranscriptionModel {
        fn provider_id(&self) -> &str {
            "fake"
        }

        fn model_id(&self) -> &str {
            "empty-transcription-model"
        }
    }

    #[async_trait::async_trait]
    impl TranscriptionModelV3 for EmptyTranscriptionModel {
        async fn transcribe(&self, _request: SttRequest) -> Result<SttResponse, LlmError> {
            Ok(SttResponse {
                text: "   ".to_string(),
                language: None,
                confidence: None,
                words: None,
                duration: None,
                metadata: HashMap::new(),
                response: Some(HttpResponseInfo {
                    timestamp: chrono::Utc::now(),
                    model_id: Some("empty-transcription-model".to_string()),
                    headers: HashMap::from([("x-test".to_string(), "1".to_string())]),
                }),
            })
        }
    }

    #[tokio::test]
    async fn transcribe_returns_no_transcript_generated_with_response_metadata() {
        let err = transcribe(
            &EmptyTranscriptionModel,
            SttRequest::from_audio(Vec::new(), "audio/wav"),
            TranscribeOptions::default(),
        )
        .await
        .expect_err("blank transcript should fail");

        match err {
            LlmError::NoTranscriptGenerated { responses } => {
                assert_eq!(responses.len(), 1);
                assert_eq!(
                    responses[0].model_id.as_deref(),
                    Some("empty-transcription-model")
                );
                assert_eq!(responses[0].headers.get("x-test"), Some(&"1".to_string()));
            }
            other => panic!("expected NoTranscriptGenerated error, got {other:?}"),
        }
    }
}
