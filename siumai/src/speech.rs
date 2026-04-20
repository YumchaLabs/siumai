//! Speech (text-to-speech) model family APIs.
//!
//! This is the recommended Rust-first surface for TTS:
//! - `synthesize`
//!
//! `synthesize` accepts the metadata-bearing `SpeechModel` family trait rather than
//! the legacy `SpeechModelV3` compatibility layer.

use crate::retry_api::{RetryOptions, retry_with};
use siumai_core::error::LlmError;
use siumai_core::types::HttpConfig;
use std::collections::HashMap;
use std::time::Duration;

pub use siumai_core::speech::{SpeechModel, SpeechModelV3};
pub use siumai_core::types::{TtsRequest, TtsResponse};

/// Options for `speech::synthesize`.
#[derive(Debug, Clone, Default)]
pub struct SynthesizeOptions {
    /// Optional retry policy applied around the model call.
    pub retry: Option<RetryOptions>,
    /// Optional per-call request timeout.
    ///
    /// This is applied via `TtsRequest.http_config.timeout`.
    pub timeout: Option<Duration>,
    /// Optional per-call extra headers.
    ///
    /// These are merged into `TtsRequest.http_config.headers`.
    pub headers: HashMap<String, String>,
}

fn apply_tts_call_options(
    mut request: TtsRequest,
    timeout: Option<Duration>,
    headers: HashMap<String, String>,
) -> TtsRequest {
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

/// Synthesize audio from text.
pub async fn synthesize<M: SpeechModel + ?Sized>(
    model: &M,
    request: TtsRequest,
    options: SynthesizeOptions,
) -> Result<TtsResponse, LlmError> {
    let request = apply_tts_call_options(request, options.timeout, options.headers);
    let response = if let Some(retry) = options.retry {
        retry_with(
            || {
                let req = request.clone();
                async move { model.synthesize(req).await }
            },
            retry,
        )
        .await
    } else {
        model.synthesize(request).await
    }?;

    if response.audio_data.is_empty() {
        return Err(LlmError::NoSpeechGenerated {
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

    struct EmptySpeechModel;

    impl ModelMetadata for EmptySpeechModel {
        fn provider_id(&self) -> &str {
            "fake"
        }

        fn model_id(&self) -> &str {
            "empty-speech-model"
        }
    }

    #[async_trait::async_trait]
    impl SpeechModelV3 for EmptySpeechModel {
        async fn synthesize(&self, _request: TtsRequest) -> Result<TtsResponse, LlmError> {
            Ok(TtsResponse {
                audio_data: Vec::new(),
                format: "mp3".to_string(),
                duration: None,
                sample_rate: None,
                metadata: HashMap::new(),
                response: Some(HttpResponseInfo {
                    timestamp: chrono::Utc::now(),
                    model_id: Some("empty-speech-model".to_string()),
                    headers: HashMap::from([("x-test".to_string(), "1".to_string())]),
                }),
            })
        }
    }

    #[tokio::test]
    async fn synthesize_returns_no_speech_generated_with_response_metadata() {
        let err = synthesize(
            &EmptySpeechModel,
            TtsRequest::new("hello".to_string()),
            SynthesizeOptions::default(),
        )
        .await
        .expect_err("empty audio should fail");

        match err {
            LlmError::NoSpeechGenerated { responses } => {
                assert_eq!(responses.len(), 1);
                assert_eq!(responses[0].model_id.as_deref(), Some("empty-speech-model"));
                assert_eq!(responses[0].headers.get("x-test"), Some(&"1".to_string()));
            }
            other => panic!("expected NoSpeechGenerated error, got {other:?}"),
        }
    }
}
