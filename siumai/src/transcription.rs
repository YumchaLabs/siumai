//! Transcription (speech-to-text) model family APIs.
//!
//! This is the recommended Rust-first surface for STT:
//! - `transcribe`
//!
//! `transcribe` accepts the metadata-bearing `TranscriptionModel` family trait rather
//! than the legacy `TranscriptionModelV3` compatibility layer, and returns a high-level
//! helper result closer in role to AI SDK `transcribe()`.

use crate::request_options::{EffectiveRequestOptions, retry_or_call_with_abort};
use crate::retry_api::RetryOptions;
use siumai_core::error::LlmError;
use siumai_core::types::{
    HttpConfig, HttpRequestInfo, HttpResponseInfo, ProviderMetadataMap, RequestOptions, Warning,
    WordTimestamp, merge_provider_metadata, provider_metadata_from_object,
};
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
    /// AI SDK-style request controls.
    ///
    /// When present, `max_retries` defaults to 2 to match AI SDK. Legacy
    /// `retry`, `timeout`, and `headers` fields override equivalent values here.
    pub request_options: Option<RequestOptions>,
}

/// Transcript segment aligned with AI SDK `TranscriptionResult.segments`.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionSegment {
    /// Segment text.
    pub text: String,
    /// Segment start time in seconds.
    pub start_second: f32,
    /// Segment end time in seconds.
    pub end_second: f32,
}

/// High-level transcription helper result, closer to AI SDK `TranscriptionResult`.
#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    /// Complete transcript text.
    pub text: String,
    /// Segment-level transcript timing information.
    pub segments: Vec<TranscriptionSegment>,
    /// Detected language.
    pub language: Option<String>,
    /// AI SDK-style duration field.
    pub duration_in_seconds: Option<f32>,
    /// Compatibility mirror for existing Rust callers.
    pub duration: Option<f32>,
    /// Confidence score when available.
    pub confidence: Option<f32>,
    /// Word-level timing information when available.
    pub words: Option<Vec<WordTimestamp>>,
    /// Legacy flat metadata map returned by the provider path.
    pub metadata: HashMap<String, serde_json::Value>,
    /// Best-effort request metadata for the final request.
    pub request: Option<HttpRequestInfo>,
    /// Non-fatal provider warnings.
    pub warnings: Vec<Warning>,
    /// Final response metadata envelopes.
    pub responses: Vec<HttpResponseInfo>,
    /// Provider-scoped metadata keyed by provider id.
    pub provider_metadata: ProviderMetadataMap,
}

/// Alias kept for explicit helper-result naming.
pub type TranscribeResult = TranscriptionResult;

impl TranscriptionResult {
    /// The first response envelope, if available.
    pub fn response(&self) -> Option<&HttpResponseInfo> {
        self.responses.first()
    }

    /// Convert this high-level helper result back into the raw stable STT response.
    pub fn into_stt_response(self) -> SttResponse {
        SttResponse {
            text: self.text,
            language: self.language,
            confidence: self.confidence,
            words: self.words,
            duration: self.duration,
            metadata: self.metadata,
            request: self.request,
            warnings: (!self.warnings.is_empty()).then_some(self.warnings),
            provider_metadata: (!self.provider_metadata.is_empty())
                .then_some(self.provider_metadata),
            response: self.responses.into_iter().next(),
        }
    }
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

fn provider_metadata_from_legacy_audio_metadata(
    provider_id: &str,
    metadata: &HashMap<String, serde_json::Value>,
    provider_metadata: Option<ProviderMetadataMap>,
) -> ProviderMetadataMap {
    let mut provider_metadata = provider_metadata.unwrap_or_default();
    if !metadata.is_empty() && !provider_metadata.contains_key(provider_id) {
        merge_provider_metadata(
            &mut provider_metadata,
            provider_metadata_from_object(provider_id.to_string(), metadata.clone()),
        );
    }
    provider_metadata
}

fn parse_segment_time(
    object: &serde_json::Map<String, serde_json::Value>,
    primary: &str,
    camel: &str,
    snake: &str,
) -> Option<f32> {
    object
        .get(primary)
        .or_else(|| object.get(camel))
        .or_else(|| object.get(snake))
        .and_then(|value| value.as_f64())
        .map(|value| value as f32)
}

fn extract_transcription_segments(
    metadata: &HashMap<String, serde_json::Value>,
    words: Option<&[WordTimestamp]>,
) -> Vec<TranscriptionSegment> {
    let from_metadata = metadata
        .get("segments")
        .and_then(|value| value.as_array())
        .map(|segments| {
            segments
                .iter()
                .filter_map(|segment| {
                    let object = segment.as_object()?;
                    let text = object.get("text")?.as_str()?.to_string();
                    let start = parse_segment_time(object, "start", "startSecond", "start_second")?;
                    let end = parse_segment_time(object, "end", "endSecond", "end_second")?;
                    Some(TranscriptionSegment {
                        text,
                        start_second: start,
                        end_second: end,
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    if !from_metadata.is_empty() {
        return from_metadata;
    }

    words
        .map(|words| {
            words
                .iter()
                .map(|word| TranscriptionSegment {
                    text: word.word.clone(),
                    start_second: word.start,
                    end_second: word.end,
                })
                .collect()
        })
        .unwrap_or_default()
}

fn transcription_result_from_response(
    provider_id: &str,
    response: SttResponse,
) -> TranscriptionResult {
    let segments = extract_transcription_segments(&response.metadata, response.words.as_deref());
    let provider_metadata = provider_metadata_from_legacy_audio_metadata(
        provider_id,
        &response.metadata,
        response.provider_metadata.clone(),
    );

    TranscriptionResult {
        text: response.text,
        segments,
        language: response.language,
        duration_in_seconds: response.duration,
        duration: response.duration,
        confidence: response.confidence,
        words: response.words,
        metadata: response.metadata,
        request: response.request,
        warnings: response.warnings.unwrap_or_default(),
        responses: response.response.into_iter().collect(),
        provider_metadata,
    }
}

/// Transcribe audio into text.
pub async fn transcribe<M: TranscriptionModel + ?Sized>(
    model: &M,
    request: SttRequest,
    options: TranscribeOptions,
) -> Result<TranscriptionResult, LlmError> {
    let effective = EffectiveRequestOptions::from_parts(
        options.request_options,
        options.retry,
        options.timeout,
        options.headers,
    );
    let request = apply_stt_call_options(request, effective.timeout(), effective.headers());
    let response = retry_or_call_with_abort(effective.retry(), effective.abort_signal(), || {
        let req = request.clone();
        async move { model.transcribe(req).await }
    })
    .await?;

    if response.text.trim().is_empty() {
        return Err(LlmError::NoTranscriptGenerated {
            responses: response.response.clone().into_iter().collect(),
        });
    }

    Ok(transcription_result_from_response(
        model.provider_id(),
        response,
    ))
}

/// Deprecated AI SDK-style alias for `transcribe`.
#[deprecated(note = "Use transcribe instead.")]
pub async fn experimental_transcribe<M: TranscriptionModel + ?Sized>(
    model: &M,
    request: SttRequest,
    options: TranscribeOptions,
) -> Result<TranscriptionResult, LlmError> {
    transcribe(model, request, options).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use siumai_core::traits::ModelMetadata;
    use siumai_core::types::{HttpRequestInfo, HttpResponseInfo};

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
                metadata: HashMap::from([("requestId".to_string(), serde_json::json!("req_1"))]),
                request: None,
                warnings: None,
                provider_metadata: None,
                response: Some(HttpResponseInfo {
                    timestamp: chrono::Utc::now(),
                    model_id: Some("empty-transcription-model".to_string()),
                    headers: HashMap::from([("x-test".to_string(), "1".to_string())]),
                    body: None,
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

    struct ReadyTranscriptionModel;

    impl ModelMetadata for ReadyTranscriptionModel {
        fn provider_id(&self) -> &str {
            "fake"
        }

        fn model_id(&self) -> &str {
            "ready-transcription-model"
        }
    }

    #[async_trait::async_trait]
    impl TranscriptionModelV3 for ReadyTranscriptionModel {
        async fn transcribe(&self, _request: SttRequest) -> Result<SttResponse, LlmError> {
            Ok(SttResponse {
                text: "hello world".to_string(),
                language: Some("en".to_string()),
                confidence: Some(0.9),
                words: Some(vec![
                    WordTimestamp {
                        word: "hello".to_string(),
                        start: 0.0,
                        end: 0.5,
                        confidence: Some(0.95),
                    },
                    WordTimestamp {
                        word: "world".to_string(),
                        start: 0.5,
                        end: 1.0,
                        confidence: Some(0.93),
                    },
                ]),
                duration: Some(1.0),
                metadata: HashMap::from([
                    ("requestId".to_string(), serde_json::json!("req_2")),
                    (
                        "segments".to_string(),
                        serde_json::json!([
                            { "text": "hello world", "start": 0.0, "end": 1.0 }
                        ]),
                    ),
                ]),
                request: Some(HttpRequestInfo {
                    body: Some(r#"{"model":"ready-transcription-model"}"#.to_string()),
                }),
                warnings: Some(vec![Warning::other("test warning")]),
                provider_metadata: None,
                response: Some(HttpResponseInfo {
                    timestamp: chrono::Utc::now(),
                    model_id: Some("ready-transcription-model".to_string()),
                    headers: HashMap::from([("x-test".to_string(), "2".to_string())]),
                    body: None,
                }),
            })
        }
    }

    #[tokio::test]
    async fn transcribe_returns_high_level_result_with_segments_and_provider_metadata() {
        let result = transcribe(
            &ReadyTranscriptionModel,
            SttRequest::from_audio(Vec::new(), "audio/wav"),
            TranscribeOptions::default(),
        )
        .await
        .expect("transcription result");

        assert_eq!(result.text, "hello world");
        assert_eq!(result.language.as_deref(), Some("en"));
        assert_eq!(result.duration_in_seconds, Some(1.0));
        assert_eq!(result.duration, Some(1.0));
        assert_eq!(result.segments.len(), 1);
        assert_eq!(result.segments[0].text, "hello world");
        assert_eq!(result.warnings.len(), 1);
        assert_eq!(result.responses.len(), 1);
        assert_eq!(
            result
                .request
                .as_ref()
                .and_then(|request| request.body.as_deref()),
            Some(r#"{"model":"ready-transcription-model"}"#)
        );
        assert_eq!(
            result
                .provider_metadata
                .get("fake")
                .and_then(|value| value.as_object())
                .and_then(|value| value.get("requestId"))
                .and_then(|value| value.as_str()),
            Some("req_2")
        );

        let raw = result.into_stt_response();
        assert_eq!(
            raw.request
                .as_ref()
                .and_then(|request| request.body.as_deref()),
            Some(r#"{"model":"ready-transcription-model"}"#)
        );
    }
}
