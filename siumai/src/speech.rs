//! Speech (text-to-speech) model family APIs.
//!
//! This is the recommended Rust-first surface for TTS:
//! - `synthesize`
//!
//! `synthesize` accepts the metadata-bearing `SpeechModel` family trait rather than
//! the legacy `SpeechModelV3` compatibility layer, and returns a high-level helper
//! result closer in role to AI SDK `generateSpeech()`.

use crate::retry_api::{RetryOptions, retry_with};
use base64::{Engine, engine::general_purpose::STANDARD};
use siumai_core::error::LlmError;
use siumai_core::types::{
    HttpConfig, HttpResponseInfo, ProviderMetadataMap, Warning, merge_provider_metadata,
    provider_metadata_from_object,
};
use siumai_core::utils::mime::guess_mime_from_bytes;
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

/// Generated audio file, closer in role to AI SDK `GeneratedAudioFile`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedAudioFile {
    data: Vec<u8>,
    /// Resolved IANA media type for the generated audio.
    pub media_type: String,
    /// Audio format such as `mp3` or `wav`.
    pub format: String,
}

impl GeneratedAudioFile {
    fn new(data: Vec<u8>, media_type: String, format: String) -> Self {
        Self {
            data,
            media_type,
            format,
        }
    }

    /// Borrow the generated audio bytes.
    pub fn as_bytes(&self) -> &[u8] {
        self.data.as_slice()
    }

    /// Clone the generated audio bytes.
    pub fn bytes(&self) -> Vec<u8> {
        self.data.clone()
    }

    /// Return the generated audio as base64.
    pub fn base64(&self) -> String {
        STANDARD.encode(&self.data)
    }
}

/// High-level speech helper result, closer to AI SDK `SpeechResult`.
#[derive(Debug, Clone)]
pub struct SpeechResult {
    /// Generated audio file.
    pub audio: GeneratedAudioFile,
    /// Compatibility mirror of the raw audio bytes.
    pub audio_data: Vec<u8>,
    /// Compatibility mirror of the raw audio format.
    pub format: String,
    /// Duration in seconds.
    pub duration: Option<f32>,
    /// Sample rate in Hz.
    pub sample_rate: Option<u32>,
    /// Legacy flat metadata map returned by the provider path.
    pub metadata: HashMap<String, serde_json::Value>,
    /// Non-fatal provider warnings.
    pub warnings: Vec<Warning>,
    /// Final response metadata envelopes.
    pub responses: Vec<HttpResponseInfo>,
    /// Provider-scoped metadata keyed by provider id.
    pub provider_metadata: ProviderMetadataMap,
}

/// AI SDK-auditable alias for the high-level speech helper result.
pub type GenerateSpeechResult = SpeechResult;

impl SpeechResult {
    /// The first response envelope, if available.
    pub fn response(&self) -> Option<&HttpResponseInfo> {
        self.responses.first()
    }

    /// Convert this high-level helper result back into the raw stable TTS response.
    pub fn into_tts_response(self) -> TtsResponse {
        TtsResponse {
            audio_data: self.audio_data,
            format: self.format,
            duration: self.duration,
            sample_rate: self.sample_rate,
            metadata: self.metadata,
            warnings: (!self.warnings.is_empty()).then_some(self.warnings),
            provider_metadata: (!self.provider_metadata.is_empty())
                .then_some(self.provider_metadata),
            response: self.responses.into_iter().next(),
        }
    }
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

fn fallback_audio_media_type(format: &str) -> String {
    match format.trim().to_ascii_lowercase().as_str() {
        "mp3" | "mpeg" => "audio/mpeg".to_string(),
        "wav" | "wave" => "audio/wav".to_string(),
        "ogg" => "audio/ogg".to_string(),
        "flac" => "audio/flac".to_string(),
        "aac" => "audio/aac".to_string(),
        "m4a" | "mp4" => "audio/mp4".to_string(),
        "webm" => "audio/webm".to_string(),
        "pcm" => "audio/pcm".to_string(),
        other if !other.is_empty() => format!("audio/{other}"),
        _ => "audio/mpeg".to_string(),
    }
}

fn normalize_generated_audio_media_type(media_type: String) -> String {
    match media_type.trim().to_ascii_lowercase().as_str() {
        "audio/x-wav" | "audio/wave" => "audio/wav".to_string(),
        _ => media_type,
    }
}

fn resolve_generated_audio_media_type(audio_data: &[u8], format: &str) -> String {
    normalize_generated_audio_media_type(
        guess_mime_from_bytes(audio_data).unwrap_or_else(|| fallback_audio_media_type(format)),
    )
}

fn speech_result_from_response(provider_id: &str, response: TtsResponse) -> SpeechResult {
    let audio_data = response.audio_data.clone();
    let format = response.format.clone();
    let media_type = resolve_generated_audio_media_type(audio_data.as_slice(), &format);
    let provider_metadata = provider_metadata_from_legacy_audio_metadata(
        provider_id,
        &response.metadata,
        response.provider_metadata.clone(),
    );

    SpeechResult {
        audio: GeneratedAudioFile::new(audio_data.clone(), media_type, format.clone()),
        audio_data,
        format,
        duration: response.duration,
        sample_rate: response.sample_rate,
        metadata: response.metadata,
        warnings: response.warnings.unwrap_or_default(),
        responses: response.response.into_iter().collect(),
        provider_metadata,
    }
}

/// Synthesize audio from text.
pub async fn synthesize<M: SpeechModel + ?Sized>(
    model: &M,
    request: TtsRequest,
    options: SynthesizeOptions,
) -> Result<SpeechResult, LlmError> {
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

    Ok(speech_result_from_response(model.provider_id(), response))
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
                metadata: HashMap::from([("requestId".to_string(), serde_json::json!("req_1"))]),
                warnings: None,
                provider_metadata: None,
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

    struct ReadySpeechModel;

    impl ModelMetadata for ReadySpeechModel {
        fn provider_id(&self) -> &str {
            "fake"
        }

        fn model_id(&self) -> &str {
            "ready-speech-model"
        }
    }

    #[async_trait::async_trait]
    impl SpeechModelV3 for ReadySpeechModel {
        async fn synthesize(&self, _request: TtsRequest) -> Result<TtsResponse, LlmError> {
            Ok(TtsResponse {
                audio_data: b"RIFF....WAVE".to_vec(),
                format: "wav".to_string(),
                duration: Some(1.5),
                sample_rate: Some(24_000),
                metadata: HashMap::from([("requestId".to_string(), serde_json::json!("req_2"))]),
                warnings: Some(vec![Warning::other("test warning")]),
                provider_metadata: None,
                response: Some(HttpResponseInfo {
                    timestamp: chrono::Utc::now(),
                    model_id: Some("ready-speech-model".to_string()),
                    headers: HashMap::from([("x-test".to_string(), "2".to_string())]),
                }),
            })
        }
    }

    #[tokio::test]
    async fn synthesize_returns_high_level_speech_result_with_derived_provider_metadata() {
        let result = synthesize(
            &ReadySpeechModel,
            TtsRequest::new("hello".to_string()),
            SynthesizeOptions::default(),
        )
        .await
        .expect("speech result");

        assert_eq!(result.audio_data, b"RIFF....WAVE".to_vec());
        assert_eq!(result.audio.format, "wav");
        assert_eq!(result.audio.media_type, "audio/wav");
        assert_eq!(result.duration, Some(1.5));
        assert_eq!(result.sample_rate, Some(24_000));
        assert_eq!(result.warnings.len(), 1);
        assert_eq!(result.responses.len(), 1);
        assert_eq!(
            result
                .provider_metadata
                .get("fake")
                .and_then(|value| value.as_object())
                .and_then(|value| value.get("requestId"))
                .and_then(|value| value.as_str()),
            Some("req_2")
        );
    }
}
