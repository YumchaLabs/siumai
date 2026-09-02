//! Portable final-result transcription over ElevenLabs Speech-to-Text.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use http::header::{ACCEPT, HeaderName, HeaderValue};
use http::{Method, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, Model, ModelDescriptor, ModelFamily, ModelId,
    ModelOperation, ProviderOptionError, PublicDiagnosticText, ResponseDiagnostics,
    ResponseMetadata, SensitiveResponse, TranscriptSegment, TranscriptionLimits,
    TranscriptionModel, TranscriptionRequest, TranscriptionResponse, TypedProviderOptions, Usage,
};
use siumai_transport::{
    MultipartBody, MultipartPart, ReplaySafety, RequestBody, RequestBuildError, RequestHeaders,
    RequestPlan, RequestTarget, ResponseHeaders, TransportResponse,
};

use super::models;
use super::profile::TRANSCRIPTION_API_MODE_ID;
use super::provider::TranscriptionRuntime;

const TARGET: &str = "v1/speech-to-text";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ElevenLabsTimestampGranularity {
    None,
    Word,
    Character,
}

impl ElevenLabsTimestampGranularity {
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Word => "word",
            Self::Character => "character",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ElevenLabsTranscriptionFileFormat {
    PcmS16le16,
    Other,
}

impl ElevenLabsTranscriptionFileFormat {
    fn as_str(self) -> &'static str {
        match self {
            Self::PcmS16le16 => "pcm_s16le_16",
            Self::Other => "other",
        }
    }
}

/// Typed ElevenLabs options for final-result/batch transcription.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ElevenLabsTranscriptionOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    tag_audio_events: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_speakers: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timestamps_granularity: Option<ElevenLabsTimestampGranularity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    diarize: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    file_format: Option<ElevenLabsTranscriptionFileFormat>,
}

impl ElevenLabsTranscriptionOptions {
    pub const fn new() -> Self {
        Self {
            tag_audio_events: None,
            num_speakers: None,
            timestamps_granularity: None,
            diarize: None,
            file_format: None,
        }
    }

    pub const fn with_tag_audio_events(mut self, value: bool) -> Self {
        self.tag_audio_events = Some(value);
        self
    }

    pub const fn with_num_speakers(mut self, value: u8) -> Self {
        self.num_speakers = Some(value);
        self
    }

    pub const fn with_timestamps_granularity(
        mut self,
        value: ElevenLabsTimestampGranularity,
    ) -> Self {
        self.timestamps_granularity = Some(value);
        self
    }

    pub const fn with_diarize(mut self, value: bool) -> Self {
        self.diarize = Some(value);
        self
    }

    pub const fn with_file_format(mut self, value: ElevenLabsTranscriptionFileFormat) -> Self {
        self.file_format = Some(value);
        self
    }

    pub(crate) fn merge_from(&mut self, other: Self) {
        if other.tag_audio_events.is_some() {
            self.tag_audio_events = other.tag_audio_events;
        }
        if other.num_speakers.is_some() {
            self.num_speakers = other.num_speakers;
        }
        if other.timestamps_granularity.is_some() {
            self.timestamps_granularity = other.timestamps_granularity;
        }
        if other.diarize.is_some() {
            self.diarize = other.diarize;
        }
        if other.file_format.is_some() {
            self.file_format = other.file_format;
        }
    }
}

impl TypedProviderOptions for ElevenLabsTranscriptionOptions {
    const NAMESPACE: &'static str = "elevenlabs";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Transcription;
    const API_MODE: Option<&'static str> = Some(TRANSCRIPTION_API_MODE_ID);

    fn validate(&self) -> Result<(), ProviderOptionError> {
        if self
            .num_speakers
            .is_some_and(|value| !(1..=32).contains(&value))
        {
            return Err(ProviderOptionError::Rejected {
                path: "num_speakers".to_string(),
                reason: "must be between 1 and 32".to_string(),
            });
        }
        Ok(())
    }
}

/// Lightweight final-result transcription model sharing one provider runtime.
#[derive(Clone)]
pub struct ElevenLabsTranscriptionModel {
    pub(crate) runtime: Arc<TranscriptionRuntime>,
    descriptor: ModelDescriptor,
}

impl ElevenLabsTranscriptionModel {
    pub(crate) fn new(runtime: Arc<TranscriptionRuntime>, model: ModelId) -> Self {
        let descriptor = ModelDescriptor::from_scope(
            runtime.scope.clone(),
            model,
            ModelFamily::Transcription,
            runtime.instance_id.clone(),
        );
        Self {
            runtime,
            descriptor,
        }
    }

    fn plan(
        &self,
        request: &TranscriptionRequest,
        options: &ElevenLabsTranscriptionOptions,
    ) -> Result<RequestPlan, Error> {
        if self.model_id().as_str() == models::SCRIBE_V2_REALTIME {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "ElevenLabs realtime transcription is not callable through the final-result adapter",
            ));
        }
        if request.prompt().is_some() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "ElevenLabs final-result transcription does not support prompt guidance",
            ));
        }
        let content_type = HeaderValue::from_str(request.media_type()).map_err(|source| {
            Error::new(
                ErrorKind::InvalidInput,
                "ElevenLabs transcription media type cannot be encoded",
            )
            .with_source(source)
        })?;
        let mut parts = vec![
            MultipartPart::file(
                "file",
                audio_file_name(request.media_type())?,
                content_type,
                request.audio().clone(),
            )
            .map_err(request_build_error)?,
            MultipartPart::field("model_id", self.model_id().as_str().as_bytes().to_vec())
                .map_err(request_build_error)?,
        ];
        if let Some(language) = request.language() {
            parts.push(
                MultipartPart::field("language_code", language.as_bytes().to_vec())
                    .map_err(request_build_error)?,
            );
        }
        for (name, value) in [
            (
                "tag_audio_events",
                options.tag_audio_events.map(|value| value.to_string()),
            ),
            (
                "num_speakers",
                options.num_speakers.map(|value| value.to_string()),
            ),
            (
                "timestamps_granularity",
                options
                    .timestamps_granularity
                    .map(|value| value.as_str().to_string()),
            ),
            ("diarize", options.diarize.map(|value| value.to_string())),
            (
                "file_format",
                options.file_format.map(|value| value.as_str().to_string()),
            ),
        ] {
            if let Some(value) = value {
                parts.push(
                    MultipartPart::field(name, value.into_bytes()).map_err(request_build_error)?,
                );
            }
        }
        let headers = RequestHeaders::new()
            .try_insert(ACCEPT, HeaderValue::from_static("application/json"))
            .map_err(request_build_error)?;
        RequestPlan::new(
            Method::POST,
            RequestTarget::new(TARGET).map_err(request_build_error)?,
        )
        .with_headers(headers)
        .with_body(RequestBody::multipart(MultipartBody::new(parts)))
        .with_replay_safety(ReplaySafety::Never)
        .map_err(request_build_error)
    }

    fn contextualize(&self, error: Error) -> Error {
        error.with_context(ErrorContext {
            operation: Some(ModelOperation::Transcribe),
            provider: Some(self.provider_id().clone()),
            route: None,
            model: Some(self.model_id().clone()),
        })
    }
}

impl std::fmt::Debug for ElevenLabsTranscriptionModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ElevenLabsTranscriptionModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for ElevenLabsTranscriptionModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl TranscriptionModel for ElevenLabsTranscriptionModel {
    fn limits(&self) -> TranscriptionLimits {
        TranscriptionLimits {
            max_audio_bytes: Some(self.runtime.transport.limits().max_request_bytes),
            max_duration_seconds: None,
        }
    }

    async fn transcribe(
        &self,
        request: TranscriptionRequest,
        call: CallOptions,
    ) -> Result<TranscriptionResponse, Error> {
        let call = call.resolve_deadline().map_err(Error::from)?;
        self.limits()
            .validate(&request)
            .map_err(|error| self.contextualize(error))?;
        let options = self
            .runtime
            .merge_options(self, &call)
            .map_err(option_error)
            .map_err(|error| self.contextualize(error))?;
        let plan = self
            .plan(&request, &options)
            .map_err(|error| self.contextualize(error))?;
        let response = self
            .runtime
            .transport
            .execute(plan, call)
            .await
            .map_err(|error| self.contextualize(error))?;
        if !response.status().is_success() {
            return Err(self.contextualize(response_error(response)));
        }
        decode_response(self.model_id(), response).map_err(|error| self.contextualize(error))
    }
}

#[derive(Deserialize)]
struct TranscriptionWire {
    text: String,
    #[serde(default)]
    language_code: Option<String>,
    #[serde(default)]
    language_probability: Option<f64>,
    #[serde(default)]
    words: Vec<WordWire>,
}

#[derive(Deserialize)]
struct WordWire {
    text: String,
    #[serde(default)]
    start: Option<f64>,
    #[serde(default)]
    end: Option<f64>,
}

fn decode_response(
    model: &ModelId,
    response: TransportResponse,
) -> Result<TranscriptionResponse, Error> {
    let attempts = response.attempts();
    let (_, headers, body) = response.into_parts();
    let wire: TranscriptionWire = serde_json::from_slice(&body).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "ElevenLabs returned malformed transcription JSON",
        )
        .with_source(source)
    })?;
    let mut duration = None;
    let segments = wire
        .words
        .iter()
        .filter_map(|word| {
            let (Some(start), Some(end)) = (word.start, word.end) else {
                return None;
            };
            duration = Some(duration.map_or(end, |current: f64| current.max(end)));
            Some(TranscriptSegment {
                start_seconds: start,
                end_seconds: end,
                text: word.text.clone(),
                confidence: None,
            })
        })
        .collect();
    let request_id = response_request_id(&headers);
    let mut provider = BTreeMap::new();
    provider.insert("transport_attempts".to_string(), Value::from(attempts));
    provider.insert("word_count".to_string(), Value::from(wire.words.len()));
    if let Some(probability) = wire.language_probability {
        provider.insert("language_probability".to_string(), Value::from(probability));
    }
    let response = TranscriptionResponse {
        text: wire.text,
        language: wire.language_code,
        confidence: wire.language_probability,
        duration_seconds: duration,
        segments,
        metadata: ResponseMetadata {
            response_id: None,
            request_id,
            model: Some(model.clone()),
        },
        usage: Usage::default(),
        warnings: Vec::new(),
        provider,
    };
    response.validate()?;
    Ok(response)
}

fn audio_file_name(media_type: &str) -> Result<String, Error> {
    let extension = match media_type.split(';').next().map(str::trim) {
        Some("audio/mpeg" | "audio/mp3") => "mp3",
        Some("audio/wav" | "audio/x-wav") => "wav",
        Some("audio/flac") => "flac",
        Some("audio/ogg") => "ogg",
        Some("audio/mp4" | "video/mp4") => "mp4",
        Some("audio/webm" | "video/webm") => "webm",
        Some("audio/pcm" | "audio/L16") => "pcm",
        _ => {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "ElevenLabs transcription media type is not supported by the portable adapter",
            ));
        }
    };
    Ok(format!("audio.{extension}"))
}

fn response_request_id(headers: &ResponseHeaders) -> Option<String> {
    ["request-id", "x-request-id"].into_iter().find_map(|name| {
        headers
            .get(&HeaderName::from_static(name))
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned)
    })
}

fn response_error(response: TransportResponse) -> Error {
    let (status, headers, body) = response.into_parts();
    let kind = match status {
        StatusCode::UNAUTHORIZED => ErrorKind::Authentication,
        StatusCode::FORBIDDEN => ErrorKind::Authorization,
        StatusCode::PAYMENT_REQUIRED => ErrorKind::QuotaExceeded,
        StatusCode::TOO_MANY_REQUESTS => ErrorKind::RateLimited,
        StatusCode::BAD_REQUEST | StatusCode::NOT_FOUND | StatusCode::UNPROCESSABLE_ENTITY => {
            ErrorKind::InvalidInput
        }
        _ => ErrorKind::Provider,
    };
    let mut diagnostics = ResponseDiagnostics::default().with_status(status.as_u16());
    if let Some(request_id) =
        response_request_id(&headers).and_then(|value| PublicDiagnosticText::new(value).ok())
    {
        diagnostics = diagnostics.with_request_id(request_id);
    }
    let raw_headers = headers
        .expose()
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.as_str().to_string(), value.to_string()))
        })
        .collect();
    Error::new(kind, "ElevenLabs transcription request failed")
        .with_diagnostics(diagnostics)
        .with_sensitive_response(SensitiveResponse::new(raw_headers, body.to_vec()))
}

fn option_error(source: ProviderOptionError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "ElevenLabs transcription options are invalid",
    )
    .with_source(source)
}

fn request_build_error(source: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "ElevenLabs transcription request violates the transport contract",
    )
    .with_source(source)
}

#[cfg(test)]
mod tests {
    use siumai_core::{TranscriptionModel as _, UsageValue};
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::configured::{
        ElevenLabsCredential, ElevenLabsProfile, ElevenLabsProvider, ElevenLabsTimestampGranularity,
    };

    fn provider(server: &MockServer) -> ElevenLabsProvider {
        ElevenLabsProvider::builder(
            ElevenLabsProfile::local_explicit(server.uri()).unwrap(),
            ElevenLabsCredential::api_key("test-key"),
        )
        .build()
        .unwrap()
    }

    #[tokio::test]
    async fn final_transcription_preserves_segments_and_unknown_usage() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/speech-to-text"))
            .and(header("xi-api-key", "test-key"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .insert_header("request-id", "request-1")
                    .set_body_json(serde_json::json!({
                        "language_code": "en",
                        "language_probability": 0.98,
                        "text": "hello world",
                        "words": [
                            { "text": "hello", "start": 0.0, "end": 0.4, "type": "word" },
                            { "text": "world", "start": 0.5, "end": 1.0, "type": "word" }
                        ]
                    })),
            )
            .expect(1)
            .mount(&server)
            .await;

        let options = ElevenLabsTranscriptionOptions::new()
            .with_diarize(true)
            .with_timestamps_granularity(ElevenLabsTimestampGranularity::Word);
        let model = provider(&server).default_transcription_model().unwrap();
        let call = CallOptions::default()
            .with_provider_options_for(&model, &options)
            .unwrap();
        let response = model
            .transcribe(
                TranscriptionRequest::new(vec![1_u8, 2, 3], "audio/mpeg")
                    .unwrap()
                    .with_language("en")
                    .unwrap(),
                call,
            )
            .await
            .unwrap();

        assert_eq!(response.text, "hello world");
        assert_eq!(response.language.as_deref(), Some("en"));
        assert_eq!(response.duration_seconds, Some(1.0));
        assert_eq!(response.segments.len(), 2);
        assert_eq!(response.metadata.request_id.as_deref(), Some("request-1"));
        assert_eq!(response.usage.input_tokens, UsageValue::Unknown);
        assert_eq!(response.usage.output_tokens, UsageValue::Unknown);

        let requests = server.received_requests().await.unwrap();
        let body = String::from_utf8_lossy(&requests[0].body);
        for expected in [
            "name=\"file\"",
            "name=\"model_id\"",
            models::SCRIBE_V2,
            "name=\"language_code\"",
            "name=\"diarize\"",
            "name=\"timestamps_granularity\"",
        ] {
            assert!(
                body.contains(expected),
                "missing multipart field {expected}"
            );
        }
    }

    #[tokio::test]
    async fn provider_error_body_is_bounded_and_redacted() {
        let server = MockServer::start().await;
        let secret = "provider-secret";
        let body = format!("{secret}{}", "x".repeat(70 * 1024));
        Mock::given(method("POST"))
            .and(path("/v1/speech-to-text"))
            .respond_with(ResponseTemplate::new(429).set_body_string(body))
            .expect(1)
            .mount(&server)
            .await;

        let error = provider(&server)
            .default_transcription_model()
            .unwrap()
            .transcribe(
                TranscriptionRequest::new(vec![1_u8, 2, 3], "audio/mpeg").unwrap(),
                CallOptions::default(),
            )
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::RateLimited);
        let sensitive = error.sensitive_response().unwrap();
        assert!(sensitive.was_truncated());
        assert!(!format!("{error:?}").contains(secret));
    }
}
