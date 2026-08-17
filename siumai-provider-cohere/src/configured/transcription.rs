use std::fmt;
use std::sync::Arc;

use bytes::Bytes;
use http::Method;
use http::header::{ACCEPT, HeaderValue};
use serde::Deserialize;
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, ModelOperation, ResponseMetadata,
    TranscriptionRequest, TranscriptionResponse, Usage,
};
use siumai_transport::{
    MultipartBody, MultipartPart, ReplaySafety, RequestBody, RequestBuildError, RequestHeaders,
    RequestPlan, RequestTarget,
};

use super::model::{provider_status_error, response_request_id};
use super::provider::CohereRuntime;

const TRANSCRIPTION_TARGET: &str = "audio/transcriptions";
const MAX_AUDIO_BYTES: usize = 25_000_000;
const MAX_LANGUAGE_BYTES: usize = 64;

/// One bounded request for Cohere's model-less v2 audio transcription resource.
#[derive(Clone, PartialEq)]
pub struct CohereTranscriptionRequest {
    audio: Bytes,
    media_type: String,
    language: String,
    temperature: Option<f32>,
}

impl CohereTranscriptionRequest {
    /// Create a transcription request with Cohere's required language field.
    pub fn new(
        audio: impl Into<Bytes>,
        media_type: impl Into<String>,
        language: impl Into<String>,
    ) -> Result<Self, Error> {
        let request = Self {
            audio: audio.into(),
            media_type: media_type.into(),
            language: language.into(),
            temperature: None,
        };
        request.validate()?;
        Ok(request)
    }

    /// Set Cohere's documented transcription sampling temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Result<Self, Error> {
        self.temperature = Some(temperature);
        self.validate()?;
        Ok(self)
    }

    pub fn audio(&self) -> &Bytes {
        &self.audio
    }

    pub fn media_type(&self) -> &str {
        &self.media_type
    }

    pub fn language(&self) -> &str {
        &self.language
    }

    pub const fn temperature(&self) -> Option<f32> {
        self.temperature
    }

    fn validate(&self) -> Result<(), Error> {
        if self.audio.is_empty() {
            return Err(invalid_input(
                "Cohere transcription audio must not be empty",
            ));
        }
        if self.audio.len() > MAX_AUDIO_BYTES {
            return Err(Error::limit_exceeded(
                siumai_core::ResourceKind::TranscriptionAudioBytes,
                self.audio.len() as u64,
                MAX_AUDIO_BYTES as u64,
            ));
        }
        if audio_file_name(&self.media_type).is_none() {
            return Err(invalid_input(
                "Cohere transcription media type must be MP3, WAV, FLAC, AAC, M4A, or OGG",
            ));
        }
        if self.language.trim().is_empty()
            || self.language.len() > MAX_LANGUAGE_BYTES
            || self.language.chars().any(char::is_control)
        {
            return Err(invalid_input(
                "Cohere transcription language must be non-empty and within the local field budget",
            ));
        }
        if self
            .temperature
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(invalid_input(
                "Cohere transcription temperature must be between 0 and 1",
            ));
        }
        Ok(())
    }
}

impl TryFrom<TranscriptionRequest> for CohereTranscriptionRequest {
    type Error = Error;

    fn try_from(request: TranscriptionRequest) -> Result<Self, Self::Error> {
        if request.prompt().is_some() {
            return Err(invalid_input(
                "Cohere audio transcription does not expose a prompt field",
            ));
        }
        let language = request.language().ok_or_else(|| {
            invalid_input("Cohere audio transcription requires an explicit language")
        })?;
        Self::new(request.audio().clone(), request.media_type(), language)
    }
}

impl fmt::Debug for CohereTranscriptionRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CohereTranscriptionRequest")
            .field("audio_bytes", &self.audio.len())
            .field("media_type_bytes", &self.media_type.len())
            .field("language_bytes", &self.language.len())
            .field("temperature", &self.temperature)
            .finish()
    }
}

/// Provider-owned access to Cohere's v2 audio transcription operation.
#[derive(Clone)]
pub struct CohereTranscriptions {
    runtime: Arc<CohereRuntime>,
}

impl CohereTranscriptions {
    pub(crate) fn new(runtime: Arc<CohereRuntime>) -> Self {
        Self { runtime }
    }

    /// Transcribe one complete audio file.
    pub async fn create(
        &self,
        request: CohereTranscriptionRequest,
        options: CallOptions,
    ) -> Result<TranscriptionResponse, Error> {
        let options = options.resolve_deadline().map_err(Error::from)?;
        request
            .validate()
            .map_err(|error| self.contextualize(error))?;
        let response = self
            .runtime
            .transport
            .execute(
                transcription_plan(&request).map_err(|error| self.contextualize(error))?,
                options,
            )
            .await
            .map_err(|error| self.contextualize(error))?;
        if !response.status().is_success() {
            return Err(self.contextualize(provider_status_error(response)));
        }
        let request_id = response_request_id(response.headers());
        let decoded = serde_json::from_slice::<TranscriptionWireResponse>(response.body())
            .map_err(|source| {
                self.contextualize(
                    Error::new(
                        ErrorKind::Protocol,
                        "Cohere returned an invalid audio transcription response",
                    )
                    .with_source(source),
                )
            })?;
        let result = TranscriptionResponse {
            text: decoded.text,
            language: None,
            confidence: None,
            duration_seconds: None,
            segments: Vec::new(),
            metadata: ResponseMetadata {
                response_id: None,
                request_id,
                model: None,
            },
            usage: Usage::default(),
            warnings: Vec::new(),
            provider: Default::default(),
        };
        result
            .validate()
            .map_err(|error| self.contextualize(error))?;
        Ok(result)
    }

    fn contextualize(&self, error: Error) -> Error {
        error.with_context(ErrorContext {
            operation: Some(ModelOperation::Transcribe),
            provider: Some(self.runtime.scope.provider_id().clone()),
            route: None,
            model: None,
        })
    }
}

impl fmt::Debug for CohereTranscriptions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CohereTranscriptions")
            .field("scope", &self.runtime.scope)
            .field("transport", &"shared")
            .finish()
    }
}

#[derive(Deserialize)]
struct TranscriptionWireResponse {
    text: String,
}

fn transcription_plan(request: &CohereTranscriptionRequest) -> Result<RequestPlan, Error> {
    let content_type = HeaderValue::from_str(request.media_type()).map_err(|source| {
        invalid_input("Cohere transcription media type cannot be encoded as multipart metadata")
            .with_source(source)
    })?;
    let mut parts = vec![
        MultipartPart::file(
            "file",
            audio_file_name(request.media_type()).expect("validated media type"),
            content_type,
            request.audio().clone(),
        )
        .map_err(request_build_error)?,
        MultipartPart::field("language", request.language().as_bytes().to_vec())
            .map_err(request_build_error)?,
    ];
    if let Some(temperature) = request.temperature() {
        parts.push(
            MultipartPart::field("temperature", temperature.to_string())
                .map_err(request_build_error)?,
        );
    }
    let headers = RequestHeaders::new()
        .try_insert(ACCEPT, HeaderValue::from_static("application/json"))
        .map_err(request_build_error)?;
    RequestPlan::new(
        Method::POST,
        RequestTarget::new(TRANSCRIPTION_TARGET).map_err(request_build_error)?,
    )
    .with_headers(headers)
    .with_body(RequestBody::multipart(MultipartBody::new(parts)))
    .with_replay_safety(ReplaySafety::Never)
    .map_err(request_build_error)
}

fn audio_file_name(media_type: &str) -> Option<&'static str> {
    match media_type.to_ascii_lowercase().as_str() {
        "audio/mpeg" | "audio/mp3" => Some("audio.mp3"),
        "audio/wav" | "audio/x-wav" => Some("audio.wav"),
        "audio/flac" | "audio/x-flac" => Some("audio.flac"),
        "audio/aac" => Some("audio.aac"),
        "audio/mp4" | "audio/x-m4a" => Some("audio.m4a"),
        "audio/ogg" => Some("audio.ogg"),
        _ => None,
    }
}

fn invalid_input(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

fn request_build_error(source: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "Cohere audio transcription request could not be encoded",
    )
    .with_source(source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use siumai_core::ErrorKind;

    #[test]
    fn request_validation_is_bounded_and_debug_is_redacted() {
        let request = CohereTranscriptionRequest::new(
            Bytes::from_static(b"canary-audio"),
            "audio/wav",
            "secret-language",
        )
        .unwrap()
        .with_temperature(0.25)
        .unwrap();
        let debug = format!("{request:?}");
        assert!(!debug.contains("canary-audio"));
        assert!(!debug.contains("secret-language"));
        assert!(debug.contains("audio_bytes"));

        assert_eq!(
            CohereTranscriptionRequest::new(vec![1_u8], "video/mp4", "en")
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidInput
        );
        assert_eq!(
            CohereTranscriptionRequest::new(vec![1_u8], "audio/wav", "en")
                .unwrap()
                .with_temperature(1.1)
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidInput
        );
    }

    #[test]
    fn portable_request_conversion_requires_language_and_rejects_prompt() {
        let missing_language = TranscriptionRequest::new(vec![1_u8], "audio/wav").unwrap();
        assert_eq!(
            CohereTranscriptionRequest::try_from(missing_language)
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidInput
        );

        let prompt = TranscriptionRequest::new(vec![1_u8], "audio/wav")
            .unwrap()
            .with_language("en")
            .unwrap()
            .with_prompt("speaker names")
            .unwrap();
        assert_eq!(
            CohereTranscriptionRequest::try_from(prompt)
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidInput
        );
    }
}
