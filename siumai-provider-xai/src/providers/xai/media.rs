//! Portable xAI image, speech, and final transcription adapters.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine as _;
use http::header::{ACCEPT, HeaderValue};
use http::{Method, StatusCode};
use serde::Deserialize;
use serde_json::Value;
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, ImageArtifact, ImageLimits, ImageModel,
    ImageRequest, ImageResponse, MediaData, Model, ModelDescriptor, ModelFamily, ModelId,
    ModelOperation, ModelPolicy, ModelPolicyContext, ModelPolicyDecision, PublicDiagnosticText,
    ResponseDiagnostics, ResponseMetadata, SensitiveResponse, SpeechLimits, SpeechModel,
    SpeechRequest, SpeechResponse, SupportState, TranscriptSegment, TranscriptionLimits,
    TranscriptionModel, TranscriptionRequest, TranscriptionResponse, UnsupportedReason, Usage,
};
use siumai_transport::{
    MultipartBody, MultipartPart, ProviderTransport, ReplaySafety, RequestBody, RequestBuildError,
    RequestHeaders, RequestPlan, RequestTarget, ResponseHeaders, TransportResponse,
};

use super::models;

pub const IMAGE_PROTOCOL_ID: &str = "xai-images";
pub const IMAGE_API_MODE_ID: &str = "image-generations";
pub const IMAGE_SOURCE: &str =
    "https://docs.x.ai/developers/rest-api-reference/images/generate-image";
pub const SPEECH_PROTOCOL_ID: &str = "xai-tts";
pub const SPEECH_API_MODE_ID: &str = "tts";
pub const SPEECH_SOURCE: &str = "https://docs.x.ai/developers/model-capabilities/audio/tts";
pub const TRANSCRIPTION_PROTOCOL_ID: &str = "xai-stt";
pub const TRANSCRIPTION_API_MODE_ID: &str = "stt";
pub const TRANSCRIPTION_SOURCE: &str = "https://docs.x.ai/developers/model-capabilities/audio/stt";
pub const MEDIA_VERIFIED_ON: &str = "2026-08-09";

const IMAGE_TARGET: &str = "images/generations";
const SPEECH_TARGET: &str = "tts";
const TRANSCRIPTION_TARGET: &str = "stt";
const MAX_IMAGES_PER_CALL: u32 = 10;
const MAX_SPEECH_TEXT_CHARACTERS: usize = 15_000;
const MIN_SPEECH_SPEED: f32 = 0.7;
const MAX_SPEECH_SPEED: f32 = 1.5;

pub(crate) struct XaiImageRuntime {
    pub(crate) scope: Arc<siumai_core::ProviderScope>,
    pub(crate) transport: ProviderTransport,
    pub(crate) policy: Arc<XaiImagePolicy>,
}

impl XaiImageRuntime {
    pub(crate) fn new(
        scope: Arc<siumai_core::ProviderScope>,
        transport: ProviderTransport,
        verified_endpoint: bool,
    ) -> Self {
        Self {
            policy: Arc::new(XaiImagePolicy {
                expected_scope: scope.clone(),
                verified_endpoint,
            }),
            scope,
            transport,
        }
    }
}

pub(crate) struct XaiSpeechRuntime {
    pub(crate) scope: Arc<siumai_core::ProviderScope>,
    pub(crate) transport: ProviderTransport,
    pub(crate) policy: Arc<XaiSpeechPolicy>,
}

impl XaiSpeechRuntime {
    pub(crate) fn new(
        scope: Arc<siumai_core::ProviderScope>,
        transport: ProviderTransport,
        verified_endpoint: bool,
    ) -> Self {
        Self {
            policy: Arc::new(XaiSpeechPolicy {
                expected_scope: scope.clone(),
                verified_endpoint,
            }),
            scope,
            transport,
        }
    }
}

pub(crate) struct XaiTranscriptionRuntime {
    pub(crate) scope: Arc<siumai_core::ProviderScope>,
    pub(crate) transport: ProviderTransport,
    pub(crate) policy: Arc<XaiTranscriptionPolicy>,
}

impl XaiTranscriptionRuntime {
    pub(crate) fn new(
        scope: Arc<siumai_core::ProviderScope>,
        transport: ProviderTransport,
        verified_endpoint: bool,
    ) -> Self {
        Self {
            policy: Arc::new(XaiTranscriptionPolicy {
                expected_scope: scope.clone(),
                verified_endpoint,
            }),
            scope,
            transport,
        }
    }
}

macro_rules! runtime_debug {
    ($runtime:ty, $name:literal) => {
        impl fmt::Debug for $runtime {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter
                    .debug_struct($name)
                    .field("scope", &self.scope)
                    .field("transport", &"shared")
                    .finish()
            }
        }
    };
}

runtime_debug!(XaiImageRuntime, "XaiImageRuntime");
runtime_debug!(XaiSpeechRuntime, "XaiSpeechRuntime");
runtime_debug!(XaiTranscriptionRuntime, "XaiTranscriptionRuntime");

pub(crate) struct XaiImagePolicy {
    expected_scope: Arc<siumai_core::ProviderScope>,
    verified_endpoint: bool,
}

pub(crate) struct XaiSpeechPolicy {
    expected_scope: Arc<siumai_core::ProviderScope>,
    verified_endpoint: bool,
}

pub(crate) struct XaiTranscriptionPolicy {
    expected_scope: Arc<siumai_core::ProviderScope>,
    verified_endpoint: bool,
}

impl ModelPolicy for XaiImagePolicy {
    fn evaluate(&self, context: &ModelPolicyContext) -> ModelPolicyDecision {
        evaluate_policy(
            context,
            &self.expected_scope,
            self.verified_endpoint,
            ModelOperation::GenerateImage,
            |model| models::image::HINTS.contains(&model),
            false,
        )
    }
}

impl ModelPolicy for XaiSpeechPolicy {
    fn evaluate(&self, context: &ModelPolicyContext) -> ModelPolicyDecision {
        evaluate_policy(
            context,
            &self.expected_scope,
            self.verified_endpoint,
            ModelOperation::SynthesizeSpeech,
            |model| model == models::speech::TTS,
            true,
        )
    }
}

impl ModelPolicy for XaiTranscriptionPolicy {
    fn evaluate(&self, context: &ModelPolicyContext) -> ModelPolicyDecision {
        evaluate_policy(
            context,
            &self.expected_scope,
            self.verified_endpoint,
            ModelOperation::Transcribe,
            |model| model == models::transcription::STT,
            true,
        )
    }
}

fn evaluate_policy(
    context: &ModelPolicyContext,
    expected_scope: &siumai_core::ProviderScope,
    verified_endpoint: bool,
    operation: ModelOperation,
    known: impl Fn(&str) -> bool,
    fixed_endpoint: bool,
) -> ModelPolicyDecision {
    if context.scope() != expected_scope {
        return ModelPolicyDecision::unsupported(UnsupportedReason::ApiModeMismatch);
    }
    if context.operation() != operation {
        return ModelPolicyDecision::unsupported(UnsupportedReason::OperationNotImplemented);
    }
    if fixed_endpoint && !known(context.model().as_str()) {
        return ModelPolicyDecision::unsupported(UnsupportedReason::ProviderRestriction);
    }
    if verified_endpoint && known(context.model().as_str()) {
        ModelPolicyDecision::supported()
    } else {
        ModelPolicyDecision::unknown_model()
    }
}

/// Lightweight text-to-image model over xAI Images.
#[derive(Clone)]
pub struct XaiImageModel {
    pub(crate) runtime: Arc<XaiImageRuntime>,
    descriptor: ModelDescriptor,
}

impl XaiImageModel {
    pub(crate) fn new(runtime: Arc<XaiImageRuntime>, model: ModelId) -> Self {
        let descriptor =
            ModelDescriptor::from_scope(runtime.scope.clone(), model, ModelFamily::Image);
        Self {
            runtime,
            descriptor,
        }
    }

    fn plan(&self, request: &ImageRequest) -> Result<RequestPlan, Error> {
        if request.size().is_some() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "xAI Images exposes aspect ratio rather than exact portable dimensions",
            ));
        }
        if request.format().is_some() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "xAI Images does not expose portable output-format selection",
            ));
        }
        let body = serde_json::json!({
            "model": self.model_id().as_str(),
            "prompt": request.prompt(),
            "n": request.count(),
            "response_format": "url"
        });
        RequestPlan::new(
            Method::POST,
            RequestTarget::new(IMAGE_TARGET).map_err(request_build_error)?,
        )
        .with_body(RequestBody::json(&body).map_err(request_build_error)?)
        .with_replay_safety(ReplaySafety::Never)
        .map_err(request_build_error)
    }

    fn contextualize(&self, error: Error) -> Error {
        contextualize(error, ModelOperation::GenerateImage, self.descriptor())
    }
}

impl fmt::Debug for XaiImageModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("XaiImageModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for XaiImageModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl ImageModel for XaiImageModel {
    fn limits(&self) -> ImageLimits {
        ImageLimits {
            max_outputs_per_call: Some(MAX_IMAGES_PER_CALL),
        }
    }

    async fn generate_image(
        &self,
        request: ImageRequest,
        call: CallOptions,
    ) -> Result<ImageResponse, Error> {
        self.limits()
            .validate(&request)
            .map_err(|error| self.contextualize(error))?;
        require_supported(
            &*self.runtime.policy,
            self.descriptor(),
            ModelOperation::GenerateImage,
            "xAI model policy rejected image generation",
        )?;
        let plan = self
            .plan(&request)
            .map_err(|error| self.contextualize(error))?;
        let response = self
            .runtime
            .transport
            .execute(plan, call)
            .await
            .map_err(|error| self.contextualize(error))?;
        if !response.status().is_success() {
            return Err(self.contextualize(response_error("xAI image request failed", response)));
        }
        let decoded = decode_image_response(self.model_id(), response, request.count())
            .map_err(|error| self.contextualize(error))?;
        decoded
            .validate(&request)
            .map_err(|error| self.contextualize(error))?;
        Ok(decoded)
    }
}

/// Lightweight buffered speech model over xAI TTS.
#[derive(Clone)]
pub struct XaiSpeechModel {
    pub(crate) runtime: Arc<XaiSpeechRuntime>,
    descriptor: ModelDescriptor,
}

impl XaiSpeechModel {
    pub(crate) fn new(runtime: Arc<XaiSpeechRuntime>, model: ModelId) -> Self {
        let descriptor =
            ModelDescriptor::from_scope(runtime.scope.clone(), model, ModelFamily::Speech);
        Self {
            runtime,
            descriptor,
        }
    }

    fn plan(&self, request: &SpeechRequest) -> Result<(RequestPlan, SpeechFormat), Error> {
        let format = SpeechFormat::parse(request.format())?;
        let voice = request.voice().unwrap_or("eve");
        validate_text(voice, 128, "xAI speech voice is invalid")?;
        let language = request.language().unwrap_or("auto");
        validate_text(language, 64, "xAI speech language is invalid")?;
        let mut body = serde_json::Map::from_iter([
            (
                "text".to_string(),
                Value::String(request.text().to_string()),
            ),
            ("voice_id".to_string(), Value::String(voice.to_string())),
            ("language".to_string(), Value::String(language.to_string())),
            (
                "output_format".to_string(),
                serde_json::json!({"codec": format.codec()}),
            ),
        ]);
        if let Some(speed) = request.speed() {
            if !(MIN_SPEECH_SPEED..=MAX_SPEECH_SPEED).contains(&speed) {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "xAI speech speed must be between 0.7 and 1.5",
                ));
            }
            body.insert("speed".to_string(), Value::from(speed));
        }
        let headers = RequestHeaders::new()
            .try_insert(ACCEPT, HeaderValue::from_static("audio/*"))
            .map_err(request_build_error)?;
        let plan = RequestPlan::new(
            Method::POST,
            RequestTarget::new(SPEECH_TARGET).map_err(request_build_error)?,
        )
        .with_headers(headers)
        .with_body(RequestBody::json(&Value::Object(body)).map_err(request_build_error)?)
        .with_replay_safety(ReplaySafety::Never)
        .map_err(request_build_error)?;
        Ok((plan, format))
    }

    fn contextualize(&self, error: Error) -> Error {
        contextualize(error, ModelOperation::SynthesizeSpeech, self.descriptor())
    }
}

impl fmt::Debug for XaiSpeechModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("XaiSpeechModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for XaiSpeechModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl SpeechModel for XaiSpeechModel {
    fn limits(&self) -> SpeechLimits {
        SpeechLimits {
            max_text_bytes: None,
            max_text_chars: Some(MAX_SPEECH_TEXT_CHARACTERS),
        }
    }

    async fn synthesize(
        &self,
        request: SpeechRequest,
        call: CallOptions,
    ) -> Result<SpeechResponse, Error> {
        self.limits()
            .validate(&request)
            .map_err(|error| self.contextualize(error))?;
        require_supported(
            &*self.runtime.policy,
            self.descriptor(),
            ModelOperation::SynthesizeSpeech,
            "xAI speech uses a fixed endpoint handle",
        )?;
        let voice = request.voice().unwrap_or("eve").to_string();
        let (plan, format) = self
            .plan(&request)
            .map_err(|error| self.contextualize(error))?;
        let response = self
            .runtime
            .transport
            .execute(plan, call)
            .await
            .map_err(|error| self.contextualize(error))?;
        if !response.status().is_success() {
            return Err(self.contextualize(response_error("xAI speech request failed", response)));
        }
        let attempts = response.attempts();
        let (_, headers, audio) = response.into_parts();
        let mut provider = BTreeMap::new();
        provider.insert("voice".to_string(), Value::String(voice));
        provider.insert(
            "codec".to_string(),
            Value::String(format.codec().to_string()),
        );
        provider.insert("transport_attempts".to_string(), Value::from(attempts));
        let response = SpeechResponse {
            media_type: response_media_type(&headers)
                .unwrap_or(format.media_type())
                .to_string(),
            audio,
            duration_seconds: None,
            sample_rate_hz: None,
            metadata: ResponseMetadata {
                response_id: None,
                request_id: response_request_id(&headers),
                model: Some(self.model_id().clone()),
            },
            usage: Usage::default(),
            warnings: Vec::new(),
            provider,
        };
        response
            .validate()
            .map_err(|error| self.contextualize(error))?;
        Ok(response)
    }
}

/// Lightweight final-result transcription model over xAI STT.
#[derive(Clone)]
pub struct XaiTranscriptionModel {
    pub(crate) runtime: Arc<XaiTranscriptionRuntime>,
    descriptor: ModelDescriptor,
}

impl XaiTranscriptionModel {
    pub(crate) fn new(runtime: Arc<XaiTranscriptionRuntime>, model: ModelId) -> Self {
        let descriptor =
            ModelDescriptor::from_scope(runtime.scope.clone(), model, ModelFamily::Transcription);
        Self {
            runtime,
            descriptor,
        }
    }

    fn plan(&self, request: &TranscriptionRequest) -> Result<RequestPlan, Error> {
        if request.prompt().is_some() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "xAI STT does not expose portable transcription prompts",
            ));
        }
        let mut parts = Vec::new();
        if let Some(language) = request.language() {
            validate_text(language, 64, "xAI transcription language is invalid")?;
            parts.push(text_part("language", language)?);
        }
        let content_type = HeaderValue::from_str(request.media_type()).map_err(|source| {
            Error::new(
                ErrorKind::InvalidInput,
                "xAI transcription media type is not a valid multipart content type",
            )
            .with_source(source)
        })?;
        parts.push(
            MultipartPart::file(
                "file",
                audio_file_name(request.media_type()),
                content_type,
                request.audio().clone(),
            )
            .map_err(request_build_error)?,
        );
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

    fn contextualize(&self, error: Error) -> Error {
        contextualize(error, ModelOperation::Transcribe, self.descriptor())
    }
}

impl fmt::Debug for XaiTranscriptionModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("XaiTranscriptionModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for XaiTranscriptionModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl TranscriptionModel for XaiTranscriptionModel {
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
        self.limits()
            .validate(&request)
            .map_err(|error| self.contextualize(error))?;
        require_supported(
            &*self.runtime.policy,
            self.descriptor(),
            ModelOperation::Transcribe,
            "xAI transcription uses a fixed endpoint handle",
        )?;
        let plan = self
            .plan(&request)
            .map_err(|error| self.contextualize(error))?;
        let response = self
            .runtime
            .transport
            .execute(plan, call)
            .await
            .map_err(|error| self.contextualize(error))?;
        if !response.status().is_success() {
            return Err(
                self.contextualize(response_error("xAI transcription request failed", response))
            );
        }
        decode_transcription_response(self.model_id(), response)
            .map_err(|error| self.contextualize(error))
    }
}

fn require_supported(
    policy: &dyn ModelPolicy,
    descriptor: &ModelDescriptor,
    operation: ModelOperation,
    message: &'static str,
) -> Result<(), Error> {
    let decision = policy.evaluate(&ModelPolicyContext::new(
        Arc::new(descriptor.scope().clone()),
        descriptor.model().clone(),
        operation,
    ));
    if matches!(decision.state(), SupportState::Unsupported { .. }) {
        return Err(contextualize(
            Error::new(ErrorKind::Unsupported, message),
            operation,
            descriptor,
        ));
    }
    Ok(())
}

fn contextualize(error: Error, operation: ModelOperation, descriptor: &ModelDescriptor) -> Error {
    error.with_context(ErrorContext {
        operation: Some(operation),
        provider: Some(descriptor.provider().clone()),
        route: None,
        model: Some(descriptor.model().clone()),
    })
}

#[derive(Debug, Deserialize)]
struct ImageResponseWire {
    data: Vec<ImageItemWire>,
    #[serde(default)]
    usage: Option<ImageUsageWire>,
}

#[derive(Debug, Deserialize)]
struct ImageItemWire {
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    b64_json: Option<String>,
    #[serde(default)]
    revised_prompt: Option<String>,
    #[serde(default)]
    mime_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ImageUsageWire {
    #[serde(default)]
    cost_in_usd_ticks: Option<u64>,
}

fn decode_image_response(
    model: &ModelId,
    response: TransportResponse,
    expected: u32,
) -> Result<ImageResponse, Error> {
    let attempts = response.attempts();
    let (_, headers, body) = response.into_parts();
    let wire: ImageResponseWire = serde_json::from_slice(&body).map_err(|source| {
        Error::new(ErrorKind::Protocol, "xAI returned malformed image JSON").with_source(source)
    })?;
    let images = wire
        .data
        .into_iter()
        .map(|item| {
            let media_type = item
                .mime_type
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| {
                    Error::protocol_violation("xAI image response omitted the MIME type")
                })?;
            let data = match (item.url, item.b64_json) {
                (Some(url), _) if !url.trim().is_empty() => MediaData::Url(url),
                (_, Some(encoded)) => MediaData::Bytes(
                    base64::engine::general_purpose::STANDARD
                        .decode(encoded)
                        .map(bytes::Bytes::from)
                        .map_err(|source| {
                            Error::new(
                                ErrorKind::Protocol,
                                "xAI image response contains invalid base64",
                            )
                            .with_source(source)
                        })?,
                ),
                _ => {
                    return Err(Error::protocol_violation(
                        "xAI image response omitted both URL and base64 data",
                    ));
                }
            };
            Ok(ImageArtifact {
                media_type,
                data,
                revised_prompt: item.revised_prompt,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    if images.len() != usize::try_from(expected).unwrap_or(usize::MAX) {
        return Err(Error::protocol_violation(
            "xAI image response count does not match the request",
        ));
    }
    let mut provider = BTreeMap::new();
    provider.insert("transport_attempts".to_string(), Value::from(attempts));
    if let Some(cost) = wire.usage.and_then(|usage| usage.cost_in_usd_ticks) {
        provider.insert("cost_in_usd_ticks".to_string(), Value::from(cost));
    }
    Ok(ImageResponse {
        images,
        metadata: ResponseMetadata {
            response_id: None,
            request_id: response_request_id(&headers),
            model: Some(model.clone()),
        },
        usage: Usage::default(),
        warnings: Vec::new(),
        provider,
    })
}

#[derive(Debug, Deserialize)]
struct TranscriptionResponseWire {
    text: String,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    duration: Option<f64>,
    #[serde(default)]
    words: Vec<TranscriptionWordWire>,
}

#[derive(Debug, Deserialize)]
struct TranscriptionWordWire {
    text: String,
    start: f64,
    end: f64,
}

fn decode_transcription_response(
    model: &ModelId,
    response: TransportResponse,
) -> Result<TranscriptionResponse, Error> {
    let attempts = response.attempts();
    let (_, headers, body) = response.into_parts();
    let wire: TranscriptionResponseWire = serde_json::from_slice(&body).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "xAI returned malformed transcription JSON",
        )
        .with_source(source)
    })?;
    let segments = wire
        .words
        .into_iter()
        .map(|word| TranscriptSegment {
            start_seconds: word.start,
            end_seconds: word.end,
            text: word.text,
            confidence: None,
        })
        .collect();
    let mut provider = BTreeMap::new();
    provider.insert("transport_attempts".to_string(), Value::from(attempts));
    let response = TranscriptionResponse {
        text: wire.text,
        language: wire.language,
        confidence: None,
        duration_seconds: wire.duration,
        segments,
        metadata: ResponseMetadata {
            response_id: None,
            request_id: response_request_id(&headers),
            model: Some(model.clone()),
        },
        usage: Usage::default(),
        warnings: Vec::new(),
        provider,
    };
    response.validate()?;
    Ok(response)
}

#[derive(Clone, Copy)]
enum SpeechFormat {
    Mp3,
    Wav,
    Pcm,
    Mulaw,
    Alaw,
}

impl SpeechFormat {
    fn parse(value: Option<&str>) -> Result<Self, Error> {
        match value.map(str::to_ascii_lowercase).as_deref() {
            None | Some("mp3") | Some("audio/mpeg") => Ok(Self::Mp3),
            Some("wav") | Some("audio/wav") => Ok(Self::Wav),
            Some("pcm") | Some("audio/pcm") => Ok(Self::Pcm),
            Some("mulaw") | Some("mu-law") => Ok(Self::Mulaw),
            Some("alaw") | Some("a-law") => Ok(Self::Alaw),
            _ => Err(Error::new(
                ErrorKind::Unsupported,
                "xAI speech supports mp3, wav, pcm, mulaw, or alaw output",
            )),
        }
    }

    fn codec(self) -> &'static str {
        match self {
            Self::Mp3 => "mp3",
            Self::Wav => "wav",
            Self::Pcm => "pcm",
            Self::Mulaw => "mulaw",
            Self::Alaw => "alaw",
        }
    }

    fn media_type(self) -> &'static str {
        match self {
            Self::Mp3 => "audio/mpeg",
            Self::Wav => "audio/wav",
            Self::Pcm => "audio/L16",
            Self::Mulaw | Self::Alaw => "audio/basic",
        }
    }
}

fn text_part(name: &str, value: &str) -> Result<MultipartPart, Error> {
    MultipartPart::field(name, value.as_bytes().to_vec()).map_err(request_build_error)
}

fn audio_file_name(media_type: &str) -> &'static str {
    match media_type.to_ascii_lowercase().as_str() {
        "audio/mpeg" | "audio/mp3" => "audio.mp3",
        "audio/mp4" | "audio/x-m4a" => "audio.m4a",
        "audio/ogg" => "audio.ogg",
        "audio/flac" => "audio.flac",
        "audio/webm" => "audio.webm",
        _ => "audio.wav",
    }
}

fn validate_text(value: &str, maximum: usize, message: &'static str) -> Result<(), Error> {
    if value.trim().is_empty() || value.len() > maximum || value.chars().any(char::is_control) {
        return Err(Error::new(ErrorKind::InvalidInput, message));
    }
    Ok(())
}

fn response_error(message: &'static str, response: TransportResponse) -> Error {
    let (status, headers, body) = response.into_parts();
    let kind = match status {
        StatusCode::UNAUTHORIZED => ErrorKind::Authentication,
        StatusCode::FORBIDDEN => ErrorKind::Authorization,
        StatusCode::PAYMENT_REQUIRED => ErrorKind::QuotaExceeded,
        StatusCode::TOO_MANY_REQUESTS => ErrorKind::RateLimited,
        StatusCode::REQUEST_TIMEOUT | StatusCode::GATEWAY_TIMEOUT => ErrorKind::Timeout,
        StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY => ErrorKind::InvalidInput,
        _ => ErrorKind::Provider,
    };
    let mut diagnostics = ResponseDiagnostics::default().with_status(status.as_u16());
    if let Some(request_id) = response_request_id(&headers)
        && let Ok(request_id) = PublicDiagnosticText::new(request_id)
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
    Error::new(kind, message)
        .with_diagnostics(diagnostics)
        .with_sensitive_response(SensitiveResponse::new(raw_headers, body.to_vec()))
}

fn response_request_id(headers: &ResponseHeaders) -> Option<String> {
    ["x-request-id", "request-id"].into_iter().find_map(|name| {
        headers
            .get(&http::header::HeaderName::from_static(name))
            .and_then(|value| value.to_str().ok())
            .and_then(|value| PublicDiagnosticText::new(value.to_owned()).ok())
            .map(|value| value.as_str().to_owned())
    })
}

fn response_media_type(headers: &ResponseHeaders) -> Option<&str> {
    headers
        .get(&http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn request_build_error(source: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "xAI media request violates the transport contract",
    )
    .with_source(source)
}

#[cfg(test)]
mod tests {
    use mockito::Matcher;
    use siumai_core::{ImageModel as _, SpeechModel as _, TranscriptionModel as _, UsageValue};
    use siumai_transport::EndpointConfig;

    use super::*;
    use crate::{XaiCredential, XaiProvider};

    fn provider(server: &mockito::ServerGuard) -> XaiProvider {
        XaiProvider::builder(XaiCredential::unauthenticated())
            .with_endpoint(EndpointConfig::local_explicit(format!("{}/v1", server.url())).unwrap())
            .with_replay_domain(siumai_core::ReplayDomain::custom(
                siumai_core::ReplayDomainId::new("test-xai-media").unwrap(),
            ))
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn portable_image_speech_and_transcription_use_native_endpoints() {
        let mut server = mockito::Server::new_async().await;
        let image = server
            .mock("POST", "/v1/images/generations")
            .match_body(Matcher::Json(serde_json::json!({
                "model": models::image::GROK_IMAGINE_IMAGE_QUALITY,
                "prompt": "a rust crab",
                "n": 1,
                "response_format": "url"
            })))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "data":[{
                        "url":"https://cdn.example.com/image.png",
                        "mime_type":"image/png",
                        "revised_prompt":"a polished rust crab"
                    }],
                    "usage":{"cost_in_usd_ticks":12}
                })
                .to_string(),
            )
            .create_async()
            .await;
        let speech = server
            .mock("POST", "/v1/tts")
            .match_body(Matcher::Json(serde_json::json!({
                "text":"hello",
                "voice_id":"eve",
                "language":"auto",
                "output_format":{"codec":"mp3"}
            })))
            .with_status(200)
            .with_header("content-type", "audio/mpeg")
            .with_body(*b"ID3")
            .create_async()
            .await;
        let transcription = server
            .mock("POST", "/v1/stt")
            .match_body(Matcher::Regex(
                "name=\"language\"[\\s\\S]*english[\\s\\S]*name=\"file\"".to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "text":"hello",
                    "language":"english",
                    "duration":0.5,
                    "words":[{"text":"hello","start":0.0,"end":0.5}]
                })
                .to_string(),
            )
            .create_async()
            .await;
        let provider = provider(&server);

        let image_response = provider
            .image(models::image::GROK_IMAGINE_IMAGE_QUALITY)
            .unwrap()
            .generate_image(
                ImageRequest::new("a rust crab").unwrap(),
                CallOptions::default(),
            )
            .await
            .unwrap();
        let speech_response = provider
            .speech()
            .synthesize(SpeechRequest::new("hello").unwrap(), CallOptions::default())
            .await
            .unwrap();
        let transcription_response = provider
            .transcription()
            .transcribe(
                TranscriptionRequest::new(bytes::Bytes::from_static(b"RIFF"), "audio/wav")
                    .unwrap()
                    .with_language("english")
                    .unwrap(),
                CallOptions::default(),
            )
            .await
            .unwrap();

        image.assert_async().await;
        speech.assert_async().await;
        transcription.assert_async().await;
        assert!(matches!(image_response.images[0].data, MediaData::Url(_)));
        assert_eq!(speech_response.audio.as_ref(), b"ID3");
        assert_eq!(transcription_response.segments.len(), 1);
        assert_eq!(image_response.usage.input_tokens, UsageValue::Unknown);
        assert_eq!(speech_response.usage.input_tokens, UsageValue::Unknown);
        assert_eq!(
            transcription_response.usage.input_tokens,
            UsageValue::Unknown
        );

        let speech = provider.speech();
        assert!(
            speech
                .plan(
                    &SpeechRequest::new("hello")
                        .unwrap()
                        .with_speed(1.6)
                        .unwrap()
                )
                .is_err()
        );
        assert!(
            speech
                .limits()
                .validate(&SpeechRequest::new("x".repeat(15_001)).unwrap())
                .is_err()
        );
    }
}
