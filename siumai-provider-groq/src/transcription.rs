//! Groq final-result audio transcription model.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use http::header::{ACCEPT, HeaderName, HeaderValue};
use http::{Method, StatusCode};
use serde::Deserialize;
use serde_json::{Map, Value};
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, Model, ModelDescriptor, ModelFamily, ModelId,
    ModelOperation, ProviderInstanceId, ProviderOptionError, ProviderOptionSelection,
    ProviderOptions, ProviderScope, PublicDiagnosticText, ResponseMetadata, SensitiveResponse,
    TranscriptSegment, TranscriptionLimits, TranscriptionModel, TranscriptionRequest,
    TranscriptionResponse, TypedProviderOptions, Usage,
};
use siumai_transport::{
    MultipartBody, MultipartPart, ProviderTransport, ReplaySafety, RequestBody, RequestBuildError,
    RequestHeaders, RequestPlan, RequestTarget, ResponseHeaders, TransportResponse,
};

use crate::options::{
    GroqTimestampGranularity, GroqTranscriptionOptions, GroqTranscriptionResponseFormat,
};

pub const TRANSCRIPTION_PROTOCOL_ID: &str = "groq-audio-transcriptions";
pub const TRANSCRIPTION_API_MODE_ID: &str = "audio-transcriptions";
pub const TRANSCRIPTION_SOURCE: &str = "https://console.groq.com/docs/speech-to-text";

const TRANSCRIPTION_TARGET: &str = "audio/transcriptions";

/// Lightweight Groq transcription model over one shared provider runtime.
#[derive(Clone)]
pub struct GroqTranscriptionModel {
    pub(crate) runtime: Arc<GroqTranscriptionRuntime>,
    descriptor: ModelDescriptor,
}

impl GroqTranscriptionModel {
    pub(crate) fn new(runtime: Arc<GroqTranscriptionRuntime>, model: ModelId) -> Self {
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
        options: &GroqTranscriptionOptions,
    ) -> Result<RequestPlan, Error> {
        let content_type = HeaderValue::from_str(request.media_type()).map_err(|source| {
            Error::new(
                ErrorKind::InvalidInput,
                "transcription media type cannot be encoded as a multipart content type",
            )
            .with_source(source)
        })?;
        let mut parts = vec![
            MultipartPart::file(
                "file",
                audio_file_name(request.media_type()),
                content_type,
                request.audio().clone(),
            )
            .map_err(request_build_error)?,
            text_part("model", self.model_id().as_str())?,
            text_part(
                "response_format",
                response_format(options.response_format.unwrap_or_default()),
            )?,
        ];
        if let Some(language) = request.language() {
            parts.push(text_part("language", language)?);
        }
        if let Some(prompt) = request.prompt() {
            parts.push(text_part("prompt", prompt)?);
        }
        if let Some(temperature) = options.temperature {
            parts.push(text_part("temperature", temperature.to_string())?);
        }
        for granularity in &options.timestamp_granularities {
            parts.push(text_part(
                "timestamp_granularities[]",
                timestamp_granularity(*granularity),
            )?);
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

    fn contextualize(&self, error: Error) -> Error {
        error.with_context(ErrorContext {
            operation: Some(ModelOperation::Transcribe),
            provider: Some(self.provider_id().clone()),
            route: None,
            model: Some(self.model_id().clone()),
        })
    }
}

impl fmt::Debug for GroqTranscriptionModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GroqTranscriptionModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for GroqTranscriptionModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl TranscriptionModel for GroqTranscriptionModel {
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
        let options = self
            .runtime
            .options(self, &call)
            .map_err(option_error)
            .map_err(|error| self.contextualize(error))?;
        let response_format = options.response_format.unwrap_or_default();
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
            return Err(self.contextualize(provider_response_error(response)));
        }
        decode_response(self.model_id(), response, response_format)
            .map_err(|error| self.contextualize(error))
    }
}

pub(crate) struct GroqTranscriptionRuntime {
    pub(crate) instance_id: ProviderInstanceId,
    pub(crate) scope: Arc<ProviderScope>,
    pub(crate) transport: ProviderTransport,
    default_options: ProviderOptions,
}

impl GroqTranscriptionRuntime {
    pub(crate) fn new(
        instance_id: ProviderInstanceId,
        scope: Arc<ProviderScope>,
        transport: ProviderTransport,
        default_options: ProviderOptions,
    ) -> Self {
        Self {
            instance_id,
            scope,
            transport,
            default_options,
        }
    }

    fn options<M: Model + ?Sized>(
        &self,
        model: &M,
        call: &CallOptions,
    ) -> Result<GroqTranscriptionOptions, ProviderOptionError> {
        let selection = call.provider_options_for(model)?;
        merge_options(&self.default_options, &selection)
    }
}

impl fmt::Debug for GroqTranscriptionRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GroqTranscriptionRuntime")
            .field("scope", &self.scope)
            .field("transport", &"shared")
            .field("default_options", &self.default_options)
            .finish()
    }
}

fn merge_options(
    defaults: &ProviderOptions,
    selection: &ProviderOptionSelection<'_>,
) -> Result<GroqTranscriptionOptions, ProviderOptionError> {
    let mut merged = defaults.value().clone();
    for options in selection.typed() {
        decode_options(options.value())?.validate()?;
        merged.extend(options.value().clone());
    }
    if selection.raw_override().is_some() {
        return Err(ProviderOptionError::Rejected {
            path: "$".to_string(),
            reason: "Groq transcription only accepts typed provider options".to_string(),
        });
    }
    let options = decode_options(&merged)?;
    options.validate()?;
    Ok(options)
}

fn decode_options(
    options: &Map<String, Value>,
) -> Result<GroqTranscriptionOptions, ProviderOptionError> {
    serde_json::from_value(Value::Object(options.clone())).map_err(|error| {
        ProviderOptionError::Rejected {
            path: "$".to_string(),
            reason: error.to_string(),
        }
    })
}

fn text_part(name: &str, value: impl AsRef<str>) -> Result<MultipartPart, Error> {
    MultipartPart::field(name, value.as_ref().as_bytes().to_vec()).map_err(request_build_error)
}

fn response_format(format: GroqTranscriptionResponseFormat) -> &'static str {
    match format {
        GroqTranscriptionResponseFormat::Json => "json",
        GroqTranscriptionResponseFormat::VerboseJson => "verbose_json",
        GroqTranscriptionResponseFormat::Text => "text",
    }
}

fn timestamp_granularity(granularity: GroqTimestampGranularity) -> &'static str {
    match granularity {
        GroqTimestampGranularity::Segment => "segment",
        GroqTimestampGranularity::Word => "word",
    }
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

fn decode_response(
    requested_model: &ModelId,
    response: TransportResponse,
    format: GroqTranscriptionResponseFormat,
) -> Result<TranscriptionResponse, Error> {
    let (_, headers, body) = response.into_parts();
    decode_response_parts(requested_model, Some(&headers), &body, format)
}

fn decode_response_parts(
    requested_model: &ModelId,
    headers: Option<&ResponseHeaders>,
    body: &[u8],
    format: GroqTranscriptionResponseFormat,
) -> Result<TranscriptionResponse, Error> {
    if format == GroqTranscriptionResponseFormat::Text {
        return decode_text_response(requested_model, headers, body);
    }
    let wire = serde_json::from_slice::<GroqTranscriptionWire>(body).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "Groq returned malformed transcription JSON",
        )
        .with_source(source)
        .with_sensitive_response(SensitiveResponse::new(BTreeMap::new(), body.to_vec()))
    })?;
    let request_id = checked_identifier(
        wire.x_groq
            .as_ref()
            .and_then(|value| value.get("id"))
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| headers.and_then(response_request_id)),
    )?;
    let mut segments = wire
        .segments
        .into_iter()
        .map(|segment| TranscriptSegment {
            start_seconds: segment.start,
            end_seconds: segment.end,
            text: segment.text,
            confidence: segment.confidence,
        })
        .collect::<Vec<_>>();
    if segments.is_empty() {
        segments.extend(wire.words.into_iter().map(|word| TranscriptSegment {
            start_seconds: word.start,
            end_seconds: word.end,
            text: word.word,
            confidence: word.confidence,
        }));
    }
    let raw_usage = wire.usage.or_else(|| {
        wire.x_groq
            .as_ref()
            .and_then(|value| value.get("usage"))
            .cloned()
    });
    let duration_seconds = wire.duration.or_else(|| {
        raw_usage
            .as_ref()
            .and_then(|value| value.get("total_time"))
            .and_then(Value::as_f64)
    });
    let mut usage = Usage::default();
    if let Some(raw_usage) = &raw_usage {
        usage.provider.insert("groq".to_string(), raw_usage.clone());
    }
    if let Some(duration) = duration_seconds {
        usage
            .provider
            .insert("audio_duration_seconds".to_string(), Value::from(duration));
    }
    let mut metadata = Map::new();
    if let Some(request_id) = &request_id {
        metadata.insert("requestId".to_string(), Value::String(request_id.clone()));
    }
    if let Some(task) = wire.task {
        metadata.insert("task".to_string(), Value::String(task));
    }
    if let Some(x_groq) = wire.x_groq {
        metadata.insert("xGroq".to_string(), x_groq);
    }
    if let Some(raw_usage) = raw_usage {
        metadata.insert("rawUsage".to_string(), raw_usage);
    }
    let provider = if metadata.is_empty() {
        BTreeMap::new()
    } else {
        BTreeMap::from([("groq".to_string(), Value::Object(metadata))])
    };
    let result = TranscriptionResponse {
        text: wire.text,
        language: wire.language,
        confidence: wire.confidence,
        duration_seconds,
        segments,
        metadata: ResponseMetadata {
            response_id: None,
            request_id,
            model: Some(requested_model.clone()),
        },
        usage,
        warnings: Vec::new(),
        provider,
    };
    result.validate()?;
    Ok(result)
}

fn decode_text_response(
    requested_model: &ModelId,
    headers: Option<&ResponseHeaders>,
    body: &[u8],
) -> Result<TranscriptionResponse, Error> {
    let text = String::from_utf8(body.to_vec()).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "Groq returned non-UTF-8 transcription text",
        )
        .with_source(source)
        .with_sensitive_response(SensitiveResponse::new(BTreeMap::new(), body.to_vec()))
    })?;
    let request_id = checked_identifier(headers.and_then(response_request_id))?;
    let provider = request_id
        .as_ref()
        .map_or_else(BTreeMap::new, |request_id| {
            BTreeMap::from([(
                "groq".to_string(),
                serde_json::json!({"requestId": request_id}),
            )])
        });
    let result = TranscriptionResponse {
        text,
        language: None,
        confidence: None,
        duration_seconds: None,
        segments: Vec::new(),
        metadata: ResponseMetadata {
            response_id: None,
            request_id,
            model: Some(requested_model.clone()),
        },
        usage: Usage::default(),
        warnings: Vec::new(),
        provider,
    };
    result.validate()?;
    Ok(result)
}

#[derive(Debug, Deserialize)]
struct GroqTranscriptionWire {
    text: String,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    confidence: Option<f64>,
    #[serde(default)]
    duration: Option<f64>,
    #[serde(default)]
    task: Option<String>,
    #[serde(default)]
    segments: Vec<GroqTranscriptSegment>,
    #[serde(default)]
    words: Vec<GroqTranscriptWord>,
    #[serde(default)]
    usage: Option<Value>,
    #[serde(default)]
    x_groq: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct GroqTranscriptSegment {
    start: f64,
    end: f64,
    text: String,
    #[serde(default)]
    confidence: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct GroqTranscriptWord {
    word: String,
    start: f64,
    end: f64,
    #[serde(default)]
    confidence: Option<f64>,
}

fn option_error(source: ProviderOptionError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "Groq transcription options are invalid",
    )
    .with_source(source)
}

fn request_build_error(source: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "Groq transcription request violates the transport contract",
    )
    .with_source(source)
}

fn provider_response_error(response: TransportResponse) -> Error {
    let (status, headers, body) = response.into_parts();
    let parsed = serde_json::from_slice::<GroqErrorWire>(&body).ok();
    let request_id = parsed
        .as_ref()
        .and_then(|error| error.request_id.clone())
        .or_else(|| response_request_id(&headers));
    let provider_code = parsed
        .as_ref()
        .and_then(|error| error.error.as_ref())
        .and_then(|error| error.code.as_ref())
        .and_then(value_text);
    let provider_type = parsed
        .as_ref()
        .and_then(|error| error.error.as_ref())
        .and_then(|error| error.kind.clone());
    let mut diagnostics = headers.diagnostics().with_status(status.as_u16());
    if let Some(value) = request_id.and_then(public_text) {
        diagnostics = diagnostics.with_request_id(value);
    }
    if let Some(value) = provider_code.and_then(public_text) {
        diagnostics = diagnostics.with_provider_code(value);
    }
    if let Some(value) = provider_type.and_then(public_text) {
        diagnostics = diagnostics.with_provider_type(value);
    }
    let raw_headers = headers
        .expose()
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.to_string(), value.to_string()))
        })
        .collect();
    Error::new(status_error_kind(status), "Groq rejected transcription")
        .with_diagnostics(diagnostics)
        .with_sensitive_response(SensitiveResponse::new(raw_headers, body.to_vec()))
}

#[derive(Debug, Deserialize)]
struct GroqErrorWire {
    #[serde(default)]
    error: Option<GroqNestedError>,
    #[serde(default)]
    request_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GroqNestedError {
    #[serde(default)]
    code: Option<Value>,
    #[serde(default, rename = "type")]
    kind: Option<String>,
}

fn value_text(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        _ => None,
    }
}

fn public_text(value: String) -> Option<PublicDiagnosticText> {
    PublicDiagnosticText::new(value).ok()
}

fn checked_identifier(value: Option<String>) -> Result<Option<String>, Error> {
    value
        .map(|value| {
            PublicDiagnosticText::new(value.clone())
                .map(|_| value)
                .map_err(|source| {
                    Error::protocol_violation(
                        "Groq response contains an invalid request identifier",
                    )
                    .with_source(source)
                })
        })
        .transpose()
}

fn status_error_kind(status: StatusCode) -> ErrorKind {
    match status {
        StatusCode::UNAUTHORIZED => ErrorKind::Authentication,
        StatusCode::FORBIDDEN => ErrorKind::Authorization,
        StatusCode::PAYMENT_REQUIRED => ErrorKind::QuotaExceeded,
        StatusCode::TOO_MANY_REQUESTS => ErrorKind::RateLimited,
        StatusCode::REQUEST_TIMEOUT | StatusCode::GATEWAY_TIMEOUT => ErrorKind::Timeout,
        _ => ErrorKind::Provider,
    }
}

fn response_request_id(headers: &ResponseHeaders) -> Option<String> {
    ["x-request-id", "request-id"].into_iter().find_map(|name| {
        let name = HeaderName::from_static(name);
        headers
            .get(&name)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned)
    })
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use siumai_core::{ApiModeId, ProtocolId, ProviderId};

    use super::*;

    #[test]
    fn transcription_plan_is_multipart_and_never_replayable() {
        let endpoint =
            siumai_transport::EndpointConfig::public_custom("https://example.com/v1").unwrap();
        let transport = siumai_transport::ProviderTransport::builder(endpoint)
            .build()
            .unwrap();
        let scope = Arc::new(
            ProviderScope::new(ProviderId::new("groq").unwrap())
                .with_protocol(ProtocolId::new(TRANSCRIPTION_PROTOCOL_ID).unwrap())
                .with_api_mode(ApiModeId::new(TRANSCRIPTION_API_MODE_ID).unwrap()),
        );
        let defaults = ProviderOptions::typed(&GroqTranscriptionOptions::new()).unwrap();
        let model = GroqTranscriptionModel::new(
            Arc::new(GroqTranscriptionRuntime::new(
                ProviderInstanceId::new(),
                scope,
                transport,
                defaults,
            )),
            ModelId::new("future-whisper").unwrap(),
        );
        let request = TranscriptionRequest::new(Bytes::from_static(b"audio"), "audio/wav").unwrap();
        let plan = model
            .plan(&request, &GroqTranscriptionOptions::new())
            .unwrap();

        assert_eq!(plan.target().as_str(), TRANSCRIPTION_TARGET);
        assert!(matches!(plan.body(), RequestBody::Multipart(_)));
        assert_eq!(plan.replay_safety(), &ReplaySafety::Never);
    }

    #[test]
    fn response_decoder_preserves_segments_usage_and_metadata() {
        let body = serde_json::to_vec(&serde_json::json!({
            "text": "hello world",
            "language": "en",
            "duration": 1.5,
            "task": "transcribe",
            "segments": [{"start":0.0,"end":1.5,"text":"hello world"}],
            "x_groq": {"id":"req-1","usage":{"total_time":1.5}}
        }))
        .unwrap();
        let decoded = decode_response_parts(
            &ModelId::new("whisper-large-v3-turbo").unwrap(),
            None,
            &body,
            GroqTranscriptionResponseFormat::VerboseJson,
        )
        .unwrap();

        assert_eq!(decoded.text, "hello world");
        assert_eq!(decoded.segments.len(), 1);
        assert_eq!(decoded.metadata.request_id.as_deref(), Some("req-1"));
        assert_eq!(decoded.provider["groq"]["task"], "transcribe");
        assert_eq!(decoded.usage.provider["audio_duration_seconds"], 1.5);
    }

    #[test]
    fn text_response_format_preserves_plain_transcript() {
        let decoded = decode_response_parts(
            &ModelId::new("whisper-large-v3-turbo").unwrap(),
            None,
            b"hello world\n",
            GroqTranscriptionResponseFormat::Text,
        )
        .unwrap();

        assert_eq!(decoded.text, "hello world\n");
        assert!(decoded.segments.is_empty());
        assert_eq!(decoded.usage, Usage::default());
    }
}
