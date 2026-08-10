//! OpenAI portable final-result audio transcription model.

use std::collections::BTreeSet;
use std::sync::Arc;

use async_trait::async_trait;
use http::Method;
use http::header::{ACCEPT, HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, Model, ModelDescriptor, ModelFamily, ModelId,
    ModelOperation, ProviderOptionContext, ProviderOptionError, ProviderOptionLayers,
    ProviderOptionMerger, ProviderOptionOrigin, ProviderOptions, ProviderScope,
    TranscriptionLimits, TranscriptionModel, TranscriptionRequest, TranscriptionResponse,
    TypedProviderOptions, Warning, WarningKind,
};
use siumai_protocol_openai::transcription::{
    API_MODE_ID, TARGET, TranscriptionConfig, TranscriptionResponseFormat,
    TranscriptionTimestampGranularity, decode_transcription_response, encode_transcription_fields,
};
use siumai_transport::{
    MultipartBody, MultipartPart, ReplaySafety, RequestBody, RequestHeaders, RequestPlan,
    RequestTarget,
};

use super::http_error::{request_build_error, response_error};
use super::provider::OpenAiRuntime;

pub const WHISPER_1: &str = "whisper-1";
pub const GPT_4O_MINI_TRANSCRIBE: &str = "gpt-4o-mini-transcribe";
pub const GPT_4O_MINI_TRANSCRIBE_2025_03_20: &str = "gpt-4o-mini-transcribe-2025-03-20";
pub const GPT_4O_MINI_TRANSCRIBE_2025_12_15: &str = "gpt-4o-mini-transcribe-2025-12-15";
pub const GPT_4O_TRANSCRIBE: &str = "gpt-4o-transcribe";
pub const GPT_4O_TRANSCRIBE_DIARIZE: &str = "gpt-4o-transcribe-diarize";

const MAX_AUDIO_BYTES: usize = 25 * 1024 * 1024;
const MAX_LANGUAGE_BYTES: usize = 128;
const MAX_PROMPT_BYTES: usize = 16 * 1024;

/// JSON response representation requested from OpenAI transcription.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum OpenAiTranscriptionResponseFormat {
    Json,
    VerboseJson,
    DiarizedJson,
}

/// Timestamp units requested in verbose transcription responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum OpenAiTranscriptionTimestampGranularity {
    Segment,
    Word,
}

/// Provider-owned controls for final-result OpenAI transcription.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiTranscriptionOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<OpenAiTranscriptionResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub timestamp_granularities: Vec<OpenAiTranscriptionTimestampGranularity>,
}

impl OpenAiTranscriptionOptions {
    pub const fn new() -> Self {
        Self {
            response_format: None,
            temperature: None,
            timestamp_granularities: Vec::new(),
        }
    }

    pub const fn with_response_format(
        mut self,
        response_format: OpenAiTranscriptionResponseFormat,
    ) -> Self {
        self.response_format = Some(response_format);
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Result<Self, ProviderOptionError> {
        self.temperature = Some(temperature);
        self.validate()?;
        Ok(self)
    }

    pub fn with_timestamp_granularities(
        mut self,
        granularities: impl IntoIterator<Item = OpenAiTranscriptionTimestampGranularity>,
    ) -> Result<Self, ProviderOptionError> {
        self.timestamp_granularities = granularities.into_iter().collect();
        self.validate()?;
        Ok(self)
    }
}

impl TypedProviderOptions for OpenAiTranscriptionOptions {
    const NAMESPACE: &'static str = "openai";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Transcription;
    const API_MODE: Option<&'static str> = Some(API_MODE_ID);

    fn validate(&self) -> Result<(), ProviderOptionError> {
        if self
            .temperature
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(rejected(
                "temperature",
                "must be finite and between 0 and 1",
            ));
        }
        if self.timestamp_granularities.len() > 2
            || self
                .timestamp_granularities
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len()
                != self.timestamp_granularities.len()
        {
            return Err(rejected(
                "timestamp_granularities",
                "must contain unique segment and/or word values",
            ));
        }
        if !self.timestamp_granularities.is_empty()
            && self.response_format != Some(OpenAiTranscriptionResponseFormat::VerboseJson)
        {
            return Err(rejected(
                "timestamp_granularities",
                "requires response_format=verbose_json",
            ));
        }
        Ok(())
    }
}

/// Lightweight OpenAI final-result transcription handle.
#[derive(Clone)]
pub struct OpenAiTranscriptionModel {
    runtime: Arc<OpenAiRuntime>,
    descriptor: ModelDescriptor,
    defaults: OpenAiTranscriptionOptions,
}

impl OpenAiTranscriptionModel {
    pub(crate) fn new(
        runtime: Arc<OpenAiRuntime>,
        scope: Arc<ProviderScope>,
        model: ModelId,
        defaults: OpenAiTranscriptionOptions,
    ) -> Self {
        let instance_id = runtime.instance_id.clone();
        Self {
            runtime,
            descriptor: ModelDescriptor::from_scope(
                scope,
                model,
                ModelFamily::Transcription,
                instance_id,
            ),
            defaults,
        }
    }

    fn options(&self, call: &CallOptions) -> Result<OpenAiTranscriptionOptions, Error> {
        let layers = call
            .apply_provider_options(self.provider_id(), ProviderOptionLayers::default())
            .map_err(option_error)?;
        layers
            .merge_for(
                ProviderOptionContext::new(
                    self.provider_id(),
                    ModelFamily::Transcription,
                    self.descriptor.scope().api_mode(),
                ),
                &OpenAiTranscriptionOptionMerger {
                    defaults: self.defaults.clone(),
                },
            )
            .map_err(option_error)
    }

    fn plan(
        &self,
        request: &TranscriptionRequest,
        options: &OpenAiTranscriptionOptions,
    ) -> Result<RequestPlan, Error> {
        if self.model_id().as_str() == "gpt-realtime-whisper"
            || self
                .model_id()
                .as_str()
                .starts_with("gpt-realtime-whisper-")
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "OpenAI realtime transcription models are not callable through the final-result REST adapter",
            ));
        }
        validate_portable_fields(request)?;
        validate_known_model_request(self.descriptor.scope(), self.model_id(), request, options)?;
        let content_type = HeaderValue::from_str(request.media_type()).map_err(|source| {
            Error::new(
                ErrorKind::InvalidInput,
                "OpenAI transcription media type cannot be encoded",
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
            .map_err(request_plan_error)?,
        ];
        let config = TranscriptionConfig {
            response_format: options
                .response_format
                .map(protocol_response_format)
                .unwrap_or_default(),
            temperature: options.temperature,
            timestamp_granularities: options
                .timestamp_granularities
                .iter()
                .copied()
                .map(protocol_timestamp_granularity)
                .collect(),
        };
        for field in encode_transcription_fields(request, self.model_id(), &config) {
            parts.push(
                MultipartPart::field(field.name(), field.value().as_bytes().to_vec())
                    .map_err(request_plan_error)?,
            );
        }
        let headers = RequestHeaders::new()
            .try_insert(ACCEPT, HeaderValue::from_static("application/json"))
            .map_err(request_plan_error)?;
        RequestPlan::new(
            Method::POST,
            RequestTarget::new(TARGET).map_err(request_plan_error)?,
        )
        .with_headers(headers)
        .with_body(RequestBody::multipart(MultipartBody::new(parts)))
        .with_replay_safety(ReplaySafety::Never)
        .map_err(request_plan_error)
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

impl std::fmt::Debug for OpenAiTranscriptionModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenAiTranscriptionModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for OpenAiTranscriptionModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl TranscriptionModel for OpenAiTranscriptionModel {
    fn limits(&self) -> TranscriptionLimits {
        TranscriptionLimits {
            max_audio_bytes: Some(
                MAX_AUDIO_BYTES.min(self.runtime.transport.limits().max_request_bytes),
            ),
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
            .options(&call)
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
            return Err(self.contextualize(response_error(
                "OpenAI rejected the transcription request",
                response,
            )));
        }
        let (_, headers, body) = response.into_parts();
        let mut decoded = decode_transcription_response(&body, self.model_id())
            .map_err(|error| self.contextualize(error))?;
        decoded.metadata.request_id = response_request_id(&headers);
        if !is_verified_model(self.descriptor.scope(), self.model_id()) {
            decoded.warnings.push(Warning::new(
                WarningKind::UnknownModel,
                "model support is not verified for OpenAI final-result transcription",
            ));
        }
        Ok(decoded)
    }
}

struct OpenAiTranscriptionOptionMerger {
    defaults: OpenAiTranscriptionOptions,
}

impl ProviderOptionMerger for OpenAiTranscriptionOptionMerger {
    type Output = OpenAiTranscriptionOptions;

    fn validate_layer(
        &self,
        _origin: ProviderOptionOrigin,
        options: &ProviderOptions,
    ) -> Result<(), ProviderOptionError> {
        decode_options(options).and_then(|options| options.validate())
    }

    fn merge(&self, layers: &ProviderOptionLayers) -> Result<Self::Output, ProviderOptionError> {
        let mut merged = serde_json::to_value(&self.defaults)
            .map_err(|error| ProviderOptionError::Serialization(error.to_string()))?
            .as_object()
            .cloned()
            .unwrap_or_else(Map::new);
        for (_, options) in layers.in_precedence_order() {
            merged.extend(options.value().clone());
        }
        let output = serde_json::from_value::<OpenAiTranscriptionOptions>(Value::Object(merged))
            .map_err(|error| ProviderOptionError::Serialization(error.to_string()))?;
        output.validate()?;
        Ok(output)
    }
}

fn decode_options(
    options: &ProviderOptions,
) -> Result<OpenAiTranscriptionOptions, ProviderOptionError> {
    serde_json::from_value(Value::Object(options.value().clone())).map_err(|_| {
        ProviderOptionError::Rejected {
            path: "openai".to_string(),
            reason: "options do not match the OpenAI transcription schema".to_string(),
        }
    })
}

fn validate_portable_fields(request: &TranscriptionRequest) -> Result<(), Error> {
    if let Some(language) = request.language()
        && (language.len() > MAX_LANGUAGE_BYTES || language.chars().any(char::is_control))
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "OpenAI transcription language is invalid",
        ));
    }
    if let Some(prompt) = request.prompt()
        && prompt.len() > MAX_PROMPT_BYTES
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "OpenAI transcription prompt exceeds the portable adapter limit",
        ));
    }
    Ok(())
}

fn validate_known_model_request(
    scope: &ProviderScope,
    model: &ModelId,
    request: &TranscriptionRequest,
    options: &OpenAiTranscriptionOptions,
) -> Result<(), Error> {
    if scope.platform().map(|value| value.as_str()) != Some("openai-api") {
        return Ok(());
    }
    match model.as_str() {
        GPT_4O_MINI_TRANSCRIBE
        | GPT_4O_MINI_TRANSCRIBE_2025_03_20
        | GPT_4O_MINI_TRANSCRIBE_2025_12_15
        | GPT_4O_TRANSCRIBE => {
            if options
                .response_format
                .is_some_and(|format| format != OpenAiTranscriptionResponseFormat::Json)
            {
                return Err(invalid_known_model_request(
                    "OpenAI GPT-4o transcription models support only JSON response format",
                ));
            }
        }
        GPT_4O_TRANSCRIBE_DIARIZE => {
            if request.prompt().is_some() {
                return Err(invalid_known_model_request(
                    "OpenAI diarized transcription does not support prompt",
                ));
            }
            if !options.timestamp_granularities.is_empty() {
                return Err(invalid_known_model_request(
                    "OpenAI diarized transcription does not support timestamp granularities",
                ));
            }
            if options.response_format == Some(OpenAiTranscriptionResponseFormat::VerboseJson) {
                return Err(invalid_known_model_request(
                    "OpenAI diarized transcription does not support verbose_json",
                ));
            }
        }
        WHISPER_1
            if options.response_format == Some(OpenAiTranscriptionResponseFormat::DiarizedJson) =>
        {
            return Err(invalid_known_model_request(
                "OpenAI whisper-1 does not support diarized_json",
            ));
        }
        _ => {}
    }
    Ok(())
}

fn invalid_known_model_request(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

fn audio_file_name(media_type: &str) -> Result<&'static str, Error> {
    match media_type.to_ascii_lowercase().as_str() {
        "audio/flac" => Ok("audio.flac"),
        "audio/mpeg" | "audio/mp3" | "audio/mpga" => Ok("audio.mp3"),
        "audio/mp4" | "audio/m4a" | "audio/x-m4a" => Ok("audio.m4a"),
        "video/mp4" => Ok("audio.mp4"),
        "audio/ogg" | "application/ogg" => Ok("audio.ogg"),
        "audio/wav" | "audio/x-wav" => Ok("audio.wav"),
        "audio/webm" | "video/webm" => Ok("audio.webm"),
        _ => Err(Error::new(
            ErrorKind::Unsupported,
            "OpenAI transcription does not support this portable audio media type",
        )),
    }
}

fn protocol_response_format(
    value: OpenAiTranscriptionResponseFormat,
) -> TranscriptionResponseFormat {
    match value {
        OpenAiTranscriptionResponseFormat::Json => TranscriptionResponseFormat::Json,
        OpenAiTranscriptionResponseFormat::VerboseJson => TranscriptionResponseFormat::VerboseJson,
        OpenAiTranscriptionResponseFormat::DiarizedJson => {
            TranscriptionResponseFormat::DiarizedJson
        }
    }
}

fn protocol_timestamp_granularity(
    value: OpenAiTranscriptionTimestampGranularity,
) -> TranscriptionTimestampGranularity {
    match value {
        OpenAiTranscriptionTimestampGranularity::Segment => {
            TranscriptionTimestampGranularity::Segment
        }
        OpenAiTranscriptionTimestampGranularity::Word => TranscriptionTimestampGranularity::Word,
    }
}

fn is_verified_model(scope: &ProviderScope, model: &ModelId) -> bool {
    scope.platform().map(|value| value.as_str()) == Some("openai-api")
        && matches!(
            model.as_str(),
            WHISPER_1
                | GPT_4O_MINI_TRANSCRIBE
                | GPT_4O_MINI_TRANSCRIBE_2025_03_20
                | GPT_4O_MINI_TRANSCRIBE_2025_12_15
                | GPT_4O_TRANSCRIBE
                | GPT_4O_TRANSCRIBE_DIARIZE
        )
}

fn rejected(path: &str, reason: &str) -> ProviderOptionError {
    ProviderOptionError::Rejected {
        path: path.to_string(),
        reason: reason.to_string(),
    }
}

fn option_error(source: ProviderOptionError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "provider options are invalid for OpenAI transcription",
    )
    .with_source(source)
}

fn request_plan_error(source: siumai_transport::RequestBuildError) -> Error {
    request_build_error(
        "OpenAI transcription request violates the transport contract",
        source,
    )
}

fn response_request_id(headers: &siumai_transport::ResponseHeaders) -> Option<String> {
    ["x-request-id", "request-id"].into_iter().find_map(|name| {
        headers
            .get(&HeaderName::from_static(name))
            .and_then(|value| value.to_str().ok())
            .filter(|value| {
                !value.is_empty()
                    && value.len() <= 256
                    && value.bytes().all(|byte| {
                        byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':')
                    })
            })
            .map(str::to_owned)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configured::profile::OpenAiProfile;

    #[test]
    fn options_require_verbose_json_for_timestamps() {
        let invalid = OpenAiTranscriptionOptions::new()
            .with_timestamp_granularities([OpenAiTranscriptionTimestampGranularity::Word]);
        assert!(invalid.is_err());

        let valid = OpenAiTranscriptionOptions::new()
            .with_response_format(OpenAiTranscriptionResponseFormat::VerboseJson)
            .with_timestamp_granularities([OpenAiTranscriptionTimestampGranularity::Word]);
        assert!(valid.is_ok());
    }

    #[test]
    fn supported_media_types_map_to_safe_static_file_names() {
        assert_eq!(audio_file_name("audio/wav").unwrap(), "audio.wav");
        assert!(audio_file_name("application/octet-stream").is_err());
    }

    #[test]
    fn official_known_models_reject_unsupported_transcription_controls() {
        let profile = OpenAiProfile::current().unwrap();
        let scope = profile
            .family_provider_scope(ModelFamily::Transcription)
            .unwrap();
        let request = TranscriptionRequest::new(vec![1_u8], "audio/wav").unwrap();
        let prompted = request.clone().with_prompt("speaker context").unwrap();

        assert!(
            validate_known_model_request(
                scope,
                &ModelId::new(GPT_4O_TRANSCRIBE).unwrap(),
                &request,
                &OpenAiTranscriptionOptions::new()
                    .with_response_format(OpenAiTranscriptionResponseFormat::VerboseJson),
            )
            .is_err()
        );
        assert!(
            validate_known_model_request(
                scope,
                &ModelId::new(GPT_4O_TRANSCRIBE_DIARIZE).unwrap(),
                &prompted,
                &OpenAiTranscriptionOptions::default(),
            )
            .is_err()
        );
        assert!(
            validate_known_model_request(
                scope,
                &ModelId::new(WHISPER_1).unwrap(),
                &request,
                &OpenAiTranscriptionOptions::new()
                    .with_response_format(OpenAiTranscriptionResponseFormat::DiarizedJson),
            )
            .is_err()
        );
    }

    #[test]
    fn custom_transcription_endpoints_keep_open_model_baseline_behavior() {
        let profile = OpenAiProfile::custom(siumai_core::ReplayDomain::custom(
            siumai_core::ReplayDomainId::new("custom-transcription-fixture").unwrap(),
        ))
        .unwrap();
        let scope = profile
            .family_provider_scope(ModelFamily::Transcription)
            .unwrap();
        let request = TranscriptionRequest::new(vec![1_u8], "audio/wav")
            .unwrap()
            .with_prompt("speaker context")
            .unwrap();

        assert!(
            validate_known_model_request(
                scope,
                &ModelId::new(GPT_4O_TRANSCRIBE_DIARIZE).unwrap(),
                &request,
                &OpenAiTranscriptionOptions::new()
                    .with_response_format(OpenAiTranscriptionResponseFormat::VerboseJson),
            )
            .is_ok()
        );
    }
}
