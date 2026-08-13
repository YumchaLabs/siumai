//! OpenAI portable buffered text-to-speech model.

use std::sync::Arc;

use async_trait::async_trait;
use http::Method;
use http::header::{ACCEPT, CONTENT_TYPE, HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, Model, ModelDescriptor, ModelFamily, ModelId,
    ModelOperation, ProviderOptionError, ProviderOptionSelection, ProviderOptions, ProviderScope,
    SpeechLimits, SpeechModel, SpeechRequest, SpeechResponse, TypedProviderOptions,
};
use siumai_protocol_openai::speech::{
    API_MODE_ID, SpeechConfig, SpeechFormat, TARGET, decode_speech_response, encode_speech_request,
};
use siumai_transport::{ReplaySafety, RequestBody, RequestHeaders, RequestPlan, RequestTarget};

use super::http_error::{request_build_error, response_error};
use super::provider::OpenAiRuntime;

pub const GPT_4O_MINI_TTS: &str = "gpt-4o-mini-tts";
pub const GPT_4O_MINI_TTS_2025_03_20: &str = "gpt-4o-mini-tts-2025-03-20";
pub const GPT_4O_MINI_TTS_2025_12_15: &str = "gpt-4o-mini-tts-2025-12-15";
pub const TTS_1: &str = "tts-1";
pub const TTS_1_1106: &str = "tts-1-1106";
pub const TTS_1_HD: &str = "tts-1-hd";
pub const TTS_1_HD_1106: &str = "tts-1-hd-1106";

const MAX_TEXT_CHARS: usize = 4_096;
const MAX_VOICE_BYTES: usize = 2_048;
const MAX_INSTRUCTIONS_CHARS: usize = 4_096;

/// Provider-owned controls for buffered OpenAI speech synthesis.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiSpeechOptions {
    /// Voice-direction instructions supported by GPT-4o speech models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

impl OpenAiSpeechOptions {
    pub const fn new() -> Self {
        Self { instructions: None }
    }

    pub fn with_instructions(
        mut self,
        instructions: impl Into<String>,
    ) -> Result<Self, ProviderOptionError> {
        self.instructions = Some(instructions.into());
        self.validate()?;
        Ok(self)
    }
}

impl TypedProviderOptions for OpenAiSpeechOptions {
    const NAMESPACE: &'static str = "openai";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Speech;
    const API_MODE: Option<&'static str> = Some(API_MODE_ID);

    fn validate(&self) -> Result<(), ProviderOptionError> {
        if let Some(instructions) = self.instructions.as_deref()
            && (instructions.trim().is_empty()
                || instructions.chars().count() > MAX_INSTRUCTIONS_CHARS)
        {
            return Err(ProviderOptionError::Rejected {
                path: "instructions".to_string(),
                reason: "must be non-empty and at most 4096 characters".to_string(),
            });
        }
        Ok(())
    }
}

/// Lightweight OpenAI buffered speech handle.
#[derive(Clone)]
pub struct OpenAiSpeechModel {
    runtime: Arc<OpenAiRuntime>,
    descriptor: ModelDescriptor,
    defaults: OpenAiSpeechOptions,
}

impl OpenAiSpeechModel {
    pub(crate) fn new(
        runtime: Arc<OpenAiRuntime>,
        scope: Arc<ProviderScope>,
        model: ModelId,
        defaults: OpenAiSpeechOptions,
    ) -> Self {
        let instance_id = runtime.instance_id.clone();
        Self {
            runtime,
            descriptor: ModelDescriptor::from_scope(scope, model, ModelFamily::Speech, instance_id),
            defaults,
        }
    }

    fn options(&self, call: &CallOptions) -> Result<OpenAiSpeechOptions, Error> {
        let selection = call.provider_options_for(self).map_err(option_error)?;
        merge_options(&self.defaults, &selection).map_err(option_error)
    }

    fn plan(
        &self,
        request: &SpeechRequest,
        options: &OpenAiSpeechOptions,
    ) -> Result<(RequestPlan, SpeechFormat), Error> {
        if request.language().is_some() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "OpenAI speech synthesis does not expose a language override",
            ));
        }
        if options.instructions.is_some()
            && is_official_legacy_tts(self.descriptor.scope(), self.model_id())
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "OpenAI tts-1 models do not support speech instructions",
            ));
        }
        let voice = request.voice().ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidInput,
                "OpenAI speech synthesis requires an explicit voice",
            )
        })?;
        validate_voice(voice)?;
        let format = parse_format(request.format())?;
        if let Some(speed) = request.speed()
            && !(0.25..=4.0).contains(&speed)
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "OpenAI speech speed must be between 0.25 and 4.0",
            ));
        }
        let body = encode_speech_request(
            self.model_id(),
            request.text(),
            &SpeechConfig {
                voice: voice.to_string(),
                format,
                speed: request.speed(),
                instructions: options.instructions.clone(),
            },
        )?;
        let headers = RequestHeaders::new()
            .try_insert(ACCEPT, HeaderValue::from_static("audio/*"))
            .map_err(request_plan_error)?;
        let plan = RequestPlan::new(
            Method::POST,
            RequestTarget::new(TARGET).map_err(request_plan_error)?,
        )
        .with_headers(headers)
        .with_body(RequestBody::json(&body).map_err(request_plan_error)?)
        .with_replay_safety(ReplaySafety::Never)
        .map_err(request_plan_error)?;
        Ok((plan, format))
    }

    fn contextualize(&self, error: Error) -> Error {
        error.with_context(ErrorContext {
            operation: Some(ModelOperation::SynthesizeSpeech),
            provider: Some(self.provider_id().clone()),
            route: None,
            model: Some(self.model_id().clone()),
        })
    }
}

impl std::fmt::Debug for OpenAiSpeechModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenAiSpeechModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for OpenAiSpeechModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl SpeechModel for OpenAiSpeechModel {
    fn limits(&self) -> SpeechLimits {
        SpeechLimits {
            max_text_bytes: None,
            max_text_chars: Some(MAX_TEXT_CHARS),
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
        let options = self
            .options(&call)
            .map_err(|error| self.contextualize(error))?;
        let (plan, format) = self
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
                "OpenAI rejected the speech request",
                response,
            )));
        }
        let (_, headers, body) = response.into_parts();
        let media_type = response_media_type(&headers).unwrap_or(format.media_type());
        let mut decoded = decode_speech_response(body, self.model_id(), media_type)
            .map_err(|error| self.contextualize(error))?;
        decoded.metadata.request_id = response_request_id(&headers);
        Ok(decoded)
    }
}

fn merge_options(
    defaults: &OpenAiSpeechOptions,
    selection: &ProviderOptionSelection<'_>,
) -> Result<OpenAiSpeechOptions, ProviderOptionError> {
    let mut merged = defaults.clone();
    for options in selection.typed() {
        let options = decode_options(options)?;
        options.validate()?;
        if options.instructions.is_some() {
            merged.instructions = options.instructions;
        }
    }
    if selection.raw_override().is_some() {
        return Err(ProviderOptionError::Rejected {
            path: "$".to_string(),
            reason: "OpenAI speech only accepts typed provider options".to_string(),
        });
    }
    merged.validate()?;
    Ok(merged)
}

fn decode_options(options: &ProviderOptions) -> Result<OpenAiSpeechOptions, ProviderOptionError> {
    serde_json::from_value(Value::Object(options.value().clone())).map_err(|_| {
        ProviderOptionError::Rejected {
            path: "openai".to_string(),
            reason: "options do not match the OpenAI speech schema".to_string(),
        }
    })
}

fn parse_format(format: Option<&str>) -> Result<SpeechFormat, Error> {
    match format.map(str::to_ascii_lowercase).as_deref() {
        None | Some("mp3") | Some("audio/mpeg") => Ok(SpeechFormat::Mp3),
        Some("opus") | Some("audio/ogg") | Some("audio/opus") => Ok(SpeechFormat::Opus),
        Some("aac") | Some("audio/aac") => Ok(SpeechFormat::Aac),
        Some("flac") | Some("audio/flac") => Ok(SpeechFormat::Flac),
        Some("wav") | Some("wave") | Some("audio/wav") | Some("audio/x-wav") => {
            Ok(SpeechFormat::Wav)
        }
        Some("pcm") | Some("audio/pcm") => Ok(SpeechFormat::Pcm),
        Some(_) => Err(Error::new(
            ErrorKind::Unsupported,
            "OpenAI speech supports mp3, opus, aac, flac, wav, or pcm output",
        )),
    }
}

fn validate_voice(voice: &str) -> Result<(), Error> {
    if voice.trim().is_empty()
        || voice.len() > MAX_VOICE_BYTES
        || voice.chars().any(char::is_control)
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "OpenAI speech voice is invalid",
        ));
    }
    Ok(())
}

fn response_media_type(headers: &siumai_transport::ResponseHeaders) -> Option<&str> {
    headers
        .get(&CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .map(str::trim)
        .filter(|value| value.starts_with("audio/"))
}

fn is_official_legacy_tts(scope: &ProviderScope, model: &ModelId) -> bool {
    scope.platform().map(|value| value.as_str()) == Some("openai-api")
        && matches!(
            model.as_str(),
            TTS_1 | TTS_1_1106 | TTS_1_HD | TTS_1_HD_1106
        )
}

fn option_error(source: ProviderOptionError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "provider options are invalid for OpenAI speech",
    )
    .with_source(source)
}

fn request_plan_error(source: siumai_transport::RequestBuildError) -> Error {
    request_build_error(
        "OpenAI speech request violates the transport contract",
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
    fn portable_formats_are_strict_and_language_is_not_misrepresented() {
        assert_eq!(parse_format(Some("audio/wav")).unwrap(), SpeechFormat::Wav);
        assert!(parse_format(Some("ogg-vorbis")).is_err());
        assert!(OpenAiSpeechOptions::new().with_instructions("Warm").is_ok());
    }

    #[test]
    fn instructions_enforce_the_official_character_limit() {
        assert!(
            OpenAiSpeechOptions::new()
                .with_instructions("a".repeat(MAX_INSTRUCTIONS_CHARS))
                .is_ok()
        );
        assert!(
            OpenAiSpeechOptions::new()
                .with_instructions("a".repeat(MAX_INSTRUCTIONS_CHARS + 1))
                .is_err()
        );
        assert!(
            OpenAiSpeechOptions::new()
                .with_instructions("语".repeat(MAX_INSTRUCTIONS_CHARS))
                .is_ok()
        );
    }

    #[test]
    fn legacy_tts_instruction_restriction_is_scoped_to_the_official_platform() {
        let official = OpenAiProfile::current().unwrap();
        assert!(is_official_legacy_tts(
            official.family_provider_scope(ModelFamily::Speech).unwrap(),
            &ModelId::new(TTS_1).unwrap(),
        ));

        let custom = OpenAiProfile::custom(siumai_core::ReplayDomain::custom(
            siumai_core::ReplayDomainId::new("custom-speech-fixture").unwrap(),
        ))
        .unwrap();
        assert!(!is_official_legacy_tts(
            custom.family_provider_scope(ModelFamily::Speech).unwrap(),
            &ModelId::new(TTS_1).unwrap(),
        ));
    }
}
