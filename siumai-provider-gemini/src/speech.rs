use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use http::Method;
use http::header::{ACCEPT, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, Model, ModelDescriptor, ModelFamily, ModelId,
    ModelOperation, ProviderOptionError, ProviderOptionSelection, ProviderOptions, SpeechLimits,
    SpeechModel, SpeechRequest, SpeechResponse, TypedProviderOptions,
};
use siumai_protocol_gemini::interactions::{
    InteractionSpeechConfig, V1BETA_SPEECH_TARGET, decode_speech_response, encode_speech_request,
};
use siumai_transport::{
    ReplaySafety, RequestBody, RequestBuildError, RequestHeaders, RequestPlan, RequestTarget,
};

use crate::http::{response_error, response_request_id};
use crate::provider::ProviderRuntime;

/// Current Gemini Interactions TTS model hint.
pub const GEMINI_3_1_FLASH_TTS_PREVIEW: &str = "gemini-3.1-flash-tts-preview";
/// Older Gemini Flash TTS compatibility hint.
pub const GEMINI_2_5_FLASH_PREVIEW_TTS: &str = "gemini-2.5-flash-preview-tts";
/// Older Gemini Pro TTS compatibility hint.
pub const GEMINI_2_5_PRO_PREVIEW_TTS: &str = "gemini-2.5-pro-preview-tts";
/// Provider API mode used by the current Interactions TTS slice.
pub const GEMINI_SPEECH_API_MODE_ID: &str = "interactions-speech";

/// Provider-owned defaults for Gemini's current single-speaker Interactions TTS slice.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GeminiSpeechOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
}

impl GeminiSpeechOptions {
    pub const fn new() -> Self {
        Self { voice: None }
    }

    pub fn with_voice(mut self, voice: impl Into<String>) -> Result<Self, Error> {
        let voice = voice.into();
        validate_voice(&voice)?;
        self.voice = Some(voice);
        Ok(self)
    }
}

impl TypedProviderOptions for GeminiSpeechOptions {
    const NAMESPACE: &'static str = "google";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Speech;
    const API_MODE: Option<&'static str> = Some(GEMINI_SPEECH_API_MODE_ID);
}

/// Portable buffered speech adapter over Gemini's current Interactions TTS API.
#[derive(Clone)]
pub struct GeminiSpeechModel {
    runtime: Arc<ProviderRuntime>,
    descriptor: ModelDescriptor,
}

impl GeminiSpeechModel {
    pub(crate) fn new(runtime: Arc<ProviderRuntime>, model: ModelId) -> Self {
        let descriptor = ModelDescriptor::from_scope(
            runtime.speech_scope.clone(),
            model,
            ModelFamily::Speech,
            runtime.instance_id.clone(),
        );
        Self {
            runtime,
            descriptor,
        }
    }

    fn options(&self, call: &CallOptions) -> Result<GeminiSpeechOptions, Error> {
        let selection = call.provider_options_for(self).map_err(option_error)?;
        merge_options(&self.runtime.speech_defaults, &selection).map_err(option_error)
    }

    fn plan(
        &self,
        request: &SpeechRequest,
        options: &GeminiSpeechOptions,
    ) -> Result<RequestPlan, Error> {
        if request.speed().is_some() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Gemini Interactions TTS does not expose a numeric speech-speed control",
            ));
        }
        if request.format().is_some() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Gemini Interactions TTS portable mode currently supports only its default PCM output",
            ));
        }
        if request.language().is_some() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Gemini Interactions TTS detects language from the input and has no portable language override",
            ));
        }
        let voice = request
            .voice()
            .or(options.voice.as_deref())
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "Gemini Interactions TTS requires an explicit voice",
                )
            })?;
        let config = InteractionSpeechConfig::new(voice)?;
        let body = encode_speech_request(self.model_id(), request.text(), &config)?;
        let headers = RequestHeaders::new()
            .try_insert(ACCEPT, HeaderValue::from_static("application/json"))
            .map_err(request_build_error)?;
        RequestPlan::new(
            Method::POST,
            RequestTarget::new(V1BETA_SPEECH_TARGET).map_err(request_build_error)?,
        )
        .with_headers(headers)
        .with_body(RequestBody::json(&body).map_err(request_build_error)?)
        .with_replay_safety(ReplaySafety::Never)
        .map_err(request_build_error)
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

impl fmt::Debug for GeminiSpeechModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GeminiSpeechModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for GeminiSpeechModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl SpeechModel for GeminiSpeechModel {
    fn limits(&self) -> SpeechLimits {
        SpeechLimits::default()
    }

    async fn synthesize(
        &self,
        request: SpeechRequest,
        options: CallOptions,
    ) -> Result<SpeechResponse, Error> {
        let provider_options = self.options(&options)?;
        let plan = self
            .plan(&request, &provider_options)
            .map_err(|error| self.contextualize(error))?;
        let response = self
            .runtime
            .transport
            .execute(plan, options)
            .await
            .map_err(|error| self.contextualize(error))?;
        if !response.status().is_success() {
            return Err(self.contextualize(response_error(
                response,
                "Gemini rejected the Interactions speech request",
            )));
        }
        let (_, headers, body) = response.into_parts();
        let mut response = decode_speech_response(&body, self.descriptor.scope(), self.model_id())
            .map_err(|error| self.contextualize(error))?;
        response.metadata.request_id = response_request_id(&headers);
        Ok(response)
    }
}

fn merge_options(
    defaults: &GeminiSpeechOptions,
    selection: &ProviderOptionSelection<'_>,
) -> Result<GeminiSpeechOptions, ProviderOptionError> {
    let mut merged = serde_json::to_value(defaults)
        .map_err(|error| ProviderOptionError::Serialization(error.to_string()))?
        .as_object()
        .cloned()
        .unwrap_or_else(Map::new);
    for options in selection.typed() {
        decode_options(options)?;
        merged.extend(options.value().clone());
    }
    if selection.raw_override().is_some() {
        return Err(ProviderOptionError::Rejected {
            path: "$".to_string(),
            reason: "Gemini Interactions speech only accepts typed provider options".to_string(),
        });
    }
    decode_options_value(merged)
}

fn decode_options(options: &ProviderOptions) -> Result<GeminiSpeechOptions, ProviderOptionError> {
    if let Some(field) = options
        .value()
        .keys()
        .find(|field| field.as_str() != "voice")
    {
        return Err(ProviderOptionError::Rejected {
            path: field.clone(),
            reason: "field is not valid for Gemini Interactions speech".to_string(),
        });
    }
    decode_options_value(options.value().clone())
}

fn decode_options_value(
    options: Map<String, Value>,
) -> Result<GeminiSpeechOptions, ProviderOptionError> {
    let output = serde_json::from_value::<GeminiSpeechOptions>(Value::Object(options))
        .map_err(|error| ProviderOptionError::Serialization(error.to_string()))?;
    if let Some(voice) = &output.voice {
        validate_voice(voice).map_err(|error| ProviderOptionError::Rejected {
            path: "voice".to_string(),
            reason: error.to_string(),
        })?;
    }
    Ok(output)
}

fn validate_voice(voice: &str) -> Result<(), Error> {
    if voice.trim().is_empty()
        || voice.len() > 256
        || voice.chars().any(|character| character.is_control())
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Gemini speech voice is invalid",
        ));
    }
    Ok(())
}

fn option_error(source: ProviderOptionError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "provider options are invalid for Gemini Interactions speech",
    )
    .with_source(source)
}

fn request_build_error(source: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "Gemini speech request violates the transport contract",
    )
    .with_source(source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_require_a_bounded_voice() {
        assert!(GeminiSpeechOptions::new().with_voice("Kore").is_ok());
        assert!(GeminiSpeechOptions::new().with_voice("\n").is_err());
    }
}
