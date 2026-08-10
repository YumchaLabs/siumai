//! Portable buffered text-to-speech over Groq Orpheus.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use http::header::{ACCEPT, HeaderValue};
use http::{Method, StatusCode};
use serde_json::Value;
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, Model, ModelDescriptor, ModelFamily, ModelId,
    ModelOperation, ModelPolicy, ModelPolicyContext, PublicDiagnosticText, ResponseDiagnostics,
    ResponseMetadata, SensitiveResponse, SpeechLimits, SpeechModel, SpeechRequest, SpeechResponse,
    SupportState, Usage, Warning,
};
use siumai_transport::{
    ReplaySafety, RequestBody, RequestBuildError, RequestHeaders, RequestPlan, RequestTarget,
    ResponseHeaders, TransportResponse,
};

use crate::provider::GroqSpeechRuntime;

pub const SPEECH_PROTOCOL_ID: &str = "groq-orpheus";
pub const SPEECH_API_MODE_ID: &str = "audio-speech";
pub const SPEECH_SOURCE: &str = "https://console.groq.com/docs/text-to-speech";

const SPEECH_TARGET: &str = "audio/speech";
const MAX_TEXT_CHARACTERS: usize = 200;

/// Lightweight buffered speech model over Groq's Orpheus API.
#[derive(Clone)]
pub struct GroqSpeechModel {
    pub(crate) runtime: Arc<GroqSpeechRuntime>,
    descriptor: ModelDescriptor,
}

impl GroqSpeechModel {
    pub(crate) fn new(runtime: Arc<GroqSpeechRuntime>, model: ModelId) -> Self {
        let descriptor = ModelDescriptor::from_scope(
            runtime.scope.clone(),
            model,
            ModelFamily::Speech,
            runtime.instance_id.clone(),
        );
        Self {
            runtime,
            descriptor,
        }
    }

    fn policy(&self) -> Result<Vec<Warning>, Error> {
        let decision = self.runtime.policy.evaluate(&ModelPolicyContext::new(
            self.runtime.scope.clone(),
            self.model_id().clone(),
            ModelOperation::SynthesizeSpeech,
        ));
        if matches!(decision.state(), SupportState::Unsupported { .. }) {
            return Err(self.contextualize(Error::new(
                ErrorKind::Unsupported,
                "Groq model policy rejected speech synthesis",
            )));
        }
        Ok(Vec::new())
    }

    fn plan(&self, request: &SpeechRequest) -> Result<RequestPlan, Error> {
        let voice = request.voice().ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidInput,
                "Groq Orpheus speech synthesis requires an explicit voice",
            )
        })?;
        if voice.len() > 128 || voice.chars().any(char::is_control) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Groq speech voice is invalid",
            ));
        }
        if request.language().is_some() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Groq Orpheus model IDs select the language; request language is not portable",
            ));
        }
        if request.speed().is_some() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Groq Orpheus does not expose portable speech speed control",
            ));
        }
        let format = request.format().unwrap_or("wav");
        if !format.eq_ignore_ascii_case("wav") {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Groq Orpheus portable speech supports WAV output only",
            ));
        }
        let headers = RequestHeaders::new()
            .try_insert(ACCEPT, HeaderValue::from_static("audio/wav"))
            .map_err(request_build_error)?;
        let body = serde_json::json!({
            "model": self.model_id().as_str(),
            "input": request.text(),
            "voice": voice,
            "response_format": "wav"
        });
        RequestPlan::new(
            Method::POST,
            RequestTarget::new(SPEECH_TARGET).map_err(request_build_error)?,
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

impl fmt::Debug for GroqSpeechModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GroqSpeechModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for GroqSpeechModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl SpeechModel for GroqSpeechModel {
    fn limits(&self) -> SpeechLimits {
        SpeechLimits {
            max_text_bytes: None,
            max_text_chars: Some(MAX_TEXT_CHARACTERS),
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
        let warnings = self.policy()?;
        let voice = request.voice().map(str::to_owned);
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
            return Err(self.contextualize(response_error(response)));
        }
        let attempts = response.attempts();
        let (status, headers, audio) = response.into_parts();
        debug_assert!(status.is_success());
        let mut provider = BTreeMap::new();
        provider.insert(
            "model".to_string(),
            Value::String(self.model_id().to_string()),
        );
        provider.insert("format".to_string(), Value::String("wav".to_string()));
        provider.insert("transport_attempts".to_string(), Value::from(attempts));
        if let Some(voice) = voice {
            provider.insert("voice".to_string(), Value::String(voice));
        }
        let response = SpeechResponse {
            media_type: response_media_type(&headers),
            audio,
            duration_seconds: None,
            sample_rate_hz: None,
            metadata: ResponseMetadata {
                response_id: None,
                request_id: response_request_id(&headers),
                model: Some(self.model_id().clone()),
            },
            usage: Usage::default(),
            warnings,
            provider,
        };
        response
            .validate()
            .map_err(|error| self.contextualize(error))?;
        Ok(response)
    }
}

fn response_media_type(headers: &ResponseHeaders) -> String {
    headers
        .get(&http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("audio/wav")
        .to_string()
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

fn response_error(response: TransportResponse) -> Error {
    let (status, headers, body) = response.into_parts();
    let kind = match status {
        StatusCode::UNAUTHORIZED => ErrorKind::Authentication,
        StatusCode::FORBIDDEN => ErrorKind::Authorization,
        StatusCode::TOO_MANY_REQUESTS => ErrorKind::RateLimited,
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
    Error::new(kind, "Groq speech request failed")
        .with_diagnostics(diagnostics)
        .with_sensitive_response(SensitiveResponse::new(raw_headers, body.to_vec()))
}

fn request_build_error(source: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "Groq speech request violates the transport contract",
    )
    .with_source(source)
}

#[cfg(test)]
mod tests {
    use mockito::Matcher;
    use siumai_core::{SpeechModel as _, SpeechRequest, UsageValue};
    use siumai_transport::EndpointConfig;

    use super::*;
    use crate::{GroqCredential, GroqProvider};

    #[tokio::test]
    async fn synthesizes_orpheus_wav_and_preserves_unknown_usage() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/openai/v1/audio/speech")
            .match_header("accept", "audio/wav")
            .match_body(Matcher::Json(serde_json::json!({
                "model": crate::models::speech::ORPHEUS_V1_ENGLISH,
                "input": "hello",
                "voice": "autumn",
                "response_format": "wav"
            })))
            .with_status(200)
            .with_header("content-type", "audio/wav")
            .with_header("x-request-id", "request-1")
            .with_body(*b"RIFF")
            .create_async()
            .await;
        let provider = GroqProvider::builder(GroqCredential::unauthenticated())
            .with_endpoint(
                EndpointConfig::local_explicit(format!("{}/openai/v1", server.url())).unwrap(),
            )
            .with_replay_domain(siumai_core::ReplayDomain::custom(
                siumai_core::ReplayDomainId::new("test-groq").unwrap(),
            ))
            .build()
            .unwrap();
        let response = provider
            .default_speech_model()
            .unwrap()
            .synthesize(
                SpeechRequest::new("hello")
                    .unwrap()
                    .with_voice("autumn")
                    .unwrap(),
                CallOptions::default(),
            )
            .await
            .unwrap();

        mock.assert_async().await;
        assert_eq!(response.media_type, "audio/wav");
        assert_eq!(response.audio.as_ref(), b"RIFF");
        assert_eq!(response.metadata.request_id.as_deref(), Some("request-1"));
        assert_eq!(response.usage.input_tokens, UsageValue::Unknown);
        assert_eq!(response.usage.output_tokens, UsageValue::Unknown);
    }

    #[test]
    fn rejects_non_wav_and_missing_voice_before_transport() {
        let provider = GroqProvider::builder(GroqCredential::api_key("secret"))
            .build()
            .unwrap();
        let model = provider.default_speech_model().unwrap();
        assert!(model.plan(&SpeechRequest::new("hello").unwrap()).is_err());
        assert!(
            model
                .plan(
                    &SpeechRequest::new("hello")
                        .unwrap()
                        .with_voice("autumn")
                        .unwrap()
                        .with_format("mp3")
                        .unwrap()
                )
                .is_err()
        );
        assert!(!format!("{model:?}").contains("secret"));
    }
}
