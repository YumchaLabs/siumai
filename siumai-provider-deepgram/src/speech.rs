//! Portable buffered text-to-speech over Deepgram Aura.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use http::header::{ACCEPT, HeaderValue};
use http::{Method, StatusCode};
use serde_json::Value;
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, Model, ModelDescriptor, ModelFamily, ModelId,
    ModelOperation, PublicDiagnosticText, ResponseDiagnostics, ResponseMetadata, SensitiveResponse,
    SpeechLimits, SpeechModel, SpeechRequest, SpeechResponse, Usage,
};
use siumai_transport::{
    ReplaySafety, RequestBody, RequestBuildError, RequestHeaders, RequestPlan, RequestTarget,
    ResponseHeaders, TransportResponse,
};

use crate::provider::DeepgramSpeechRuntime;

const SPEAK_TARGET: &str = "v1/speak";
const MAX_TEXT_CHARACTERS: usize = 2_000;

/// Lightweight buffered speech model over the Deepgram Aura API.
#[derive(Clone)]
pub struct DeepgramSpeechModel {
    pub(crate) runtime: Arc<DeepgramSpeechRuntime>,
    descriptor: ModelDescriptor,
}

impl DeepgramSpeechModel {
    pub(crate) fn new(runtime: Arc<DeepgramSpeechRuntime>, model: ModelId) -> Self {
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

    fn plan(&self, request: &SpeechRequest) -> Result<PreparedSpeechRequest, Error> {
        if request.language().is_some() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Deepgram Aura model IDs select the language; request language is not portable",
            ));
        }
        let model = request.voice().unwrap_or(self.model_id().as_str());
        if model.trim().is_empty() || model.len() > 128 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Deepgram speech voice/model identifier is invalid",
            ));
        }
        let format = request.format().unwrap_or("mp3");
        let output = OutputFormat::parse(format)?;
        let mut query = url::form_urlencoded::Serializer::new(String::new());
        query.append_pair("model", model);
        query.append_pair("encoding", output.encoding);
        if let Some(container) = output.container {
            query.append_pair("container", container);
        }
        if let Some(sample_rate) = output.sample_rate {
            query.append_pair("sample_rate", &sample_rate.to_string());
        }
        if let Some(speed) = request.speed() {
            if !(0.7..=1.5).contains(&speed) {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "Deepgram Aura speech speed must be between 0.7 and 1.5",
                ));
            }
            query.append_pair("speed", &speed.to_string());
        }
        let target = RequestTarget::new(format!("{SPEAK_TARGET}?{}", query.finish()))
            .map_err(request_build_error)?;
        let headers = RequestHeaders::new()
            .try_insert(ACCEPT, HeaderValue::from_static("audio/*"))
            .map_err(request_build_error)?;
        let body = serde_json::json!({ "text": request.text() });
        let plan = RequestPlan::new(Method::POST, target)
            .with_headers(headers)
            .with_body(RequestBody::json(&body).map_err(request_build_error)?)
            .with_replay_safety(ReplaySafety::Never)
            .map_err(request_build_error)?;
        Ok(PreparedSpeechRequest {
            plan,
            media_type: output.media_type.to_string(),
            sample_rate_hz: output.sample_rate,
            model: model.to_string(),
            format: format.to_string(),
        })
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

impl std::fmt::Debug for DeepgramSpeechModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeepgramSpeechModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for DeepgramSpeechModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl SpeechModel for DeepgramSpeechModel {
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
        let call = call.resolve_deadline().map_err(Error::from)?;
        self.limits()
            .validate(&request)
            .map_err(|error| self.contextualize(error))?;
        let prepared = self
            .plan(&request)
            .map_err(|error| self.contextualize(error))?;
        let response = self
            .runtime
            .transport
            .execute(prepared.plan, call)
            .await
            .map_err(|error| self.contextualize(error))?;
        if !response.status().is_success() {
            return Err(self.contextualize(response_error(response)));
        }
        let attempts = response.attempts();
        let (status, headers, audio) = response.into_parts();
        debug_assert!(status.is_success());
        let request_id = response_request_id(&headers);
        let mut provider = BTreeMap::new();
        provider.insert("model".to_string(), Value::String(prepared.model));
        provider.insert("format".to_string(), Value::String(prepared.format));
        provider.insert("transport_attempts".to_string(), Value::from(attempts));
        if let Some(characters) = response_header_u64(&headers, "dg-char-count") {
            provider.insert("input_characters".to_string(), Value::from(characters));
        }
        let response = SpeechResponse {
            media_type: response_media_type(&headers, &prepared.media_type),
            audio,
            duration_seconds: None,
            sample_rate_hz: prepared.sample_rate_hz,
            metadata: ResponseMetadata {
                response_id: None,
                request_id,
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

struct PreparedSpeechRequest {
    plan: RequestPlan,
    media_type: String,
    sample_rate_hz: Option<u32>,
    model: String,
    format: String,
}

#[derive(Clone, Copy)]
struct OutputFormat {
    encoding: &'static str,
    container: Option<&'static str>,
    sample_rate: Option<u32>,
    media_type: &'static str,
}

impl OutputFormat {
    fn parse(value: &str) -> Result<Self, Error> {
        let normalized = value.trim().to_ascii_lowercase();
        let output = match normalized.as_str() {
            "mp3" => Self {
                encoding: "mp3",
                container: None,
                sample_rate: None,
                media_type: "audio/mpeg",
            },
            "wav" | "linear16" => Self {
                encoding: "linear16",
                container: Some("wav"),
                sample_rate: Some(24_000),
                media_type: "audio/wav",
            },
            "pcm" => Self {
                encoding: "linear16",
                container: Some("none"),
                sample_rate: Some(24_000),
                media_type: "audio/L16",
            },
            "mulaw" | "mu-law" => Self {
                encoding: "mulaw",
                container: Some("wav"),
                sample_rate: Some(8_000),
                media_type: "audio/basic",
            },
            "alaw" | "a-law" => Self {
                encoding: "alaw",
                container: Some("wav"),
                sample_rate: Some(8_000),
                media_type: "audio/basic",
            },
            "opus" | "ogg" => Self {
                encoding: "opus",
                container: Some("ogg"),
                sample_rate: Some(48_000),
                media_type: "audio/ogg",
            },
            "flac" => Self {
                encoding: "flac",
                container: None,
                sample_rate: None,
                media_type: "audio/flac",
            },
            "aac" => Self {
                encoding: "aac",
                container: None,
                sample_rate: None,
                media_type: "audio/aac",
            },
            _ => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "Deepgram speech format is not supported by the portable adapter",
                ));
            }
        };
        Ok(output)
    }
}

fn response_media_type(headers: &ResponseHeaders, fallback: &str) -> String {
    headers
        .get(&http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(fallback)
        .to_string()
}

fn response_request_id(headers: &ResponseHeaders) -> Option<String> {
    ["dg-request-id", "x-dg-request-id"]
        .into_iter()
        .find_map(|name| {
            headers
                .get(&http::header::HeaderName::from_static(name))
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned)
        })
}

fn response_header_u64(headers: &ResponseHeaders, name: &'static str) -> Option<u64> {
    headers
        .get(&http::header::HeaderName::from_static(name))
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse().ok())
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
    Error::new(kind, "Deepgram speech request failed")
        .with_diagnostics(diagnostics)
        .with_sensitive_response(SensitiveResponse::new(raw_headers, body.to_vec()))
}

fn request_build_error(source: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "Deepgram speech request violates the transport contract",
    )
    .with_source(source)
}

#[cfg(test)]
mod tests {
    use mockito::Matcher;
    use siumai_core::{SpeechModel as _, SpeechRequest, UsageValue};
    use siumai_transport::EndpointConfig;

    use super::*;
    use crate::{DeepgramCredential, DeepgramProvider};

    #[tokio::test]
    async fn synthesizes_buffered_aura_audio_with_unknown_usage() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock(
                "POST",
                "/v1/speak?model=aura-2-thalia-en&encoding=mp3&speed=1.2",
            )
            .match_header("accept", "audio/*")
            .match_body(Matcher::Json(serde_json::json!({ "text": "hello" })))
            .with_status(200)
            .with_header("content-type", "audio/mpeg")
            .with_header("dg-request-id", "request-1")
            .with_header("dg-char-count", "5")
            .with_body([b'I', b'D', b'3', 4])
            .create_async()
            .await;
        let provider = DeepgramProvider::builder(DeepgramCredential::unauthenticated())
            .with_endpoint(EndpointConfig::local_explicit(server.url()).unwrap())
            .build()
            .unwrap();
        let response = provider
            .default_speech_model()
            .unwrap()
            .synthesize(
                SpeechRequest::new("hello")
                    .unwrap()
                    .with_speed(1.2)
                    .unwrap(),
                CallOptions::default(),
            )
            .await
            .unwrap();

        mock.assert_async().await;
        assert_eq!(response.media_type, "audio/mpeg");
        assert_eq!(response.audio.as_ref(), [b'I', b'D', b'3', 4]);
        assert_eq!(response.metadata.request_id.as_deref(), Some("request-1"));
        assert_eq!(
            response.provider.get("input_characters"),
            Some(&Value::from(5_u64))
        );
        assert_eq!(response.usage.input_tokens, UsageValue::Unknown);
        assert_eq!(response.usage.output_tokens, UsageValue::Unknown);
    }

    #[test]
    fn speech_debug_does_not_expose_transport_or_credentials() {
        let provider = DeepgramProvider::builder(DeepgramCredential::api_key("secret-key"))
            .build()
            .unwrap();
        let debug = format!("{:?}", provider.default_speech_model().unwrap());
        assert!(!debug.contains("secret-key"));
    }
}
