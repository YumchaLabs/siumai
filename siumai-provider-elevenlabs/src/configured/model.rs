use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use http::header::{ACCEPT, CONTENT_TYPE, HeaderName, HeaderValue};
use http::{Method, StatusCode};
use serde_json::{Map, Value};
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, Model, ModelDescriptor, ModelFamily, ModelId,
    ModelOperation, ProviderOptionError, PublicDiagnosticText, ResponseDiagnostics,
    ResponseMetadata, SensitiveResponse, SpeechLimits, SpeechModel, SpeechRequest, SpeechResponse,
    Usage,
};
use siumai_transport::{
    ReplaySafety, RequestBody, RequestBuildError, RequestHeaders, RequestPlan, RequestTarget,
    ResponseHeaders, TransportResponse,
};

use super::options::ElevenLabsSpeechOptions;
use super::provider::{ProviderRuntime, normalized_voice_id};

const ERROR_CAPTURE_BYTES: usize = 64 * 1024;
const DEFAULT_OUTPUT_FORMAT: &str = "mp3_44100_128";

/// Lightweight ElevenLabs speech model sharing one configured provider runtime.
#[derive(Clone)]
pub struct ElevenLabsSpeechModel {
    runtime: Arc<ProviderRuntime>,
    descriptor: ModelDescriptor,
}

impl ElevenLabsSpeechModel {
    pub(crate) fn new(runtime: Arc<ProviderRuntime>, model: ModelId) -> Self {
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

    fn plan(
        &self,
        request: &SpeechRequest,
        options: &ElevenLabsSpeechOptions,
    ) -> Result<PreparedSpeechRequest, Error> {
        let voice = request
            .voice()
            .map(str::to_owned)
            .unwrap_or_else(|| self.runtime.default_voice.clone());
        let voice = normalized_voice_id(&voice).ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidInput,
                "ElevenLabs voice ID contains invalid path characters",
            )
        })?;
        let output_format = normalize_output_format(request.format())?;

        let mut body = options.body_fields().map_err(request_serialization_error)?;
        body.insert(
            "text".to_string(),
            Value::String(request.text().to_string()),
        );
        body.insert(
            "model_id".to_string(),
            Value::String(self.model_id().as_str().to_string()),
        );
        if let Some(language) = request.language() {
            body.insert(
                "language_code".to_string(),
                Value::String(language.to_string()),
            );
        }

        let mut voice_settings = match options.voice_settings() {
            Some(settings) => serde_json::to_value(settings)
                .map_err(request_serialization_error)?
                .as_object()
                .cloned()
                .unwrap_or_default(),
            None => Map::new(),
        };
        if let Some(speed) = request.speed() {
            voice_settings.insert("speed".to_string(), Value::from(speed));
        }
        if !voice_settings.is_empty() {
            body.insert("voice_settings".to_string(), Value::Object(voice_settings));
        }

        let mut target = format!("v1/text-to-speech/{voice}?output_format={output_format}");
        if let Some(enable_logging) = options.enable_logging() {
            target.push_str("&enable_logging=");
            target.push_str(if enable_logging { "true" } else { "false" });
        }
        let headers = RequestHeaders::new()
            .try_insert(ACCEPT, HeaderValue::from_static("audio/*"))
            .map_err(request_build_error)?;
        let plan = RequestPlan::new(
            Method::POST,
            RequestTarget::new(target).map_err(request_build_error)?,
        )
        .with_headers(headers)
        .with_body(RequestBody::json(&Value::Object(body)).map_err(request_build_error)?)
        .with_replay_safety(ReplaySafety::Never)
        .map_err(request_build_error)?;

        Ok(PreparedSpeechRequest {
            plan,
            voice,
            output_format,
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

impl std::fmt::Debug for ElevenLabsSpeechModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ElevenLabsSpeechModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for ElevenLabsSpeechModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl SpeechModel for ElevenLabsSpeechModel {
    fn limits(&self) -> SpeechLimits {
        self.runtime.speech_limits(self.model_id())
    }

    async fn synthesize(
        &self,
        request: SpeechRequest,
        options: CallOptions,
    ) -> Result<SpeechResponse, Error> {
        self.limits()
            .validate(&request)
            .map_err(|error| self.contextualize(error))?;
        let provider_options = self
            .runtime
            .merge_options(self, &options)
            .map_err(option_error)
            .map_err(|error| self.contextualize(error))?;
        let prepared = self
            .plan(&request, &provider_options)
            .map_err(|error| self.contextualize(error))?;
        let response = self
            .runtime
            .transport
            .execute(prepared.plan, options)
            .await
            .map_err(|error| self.contextualize(error))?;
        if !response.status().is_success() {
            return Err(self.contextualize(response_error(response)));
        }

        let attempts = response.attempts();
        let (status, headers, audio) = response.into_parts();
        debug_assert!(status.is_success());
        let request_id = response_request_id(&headers);
        let media_type = response_media_type(&headers, &prepared.output_format);
        let mut provider = BTreeMap::new();
        provider.insert(
            "output_format".to_string(),
            Value::String(prepared.output_format.clone()),
        );
        provider.insert("voice_id".to_string(), Value::String(prepared.voice));
        provider.insert("transport_attempts".to_string(), Value::from(attempts));
        let response = SpeechResponse {
            media_type,
            audio,
            duration_seconds: None,
            sample_rate_hz: sample_rate(&prepared.output_format),
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
    voice: String,
    output_format: String,
}

fn normalize_output_format(format: Option<&str>) -> Result<String, Error> {
    let format = format.unwrap_or(DEFAULT_OUTPUT_FORMAT).trim();
    let mapped = match format {
        "mp3" | "mp3_128" => DEFAULT_OUTPUT_FORMAT,
        "mp3_32" => "mp3_44100_32",
        "mp3_64" => "mp3_44100_64",
        "mp3_96" => "mp3_44100_96",
        "mp3_192" => "mp3_44100_192",
        "pcm" => "pcm_44100",
        "ulaw" => "ulaw_8000",
        other => other,
    };
    if mapped.is_empty()
        || mapped.len() > 64
        || !mapped
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "ElevenLabs output format is invalid",
        ));
    }
    Ok(mapped.to_string())
}

fn response_media_type(headers: &ResponseHeaders, output_format: &str) -> String {
    headers
        .get(&CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| inferred_media_type(output_format).to_string())
}

fn inferred_media_type(output_format: &str) -> &'static str {
    match output_format.split('_').next() {
        Some("mp3") => "audio/mpeg",
        Some("wav") => "audio/wav",
        Some("opus") => "audio/ogg",
        Some("ulaw" | "alaw") => "audio/basic",
        Some("pcm") => "audio/pcm",
        _ => "application/octet-stream",
    }
}

fn sample_rate(output_format: &str) -> Option<u32> {
    output_format
        .split('_')
        .nth(1)
        .and_then(|value| value.parse().ok())
}

fn option_error(source: ProviderOptionError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "provider options are invalid for ElevenLabs speech synthesis",
    )
    .with_source(source)
}

fn request_serialization_error(source: serde_json::Error) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "ElevenLabs speech request could not be encoded",
    )
    .with_source(source)
}

fn request_build_error(source: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "ElevenLabs speech request violates the transport contract",
    )
    .with_source(source)
}

fn response_error(response: TransportResponse) -> Error {
    let (status, headers, body) = response.into_parts();
    let provider_code = provider_error_code(&body);
    let kind = match (status, provider_code.as_deref()) {
        (_, Some("quota_exceeded")) | (StatusCode::PAYMENT_REQUIRED, _) => ErrorKind::QuotaExceeded,
        (_, Some("too_many_concurrent_requests")) | (StatusCode::TOO_MANY_REQUESTS, _) => {
            ErrorKind::RateLimited
        }
        (StatusCode::BAD_REQUEST | StatusCode::NOT_FOUND | StatusCode::UNPROCESSABLE_ENTITY, _) => {
            ErrorKind::InvalidInput
        }
        (StatusCode::UNAUTHORIZED, _) => ErrorKind::Authentication,
        (StatusCode::FORBIDDEN, _) => ErrorKind::Authorization,
        _ => ErrorKind::Provider,
    };
    let request_id = response_request_id(&headers);
    let mut diagnostics = ResponseDiagnostics::default()
        .with_status(status.as_u16())
        .with_body_truncated(body.len() > ERROR_CAPTURE_BYTES);
    if let Some(code) = provider_code.and_then(|code| PublicDiagnosticText::new(code).ok()) {
        diagnostics = diagnostics.with_provider_code(code);
    }
    if let Some(request_id) =
        request_id.and_then(|request_id| PublicDiagnosticText::new(request_id).ok())
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
                .map(|value| (name.to_string(), value.to_string()))
        })
        .collect();
    Error::new(kind, "ElevenLabs rejected the speech synthesis request")
        .with_diagnostics(diagnostics)
        .with_sensitive_response(SensitiveResponse::with_limit(
            raw_headers,
            body.to_vec(),
            ERROR_CAPTURE_BYTES,
        ))
}

fn provider_error_code(body: &[u8]) -> Option<String> {
    let value = serde_json::from_slice::<Value>(body).ok()?;
    value
        .pointer("/detail/status")
        .or_else(|| value.get("status"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty() && value.len() <= 256)
        .map(str::to_owned)
}

fn response_request_id(headers: &ResponseHeaders) -> Option<String> {
    [
        HeaderName::from_static("request-id"),
        HeaderName::from_static("x-request-id"),
    ]
    .iter()
    .find_map(|name| headers.get(name))
    .and_then(|value| value.to_str().ok())
    .filter(|value| !value.is_empty() && value.len() <= 1024)
    .map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;
    use siumai_core::{Cancellation, ErrorDetail, ResourceKind, UsageValue};
    use wiremock::matchers::{body_json, header, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::configured::{
        ElevenLabsCredential, ElevenLabsProfile, ElevenLabsProvider, ElevenLabsSpeechOptions,
        ElevenLabsVoiceSettings, models,
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
    async fn synthesis_uses_one_native_post_and_returns_owned_audio() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/text-to-speech/voice_1"))
            .and(query_param("output_format", "mp3_44100_128"))
            .and(query_param("enable_logging", "false"))
            .and(header("xi-api-key", "test-key"))
            .and(body_json(serde_json::json!({
                "text": "hello world",
                "model_id": models::ELEVEN_MULTILINGUAL_V2,
                "language_code": "en",
                "seed": 7,
                "voice_settings": {
                    "stability": 0.4,
                    "speed": 1.0
                }
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "audio/mpeg")
                    .insert_header("request-id", "req-123")
                    .set_body_bytes(vec![1_u8, 2, 3, 4]),
            )
            .expect(1)
            .mount(&server)
            .await;

        let provider = provider(&server);
        let model = provider.speech(models::ELEVEN_MULTILINGUAL_V2).unwrap();
        let request = SpeechRequest::new("hello world")
            .unwrap()
            .with_voice("voice_1")
            .unwrap()
            .with_format("mp3")
            .unwrap()
            .with_language("en")
            .unwrap()
            .with_speed(1.0)
            .unwrap();
        let provider_options = ElevenLabsSpeechOptions::new()
            .with_seed(7)
            .with_logging(false)
            .with_voice_settings(ElevenLabsVoiceSettings::new().with_stability(0.4));
        let call = CallOptions::default()
            .with_provider_options_for(&model, &provider_options)
            .unwrap();
        let response = model.synthesize(request, call).await.unwrap();

        assert_eq!(response.media_type, "audio/mpeg");
        assert_eq!(response.sample_rate_hz, Some(44_100));
        assert_eq!(response.metadata.request_id.as_deref(), Some("req-123"));
        assert_eq!(response.usage.audio_output_tokens, UsageValue::Unknown);
        assert!(response.warnings.is_empty());
        let audio = response.audio;
        let audio = tokio::spawn(async move { audio.to_vec() }).await.unwrap();
        assert_eq!(audio, vec![1, 2, 3, 4]);
    }

    #[tokio::test]
    async fn intrinsic_and_provider_limits_fail_before_the_wire() {
        assert!(SpeechRequest::new("  ").is_err());

        let server = MockServer::start().await;
        let provider = ElevenLabsProvider::builder(
            ElevenLabsProfile::local_explicit(server.uri()).unwrap(),
            ElevenLabsCredential::api_key("test-key"),
        )
        .with_speech_limits(SpeechLimits {
            max_text_bytes: None,
            max_text_chars: Some(4),
        })
        .build()
        .unwrap();
        let model = provider.speech(models::DEFAULT).unwrap();
        let error = model
            .synthesize(SpeechRequest::new("hello").unwrap(), CallOptions::default())
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::LimitExceeded);
        assert_eq!(
            error.detail(),
            Some(&ErrorDetail::LimitExceeded {
                resource: ResourceKind::SpeechTextCharacters,
                actual: 5,
                maximum: 4,
            })
        );
        assert_eq!(server.received_requests().await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn cancellation_and_deadline_reach_transport_controls() {
        let model = ElevenLabsProvider::builder(
            ElevenLabsProfile::local_explicit("http://127.0.0.1:9").unwrap(),
            ElevenLabsCredential::api_key("test-key"),
        )
        .build()
        .unwrap()
        .speech(models::DEFAULT)
        .unwrap();
        let request = SpeechRequest::new("hello").unwrap();
        let cancellation = Cancellation::new();
        cancellation.cancel();

        let error = model
            .synthesize(
                request.clone(),
                CallOptions::default().with_cancellation(cancellation),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Cancelled);

        let error = model
            .synthesize(
                request,
                CallOptions::default().with_deadline(Instant::now() - Duration::from_millis(1)),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Timeout);
    }

    #[tokio::test]
    async fn provider_errors_are_typed_and_raw_body_is_explicitly_sensitive() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(format!(
                "/v1/text-to-speech/{}",
                models::DEFAULT_VOICE
            )))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("request-id", "req-rate-limited")
                    .set_body_json(serde_json::json!({
                        "detail": {
                            "status": "too_many_concurrent_requests",
                            "message": "sensitive provider message"
                        }
                    })),
            )
            .expect(1)
            .mount(&server)
            .await;

        let model = provider(&server).speech(models::DEFAULT).unwrap();
        let error = model
            .synthesize(SpeechRequest::new("hello").unwrap(), CallOptions::default())
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::RateLimited);
        let diagnostics = error.diagnostics().unwrap();
        assert_eq!(diagnostics.status(), Some(429));
        assert_eq!(
            diagnostics.provider_code(),
            Some("too_many_concurrent_requests")
        );
        assert_eq!(diagnostics.request_id(), Some("req-rate-limited"));
        let (_, body) = error.sensitive_response().unwrap().expose();
        assert!(String::from_utf8_lossy(body).contains("sensitive provider message"));
        assert!(!format!("{error:?}").contains("sensitive provider message"));
    }

    #[tokio::test]
    async fn empty_success_audio_is_a_protocol_violation() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "audio/mpeg")
                    .set_body_bytes(Vec::<u8>::new()),
            )
            .expect(1)
            .mount(&server)
            .await;

        let error = provider(&server)
            .speech(models::DEFAULT)
            .unwrap()
            .synthesize(SpeechRequest::new("hello").unwrap(), CallOptions::default())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ProtocolViolation);
    }
}
