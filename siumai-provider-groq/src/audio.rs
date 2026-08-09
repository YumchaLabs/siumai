//! Provider-owned Groq URL-audio transcription and translation operations.

use std::collections::BTreeMap;
use std::fmt;

use http::header::{ACCEPT, HeaderValue};
use http::{Method, StatusCode};
use serde::Deserialize;
use serde_json::{Map, Value};
use siumai_core::{
    CallOptions, Error, ErrorKind, ModelId, PublicDiagnosticText, ResponseDiagnostics,
    ResponseMetadata, SensitiveResponse, Usage,
};
use siumai_transport::{
    ProviderTransport, ReplaySafety, RequestBody, RequestBuildError, RequestHeaders, RequestPlan,
    RequestTarget, ResponseHeaders, TransportResponse,
};

use crate::models;

const TRANSCRIPTIONS_TARGET: &str = "audio/transcriptions";
const TRANSLATIONS_TARGET: &str = "audio/translations";
const MAX_URL_BYTES: usize = 4 * 1024;
const MAX_PROMPT_BYTES: usize = 8 * 1024;

/// Typed request for provider-fetched audio.
#[derive(Clone, PartialEq)]
pub struct GroqUrlAudioRequest {
    model: ModelId,
    url: String,
    language: Option<String>,
    prompt: Option<String>,
    temperature: Option<f64>,
}

impl GroqUrlAudioRequest {
    pub fn new(model: impl Into<String>, url: impl Into<String>) -> Result<Self, Error> {
        let model = ModelId::new(model.into()).map_err(|source| {
            Error::new(ErrorKind::InvalidInput, "Groq audio model ID is invalid")
                .with_source(source)
        })?;
        let url = url.into();
        validate_remote_url(&url)?;
        Ok(Self {
            model,
            url,
            language: None,
            prompt: None,
            temperature: None,
        })
    }

    pub fn with_language(mut self, language: impl Into<String>) -> Result<Self, Error> {
        let language = language.into();
        validate_text(&language, 64, "Groq audio language is invalid or too long")?;
        self.language = Some(language);
        Ok(self)
    }

    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Result<Self, Error> {
        let prompt = prompt.into();
        validate_text(
            &prompt,
            MAX_PROMPT_BYTES,
            "Groq audio prompt is invalid or too long",
        )?;
        self.prompt = Some(prompt);
        Ok(self)
    }

    pub fn with_temperature(mut self, temperature: f64) -> Result<Self, Error> {
        if !temperature.is_finite() || !(0.0..=1.0).contains(&temperature) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Groq audio temperature must be between zero and one",
            ));
        }
        self.temperature = Some(temperature);
        Ok(self)
    }

    pub fn model(&self) -> &ModelId {
        &self.model
    }

    pub fn url(&self) -> &str {
        &self.url
    }
}

impl fmt::Debug for GroqUrlAudioRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GroqUrlAudioRequest")
            .field("model", &self.model)
            .field("url", &"[REDACTED]")
            .field("language", &self.language)
            .field("prompt_bytes", &self.prompt.as_ref().map(String::len))
            .field("temperature", &self.temperature)
            .finish()
    }
}

/// Typed result shared by Groq URL transcription and audio translation.
#[derive(Debug, Clone, PartialEq)]
pub struct GroqAudioResponse {
    pub text: String,
    pub language: Option<String>,
    pub duration_seconds: Option<f64>,
    pub metadata: ResponseMetadata,
    pub usage: Usage,
    pub provider: BTreeMap<String, Value>,
}

/// Provider-owned Groq audio resource for URL input and translation semantics.
#[derive(Clone)]
pub struct GroqAudio {
    transport: ProviderTransport,
    verified_endpoint: bool,
}

impl GroqAudio {
    pub(crate) fn new(transport: ProviderTransport, verified_endpoint: bool) -> Self {
        Self {
            transport,
            verified_endpoint,
        }
    }

    pub async fn transcribe_url(
        &self,
        request: GroqUrlAudioRequest,
        call: CallOptions,
    ) -> Result<GroqAudioResponse, Error> {
        self.execute(GroqAudioOperation::Transcribe, request, call)
            .await
    }

    pub async fn translate_url(
        &self,
        request: GroqUrlAudioRequest,
        call: CallOptions,
    ) -> Result<GroqAudioResponse, Error> {
        self.execute(GroqAudioOperation::Translate, request, call)
            .await
    }

    async fn execute(
        &self,
        operation: GroqAudioOperation,
        request: GroqUrlAudioRequest,
        call: CallOptions,
    ) -> Result<GroqAudioResponse, Error> {
        operation.validate_model(&request.model, self.verified_endpoint)?;
        if operation == GroqAudioOperation::Translate && request.language.is_some() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Groq translation detects the source language and does not accept language",
            ));
        }
        let plan = operation.plan(&request)?;
        let response = self.transport.execute(plan, call).await?;
        if !response.status().is_success() {
            return Err(provider_response_error(operation, response));
        }
        decode_response(operation, request.model, response)
    }
}

impl fmt::Debug for GroqAudio {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GroqAudio")
            .field("transport", &"shared")
            .field("verified_endpoint", &self.verified_endpoint)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GroqAudioOperation {
    Transcribe,
    Translate,
}

impl GroqAudioOperation {
    fn target(self) -> &'static str {
        match self {
            Self::Transcribe => TRANSCRIPTIONS_TARGET,
            Self::Translate => TRANSLATIONS_TARGET,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Transcribe => "URL transcription",
            Self::Translate => "audio translation",
        }
    }

    fn error_message(self) -> &'static str {
        match self {
            Self::Transcribe => "Groq URL transcription request failed",
            Self::Translate => "Groq audio translation request failed",
        }
    }

    fn validate_model(self, model: &ModelId, verified_endpoint: bool) -> Result<(), Error> {
        if !verified_endpoint {
            return Ok(());
        }
        if self == Self::Translate
            && models::is_known_transcription(model.as_str())
            && model.as_str() != models::transcription::WHISPER_LARGE_V3
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Groq audio translation does not support the selected known model",
            ));
        }
        Ok(())
    }

    fn plan(self, request: &GroqUrlAudioRequest) -> Result<RequestPlan, Error> {
        let mut body = Map::new();
        body.insert("url".to_string(), Value::String(request.url.clone()));
        body.insert(
            "model".to_string(),
            Value::String(request.model.to_string()),
        );
        body.insert(
            "response_format".to_string(),
            Value::String("verbose_json".to_string()),
        );
        if let Some(language) = &request.language {
            body.insert("language".to_string(), Value::String(language.clone()));
        }
        if let Some(prompt) = &request.prompt {
            body.insert("prompt".to_string(), Value::String(prompt.clone()));
        }
        if let Some(temperature) = request.temperature {
            body.insert("temperature".to_string(), Value::from(temperature));
        }
        let headers = RequestHeaders::new()
            .try_insert(ACCEPT, HeaderValue::from_static("application/json"))
            .map_err(request_build_error)?;
        RequestPlan::new(
            Method::POST,
            RequestTarget::new(self.target()).map_err(request_build_error)?,
        )
        .with_headers(headers)
        .with_body(RequestBody::json(&Value::Object(body)).map_err(request_build_error)?)
        .with_replay_safety(ReplaySafety::Never)
        .map_err(request_build_error)
    }
}

#[derive(Debug, Deserialize)]
struct GroqAudioResponseWire {
    text: String,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    duration: Option<f64>,
    #[serde(default)]
    segments: Option<Value>,
    #[serde(default)]
    words: Option<Value>,
    #[serde(default)]
    x_groq: Option<Value>,
}

fn decode_response(
    operation: GroqAudioOperation,
    model: ModelId,
    response: TransportResponse,
) -> Result<GroqAudioResponse, Error> {
    let attempts = response.attempts();
    let (_, headers, body) = response.into_parts();
    let wire: GroqAudioResponseWire = serde_json::from_slice(&body).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "Groq returned malformed URL-audio JSON",
        )
        .with_source(source)
    })?;
    if wire.text.trim().is_empty()
        || wire
            .duration
            .is_some_and(|duration| !duration.is_finite() || duration < 0.0)
    {
        return Err(Error::protocol_violation(
            "Groq URL-audio response contains invalid text or duration",
        ));
    }
    let request_id = response_request_id(&headers);
    let mut provider = BTreeMap::new();
    provider.insert(
        "operation".to_string(),
        Value::String(operation.label().to_string()),
    );
    provider.insert("transport_attempts".to_string(), Value::from(attempts));
    if let Some(segments) = wire.segments {
        provider.insert("segments".to_string(), segments);
    }
    if let Some(words) = wire.words {
        provider.insert("words".to_string(), words);
    }
    if let Some(x_groq) = wire.x_groq {
        provider.insert("x_groq".to_string(), x_groq);
    }
    Ok(GroqAudioResponse {
        text: wire.text,
        language: wire.language,
        duration_seconds: wire.duration,
        metadata: ResponseMetadata {
            response_id: None,
            request_id,
            model: Some(model),
        },
        usage: Usage::default(),
        provider,
    })
}

fn validate_remote_url(value: &str) -> Result<(), Error> {
    if value.len() > MAX_URL_BYTES || value.chars().any(char::is_control) {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Groq audio URL is invalid or too long",
        ));
    }
    let url = url::Url::parse(value).map_err(|source| {
        Error::new(
            ErrorKind::InvalidInput,
            "Groq audio URL must be an absolute HTTPS URL",
        )
        .with_source(source)
    })?;
    if url.scheme() != "https"
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Groq audio URL must use HTTPS without embedded credentials",
        ));
    }
    Ok(())
}

fn validate_text(value: &str, maximum: usize, message: &'static str) -> Result<(), Error> {
    if value.trim().is_empty() || value.len() > maximum || value.chars().any(char::is_control) {
        return Err(Error::new(ErrorKind::InvalidInput, message));
    }
    Ok(())
}

fn provider_response_error(operation: GroqAudioOperation, response: TransportResponse) -> Error {
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
    Error::new(kind, operation.error_message())
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

fn request_build_error(source: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "Groq URL-audio request violates the transport contract",
    )
    .with_source(source)
}

#[cfg(test)]
mod tests {
    use mockito::Matcher;
    use siumai_core::{ReplayDomain, ReplayDomainId, UsageValue};
    use siumai_transport::EndpointConfig;

    use super::*;
    use crate::{GroqCredential, GroqProvider};

    #[tokio::test]
    async fn translates_provider_fetched_audio_with_unknown_usage() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/openai/v1/audio/translations")
            .match_body(Matcher::Json(serde_json::json!({
                "url": "https://cdn.example.com/audio.wav",
                "model": models::transcription::WHISPER_LARGE_V3,
                "response_format": "verbose_json",
                "prompt": "product names"
            })))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_header("x-request-id", "request-translation")
            .with_body(
                serde_json::json!({
                    "text": "translated text",
                    "language": "english",
                    "duration": 1.25
                })
                .to_string(),
            )
            .create_async()
            .await;
        let provider = GroqProvider::builder(GroqCredential::unauthenticated())
            .with_endpoint(
                EndpointConfig::local_explicit(format!("{}/openai/v1", server.url())).unwrap(),
            )
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("test-groq-audio").unwrap(),
            ))
            .build()
            .unwrap();
        let request = GroqUrlAudioRequest::new(
            models::transcription::WHISPER_LARGE_V3,
            "https://cdn.example.com/audio.wav",
        )
        .unwrap()
        .with_prompt("product names")
        .unwrap();
        let response = provider
            .audio()
            .translate_url(request, CallOptions::default())
            .await
            .unwrap();

        mock.assert_async().await;
        assert_eq!(response.text, "translated text");
        assert_eq!(
            response.metadata.request_id.as_deref(),
            Some("request-translation")
        );
        assert_eq!(response.usage.input_tokens, UsageValue::Unknown);
        assert_eq!(response.usage.output_tokens, UsageValue::Unknown);
    }

    #[test]
    fn translation_rejects_known_unsupported_model_and_unsafe_url() {
        let provider = GroqProvider::builder(GroqCredential::api_key("secret"))
            .build()
            .unwrap();
        let request = GroqUrlAudioRequest::new(
            models::transcription::WHISPER_LARGE_V3_TURBO,
            "https://cdn.example.com/audio.wav",
        )
        .unwrap();
        assert!(
            GroqAudioOperation::Translate
                .validate_model(request.model(), true)
                .is_err()
        );
        assert!(GroqUrlAudioRequest::new("future-model", "http://127.0.0.1/audio.wav").is_err());
        assert!(!format!("{:?}", provider.audio()).contains("secret"));
    }
}
