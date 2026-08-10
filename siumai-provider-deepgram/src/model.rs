use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use http::header::{ACCEPT, HeaderName, HeaderValue, RETRY_AFTER};
use http::{Method, StatusCode};
use serde::Deserialize;
use serde_json::Value;
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, Model, ModelAdvisory, ModelDescriptor,
    ModelFamily, ModelId, ModelOperation, ModelPolicy, ModelPolicyDecision, ProviderOptionError,
    PublicDiagnosticText, ResponseMetadata, SensitiveResponse, SupportState, TranscriptSegment,
    TranscriptionLimits, TranscriptionModel, TranscriptionRequest, TranscriptionResponse, Usage,
    Warning, WarningKind,
};
use siumai_transport::{
    ReplaySafety, RequestBody, RequestBuildError, RequestHeaders, RequestPlan, RequestTarget,
    ResponseHeaders, TransportResponse,
};

use crate::options::{
    DeepgramDiarizeModel, DeepgramRedaction, DeepgramSummarizeOption, DeepgramTranscriptionOptions,
};
use crate::provider::ProviderRuntime;

const LISTEN_TARGET: &str = "v1/listen";
const SENSITIVE_BODY_LIMIT: usize = 64 * 1024;

/// Lightweight final-result transcription handle over a shared Deepgram runtime.
#[derive(Clone)]
pub struct DeepgramTranscriptionModel {
    pub(crate) runtime: Arc<ProviderRuntime>,
    descriptor: ModelDescriptor,
}

impl DeepgramTranscriptionModel {
    pub(crate) fn new(runtime: Arc<ProviderRuntime>, model: ModelId) -> Self {
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

    fn policy(&self) -> Result<Vec<Warning>, Error> {
        let decision = self
            .runtime
            .policy
            .evaluate(&siumai_core::ModelPolicyContext::new(
                self.runtime.scope.clone(),
                self.model_id().clone(),
                ModelOperation::Transcribe,
            ));
        if let SupportState::Unsupported { .. } = decision.state() {
            return Err(self.contextualize(Error::new(
                ErrorKind::Unsupported,
                "Deepgram model policy rejected transcription",
            )));
        }
        Ok(policy_warnings(&decision))
    }

    fn plan(
        &self,
        request: &TranscriptionRequest,
        options: &DeepgramTranscriptionOptions,
    ) -> Result<RequestPlan, Error> {
        if request.prompt().is_some() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Deepgram prerecorded transcription does not support prompt guidance",
            ));
        }
        if request.language().is_some() && options.detect_language() == Some(true) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Deepgram language and automatic language detection are mutually exclusive",
            ));
        }

        let target = RequestTarget::new(build_target(self.model_id(), request, options))
            .map_err(request_build_error)?;
        let content_type = HeaderValue::from_str(request.media_type()).map_err(|source| {
            Error::new(
                ErrorKind::InvalidInput,
                "transcription media type cannot be encoded as an HTTP header",
            )
            .with_source(source)
        })?;
        let headers = RequestHeaders::new()
            .try_insert(ACCEPT, HeaderValue::from_static("application/json"))
            .map_err(request_build_error)?;
        RequestPlan::new(Method::POST, target)
            .with_headers(headers)
            .with_body(RequestBody::bytes_with_content_type(
                request.audio().clone(),
                content_type,
            ))
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

impl std::fmt::Debug for DeepgramTranscriptionModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeepgramTranscriptionModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for DeepgramTranscriptionModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl TranscriptionModel for DeepgramTranscriptionModel {
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
        let warnings = self.policy()?;
        let options = self
            .runtime
            .options(&call)
            .map_err(option_error)
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
            return Err(self.contextualize(provider_response_error(response)));
        }
        decode_response(self.model_id(), response, warnings)
            .map_err(|error| self.contextualize(error))
    }
}

fn build_target(
    model: &ModelId,
    request: &TranscriptionRequest,
    options: &DeepgramTranscriptionOptions,
) -> String {
    let mut query = url::form_urlencoded::Serializer::new(String::new());
    query.append_pair("model", model.as_str());
    if let Some(language) = request.language() {
        query.append_pair("language", language);
    }
    append_bool(&mut query, "detect_language", options.detect_language);
    append_bool(&mut query, "smart_format", options.smart_format);
    append_bool(&mut query, "punctuate", options.punctuate);
    append_bool(&mut query, "paragraphs", options.paragraphs);
    if let Some(summarize) = options.summarize {
        query.append_pair(
            "summarize",
            match summarize {
                DeepgramSummarizeOption::Disabled => "false",
                DeepgramSummarizeOption::Version2 => "v2",
            },
        );
    }
    append_bool(&mut query, "topics", options.topics);
    append_bool(&mut query, "intents", options.intents);
    append_bool(&mut query, "sentiment", options.sentiment);
    append_bool(&mut query, "detect_entities", options.detect_entities);
    if let Some(redaction) = &options.redact {
        match redaction {
            DeepgramRedaction::Single(value) => {
                query.append_pair("redact", value);
            }
            DeepgramRedaction::Multiple(values) => {
                for value in values {
                    query.append_pair("redact", value);
                }
            }
        }
    }
    append_text(&mut query, "replace", options.replace.as_deref());
    append_text(&mut query, "search", options.search.as_deref());
    append_text(&mut query, "keyterm", options.keyterm.as_deref());
    if let Some(model) = options.diarize_model {
        query.append_pair(
            "diarize_model",
            match model {
                DeepgramDiarizeModel::Latest => "latest",
                DeepgramDiarizeModel::V1 => "v1",
            },
        );
    }
    append_bool(&mut query, "utterances", options.utterances);
    if let Some(value) = options.utt_split {
        query.append_pair("utt_split", &value.to_string());
    }
    append_bool(&mut query, "filler_words", options.filler_words);
    format!("{LISTEN_TARGET}?{}", query.finish())
}

fn append_bool<T>(
    query: &mut url::form_urlencoded::Serializer<'_, T>,
    name: &str,
    value: Option<bool>,
) where
    T: url::form_urlencoded::Target,
{
    if let Some(value) = value {
        query.append_pair(name, if value { "true" } else { "false" });
    }
}

fn append_text<T>(
    query: &mut url::form_urlencoded::Serializer<'_, T>,
    name: &str,
    value: Option<&str>,
) where
    T: url::form_urlencoded::Target,
{
    if let Some(value) = value {
        query.append_pair(name, value);
    }
}

fn decode_response(
    requested_model: &ModelId,
    response: TransportResponse,
    warnings: Vec<Warning>,
) -> Result<TranscriptionResponse, Error> {
    let (_, headers, body) = response.into_parts();
    let raw: Value = serde_json::from_slice(&body).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "Deepgram returned malformed transcription JSON",
        )
        .with_source(source)
    })?;
    let wire: DeepgramResponse = serde_json::from_value(raw.clone()).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "Deepgram transcription response does not match the wire contract",
        )
        .with_source(source)
    })?;
    let channel = wire.results.channels.into_iter().next().ok_or_else(|| {
        Error::protocol_violation("Deepgram transcription response has no audio channel")
    })?;
    let alternative = channel.alternatives.into_iter().next().ok_or_else(|| {
        Error::protocol_violation("Deepgram transcription response has no alternative")
    })?;
    let segments = alternative
        .words
        .into_iter()
        .map(|word| TranscriptSegment {
            start_seconds: word.start,
            end_seconds: word.end,
            text: word.punctuated_word.unwrap_or(word.word),
            confidence: word.confidence,
        })
        .collect();
    let duration_seconds = wire
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.duration);
    let body_request_id = wire
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.request_id.clone());
    let request_id =
        checked_response_identifier(body_request_id.or_else(|| response_request_id(&headers)))?;
    let mut usage = Usage::default();
    if let Some(duration) = duration_seconds {
        usage
            .provider
            .insert("audio_duration_seconds".to_string(), Value::from(duration));
    }
    let mut provider = BTreeMap::new();
    provider.insert("raw_response".to_string(), raw);
    let result = TranscriptionResponse {
        text: alternative.transcript,
        language: channel.detected_language,
        confidence: alternative.confidence,
        duration_seconds,
        segments,
        metadata: ResponseMetadata {
            response_id: None,
            request_id,
            model: Some(requested_model.clone()),
        },
        usage,
        warnings,
        provider,
    };
    result.validate()?;
    Ok(result)
}

#[derive(Deserialize)]
struct DeepgramResponse {
    #[serde(default)]
    metadata: Option<DeepgramMetadata>,
    results: DeepgramResults,
}

#[derive(Deserialize)]
struct DeepgramMetadata {
    #[serde(default)]
    request_id: Option<String>,
    #[serde(default)]
    duration: Option<f64>,
}

#[derive(Deserialize)]
struct DeepgramResults {
    channels: Vec<DeepgramChannel>,
}

#[derive(Deserialize)]
struct DeepgramChannel {
    #[serde(default)]
    detected_language: Option<String>,
    alternatives: Vec<DeepgramAlternative>,
}

#[derive(Deserialize)]
struct DeepgramAlternative {
    transcript: String,
    #[serde(default)]
    confidence: Option<f64>,
    #[serde(default)]
    words: Vec<DeepgramWord>,
}

#[derive(Deserialize)]
struct DeepgramWord {
    word: String,
    #[serde(default)]
    punctuated_word: Option<String>,
    start: f64,
    end: f64,
    #[serde(default)]
    confidence: Option<f64>,
}

fn policy_warnings(decision: &ModelPolicyDecision) -> Vec<Warning> {
    decision
        .advisories()
        .iter()
        .map(|advisory| match advisory {
            ModelAdvisory::UnknownModel => Warning::new(
                WarningKind::UnknownModel,
                "model is absent from the verified Deepgram advisory catalog",
            ),
            ModelAdvisory::Deprecated { .. } => {
                Warning::new(WarningKind::DeprecatedModel, "Deepgram model is deprecated")
            }
            ModelAdvisory::Retired { .. } => {
                Warning::new(WarningKind::RetiredModel, "Deepgram model is retired")
            }
            ModelAdvisory::RollingAlias => Warning::new(
                WarningKind::RollingModelAlias,
                "Deepgram model ID is a rolling alias",
            ),
            _ => Warning::provider(
                "model_advisory",
                "Deepgram policy returned a model advisory",
            ),
        })
        .collect()
}

fn option_error(source: ProviderOptionError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "Deepgram transcription options are invalid",
    )
    .with_source(source)
}

fn request_build_error(source: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "Deepgram transcription request violates the transport contract",
    )
    .with_source(source)
}

fn provider_response_error(response: TransportResponse) -> Error {
    let (status, headers, body) = response.into_parts();
    let parsed = serde_json::from_slice::<DeepgramErrorResponse>(&body).ok();
    let request_id = parsed
        .as_ref()
        .and_then(DeepgramErrorResponse::request_id)
        .map(str::to_owned)
        .or_else(|| response_request_id(&headers));
    let provider_code = parsed
        .as_ref()
        .and_then(DeepgramErrorResponse::provider_code);
    let provider_type = parsed
        .as_ref()
        .and_then(DeepgramErrorResponse::provider_type);
    let mut diagnostics = headers
        .diagnostics()
        .with_status(status.as_u16())
        .with_body_truncated(body.len() > SENSITIVE_BODY_LIMIT);
    if let Some(value) = request_id.and_then(public_text) {
        diagnostics = diagnostics.with_request_id(value);
    }
    if let Some(value) = provider_code.and_then(public_text) {
        diagnostics = diagnostics.with_provider_code(value);
    }
    if let Some(value) = provider_type.and_then(public_text) {
        diagnostics = diagnostics.with_provider_type(value);
    }
    if let Some(seconds) = retry_after(&headers) {
        diagnostics = diagnostics.with_retry_after(Duration::from_secs(seconds));
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
    Error::new(status_error_kind(status), "Deepgram rejected transcription")
        .with_diagnostics(diagnostics)
        .with_sensitive_response(SensitiveResponse::new(raw_headers, body.to_vec()))
}

#[derive(Deserialize)]
struct DeepgramErrorResponse {
    #[serde(default)]
    error: Option<DeepgramNestedError>,
    #[serde(default)]
    err_code: Option<Value>,
    #[serde(default)]
    request_id: Option<String>,
}

impl DeepgramErrorResponse {
    fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    fn provider_code(&self) -> Option<String> {
        self.err_code.as_ref().and_then(value_text).or_else(|| {
            self.error
                .as_ref()
                .and_then(|error| error.code.as_ref())
                .and_then(value_text)
        })
    }

    fn provider_type(&self) -> Option<String> {
        self.error.as_ref().and_then(|error| error.kind.clone())
    }
}

#[derive(Deserialize)]
struct DeepgramNestedError {
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

fn checked_response_identifier(value: Option<String>) -> Result<Option<String>, Error> {
    value
        .map(|value| {
            PublicDiagnosticText::new(value.clone())
                .map(|_| value)
                .map_err(|source| {
                    Error::protocol_violation(
                        "Deepgram response contains an invalid request identifier",
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
    ["x-request-id", "dg-request-id", "request-id"]
        .into_iter()
        .find_map(|name| {
            let name = HeaderName::from_static(name);
            headers
                .get(&name)
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned)
        })
}

fn retry_after(headers: &ResponseHeaders) -> Option<u64> {
    headers
        .get(&RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse().ok())
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use bytes::Bytes;
    use mockito::Matcher;
    use siumai_core::{
        Cancellation, ErrorDetail, ProviderOptions, ResourceKind, TranscriptionModel as _,
        UsageValue,
    };
    use siumai_transport::{EndpointConfig, TransportLimits};

    use crate::{DeepgramCredential, DeepgramProvider};

    use super::*;

    fn provider_at(base_url: &str) -> DeepgramProvider {
        DeepgramProvider::builder(DeepgramCredential::api_key("test-key"))
            .with_endpoint(EndpointConfig::local_explicit(base_url).unwrap())
            .build()
            .unwrap()
    }

    fn response_fixture(duration: Option<f64>) -> String {
        let metadata = duration.map_or_else(
            || serde_json::json!({"request_id": "request-1"}),
            |duration| serde_json::json!({"request_id": "request-1", "duration": duration}),
        );
        serde_json::json!({
            "metadata": metadata,
            "results": {
                "channels": [{
                    "detected_language": "en",
                    "alternatives": [{
                        "transcript": "hello world",
                        "confidence": 0.98,
                        "words": [
                            {"word": "hello", "punctuated_word": "Hello", "start": 0.0, "end": 0.4, "confidence": 0.97},
                            {"word": "world", "punctuated_word": "world.", "start": 0.4, "end": 0.8, "confidence": 0.99}
                        ]
                    }]
                }]
            }
        })
        .to_string()
    }

    #[tokio::test]
    async fn direct_and_registered_models_share_the_authenticated_wire_contract() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/listen")
            .match_header("authorization", "Token test-key")
            .match_header("content-type", "audio/wav")
            .match_header("accept", "application/json")
            .match_query("model=nova-3&language=en&smart_format=true&diarize_model=latest")
            .match_body("owned-audio")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(response_fixture(Some(0.8)))
            .expect(2)
            .create_async()
            .await;
        let provider = provider_at(&server.url());
        let provider_options = ProviderOptions::typed(
            &DeepgramTranscriptionOptions::new()
                .with_smart_format(true)
                .with_diarize_model(DeepgramDiarizeModel::Latest),
        )
        .unwrap();
        let request = TranscriptionRequest::new(Bytes::from_static(b"owned-audio"), "audio/wav")
            .unwrap()
            .with_language("en")
            .unwrap();
        let call = CallOptions::default().with_provider_options(provider_options);

        let direct = provider.transcription("nova-3").unwrap();
        let direct_response = direct
            .transcribe(request.clone(), call.clone())
            .await
            .unwrap();
        let erased = provider
            .registration()
            .transcription_model(ModelId::new("nova-3").unwrap())
            .unwrap();
        let erased_response = erased.transcribe(request, call).await.unwrap();

        assert_eq!(direct_response, erased_response);
        assert_eq!(direct_response.text, "hello world");
        assert_eq!(direct_response.segments[0].text, "Hello");
        assert_eq!(
            direct_response.metadata.request_id.as_deref(),
            Some("request-1")
        );
        assert_eq!(direct_response.duration_seconds, Some(0.8));
        assert_eq!(
            direct_response.usage.provider["audio_duration_seconds"],
            serde_json::json!(0.8)
        );
        mock.assert_async().await;
    }

    #[test]
    fn empty_audio_is_rejected_before_a_model_call() {
        let error = TranscriptionRequest::new(Bytes::new(), "audio/wav").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }

    #[tokio::test]
    async fn model_limit_and_unsupported_prompt_are_typed() {
        let limits = TransportLimits {
            max_request_bytes: 4,
            ..TransportLimits::default()
        };
        let provider = DeepgramProvider::builder(DeepgramCredential::unauthenticated())
            .with_endpoint(EndpointConfig::local_explicit("http://127.0.0.1:9").unwrap())
            .with_limits(limits)
            .build()
            .unwrap();
        let model = provider.transcription("nova-3").unwrap();
        let error = model
            .transcribe(
                TranscriptionRequest::new(Bytes::from_static(b"12345"), "audio/wav").unwrap(),
                CallOptions::default(),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::LimitExceeded);
        assert!(matches!(
            error.detail(),
            Some(ErrorDetail::LimitExceeded {
                resource: ResourceKind::TranscriptionAudioBytes,
                actual: 5,
                maximum: 4
            })
        ));

        let error = model
            .transcribe(
                TranscriptionRequest::new(Bytes::from_static(b"1234"), "audio/wav")
                    .unwrap()
                    .with_prompt("names: Ada")
                    .unwrap(),
                CallOptions::default(),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(error.context().operation, Some(ModelOperation::Transcribe));
    }

    #[tokio::test]
    async fn absent_duration_preserves_unknown_usage() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/listen")
            .match_query("model=nova-3")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(response_fixture(None))
            .expect(1)
            .create_async()
            .await;
        let response = provider_at(&server.url())
            .transcription("nova-3")
            .unwrap()
            .transcribe(
                TranscriptionRequest::new(Bytes::from_static(b"audio"), "audio/wav").unwrap(),
                CallOptions::default(),
            )
            .await
            .unwrap();

        assert_eq!(response.duration_seconds, None);
        assert_eq!(response.usage.audio_input_tokens, UsageValue::Unknown);
        assert!(
            !response
                .usage
                .provider
                .contains_key("audio_duration_seconds")
        );
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn cancellation_and_deadline_reach_the_shared_transport() {
        let provider = provider_at("http://127.0.0.1:9");
        let model = provider.transcription("nova-3").unwrap();
        let request = TranscriptionRequest::new(Bytes::from_static(b"audio"), "audio/wav").unwrap();
        let cancellation = Cancellation::new();
        cancellation.cancel();
        let error = model
            .transcribe(
                request.clone(),
                CallOptions::default().with_cancellation(cancellation),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Cancelled);
        assert_eq!(
            error.context().provider.as_ref().map(|id| id.as_str()),
            Some("deepgram")
        );

        let error = model
            .transcribe(
                request,
                CallOptions::default().with_deadline(Instant::now()),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Timeout);
    }

    #[tokio::test]
    async fn provider_errors_are_typed_bounded_and_redacted_by_default() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/listen")
            .match_query("model=nova-3")
            .with_status(400)
            .with_header("content-type", "application/json")
            .with_header("x-request-id", "request-400")
            .with_header("retry-after", "7")
            .with_header("x-secret", "canary-header-secret")
            .with_body(
                r#"{"err_code":"INVALID_AUDIO","err_msg":"canary-body-secret","request_id":"request-400"}"#,
            )
            .expect(1)
            .create_async()
            .await;
        let error = provider_at(&server.url())
            .transcription("nova-3")
            .unwrap()
            .transcribe(
                TranscriptionRequest::new(Bytes::from_static(b"audio"), "audio/wav").unwrap(),
                CallOptions::default(),
            )
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Provider);
        let diagnostics = error.diagnostics().unwrap();
        assert_eq!(diagnostics.status(), Some(400));
        assert_eq!(diagnostics.provider_code(), Some("INVALID_AUDIO"));
        assert_eq!(diagnostics.request_id(), Some("request-400"));
        assert_eq!(diagnostics.retry_after(), Some(Duration::from_secs(7)));
        let debug = format!("{error:?}");
        assert!(!debug.contains("canary-header-secret"));
        assert!(!debug.contains("canary-body-secret"));
        assert!(
            std::str::from_utf8(error.sensitive_response().unwrap().expose().1)
                .unwrap()
                .contains("canary-body-secret")
        );
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn non_idempotent_audio_post_is_never_replayed() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/listen")
            .match_query("model=nova-3")
            .with_status(500)
            .with_header("content-type", "application/json")
            .with_body(r#"{"err_code":"UNAVAILABLE"}"#)
            .expect(1)
            .create_async()
            .await;
        let error = provider_at(&server.url())
            .transcription("nova-3")
            .unwrap()
            .transcribe(
                TranscriptionRequest::new(Bytes::from_static(b"audio"), "audio/wav").unwrap(),
                CallOptions::default(),
            )
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Provider);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn request_bytes_outlive_the_source_buffer_and_cross_a_task_boundary() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/listen")
            .match_query("model=future-model")
            .match_body(Matcher::Exact("owned-copy".to_string()))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(response_fixture(Some(0.8)))
            .expect(1)
            .create_async()
            .await;
        let source = b"owned-copy".to_vec();
        let audio = Bytes::copy_from_slice(&source);
        drop(source);
        let model = provider_at(&server.url())
            .transcription("future-model")
            .unwrap();
        let response = tokio::spawn(async move {
            model
                .transcribe(
                    TranscriptionRequest::new(audio, "audio/wav").unwrap(),
                    CallOptions::default(),
                )
                .await
        })
        .await
        .unwrap()
        .unwrap();

        assert!(matches!(
            response.warnings[0].kind(),
            WarningKind::UnknownModel
        ));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn malformed_success_is_a_protocol_error() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/listen")
            .match_query("model=nova-3")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"results":{"channels":[]}}"#)
            .expect(1)
            .create_async()
            .await;
        let error = provider_at(&server.url())
            .transcription("nova-3")
            .unwrap()
            .transcribe(
                TranscriptionRequest::new(Bytes::from_static(b"audio"), "audio/wav").unwrap(),
                CallOptions::default(),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ProtocolViolation);
        mock.assert_async().await;
    }

    #[test]
    fn native_query_encoding_preserves_repeated_redaction_values() {
        let request = TranscriptionRequest::new(Bytes::from_static(b"audio"), "audio/wav").unwrap();
        let options = DeepgramTranscriptionOptions::new()
            .with_redactions(["pci", "numbers"])
            .with_diarize_model(DeepgramDiarizeModel::V1);

        assert_eq!(
            build_target(&ModelId::new("nova-3").unwrap(), &request, &options),
            "v1/listen?model=nova-3&redact=pci&redact=numbers&diarize_model=v1"
        );
    }
}
