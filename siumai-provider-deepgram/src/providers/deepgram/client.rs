use super::config::DeepgramConfig;
use super::options::{
    DeepgramSpeechModelOptions, DeepgramSummarizeOption, DeepgramTranscriptionModelOptions,
};
use crate::core_compat::client::LlmClient;
use crate::error::LlmError;
use crate::execution::http::headers::headermap_to_hashmap;
use crate::execution::http::interceptor::{HttpRequestContext, generate_request_id};
use crate::execution::http::transport::{
    HttpTransportMultipartRequest, HttpTransportRequest, HttpTransportResponse,
};
use crate::retry_api::RetryOptions;
use crate::traits::{
    AudioCapability, ModelMetadata, ProviderCapabilities, SpeechCapability, TranscriptionCapability,
};
use crate::types::{
    AudioFeature, HttpConfig, HttpRequestInfo, HttpResponseInfo, SttRequest, SttResponse,
    TtsRequest, TtsResponse, Warning, WordTimestamp, provider_metadata_from_object,
};
use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, CONTENT_LENGTH, CONTENT_TYPE, HeaderMap, HeaderValue};
use secrecy::ExposeSecret;
use serde::Deserialize;
use serde_json::{Map, Value};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

const PROVIDER_ID: &str = "deepgram";
const FEATURES: &[AudioFeature] = &[
    AudioFeature::TextToSpeech,
    AudioFeature::SpeechToText,
    AudioFeature::SpeakerDiarization,
];

/// Provider-owned Deepgram client.
#[derive(Clone)]
pub struct DeepgramClient {
    config: DeepgramConfig,
    http_client: reqwest::Client,
    retry_options: Option<RetryOptions>,
}

impl std::fmt::Debug for DeepgramClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeepgramClient")
            .field("config", &self.config)
            .field("retry_options", &self.retry_options)
            .finish()
    }
}

impl DeepgramClient {
    pub fn from_config(config: DeepgramConfig) -> Result<Self, LlmError> {
        config.validate()?;
        let http_client =
            crate::execution::http::client::build_http_client_from_config(&config.http_config)?;
        Self::with_http_client(config, http_client)
    }

    pub fn with_http_client(
        config: DeepgramConfig,
        http_client: reqwest::Client,
    ) -> Result<Self, LlmError> {
        config.validate()?;
        Ok(Self {
            config,
            http_client,
            retry_options: None,
        })
    }

    pub fn with_retry_options(mut self, retry_options: RetryOptions) -> Self {
        self.retry_options = Some(retry_options);
        self
    }

    pub fn with_retry(self, retry_options: RetryOptions) -> Self {
        self.with_retry_options(retry_options)
    }

    pub fn set_retry_options(&mut self, options: Option<RetryOptions>) {
        self.retry_options = options;
    }

    pub fn retry_options(&self) -> Option<RetryOptions> {
        self.retry_options.clone()
    }

    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    pub fn http_client(&self) -> reqwest::Client {
        self.http_client.clone()
    }

    pub fn http_transport(
        &self,
    ) -> Option<Arc<dyn crate::execution::http::transport::HttpTransport>> {
        self.config.http_transport.clone()
    }

    pub fn speech_model(&self, model_id: impl Into<String>) -> DeepgramSpeechModel {
        DeepgramSpeechModel {
            client: self.clone(),
            model_id: model_id.into(),
        }
    }

    pub fn default_speech_model(&self) -> DeepgramSpeechModel {
        self.speech_model(self.config.speech_model.clone())
    }

    pub fn transcription_model(&self, model_id: impl Into<String>) -> DeepgramTranscriptionModel {
        DeepgramTranscriptionModel {
            client: self.clone(),
            model_id: model_id.into(),
        }
    }

    pub fn default_transcription_model(&self) -> DeepgramTranscriptionModel {
        self.transcription_model(self.config.transcription_model.clone())
    }

    fn endpoint_url(&self, endpoint: &str, query: &BTreeMap<String, String>) -> String {
        let base = crate::provider_utils::join_url(&self.config.base_url, endpoint);
        crate::provider_utils::with_query_params(&base, query)
    }

    fn request_context(&self, url: &str) -> HttpRequestContext {
        HttpRequestContext {
            request_id: generate_request_id(),
            provider_id: PROVIDER_ID.to_string(),
            url: url.to_string(),
            stream: false,
        }
    }

    fn auth_headers(&self, content_type: Option<&str>) -> Result<HeaderMap, LlmError> {
        let mut headers = HeaderMap::new();
        let auth = format!("Token {}", self.config.api_key.expose_secret());
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&auth).map_err(|e| {
                LlmError::ConfigurationError(format!("Invalid Deepgram API key format: {e}"))
            })?,
        );
        if let Some(content_type) = content_type {
            headers.insert(
                CONTENT_TYPE,
                HeaderValue::from_str(content_type).map_err(|e| {
                    LlmError::ConfigurationError(format!(
                        "Invalid Deepgram content type '{content_type}': {e}"
                    ))
                })?,
            );
        }
        apply_extra_headers(&mut headers, &self.config.http_config.headers)?;
        Ok(headers)
    }

    fn merge_request_headers(
        &self,
        mut headers: HeaderMap,
        http_config: Option<&HttpConfig>,
    ) -> Result<HeaderMap, LlmError> {
        if let Some(http_config) = http_config {
            apply_extra_headers(&mut headers, &http_config.headers)?;
        }
        Ok(headers)
    }

    async fn text_to_speech_once(&self, request: TtsRequest) -> Result<TtsResponse, LlmError> {
        let model = request_model_or_default(request.model.as_deref(), &self.config.speech_model)?;
        let mut query = BTreeMap::from([("model".to_string(), model.clone())]);
        let mut warnings = Vec::new();

        apply_output_format_query(request.format.as_deref(), &mut query);
        apply_speech_provider_options(&request, &mut query)?;
        warnings.extend(speech_warnings(&request, &model));

        let url = self.endpoint_url("v1/speak", &query);
        let body = serde_json::json!({ "text": request.text });
        let headers = self.merge_request_headers(
            self.auth_headers(Some("application/json"))?,
            request.http_config.as_ref(),
        )?;
        let response = self
            .execute_json_bytes(&url, headers, body.clone(), request.http_config.as_ref())
            .await?;

        let content_type_format = response_format_from_headers(&response.headers)
            .or_else(|| request.format.clone())
            .unwrap_or_else(|| "mp3".to_string());
        let response_info = build_response_info(&response.headers, Some(&model), None);
        let request_info = serde_json::to_string(&body)
            .ok()
            .map(|body| HttpRequestInfo { body: Some(body) });

        Ok(TtsResponse {
            audio_data: response.body,
            format: content_type_format,
            duration: None,
            sample_rate: None,
            metadata: HashMap::new(),
            warnings: non_empty_warnings(warnings),
            provider_metadata: None,
            request: request_info,
            response: Some(response_info),
        })
    }

    async fn speech_to_text_once(&self, request: SttRequest) -> Result<SttResponse, LlmError> {
        let model =
            request_model_or_default(request.model.as_deref(), &self.config.transcription_model)?;
        let mut query = BTreeMap::from([
            ("model".to_string(), model.clone()),
            ("diarize".to_string(), "true".to_string()),
        ]);
        apply_transcription_provider_options(&request, &mut query)?;

        let url = self.endpoint_url("v1/listen", &query);
        let audio = request
            .audio_bytes()
            .map_err(|e| LlmError::InvalidInput(format!("Invalid Deepgram audio input: {e}")))?;
        let mut headers = self.auth_headers(Some(&request.media_type))?;
        headers.insert(
            CONTENT_LENGTH,
            HeaderValue::from_str(&audio.len().to_string()).map_err(|e| {
                LlmError::ConfigurationError(format!("Invalid content length: {e}"))
            })?,
        );
        let headers = self.merge_request_headers(headers, request.http_config.as_ref())?;
        let response = self
            .execute_raw_audio_json(&url, headers, audio, request.http_config.as_ref())
            .await?;

        let parsed: DeepgramListenResponse =
            serde_json::from_slice(&response.body).map_err(|e| {
                LlmError::ParseError(format!("Invalid Deepgram transcription response: {e}"))
            })?;
        let raw_json = serde_json::from_slice::<Value>(&response.body).map_err(|e| {
            LlmError::ParseError(format!("Invalid Deepgram transcription JSON: {e}"))
        })?;

        let text = parsed.transcript().unwrap_or_default();
        let language = parsed.detected_language();
        let words = parsed.words();
        let duration = parsed
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.duration);
        let confidence = parsed.confidence();
        let provider_metadata = provider_metadata_from_object(
            PROVIDER_ID,
            Map::from_iter([
                ("raw".to_string(), raw_json.clone()),
                ("model".to_string(), serde_json::json!(model)),
            ]),
        );
        let response_info =
            build_response_info(&response.headers, Some(&model), Some(raw_json.clone()));

        Ok(SttResponse {
            text,
            language,
            confidence,
            words,
            duration,
            metadata: HashMap::from([("raw".to_string(), raw_json)]),
            warnings: None,
            provider_metadata: Some(provider_metadata),
            request: Some(HttpRequestInfo {
                body: Some(format!("<{} bytes audio>", response.request_body_size)),
            }),
            response: Some(response_info),
        })
    }

    async fn execute_json_bytes(
        &self,
        url: &str,
        headers: HeaderMap,
        body: Value,
        http_config: Option<&HttpConfig>,
    ) -> Result<DeepgramRawResponse, LlmError> {
        let run_once = || async {
            let ctx = self.request_context(url);
            let mut builder = self
                .http_client
                .post(url)
                .headers(headers.clone())
                .json(&body);
            if let Some(timeout) = http_config.and_then(|config| config.timeout) {
                builder = builder.timeout(timeout);
            }

            let mut observed_headers = headers.clone();
            if let Some(cloned) = builder.try_clone().and_then(|request| {
                request
                    .build()
                    .ok()
                    .map(|request| request.headers().clone())
            }) {
                observed_headers = cloned;
            }
            for interceptor in &self.config.http_interceptors {
                builder = interceptor.on_before_send(&ctx, builder, &body, &observed_headers)?;
            }

            if let Some(transport) = &self.config.http_transport {
                let response = transport
                    .execute_json(HttpTransportRequest {
                        ctx: ctx.clone(),
                        url: url.to_string(),
                        headers: observed_headers,
                        body: body.clone(),
                    })
                    .await?;
                classify_transport_response(&ctx, &self.config.http_interceptors, response).map(
                    |response| DeepgramRawResponse {
                        headers: response.headers,
                        body: response.body,
                        request_body_size: body.to_string().len(),
                    },
                )
            } else {
                let response = builder
                    .send()
                    .await
                    .map_err(|e| LlmError::HttpError(e.to_string()))?;
                classify_reqwest_response(
                    PROVIDER_ID,
                    &ctx,
                    &self.config.http_interceptors,
                    response,
                    body.to_string().len(),
                )
                .await
            }
        };

        crate::retry_api::maybe_retry(self.retry_options.clone(), run_once).await
    }

    async fn execute_raw_audio_json(
        &self,
        url: &str,
        headers: HeaderMap,
        audio: Vec<u8>,
        http_config: Option<&HttpConfig>,
    ) -> Result<DeepgramRawResponse, LlmError> {
        let run_once = || {
            let audio = audio.clone();
            let headers = headers.clone();
            async move {
                let ctx = self.request_context(url);
                let mut builder = self
                    .http_client
                    .post(url)
                    .headers(headers.clone())
                    .body(audio.clone());
                if let Some(timeout) = http_config.and_then(|config| config.timeout) {
                    builder = builder.timeout(timeout);
                }

                let body_for_interceptors = serde_json::json!({
                    "audioBytes": audio.len()
                });
                let mut observed_headers = headers.clone();
                if let Some(cloned) = builder.try_clone().and_then(|request| {
                    request
                        .build()
                        .ok()
                        .map(|request| request.headers().clone())
                }) {
                    observed_headers = cloned;
                }
                for interceptor in &self.config.http_interceptors {
                    builder = interceptor.on_before_send(
                        &ctx,
                        builder,
                        &body_for_interceptors,
                        &observed_headers,
                    )?;
                }

                if let Some(transport) = &self.config.http_transport {
                    let response = transport
                        .execute_multipart(HttpTransportMultipartRequest {
                            ctx: ctx.clone(),
                            url: url.to_string(),
                            headers: observed_headers,
                            body: audio.clone(),
                        })
                        .await?;
                    classify_transport_response(&ctx, &self.config.http_interceptors, response).map(
                        |response| DeepgramRawResponse {
                            headers: response.headers,
                            body: response.body,
                            request_body_size: audio.len(),
                        },
                    )
                } else {
                    let response = builder
                        .send()
                        .await
                        .map_err(|e| LlmError::HttpError(e.to_string()))?;
                    classify_reqwest_response(
                        PROVIDER_ID,
                        &ctx,
                        &self.config.http_interceptors,
                        response,
                        audio.len(),
                    )
                    .await
                }
            }
        };

        crate::retry_api::maybe_retry(self.retry_options.clone(), run_once).await
    }
}

#[async_trait]
impl AudioCapability for DeepgramClient {
    fn supported_features(&self) -> &[AudioFeature] {
        FEATURES
    }

    async fn text_to_speech(&self, request: TtsRequest) -> Result<TtsResponse, LlmError> {
        self.text_to_speech_once(request).await
    }

    async fn speech_to_text(&self, request: SttRequest) -> Result<SttResponse, LlmError> {
        self.speech_to_text_once(request).await
    }

    fn get_supported_audio_formats(&self) -> Vec<String> {
        [
            "mp3", "wav", "linear16", "mulaw", "alaw", "opus", "ogg", "flac", "aac", "pcm",
        ]
        .into_iter()
        .map(str::to_string)
        .collect()
    }
}

impl ModelMetadata for DeepgramClient {
    fn provider_id(&self) -> &str {
        PROVIDER_ID
    }

    fn model_id(&self) -> &str {
        &self.config.transcription_model
    }
}

impl LlmClient for DeepgramClient {
    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed(PROVIDER_ID)
    }

    fn supported_models(&self) -> Vec<String> {
        super::models::ALL_SPEECH
            .iter()
            .chain(super::models::ALL_TRANSCRIPTION.iter())
            .map(|model| (*model).to_string())
            .collect()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_audio()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_audio_capability(&self) -> Option<&dyn AudioCapability> {
        Some(self)
    }

    fn as_speech_capability(&self) -> Option<&dyn SpeechCapability> {
        Some(self)
    }

    fn as_transcription_capability(&self) -> Option<&dyn TranscriptionCapability> {
        Some(self)
    }
}

/// Speech-family Deepgram model object.
#[derive(Clone, Debug)]
pub struct DeepgramSpeechModel {
    client: DeepgramClient,
    model_id: String,
}

#[async_trait]
impl SpeechCapability for DeepgramSpeechModel {
    async fn tts(&self, request: TtsRequest) -> Result<TtsResponse, LlmError> {
        self.client
            .text_to_speech_once(request.with_model_if_missing(self.model_id.clone()))
            .await
    }
}

impl ModelMetadata for DeepgramSpeechModel {
    fn provider_id(&self) -> &str {
        PROVIDER_ID
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

/// Transcription-family Deepgram model object.
#[derive(Clone, Debug)]
pub struct DeepgramTranscriptionModel {
    client: DeepgramClient,
    model_id: String,
}

#[async_trait]
impl TranscriptionCapability for DeepgramTranscriptionModel {
    async fn stt(&self, request: SttRequest) -> Result<SttResponse, LlmError> {
        self.client
            .speech_to_text_once(request.with_model_if_missing(self.model_id.clone()))
            .await
    }
}

impl ModelMetadata for DeepgramTranscriptionModel {
    fn provider_id(&self) -> &str {
        PROVIDER_ID
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[derive(Debug)]
struct DeepgramRawResponse {
    headers: HeaderMap,
    body: Vec<u8>,
    request_body_size: usize,
}

fn request_model_or_default(
    request_model: Option<&str>,
    default: &str,
) -> Result<String, LlmError> {
    let model = request_model.unwrap_or(default).trim();
    if model.is_empty() {
        return Err(LlmError::ConfigurationError(
            "Deepgram request requires a non-empty model id".to_string(),
        ));
    }
    Ok(model.to_string())
}

fn apply_extra_headers(
    headers: &mut HeaderMap,
    extra: &HashMap<String, String>,
) -> Result<(), LlmError> {
    for (name, value) in extra {
        let name = reqwest::header::HeaderName::from_bytes(name.as_bytes()).map_err(|e| {
            LlmError::ConfigurationError(format!("Invalid Deepgram header name '{name}': {e}"))
        })?;
        let value = HeaderValue::from_str(value).map_err(|e| {
            LlmError::ConfigurationError(format!("Invalid Deepgram header value for '{name}': {e}"))
        })?;
        headers.insert(name, value);
    }
    Ok(())
}

fn non_empty_warnings(warnings: Vec<Warning>) -> Option<Vec<Warning>> {
    (!warnings.is_empty()).then_some(warnings)
}

fn speech_warnings(request: &TtsRequest, model: &str) -> Vec<Warning> {
    let mut warnings = Vec::new();
    if request
        .voice
        .as_deref()
        .is_some_and(|voice| voice.trim() != model)
    {
        warnings.push(Warning::unsupported(
            "voice",
            Some("Deepgram speech voices are encoded in the model id"),
        ));
    }
    if request.speed.is_some() {
        warnings.push(Warning::unsupported(
            "speed",
            Some("Deepgram speech does not support speed controls"),
        ));
    }
    if request.language.is_some() {
        warnings.push(Warning::unsupported(
            "language",
            Some("Deepgram speech language is encoded in the model id"),
        ));
    }
    if request.instructions.is_some() {
        warnings.push(Warning::unsupported(
            "instructions",
            Some("Deepgram speech does not support instructions"),
        ));
    }
    warnings
}

fn apply_output_format_query(format: Option<&str>, query: &mut BTreeMap<String, String>) {
    let Some(format) = format.map(str::trim).filter(|format| !format.is_empty()) else {
        return;
    };
    let (base, sample_rate) = split_format_and_sample_rate(format);

    match base {
        "mp3" => {
            query.insert("encoding".to_string(), "mp3".to_string());
        }
        "wav" => {
            query.insert("container".to_string(), "wav".to_string());
            query.insert("encoding".to_string(), "linear16".to_string());
        }
        "linear16" => {
            query.insert("encoding".to_string(), "linear16".to_string());
            query.insert("container".to_string(), "wav".to_string());
        }
        "mulaw" => {
            query.insert("encoding".to_string(), "mulaw".to_string());
            query.insert("container".to_string(), "wav".to_string());
        }
        "alaw" => {
            query.insert("encoding".to_string(), "alaw".to_string());
            query.insert("container".to_string(), "wav".to_string());
        }
        "opus" | "ogg" => {
            query.insert("encoding".to_string(), "opus".to_string());
            query.insert("container".to_string(), "ogg".to_string());
        }
        "flac" => {
            query.insert("encoding".to_string(), "flac".to_string());
        }
        "aac" => {
            query.insert("encoding".to_string(), "aac".to_string());
        }
        "pcm" => {
            query.insert("encoding".to_string(), "linear16".to_string());
            query.insert("container".to_string(), "none".to_string());
        }
        _ => {
            query.insert("encoding".to_string(), base.to_string());
        }
    }

    if let Some(sample_rate) = sample_rate {
        query.insert("sample_rate".to_string(), sample_rate.to_string());
    }
}

fn split_format_and_sample_rate(format: &str) -> (&str, Option<u32>) {
    let Some((base, suffix)) = format.rsplit_once('_') else {
        return (format, None);
    };
    if let Ok(sample_rate) = suffix.parse::<u32>() {
        (base, Some(sample_rate))
    } else {
        (format, None)
    }
}

fn apply_speech_provider_options(
    request: &TtsRequest,
    query: &mut BTreeMap<String, String>,
) -> Result<(), LlmError> {
    let Some(options) = request.provider_options_map.get("deepgram") else {
        return Ok(());
    };
    let options: DeepgramSpeechModelOptions =
        serde_json::from_value(options.clone()).map_err(|e| {
            LlmError::InvalidParameter(format!("Invalid Deepgram speech provider options: {e}"))
        })?;
    insert_option(query, "encoding", options.encoding);
    insert_option(query, "container", options.container);
    insert_option(
        query,
        "sample_rate",
        options.sample_rate.map(|value| value.to_string()),
    );
    insert_option(
        query,
        "bit_rate",
        options.bit_rate.map(|value| value.to_string()),
    );
    insert_option(query, "callback", options.callback);
    insert_option(query, "callback_method", options.callback_method);
    insert_option(
        query,
        "mip_opt_out",
        options.mip_opt_out.map(|value| value.to_string()),
    );
    insert_option(query, "tag", options.tag);
    Ok(())
}

fn apply_transcription_provider_options(
    request: &SttRequest,
    query: &mut BTreeMap<String, String>,
) -> Result<(), LlmError> {
    let Some(options) = request.provider_options_map.get("deepgram") else {
        return Ok(());
    };
    let options: DeepgramTranscriptionModelOptions = serde_json::from_value(options.clone())
        .map_err(|e| {
            LlmError::InvalidParameter(format!(
                "Invalid Deepgram transcription provider options: {e}"
            ))
        })?;

    insert_option(query, "language", options.language);
    insert_option(
        query,
        "detect_language",
        options.detect_language.map(|value| value.to_string()),
    );
    insert_option(
        query,
        "smart_format",
        options.smart_format.map(|value| value.to_string()),
    );
    insert_option(
        query,
        "punctuate",
        options.punctuate.map(|value| value.to_string()),
    );
    insert_option(
        query,
        "paragraphs",
        options.paragraphs.map(|value| value.to_string()),
    );
    if let Some(value) = options.summarize {
        let value = match value {
            DeepgramSummarizeOption::Bool(value) => value.to_string(),
            DeepgramSummarizeOption::Version(value) => value,
        };
        query.insert("summarize".to_string(), value);
    }
    insert_option(
        query,
        "topics",
        options.topics.map(|value| value.to_string()),
    );
    insert_option(
        query,
        "intents",
        options.intents.map(|value| value.to_string()),
    );
    insert_option(
        query,
        "sentiment",
        options.sentiment.map(|value| value.to_string()),
    );
    insert_option(
        query,
        "detect_entities",
        options.detect_entities.map(|value| value.to_string()),
    );
    insert_joined(query, "redact", options.redact);
    insert_joined(query, "replace", options.replace);
    insert_joined(query, "search", options.search);
    insert_joined(query, "keyterm", options.keyterm);
    insert_option(
        query,
        "diarize",
        options.diarize.map(|value| value.to_string()),
    );
    insert_option(
        query,
        "utterances",
        options.utterances.map(|value| value.to_string()),
    );
    insert_option(
        query,
        "utt_split",
        options.utt_split.map(|value| value.to_string()),
    );
    insert_option(
        query,
        "filler_words",
        options.filler_words.map(|value| value.to_string()),
    );
    Ok(())
}

fn insert_option(query: &mut BTreeMap<String, String>, key: &str, value: Option<String>) {
    if let Some(value) = value {
        query.insert(key.to_string(), value);
    }
}

fn insert_joined(query: &mut BTreeMap<String, String>, key: &str, value: Option<Vec<String>>) {
    if let Some(value) = value {
        query.insert(key.to_string(), value.join(","));
    }
}

fn classify_transport_response(
    ctx: &HttpRequestContext,
    interceptors: &[Arc<dyn crate::execution::http::interceptor::HttpInterceptor>],
    response: HttpTransportResponse,
) -> Result<HttpTransportResponse, LlmError> {
    if (200..300).contains(&response.status) {
        return Ok(response);
    }

    let text = String::from_utf8_lossy(&response.body);
    let fallback = reqwest::StatusCode::from_u16(response.status)
        .ok()
        .and_then(|status| status.canonical_reason());
    let error = crate::retry_api::classify_http_error(
        PROVIDER_ID,
        response.status,
        &text,
        &response.headers,
        fallback,
    );
    for interceptor in interceptors {
        interceptor.on_error(ctx, &error);
    }
    Err(error)
}

async fn classify_reqwest_response(
    provider_id: &str,
    ctx: &HttpRequestContext,
    interceptors: &[Arc<dyn crate::execution::http::interceptor::HttpInterceptor>],
    response: reqwest::Response,
    request_body_size: usize,
) -> Result<DeepgramRawResponse, LlmError> {
    if !response.status().is_success() {
        let status = response.status();
        let headers = response.headers().clone();
        let text = response.text().await.unwrap_or_default();
        let error = crate::retry_api::classify_http_error(
            provider_id,
            status.as_u16(),
            &text,
            &headers,
            status.canonical_reason(),
        );
        for interceptor in interceptors {
            interceptor.on_error(ctx, &error);
        }
        return Err(error);
    }

    for interceptor in interceptors {
        interceptor.on_response(ctx, &response)?;
    }
    let headers = response.headers().clone();
    let body = response
        .bytes()
        .await
        .map_err(|e| LlmError::HttpError(e.to_string()))?
        .to_vec();

    Ok(DeepgramRawResponse {
        headers,
        body,
        request_body_size,
    })
}

fn response_format_from_headers(headers: &HeaderMap) -> Option<String> {
    let content_type = headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())?;
    let mime = content_type.split(';').next()?.trim();
    match mime {
        "audio/mpeg" | "audio/mp3" => Some("mp3".to_string()),
        "audio/wav" | "audio/x-wav" => Some("wav".to_string()),
        "audio/ogg" => Some("ogg".to_string()),
        "audio/flac" => Some("flac".to_string()),
        "audio/aac" => Some("aac".to_string()),
        _ => None,
    }
}

fn build_response_info(
    headers: &HeaderMap,
    model_id: Option<&str>,
    body: Option<Value>,
) -> HttpResponseInfo {
    HttpResponseInfo {
        timestamp: chrono::Utc::now(),
        model_id: model_id
            .map(str::trim)
            .filter(|model_id| !model_id.is_empty())
            .map(ToOwned::to_owned),
        headers: headermap_to_hashmap(headers),
        body,
    }
}

#[derive(Debug, Deserialize)]
struct DeepgramListenResponse {
    metadata: Option<DeepgramMetadata>,
    results: Option<DeepgramResults>,
}

impl DeepgramListenResponse {
    fn first_alternative(&self) -> Option<&DeepgramAlternative> {
        self.results
            .as_ref()?
            .channels
            .as_ref()?
            .first()?
            .alternatives
            .as_ref()?
            .first()
    }

    fn transcript(&self) -> Option<String> {
        self.first_alternative()?.transcript.clone()
    }

    fn detected_language(&self) -> Option<String> {
        self.results
            .as_ref()?
            .channels
            .as_ref()?
            .first()?
            .detected_language
            .clone()
    }

    fn confidence(&self) -> Option<f32> {
        self.first_alternative()?.confidence
    }

    fn words(&self) -> Option<Vec<WordTimestamp>> {
        let words = self.first_alternative()?.words.as_ref()?;
        Some(
            words
                .iter()
                .filter_map(|word| {
                    Some(WordTimestamp {
                        word: word.word.clone()?,
                        start: word.start?,
                        end: word.end?,
                        confidence: word.confidence,
                    })
                })
                .collect(),
        )
        .filter(|words: &Vec<WordTimestamp>| !words.is_empty())
    }
}

#[derive(Debug, Deserialize)]
struct DeepgramMetadata {
    duration: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct DeepgramResults {
    channels: Option<Vec<DeepgramChannel>>,
}

#[derive(Debug, Deserialize)]
struct DeepgramChannel {
    alternatives: Option<Vec<DeepgramAlternative>>,
    detected_language: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DeepgramAlternative {
    transcript: Option<String>,
    confidence: Option<f32>,
    words: Option<Vec<DeepgramWord>>,
}

#[derive(Debug, Deserialize)]
struct DeepgramWord {
    word: Option<String>,
    start: Option<f32>,
    end: Option<f32>,
    confidence: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::deepgram::ext::{DeepgramSttRequestExt, DeepgramTtsRequestExt};
    use crate::providers::deepgram::models;
    use crate::providers::deepgram::options::{
        DeepgramSpeechModelOptions, DeepgramTranscriptionModelOptions,
    };
    use std::sync::Mutex;

    #[derive(Clone)]
    struct BytesSpeechTransport {
        body: Vec<u8>,
        content_type: &'static str,
        last_json: Arc<Mutex<Option<HttpTransportRequest>>>,
    }

    impl BytesSpeechTransport {
        fn new(body: Vec<u8>, content_type: &'static str) -> Self {
            Self {
                body,
                content_type,
                last_json: Arc::new(Mutex::new(None)),
            }
        }

        fn take_json(&self) -> Option<HttpTransportRequest> {
            self.last_json.lock().expect("json transport lock").take()
        }
    }

    #[async_trait]
    impl crate::execution::http::transport::HttpTransport for BytesSpeechTransport {
        async fn execute_json(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last_json.lock().expect("json transport lock") = Some(request);
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static(self.content_type));
            Ok(HttpTransportResponse {
                status: 200,
                headers,
                body: self.body.clone(),
            })
        }
    }

    #[derive(Clone)]
    struct JsonTranscriptionTransport {
        response: Value,
        last_raw: Arc<Mutex<Option<HttpTransportMultipartRequest>>>,
    }

    impl JsonTranscriptionTransport {
        fn new(response: Value) -> Self {
            Self {
                response,
                last_raw: Arc::new(Mutex::new(None)),
            }
        }

        fn take_raw(&self) -> Option<HttpTransportMultipartRequest> {
            self.last_raw.lock().expect("raw transport lock").take()
        }
    }

    #[async_trait]
    impl crate::execution::http::transport::HttpTransport for JsonTranscriptionTransport {
        async fn execute_json(
            &self,
            _request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "json requests are not expected in this test".to_string(),
            ))
        }

        async fn execute_multipart(
            &self,
            request: HttpTransportMultipartRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last_raw.lock().expect("raw transport lock") = Some(request);
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
            Ok(HttpTransportResponse {
                status: 200,
                headers,
                body: serde_json::to_vec(&self.response).expect("serialize response"),
            })
        }
    }

    fn query_param(url: &str, key: &str) -> Option<String> {
        let parsed = reqwest::Url::parse(url).expect("valid url");
        parsed
            .query_pairs()
            .find_map(|(name, value)| (name == key).then(|| value.into_owned()))
    }

    #[tokio::test]
    async fn deepgram_tts_posts_speak_json_with_token_auth_and_query_options() {
        let transport = BytesSpeechTransport::new(vec![1, 2, 3, 4], "audio/wav");
        let config = DeepgramConfig::new("test-key")
            .with_base_url("https://api.deepgram.test")
            .with_speech_model(models::speech::AURA_2_HELENA_EN)
            .with_http_transport(Arc::new(transport.clone()));
        let client = DeepgramClient::from_config(config).expect("client");

        let request = TtsRequest::new("hello world".to_string())
            .with_format("wav".to_string())
            .with_deepgram_tts_options(
                DeepgramSpeechModelOptions::new()
                    .with_sample_rate(24_000)
                    .with_tag("unit-test"),
            );

        let response = AudioCapability::text_to_speech(&client, request)
            .await
            .expect("tts response");

        assert_eq!(response.audio_data, vec![1, 2, 3, 4]);
        assert_eq!(response.format, "wav");

        let captured = transport.take_json().expect("captured json request");
        assert_eq!(
            captured.url.split('?').next(),
            Some("https://api.deepgram.test/v1/speak")
        );
        assert_eq!(
            query_param(&captured.url, "model").as_deref(),
            Some(models::speech::AURA_2_HELENA_EN)
        );
        assert_eq!(
            query_param(&captured.url, "encoding").as_deref(),
            Some("linear16")
        );
        assert_eq!(
            query_param(&captured.url, "container").as_deref(),
            Some("wav")
        );
        assert_eq!(
            query_param(&captured.url, "sample_rate").as_deref(),
            Some("24000")
        );
        assert_eq!(
            query_param(&captured.url, "tag").as_deref(),
            Some("unit-test")
        );
        assert_eq!(captured.body, serde_json::json!({ "text": "hello world" }));
        assert_eq!(
            captured
                .headers
                .get(AUTHORIZATION)
                .and_then(|value| value.to_str().ok()),
            Some("Token test-key")
        );
        assert_eq!(
            captured
                .headers
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("application/json")
        );
    }

    #[test]
    fn deepgram_config_supports_explicit_key_override() {
        let config = DeepgramConfig::new("initial-key").with_api_key("explicit-key");
        let client = DeepgramClient::from_config(config).expect("client");

        let headers = client
            .auth_headers(Some("application/json"))
            .expect("headers");
        assert_eq!(
            headers
                .get(AUTHORIZATION)
                .and_then(|value| value.to_str().ok()),
            Some("Token explicit-key")
        );
    }

    #[test]
    fn deepgram_config_exposes_env_key_name_for_fallback() {
        assert_eq!(DeepgramConfig::API_KEY_ENV, "DEEPGRAM_API_KEY");
    }

    #[tokio::test]
    async fn deepgram_tts_merges_global_and_request_headers() {
        let transport = BytesSpeechTransport::new(vec![9], "audio/mpeg");
        let mut request_http = HttpConfig::empty();
        request_http
            .headers
            .insert("x-request-header".to_string(), "request".to_string());
        request_http
            .headers
            .insert("x-shared".to_string(), "request-wins".to_string());

        let config = DeepgramConfig::new("test-key")
            .with_base_url("https://api.deepgram.test")
            .with_header("x-global-header", "global")
            .with_header("x-shared", "global")
            .with_http_transport(Arc::new(transport.clone()));
        let client = DeepgramClient::from_config(config).expect("client");
        let request = TtsRequest::new("hello".to_string()).with_http_config(request_http);

        AudioCapability::text_to_speech(&client, request)
            .await
            .expect("tts response");

        let captured = transport.take_json().expect("captured request");
        assert_eq!(
            captured
                .headers
                .get("x-global-header")
                .and_then(|value| value.to_str().ok()),
            Some("global")
        );
        assert_eq!(
            captured
                .headers
                .get("x-request-header")
                .and_then(|value| value.to_str().ok()),
            Some("request")
        );
        assert_eq!(
            captured
                .headers
                .get("x-shared")
                .and_then(|value| value.to_str().ok()),
            Some("request-wins")
        );
    }

    #[test]
    fn deepgram_client_exposes_runtime_helpers_and_audio_capabilities() {
        let transport = BytesSpeechTransport::new(vec![1], "audio/mpeg");
        let mut client = DeepgramClient::from_config(
            DeepgramConfig::new("test-key")
                .with_base_url("https://api.deepgram.test/")
                .with_http_transport(Arc::new(transport)),
        )
        .expect("client");

        client.set_retry_options(Some(RetryOptions::policy_default().with_max_attempts(1)));

        assert_eq!(client.base_url(), "https://api.deepgram.test");
        assert!(client.http_transport().is_some());
        assert!(client.retry_options().is_some());
        assert!(client.capabilities().supports("speech"));
        assert!(client.capabilities().supports("transcription"));
        assert!(client.as_speech_capability().is_some());
        assert!(client.as_transcription_capability().is_some());
        assert_eq!(
            crate::traits::ModelMetadata::provider_id(&client),
            "deepgram"
        );
    }

    #[tokio::test]
    async fn deepgram_stt_posts_raw_audio_to_listen_and_maps_response() {
        let transport = JsonTranscriptionTransport::new(serde_json::json!({
            "metadata": {
                "duration": 1.25
            },
            "results": {
                "channels": [
                    {
                        "detected_language": "en",
                        "alternatives": [
                            {
                                "transcript": "hello deepgram",
                                "confidence": 0.91,
                                "words": [
                                    {
                                        "word": "hello",
                                        "start": 0.0,
                                        "end": 0.4,
                                        "confidence": 0.9
                                    },
                                    {
                                        "word": "deepgram",
                                        "start": 0.45,
                                        "end": 1.0,
                                        "confidence": 0.92
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        }));
        let config = DeepgramConfig::new("test-key")
            .with_base_url("https://api.deepgram.test")
            .with_transcription_model(models::transcription::NOVA_3)
            .with_http_transport(Arc::new(transport.clone()));
        let client = DeepgramClient::from_config(config).expect("client");

        let request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg")
            .with_deepgram_stt_options(
                DeepgramTranscriptionModelOptions::new()
                    .with_language("en")
                    .with_smart_format(true)
                    .with_diarize(false),
            );

        let response = AudioCapability::speech_to_text(&client, request)
            .await
            .expect("stt response");

        assert_eq!(response.text, "hello deepgram");
        assert_eq!(response.language.as_deref(), Some("en"));
        assert_eq!(response.confidence, Some(0.91));
        assert_eq!(response.duration, Some(1.25));
        let words = response.words.expect("word timestamps");
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "hello");
        assert_eq!(words[0].start, 0.0);
        assert_eq!(words[0].end, 0.4);
        assert_eq!(words[0].confidence, Some(0.9));
        assert!(
            response
                .provider_metadata
                .as_ref()
                .and_then(|metadata| metadata.get("deepgram"))
                .is_some()
        );

        let captured = transport.take_raw().expect("captured raw audio request");
        assert_eq!(
            captured.url.split('?').next(),
            Some("https://api.deepgram.test/v1/listen")
        );
        assert_eq!(
            query_param(&captured.url, "model").as_deref(),
            Some(models::transcription::NOVA_3)
        );
        assert_eq!(
            query_param(&captured.url, "diarize").as_deref(),
            Some("false")
        );
        assert_eq!(
            query_param(&captured.url, "language").as_deref(),
            Some("en")
        );
        assert_eq!(
            query_param(&captured.url, "smart_format").as_deref(),
            Some("true")
        );
        assert_eq!(captured.body, b"abc");
        assert_eq!(
            captured
                .headers
                .get(AUTHORIZATION)
                .and_then(|value| value.to_str().ok()),
            Some("Token test-key")
        );
        assert_eq!(
            captured
                .headers
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("audio/mpeg")
        );
        assert_eq!(
            captured
                .headers
                .get(CONTENT_LENGTH)
                .and_then(|value| value.to_str().ok()),
            Some("3")
        );
    }
}
