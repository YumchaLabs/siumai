use super::config::ElevenLabsConfig;
use super::options::{
    ElevenLabsSpeechModelOptions, ElevenLabsTranscriptionModelOptions,
    ElevenLabsTranscriptionTimestampsGranularity,
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
use futures_util::StreamExt;
use reqwest::header::{CONTENT_LENGTH, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use secrecy::ExposeSecret;
use serde::Deserialize;
use serde_json::{Map, Value};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

const PROVIDER_ID: &str = "elevenlabs";
const XI_API_KEY: &str = "xi-api-key";
const FEATURES: &[AudioFeature] = &[
    AudioFeature::TextToSpeech,
    AudioFeature::SpeechToText,
    AudioFeature::SpeakerDiarization,
    AudioFeature::AudioEventDetection,
    AudioFeature::CharacterTiming,
];

/// Provider-owned ElevenLabs client.
#[derive(Clone)]
pub struct ElevenLabsClient {
    config: ElevenLabsConfig,
    http_client: reqwest::Client,
    retry_options: Option<RetryOptions>,
}

impl std::fmt::Debug for ElevenLabsClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ElevenLabsClient")
            .field("config", &self.config)
            .field("retry_options", &self.retry_options)
            .finish()
    }
}

impl ElevenLabsClient {
    pub fn from_config(config: ElevenLabsConfig) -> Result<Self, LlmError> {
        config.validate()?;
        let http_client =
            crate::execution::http::client::build_http_client_from_config(&config.http_config)?;
        Self::with_http_client(config, http_client)
    }

    pub fn with_http_client(
        config: ElevenLabsConfig,
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

    pub fn default_voice(&self) -> &str {
        &self.config.default_voice
    }

    pub fn http_client(&self) -> reqwest::Client {
        self.http_client.clone()
    }

    pub fn http_transport(
        &self,
    ) -> Option<Arc<dyn crate::execution::http::transport::HttpTransport>> {
        self.config.http_transport.clone()
    }

    /// Get provider-owned ElevenLabs voice catalog resources.
    pub fn voices(&self) -> super::voices::ElevenLabsVoices {
        super::voices::ElevenLabsVoices::new(
            self.config.clone(),
            self.http_client.clone(),
            self.retry_options.clone(),
        )
    }

    pub fn speech_model(&self, model_id: impl Into<String>) -> ElevenLabsSpeechModel {
        ElevenLabsSpeechModel {
            client: self.clone(),
            model_id: model_id.into(),
        }
    }

    pub fn default_speech_model(&self) -> ElevenLabsSpeechModel {
        self.speech_model(self.config.speech_model.clone())
    }

    pub fn transcription_model(&self, model_id: impl Into<String>) -> ElevenLabsTranscriptionModel {
        ElevenLabsTranscriptionModel {
            client: self.clone(),
            model_id: model_id.into(),
        }
    }

    pub fn default_transcription_model(&self) -> ElevenLabsTranscriptionModel {
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
        headers.insert(
            HeaderName::from_static(XI_API_KEY),
            HeaderValue::from_str(self.config.api_key.expose_secret()).map_err(|e| {
                LlmError::ConfigurationError(format!("Invalid ElevenLabs API key format: {e}"))
            })?,
        );
        if let Some(content_type) = content_type {
            headers.insert(
                CONTENT_TYPE,
                HeaderValue::from_str(content_type).map_err(|e| {
                    LlmError::ConfigurationError(format!(
                        "Invalid ElevenLabs content type '{content_type}': {e}"
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
        let mut query = BTreeMap::new();
        let mut warnings = Vec::new();
        let mut body = Map::from_iter([
            ("text".to_string(), serde_json::json!(request.text)),
            ("model_id".to_string(), serde_json::json!(model.clone())),
        ]);

        let voice_id = request
            .voice
            .as_deref()
            .map(str::trim)
            .filter(|voice| !voice.is_empty())
            .unwrap_or(&self.config.default_voice)
            .to_string();

        apply_output_format_query(request.format.as_deref(), &mut query);
        if let Some(language) = request
            .language
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            body.insert("language_code".to_string(), serde_json::json!(language));
        }

        let mut voice_settings = Map::new();
        if let Some(speed) = request.speed {
            voice_settings.insert("speed".to_string(), serde_json::json!(speed));
        }
        apply_speech_provider_options(&request, &mut body, &mut voice_settings, &mut query)?;
        if !voice_settings.is_empty() {
            body.insert(
                "voice_settings".to_string(),
                serde_json::Value::Object(voice_settings),
            );
        }
        warnings.extend(speech_warnings(&request));

        let endpoint = format!("v1/text-to-speech/{}", urlencoding::encode(&voice_id));
        let url = self.endpoint_url(&endpoint, &query);
        let body = Value::Object(body);
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
        let audio = request
            .audio_bytes()
            .map_err(|e| LlmError::InvalidInput(format!("Invalid ElevenLabs audio input: {e}")))?;
        let options = transcription_options(&request)?;
        let form = build_transcription_form(&model, &request, audio.clone(), options.as_ref())?;
        let url = self.endpoint_url("v1/speech-to-text", &BTreeMap::new());
        let headers =
            self.merge_request_headers(self.auth_headers(None)?, request.http_config.as_ref())?;
        let response = self
            .execute_multipart_json(&url, headers, form, request.http_config.as_ref())
            .await?;

        let parsed: ElevenLabsTranscriptionResponse = serde_json::from_slice(&response.body)
            .map_err(|e| {
                LlmError::ParseError(format!("Invalid ElevenLabs transcription response: {e}"))
            })?;
        let raw_json = serde_json::from_slice::<Value>(&response.body).map_err(|e| {
            LlmError::ParseError(format!("Invalid ElevenLabs transcription JSON: {e}"))
        })?;

        let words = parsed.words();
        let duration = words
            .as_ref()
            .and_then(|words| words.iter().map(|word| word.end).reduce(f32::max));
        let confidence = Some(parsed.language_probability as f32);
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
            text: parsed.text,
            language: Some(parsed.language_code),
            confidence,
            words,
            duration,
            metadata: HashMap::from([("raw".to_string(), raw_json)]),
            warnings: None,
            provider_metadata: Some(provider_metadata),
            request: Some(HttpRequestInfo {
                body: Some(format!("<{} bytes multipart>", response.request_body_size)),
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
    ) -> Result<ElevenLabsRawResponse, LlmError> {
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
                    |response| ElevenLabsRawResponse {
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

    async fn execute_multipart_json(
        &self,
        url: &str,
        headers: HeaderMap,
        form: reqwest::multipart::Form,
        http_config: Option<&HttpConfig>,
    ) -> Result<ElevenLabsRawResponse, LlmError> {
        let (multipart_body, multipart_content_type) = build_multipart_body(form).await?;
        let multipart_headers =
            with_multipart_content_headers(headers, &multipart_content_type, multipart_body.len())?;
        let request_body_size = multipart_body.len();

        let run_once = || {
            let body = multipart_body.clone();
            let headers = multipart_headers.clone();
            async move {
                let ctx = self.request_context(url);
                let body_for_interceptors = serde_json::json!({
                    "multipartBytes": body.len()
                });

                if let Some(transport) = &self.config.http_transport {
                    let response = transport
                        .execute_multipart(HttpTransportMultipartRequest {
                            ctx: ctx.clone(),
                            url: url.to_string(),
                            headers,
                            body,
                        })
                        .await?;
                    classify_transport_response(&ctx, &self.config.http_interceptors, response).map(
                        |response| ElevenLabsRawResponse {
                            headers: response.headers,
                            body: response.body,
                            request_body_size,
                        },
                    )
                } else {
                    let mut builder = self
                        .http_client
                        .post(url)
                        .headers(headers.clone())
                        .body(body);
                    if let Some(timeout) = http_config.and_then(|config| config.timeout) {
                        builder = builder.timeout(timeout);
                    }
                    for interceptor in &self.config.http_interceptors {
                        builder = interceptor.on_before_send(
                            &ctx,
                            builder,
                            &body_for_interceptors,
                            &headers,
                        )?;
                    }
                    let response = builder
                        .send()
                        .await
                        .map_err(|e| LlmError::HttpError(e.to_string()))?;
                    classify_reqwest_response(
                        PROVIDER_ID,
                        &ctx,
                        &self.config.http_interceptors,
                        response,
                        request_body_size,
                    )
                    .await
                }
            }
        };

        crate::retry_api::maybe_retry(self.retry_options.clone(), run_once).await
    }
}

#[async_trait]
impl AudioCapability for ElevenLabsClient {
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
            "mp3",
            "mp3_44100_32",
            "mp3_44100_64",
            "mp3_44100_96",
            "mp3_44100_128",
            "mp3_44100_192",
            "pcm_16000",
            "pcm_22050",
            "pcm_24000",
            "pcm_44100",
            "ulaw_8000",
        ]
        .into_iter()
        .map(str::to_string)
        .collect()
    }
}

impl ModelMetadata for ElevenLabsClient {
    fn provider_id(&self) -> &str {
        PROVIDER_ID
    }

    fn model_id(&self) -> &str {
        &self.config.transcription_model
    }
}

impl LlmClient for ElevenLabsClient {
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

/// Speech-family ElevenLabs model object.
#[derive(Clone, Debug)]
pub struct ElevenLabsSpeechModel {
    client: ElevenLabsClient,
    model_id: String,
}

#[async_trait]
impl SpeechCapability for ElevenLabsSpeechModel {
    async fn tts(&self, request: TtsRequest) -> Result<TtsResponse, LlmError> {
        self.client
            .text_to_speech_once(request.with_model_if_missing(self.model_id.clone()))
            .await
    }
}

impl ModelMetadata for ElevenLabsSpeechModel {
    fn provider_id(&self) -> &str {
        PROVIDER_ID
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

/// Transcription-family ElevenLabs model object.
#[derive(Clone, Debug)]
pub struct ElevenLabsTranscriptionModel {
    client: ElevenLabsClient,
    model_id: String,
}

#[async_trait]
impl TranscriptionCapability for ElevenLabsTranscriptionModel {
    async fn stt(&self, request: SttRequest) -> Result<SttResponse, LlmError> {
        self.client
            .speech_to_text_once(request.with_model_if_missing(self.model_id.clone()))
            .await
    }
}

impl ModelMetadata for ElevenLabsTranscriptionModel {
    fn provider_id(&self) -> &str {
        PROVIDER_ID
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[derive(Debug)]
struct ElevenLabsRawResponse {
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
            "ElevenLabs request requires a non-empty model id".to_string(),
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
            LlmError::ConfigurationError(format!("Invalid ElevenLabs header name '{name}': {e}"))
        })?;
        let value = HeaderValue::from_str(value).map_err(|e| {
            LlmError::ConfigurationError(format!(
                "Invalid ElevenLabs header value for '{name}': {e}"
            ))
        })?;
        headers.insert(name, value);
    }
    Ok(())
}

fn non_empty_warnings(warnings: Vec<Warning>) -> Option<Vec<Warning>> {
    (!warnings.is_empty()).then_some(warnings)
}

fn speech_warnings(request: &TtsRequest) -> Vec<Warning> {
    let mut warnings = Vec::new();
    if request.instructions.is_some() {
        warnings.push(Warning::unsupported(
            "instructions",
            Some("ElevenLabs speech models do not support instructions"),
        ));
    }
    warnings
}

fn apply_output_format_query(format: Option<&str>, query: &mut BTreeMap<String, String>) {
    let Some(format) = format.map(str::trim).filter(|format| !format.is_empty()) else {
        return;
    };
    let mapped = match format {
        "mp3" => "mp3_44100_128",
        "mp3_32" => "mp3_44100_32",
        "mp3_64" => "mp3_44100_64",
        "mp3_96" => "mp3_44100_96",
        "mp3_128" => "mp3_44100_128",
        "mp3_192" => "mp3_44100_192",
        "pcm" => "pcm_44100",
        "pcm_16000" => "pcm_16000",
        "pcm_22050" => "pcm_22050",
        "pcm_24000" => "pcm_24000",
        "pcm_44100" => "pcm_44100",
        "ulaw" => "ulaw_8000",
        other => other,
    };
    query.insert("output_format".to_string(), mapped.to_string());
}

fn apply_speech_provider_options(
    request: &TtsRequest,
    body: &mut Map<String, Value>,
    voice_settings: &mut Map<String, Value>,
    query: &mut BTreeMap<String, String>,
) -> Result<(), LlmError> {
    let Some(options) = request.provider_options_map.get(PROVIDER_ID) else {
        return Ok(());
    };
    let options: ElevenLabsSpeechModelOptions =
        serde_json::from_value(options.clone()).map_err(|e| {
            LlmError::InvalidParameter(format!("Invalid ElevenLabs speech provider options: {e}"))
        })?;

    if let Some(language_code) = options.language_code {
        body.entry("language_code".to_string())
            .or_insert_with(|| serde_json::json!(language_code));
    }
    if let Some(settings) = options.voice_settings {
        insert_number(voice_settings, "stability", settings.stability);
        insert_number(
            voice_settings,
            "similarity_boost",
            settings.similarity_boost,
        );
        insert_number(voice_settings, "style", settings.style);
        insert_bool(
            voice_settings,
            "use_speaker_boost",
            settings.use_speaker_boost,
        );
    }
    if let Some(locators) = options.pronunciation_dictionary_locators {
        let value = Value::Array(
            locators
                .into_iter()
                .map(|locator| {
                    let mut locator_obj = Map::from_iter([(
                        "pronunciation_dictionary_id".to_string(),
                        serde_json::json!(locator.pronunciation_dictionary_id),
                    )]);
                    if let Some(version_id) = locator.version_id {
                        locator_obj.insert("version_id".to_string(), serde_json::json!(version_id));
                    }
                    Value::Object(locator_obj)
                })
                .collect(),
        );
        body.insert("pronunciation_dictionary_locators".to_string(), value);
    }
    insert_u32_body(body, "seed", options.seed);
    insert_string_body(body, "previous_text", options.previous_text);
    insert_string_body(body, "next_text", options.next_text);
    insert_string_array_body(body, "previous_request_ids", options.previous_request_ids);
    insert_string_array_body(body, "next_request_ids", options.next_request_ids);
    if let Some(policy) = options.apply_text_normalization {
        body.insert(
            "apply_text_normalization".to_string(),
            serde_json::json!(policy.as_wire_value()),
        );
    }
    insert_bool_body(
        body,
        "apply_language_text_normalization",
        options.apply_language_text_normalization,
    );
    if let Some(enable_logging) = options.enable_logging {
        query.insert("enable_logging".to_string(), enable_logging.to_string());
    }
    Ok(())
}

fn transcription_options(
    request: &SttRequest,
) -> Result<Option<ElevenLabsTranscriptionModelOptions>, LlmError> {
    request
        .provider_options_map
        .get(PROVIDER_ID)
        .map(|options| {
            serde_json::from_value(options.clone()).map_err(|e| {
                LlmError::InvalidParameter(format!(
                    "Invalid ElevenLabs transcription provider options: {e}"
                ))
            })
        })
        .transpose()
}

fn build_transcription_form(
    model: &str,
    request: &SttRequest,
    audio: Vec<u8>,
    options: Option<&ElevenLabsTranscriptionModelOptions>,
) -> Result<reqwest::multipart::Form, LlmError> {
    let file_extension = crate::provider_utils::media_type_to_extension(&request.media_type);
    let filename = if file_extension.is_empty() {
        "audio".to_string()
    } else {
        format!("audio.{file_extension}")
    };
    let file_part = reqwest::multipart::Part::bytes(audio)
        .file_name(filename)
        .mime_str(&request.media_type)
        .map_err(|e| {
            LlmError::InvalidInput(format!(
                "Invalid ElevenLabs transcription media type '{}': {e}",
                request.media_type
            ))
        })?;

    let mut form = reqwest::multipart::Form::new()
        .text("model_id", model.to_string())
        .part("file", file_part)
        .text("diarize", "true");

    if let Some(options) = options {
        if let Some(value) = &options.language_code {
            form = form.text("language_code", value.clone());
        }
        if let Some(value) = options.tag_audio_events {
            form = form.text("tag_audio_events", value.to_string());
        }
        if let Some(value) = options.num_speakers {
            form = form.text("num_speakers", value.to_string());
        }
        if let Some(value) = &options.timestamps_granularity {
            form = form.text("timestamps_granularity", value.as_wire_value().to_string());
        } else {
            form = form.text(
                "timestamps_granularity",
                ElevenLabsTranscriptionTimestampsGranularity::Word
                    .as_wire_value()
                    .to_string(),
            );
        }
        if let Some(value) = options.diarize {
            form = form.text("diarize", value.to_string());
        }
        if let Some(value) = &options.file_format {
            form = form.text("file_format", value.as_wire_value().to_string());
        }
    }

    Ok(form)
}

fn insert_number(obj: &mut Map<String, Value>, key: &str, value: Option<f64>) {
    if let Some(value) = value {
        obj.insert(key.to_string(), serde_json::json!(value));
    }
}

fn insert_bool(obj: &mut Map<String, Value>, key: &str, value: Option<bool>) {
    if let Some(value) = value {
        obj.insert(key.to_string(), serde_json::json!(value));
    }
}

fn insert_string_body(obj: &mut Map<String, Value>, key: &str, value: Option<String>) {
    if let Some(value) = value {
        obj.insert(key.to_string(), serde_json::json!(value));
    }
}

fn insert_u32_body(obj: &mut Map<String, Value>, key: &str, value: Option<u32>) {
    if let Some(value) = value {
        obj.insert(key.to_string(), serde_json::json!(value));
    }
}

fn insert_bool_body(obj: &mut Map<String, Value>, key: &str, value: Option<bool>) {
    if let Some(value) = value {
        obj.insert(key.to_string(), serde_json::json!(value));
    }
}

fn insert_string_array_body(obj: &mut Map<String, Value>, key: &str, value: Option<Vec<String>>) {
    if let Some(value) = value {
        obj.insert(key.to_string(), serde_json::json!(value));
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
) -> Result<ElevenLabsRawResponse, LlmError> {
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

    Ok(ElevenLabsRawResponse {
        headers,
        body,
        request_body_size,
    })
}

async fn build_multipart_body(
    form: reqwest::multipart::Form,
) -> Result<(Vec<u8>, String), LlmError> {
    let boundary = form.boundary().to_string();
    let mut body = Vec::new();
    let mut stream = form.into_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| LlmError::HttpError(e.to_string()))?;
        body.extend_from_slice(&chunk);
    }

    Ok((body, format!("multipart/form-data; boundary={boundary}")))
}

fn with_multipart_content_headers(
    mut headers: HeaderMap,
    content_type: &str,
    content_length: usize,
) -> Result<HeaderMap, LlmError> {
    let content_type =
        HeaderValue::from_str(content_type).map_err(|e| LlmError::HttpError(e.to_string()))?;
    let content_length = HeaderValue::from_str(&content_length.to_string())
        .map_err(|e| LlmError::HttpError(e.to_string()))?;
    headers.insert(CONTENT_TYPE, content_type);
    headers.insert(CONTENT_LENGTH, content_length);
    Ok(headers)
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
        "audio/ulaw" => Some("ulaw".to_string()),
        "audio/pcm" => Some("pcm".to_string()),
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
struct ElevenLabsTranscriptionResponse {
    language_code: String,
    language_probability: f64,
    text: String,
    words: Option<Vec<ElevenLabsTranscriptionWord>>,
}

impl ElevenLabsTranscriptionResponse {
    fn words(&self) -> Option<Vec<WordTimestamp>> {
        let words = self.words.as_ref()?;
        Some(
            words
                .iter()
                .filter_map(|word| {
                    Some(WordTimestamp {
                        word: word.text.clone(),
                        start: word.start?,
                        end: word.end?,
                        confidence: None,
                    })
                })
                .collect(),
        )
        .filter(|words: &Vec<WordTimestamp>| !words.is_empty())
    }
}

#[derive(Debug, Deserialize)]
struct ElevenLabsTranscriptionWord {
    text: String,
    #[allow(dead_code)]
    r#type: Option<String>,
    start: Option<f32>,
    end: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::elevenlabs::ext::{ElevenLabsSttRequestExt, ElevenLabsTtsRequestExt};
    use crate::providers::elevenlabs::models;
    use crate::providers::elevenlabs::options::{
        ApplyTextNormalization, ElevenLabsPronunciationDictionaryLocator,
        ElevenLabsSpeechModelOptions, ElevenLabsTranscriptionFileFormat,
        ElevenLabsTranscriptionModelOptions, ElevenLabsTranscriptionTimestampsGranularity,
        ElevenLabsVoiceSettings,
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
        last_multipart: Arc<Mutex<Option<HttpTransportMultipartRequest>>>,
    }

    impl JsonTranscriptionTransport {
        fn new(response: Value) -> Self {
            Self {
                response,
                last_multipart: Arc::new(Mutex::new(None)),
            }
        }

        fn take_multipart(&self) -> Option<HttpTransportMultipartRequest> {
            self.last_multipart
                .lock()
                .expect("multipart transport lock")
                .take()
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
            *self
                .last_multipart
                .lock()
                .expect("multipart transport lock") = Some(request);
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
    async fn elevenlabs_tts_posts_voice_json_with_xi_api_key_and_query_options() {
        let transport = BytesSpeechTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
        let config = ElevenLabsConfig::new("test-key")
            .with_base_url("https://api.elevenlabs.test")
            .with_speech_model(models::speech::ELEVEN_MULTILINGUAL_V2)
            .with_http_transport(Arc::new(transport.clone()));
        let client = ElevenLabsClient::from_config(config).expect("client");

        let request = TtsRequest::new("hello world".to_string())
            .with_voice("voice-123".to_string())
            .with_format("mp3_64".to_string())
            .with_language("en")
            .with_speed(1.25)
            .with_instructions("ignored")
            .with_elevenlabs_tts_options(
                ElevenLabsSpeechModelOptions::new()
                    .with_voice_settings(
                        ElevenLabsVoiceSettings::new()
                            .with_stability(0.5)
                            .with_similarity_boost(0.75)
                            .with_style(0.2)
                            .with_use_speaker_boost(true),
                    )
                    .with_pronunciation_dictionary_locators(vec![
                        ElevenLabsPronunciationDictionaryLocator::new("dict-1")
                            .with_version_id("v2"),
                    ])
                    .with_seed(42)
                    .with_previous_text("before")
                    .with_next_text("after")
                    .with_previous_request_ids(vec!["prev-1".to_string()])
                    .with_next_request_ids(vec!["next-1".to_string()])
                    .with_apply_text_normalization(ApplyTextNormalization::Auto)
                    .with_apply_language_text_normalization(true)
                    .with_enable_logging(false),
            );

        let response = AudioCapability::text_to_speech(&client, request)
            .await
            .expect("tts response");

        assert_eq!(response.audio_data, vec![1, 2, 3, 4]);
        assert_eq!(response.format, "mp3");
        assert!(response.warnings.as_ref().is_some_and(|warnings| {
            warnings.iter().any(|warning| {
                matches!(
                    warning,
                    Warning::Unsupported { feature, .. } if feature == "instructions"
                )
            })
        }));

        let captured = transport.take_json().expect("captured json request");
        assert_eq!(
            captured.url.split('?').next(),
            Some("https://api.elevenlabs.test/v1/text-to-speech/voice-123")
        );
        assert_eq!(
            query_param(&captured.url, "output_format").as_deref(),
            Some("mp3_44100_64")
        );
        assert_eq!(
            query_param(&captured.url, "enable_logging").as_deref(),
            Some("false")
        );
        assert_eq!(
            captured
                .headers
                .get(XI_API_KEY)
                .and_then(|value| value.to_str().ok()),
            Some("test-key")
        );
        assert_eq!(
            captured
                .headers
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("application/json")
        );
        assert_eq!(
            captured.body,
            serde_json::json!({
                "text": "hello world",
                "model_id": models::speech::ELEVEN_MULTILINGUAL_V2,
                "language_code": "en",
                "voice_settings": {
                    "speed": 1.25,
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0.2,
                    "use_speaker_boost": true
                },
                "pronunciation_dictionary_locators": [
                    {
                        "pronunciation_dictionary_id": "dict-1",
                        "version_id": "v2"
                    }
                ],
                "seed": 42,
                "previous_text": "before",
                "next_text": "after",
                "previous_request_ids": ["prev-1"],
                "next_request_ids": ["next-1"],
                "apply_text_normalization": "auto",
                "apply_language_text_normalization": true
            })
        );
    }

    #[test]
    fn elevenlabs_config_supports_explicit_key_override() {
        let config = ElevenLabsConfig::new("initial-key").with_api_key("explicit-key");
        let client = ElevenLabsClient::from_config(config).expect("client");

        let headers = client
            .auth_headers(Some("application/json"))
            .expect("headers");
        assert_eq!(
            headers
                .get(XI_API_KEY)
                .and_then(|value| value.to_str().ok()),
            Some("explicit-key")
        );
    }

    #[test]
    fn elevenlabs_config_exposes_env_key_name_for_fallback() {
        assert_eq!(ElevenLabsConfig::API_KEY_ENV, "ELEVENLABS_API_KEY");
    }

    #[tokio::test]
    async fn elevenlabs_tts_merges_global_and_request_headers() {
        let transport = BytesSpeechTransport::new(vec![9], "audio/mpeg");
        let mut request_http = HttpConfig::empty();
        request_http
            .headers
            .insert("x-request-header".to_string(), "request".to_string());
        request_http
            .headers
            .insert("x-shared".to_string(), "request-wins".to_string());

        let config = ElevenLabsConfig::new("test-key")
            .with_base_url("https://api.elevenlabs.test")
            .with_header("x-global-header", "global")
            .with_header("x-shared", "global")
            .with_http_transport(Arc::new(transport.clone()));
        let client = ElevenLabsClient::from_config(config).expect("client");
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
    fn elevenlabs_client_exposes_runtime_helpers_and_audio_capabilities() {
        let transport = BytesSpeechTransport::new(vec![1], "audio/mpeg");
        let mut client = ElevenLabsClient::from_config(
            ElevenLabsConfig::new("test-key")
                .with_base_url("https://api.elevenlabs.test/")
                .with_default_voice("voice-default")
                .with_http_transport(Arc::new(transport)),
        )
        .expect("client");

        client.set_retry_options(Some(RetryOptions::policy_default().with_max_attempts(1)));

        assert_eq!(client.base_url(), "https://api.elevenlabs.test");
        assert_eq!(client.default_voice(), "voice-default");
        assert!(client.http_transport().is_some());
        assert!(client.retry_options().is_some());
        assert!(client.capabilities().supports("speech"));
        assert!(client.capabilities().supports("transcription"));
        assert!(client.as_speech_capability().is_some());
        assert!(client.as_transcription_capability().is_some());
        assert_eq!(
            crate::traits::ModelMetadata::provider_id(&client),
            "elevenlabs"
        );
    }

    #[tokio::test]
    async fn elevenlabs_stt_posts_multipart_to_speech_to_text_and_maps_response() {
        let transport = JsonTranscriptionTransport::new(serde_json::json!({
            "language_code": "en",
            "language_probability": 0.96,
            "text": "hello elevenlabs",
            "words": [
                {
                    "text": "hello",
                    "type": "word",
                    "start": 0.0,
                    "end": 0.4
                },
                {
                    "text": " ",
                    "type": "spacing",
                    "start": null,
                    "end": null
                },
                {
                    "text": "elevenlabs",
                    "type": "word",
                    "start": 0.45,
                    "end": 1.1
                }
            ]
        }));
        let config = ElevenLabsConfig::new("test-key")
            .with_base_url("https://api.elevenlabs.test")
            .with_transcription_model(models::transcription::SCRIBE_V1)
            .with_http_transport(Arc::new(transport.clone()));
        let client = ElevenLabsClient::from_config(config).expect("client");

        let request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg")
            .with_elevenlabs_stt_options(
                ElevenLabsTranscriptionModelOptions::new()
                    .with_language_code("en")
                    .with_tag_audio_events(false)
                    .with_num_speakers(2)
                    .with_timestamps_granularity(
                        ElevenLabsTranscriptionTimestampsGranularity::Character,
                    )
                    .with_diarize(false)
                    .with_file_format(ElevenLabsTranscriptionFileFormat::PcmS16le16),
            );

        let response = AudioCapability::speech_to_text(&client, request)
            .await
            .expect("stt response");

        assert_eq!(response.text, "hello elevenlabs");
        assert_eq!(response.language.as_deref(), Some("en"));
        assert_eq!(response.confidence, Some(0.96));
        assert_eq!(response.duration, Some(1.1));
        let words = response.words.expect("word timestamps");
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "hello");
        assert_eq!(words[0].start, 0.0);
        assert_eq!(words[0].end, 0.4);
        assert_eq!(words[1].word, "elevenlabs");
        assert!(
            response
                .provider_metadata
                .as_ref()
                .and_then(|metadata| metadata.get("elevenlabs"))
                .is_some()
        );

        let captured = transport
            .take_multipart()
            .expect("captured multipart request");
        assert_eq!(
            captured.url,
            "https://api.elevenlabs.test/v1/speech-to-text"
        );
        assert_eq!(
            captured
                .headers
                .get(XI_API_KEY)
                .and_then(|value| value.to_str().ok()),
            Some("test-key")
        );
        let content_type = captured
            .headers
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .expect("multipart content-type");
        assert!(
            content_type.starts_with("multipart/form-data; boundary="),
            "unexpected content type: {content_type}"
        );
        let content_length = captured
            .headers
            .get(CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<usize>().ok())
            .expect("multipart content length");
        assert_eq!(content_length, captured.body.len());

        let body_text = String::from_utf8_lossy(&captured.body);
        assert!(body_text.contains("name=\"model_id\""));
        assert!(body_text.contains(models::transcription::SCRIBE_V1));
        assert!(body_text.contains("name=\"file\"; filename=\"audio.mp3\""));
        assert!(body_text.contains("Content-Type: audio/mpeg"));
        assert!(body_text.contains("abc"));
        assert!(body_text.contains("name=\"diarize\""));
        assert!(body_text.contains("false"));
        assert!(body_text.contains("name=\"language_code\""));
        assert!(body_text.contains("en"));
        assert!(body_text.contains("name=\"tag_audio_events\""));
        assert!(body_text.contains("false"));
        assert!(body_text.contains("name=\"num_speakers\""));
        assert!(body_text.contains("2"));
        assert!(body_text.contains("name=\"timestamps_granularity\""));
        assert!(body_text.contains("character"));
        assert!(body_text.contains("name=\"file_format\""));
        assert!(body_text.contains("pcm_s16le_16"));
    }
}
