//! ElevenLabs voice catalog resources.
//!
//! ElevenLabs exposes voice listing and voice detail APIs outside Siumai's unified
//! speech/transcription families. This module keeps that surface provider-owned.

use crate::error::LlmError;
use crate::provider_utils::url::join_url;
use crate::retry_api::RetryOptions;
use crate::types::HttpConfig;
use reqwest::Url;
use secrecy::ExposeSecret;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;

use super::config::ElevenLabsConfig;
use super::resource_http::execute_get_json;

/// Provider-owned client for ElevenLabs voice catalog resources.
#[derive(Clone)]
pub struct ElevenLabsVoices {
    config: ElevenLabsConfig,
    http_client: reqwest::Client,
    retry_options: Option<RetryOptions>,
}

impl std::fmt::Debug for ElevenLabsVoices {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ElevenLabsVoices")
            .field("base_url", &self.config.base_url)
            .field(
                "has_api_key",
                &!self.config.api_key.expose_secret().is_empty(),
            )
            .field("has_transport", &self.config.http_transport.is_some())
            .field("interceptors", &self.config.http_interceptors.len())
            .field("has_retry", &self.retry_options.is_some())
            .finish()
    }
}

impl ElevenLabsVoices {
    pub(crate) fn new(
        config: ElevenLabsConfig,
        http_client: reqwest::Client,
        retry_options: Option<RetryOptions>,
    ) -> Self {
        Self {
            config,
            http_client,
            retry_options,
        }
    }

    /// List voices using `GET /v2/voices`.
    pub async fn list(
        &self,
        query: Option<ElevenLabsVoiceListQuery>,
    ) -> Result<ElevenLabsVoiceListResponse, LlmError> {
        let url = self.list_url(query.as_ref())?;
        let per_request_http_config = query.as_ref().and_then(|query| query.http_config.as_ref());
        execute_get_json(
            &self.config,
            self.http_client.clone(),
            self.retry_options.clone(),
            &url,
            per_request_http_config,
            "list voices",
        )
        .await
    }

    /// Retrieve a single voice using `GET /v1/voices/{voice_id}`.
    pub async fn get(&self, voice_id: impl AsRef<str>) -> Result<ElevenLabsVoice, LlmError> {
        self.get_with_http_config(voice_id, None).await
    }

    /// Retrieve a single voice with per-request HTTP configuration.
    pub async fn get_with_http_config(
        &self,
        voice_id: impl AsRef<str>,
        http_config: Option<&HttpConfig>,
    ) -> Result<ElevenLabsVoice, LlmError> {
        let voice_id = voice_id.as_ref().trim();
        if voice_id.is_empty() {
            return Err(LlmError::InvalidInput(
                "ElevenLabs voice_id cannot be empty".to_string(),
            ));
        }

        let encoded = urlencoding::encode(voice_id);
        let url = join_url(&self.base_url(), &format!("v1/voices/{encoded}"));
        execute_get_json(
            &self.config,
            self.http_client.clone(),
            self.retry_options.clone(),
            &url,
            http_config,
            "get voice",
        )
        .await
    }

    fn base_url(&self) -> String {
        self.config.base_url.trim_end_matches('/').to_string()
    }

    fn list_url(&self, query: Option<&ElevenLabsVoiceListQuery>) -> Result<String, LlmError> {
        let mut url = Url::parse(&join_url(&self.base_url(), "v2/voices"))
            .map_err(|e| LlmError::InvalidInput(format!("Invalid ElevenLabs voices URL: {e}")))?;

        if let Some(query) = query {
            query.validate()?;
            let pairs = query.to_query_pairs();
            if !pairs.is_empty() {
                url.query_pairs_mut().extend_pairs(pairs.iter());
            }
        }

        Ok(url.to_string())
    }
}

/// Query parameters for `GET /v2/voices`.
#[derive(Debug, Clone, Default)]
pub struct ElevenLabsVoiceListQuery {
    pub next_page_token: Option<String>,
    pub page_size: Option<u32>,
    pub search: Option<String>,
    pub sort: Option<String>,
    pub sort_direction: Option<String>,
    pub voice_type: Option<String>,
    pub category: Option<String>,
    pub fine_tuning_state: Option<String>,
    pub collection_id: Option<String>,
    pub include_total_count: Option<bool>,
    pub voice_ids: Vec<String>,
    pub http_config: Option<HttpConfig>,
}

impl ElevenLabsVoiceListQuery {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_next_page_token(mut self, value: impl Into<String>) -> Self {
        self.next_page_token = Some(value.into());
        self
    }

    pub fn with_page_size(mut self, value: u32) -> Self {
        self.page_size = Some(value);
        self
    }

    pub fn with_search(mut self, value: impl Into<String>) -> Self {
        self.search = Some(value.into());
        self
    }

    pub fn with_sort(mut self, value: impl Into<String>) -> Self {
        self.sort = Some(value.into());
        self
    }

    pub fn with_sort_direction(mut self, value: impl Into<String>) -> Self {
        self.sort_direction = Some(value.into());
        self
    }

    pub fn with_voice_type(mut self, value: impl Into<String>) -> Self {
        self.voice_type = Some(value.into());
        self
    }

    pub fn with_category(mut self, value: impl Into<String>) -> Self {
        self.category = Some(value.into());
        self
    }

    pub fn with_fine_tuning_state(mut self, value: impl Into<String>) -> Self {
        self.fine_tuning_state = Some(value.into());
        self
    }

    pub fn with_collection_id(mut self, value: impl Into<String>) -> Self {
        self.collection_id = Some(value.into());
        self
    }

    pub fn with_include_total_count(mut self, value: bool) -> Self {
        self.include_total_count = Some(value);
        self
    }

    pub fn with_voice_id(mut self, value: impl Into<String>) -> Self {
        self.voice_ids.push(value.into());
        self
    }

    pub fn with_voice_ids<I, S>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.voice_ids.extend(values.into_iter().map(Into::into));
        self
    }

    pub fn with_http_config(mut self, value: HttpConfig) -> Self {
        self.http_config = Some(value);
        self
    }

    fn validate(&self) -> Result<(), LlmError> {
        if self.page_size == Some(0) {
            return Err(LlmError::InvalidInput(
                "ElevenLabs voice list page_size must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }

    fn to_query_pairs(&self) -> Vec<(String, String)> {
        let mut pairs = Vec::new();
        push_optional(
            &mut pairs,
            "next_page_token",
            self.next_page_token.as_deref(),
        );
        if let Some(page_size) = self.page_size {
            pairs.push(("page_size".to_string(), page_size.to_string()));
        }
        push_optional(&mut pairs, "search", self.search.as_deref());
        push_optional(&mut pairs, "sort", self.sort.as_deref());
        push_optional(&mut pairs, "sort_direction", self.sort_direction.as_deref());
        push_optional(&mut pairs, "voice_type", self.voice_type.as_deref());
        push_optional(&mut pairs, "category", self.category.as_deref());
        push_optional(
            &mut pairs,
            "fine_tuning_state",
            self.fine_tuning_state.as_deref(),
        );
        push_optional(&mut pairs, "collection_id", self.collection_id.as_deref());
        if let Some(include_total_count) = self.include_total_count {
            pairs.push((
                "include_total_count".to_string(),
                include_total_count.to_string(),
            ));
        }
        for voice_id in &self.voice_ids {
            push_optional(&mut pairs, "voice_ids", Some(voice_id));
        }
        pairs
    }
}

fn push_optional(pairs: &mut Vec<(String, String)>, key: &str, value: Option<&str>) {
    if let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) {
        pairs.push((key.to_string(), value.to_string()));
    }
}

/// Response body for `GET /v2/voices`.
#[derive(Debug, Clone, Deserialize)]
pub struct ElevenLabsVoiceListResponse {
    #[serde(default)]
    pub voices: Vec<ElevenLabsVoice>,
    #[serde(default)]
    pub has_more: bool,
    #[serde(default)]
    pub total_count: Option<u64>,
    #[serde(default)]
    pub next_page_token: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Voice object returned by ElevenLabs voice APIs.
#[derive(Debug, Clone, Deserialize)]
pub struct ElevenLabsVoice {
    pub voice_id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub samples: Option<Vec<Value>>,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub fine_tuning: Option<Value>,
    #[serde(default)]
    pub labels: HashMap<String, Value>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub preview_url: Option<String>,
    #[serde(default)]
    pub available_for_tiers: Option<Vec<String>>,
    #[serde(default)]
    pub settings: Option<ElevenLabsVoiceSettingsResponse>,
    #[serde(default)]
    pub sharing: Option<Value>,
    #[serde(default)]
    pub high_quality_base_model_ids: Option<Vec<String>>,
    #[serde(default)]
    pub verified_languages: Option<Vec<ElevenLabsVerifiedLanguage>>,
    #[serde(default)]
    pub collection_ids: Option<Vec<String>>,
    #[serde(default)]
    pub safety_control: Option<String>,
    #[serde(default)]
    pub voice_verification: Option<Value>,
    #[serde(default)]
    pub permission_on_resource: Option<String>,
    #[serde(default)]
    pub is_owner: Option<bool>,
    #[serde(default)]
    pub is_legacy: Option<bool>,
    #[serde(default)]
    pub is_mixed: Option<bool>,
    #[serde(default)]
    pub favorited_at_unix: Option<i64>,
    #[serde(default)]
    pub created_at_unix: Option<i64>,
    #[serde(default)]
    pub is_bookmarked: Option<bool>,
    #[serde(default)]
    pub recording_quality: Option<String>,
    #[serde(default)]
    pub labelling_status: Option<String>,
    #[serde(default)]
    pub recording_quality_reason: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Voice settings returned by ElevenLabs voice APIs.
#[derive(Debug, Clone, Deserialize)]
pub struct ElevenLabsVoiceSettingsResponse {
    #[serde(default)]
    pub stability: Option<f64>,
    #[serde(default)]
    pub similarity_boost: Option<f64>,
    #[serde(default)]
    pub style: Option<f64>,
    #[serde(default)]
    pub use_speaker_boost: Option<bool>,
    #[serde(default)]
    pub speed: Option<f64>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Verified language metadata returned by ElevenLabs voice APIs.
#[derive(Debug, Clone, Deserialize)]
pub struct ElevenLabsVerifiedLanguage {
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub accent: Option<String>,
    #[serde(default)]
    pub locale: Option<String>,
    #[serde(default)]
    pub preview_url: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::transport::{
        HttpTransport, HttpTransportGetRequest, HttpTransportRequest, HttpTransportResponse,
    };
    use async_trait::async_trait;
    use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
    use serde_json::json;
    use std::sync::Arc;
    use std::sync::Mutex;

    use super::super::resource_http::XI_API_KEY;

    #[derive(Clone)]
    struct JsonGetTransport {
        response: Value,
        last_get: Arc<Mutex<Option<HttpTransportGetRequest>>>,
    }

    impl JsonGetTransport {
        fn new(response: Value) -> Self {
            Self {
                response,
                last_get: Arc::new(Mutex::new(None)),
            }
        }

        fn take_get(&self) -> HttpTransportGetRequest {
            self.last_get
                .lock()
                .expect("get transport lock")
                .take()
                .expect("captured get request")
        }
    }

    #[async_trait]
    impl HttpTransport for JsonGetTransport {
        async fn execute_json(
            &self,
            _request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "json requests are not expected in voice resource tests".to_string(),
            ))
        }

        async fn execute_get(
            &self,
            request: HttpTransportGetRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last_get.lock().expect("get transport lock") = Some(request);
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
            Ok(HttpTransportResponse {
                status: 200,
                headers,
                body: serde_json::to_vec(&self.response).expect("serialize response"),
            })
        }
    }

    fn query_values(url: &str, key: &str) -> Vec<String> {
        let parsed = Url::parse(url).expect("valid url");
        parsed
            .query_pairs()
            .filter_map(|(name, value)| (name == key).then(|| value.into_owned()))
            .collect()
    }

    fn header_value<'a>(headers: &'a reqwest::header::HeaderMap, key: &str) -> Option<&'a str> {
        headers.get(key).and_then(|value| value.to_str().ok())
    }

    #[tokio::test]
    async fn voices_list_maps_query_headers_and_response_metadata() {
        let transport = JsonGetTransport::new(json!({
            "voices": [
                {
                    "voice_id": "voice-1",
                    "name": "Rachel",
                    "category": "premade",
                    "labels": { "accent": "american" },
                    "description": "Narration voice",
                    "preview_url": "https://cdn.elevenlabs.test/preview.mp3",
                    "settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "style": 0.1,
                        "use_speaker_boost": true,
                        "speed": 1.0,
                        "future_setting": "kept"
                    },
                    "sharing": { "status": "public" },
                    "verified_languages": [
                        {
                            "language": "en",
                            "accent": "american",
                            "locale": "en-US",
                            "preview_url": "https://cdn.elevenlabs.test/en.mp3",
                            "future_language": "kept"
                        }
                    ],
                    "created_at_unix": 1_700_000_000,
                    "unknown_voice": "kept"
                }
            ],
            "has_more": true,
            "total_count": 1,
            "next_page_token": "next-token",
            "unknown_list": "kept"
        }));
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
        let voices = ElevenLabsVoices::new(
            config,
            reqwest::Client::new(),
            Some(RetryOptions::policy_default().with_max_attempts(1)),
        );

        let response = voices
            .list(Some(
                ElevenLabsVoiceListQuery::new()
                    .with_next_page_token("previous-token")
                    .with_page_size(25)
                    .with_search("rachel")
                    .with_sort("created_at_unix")
                    .with_sort_direction("desc")
                    .with_voice_type("default")
                    .with_category("premade")
                    .with_fine_tuning_state("fine_tuned")
                    .with_collection_id("collection-1")
                    .with_include_total_count(true)
                    .with_voice_ids(["voice-1", "voice-2"])
                    .with_http_config(request_http),
            ))
            .await
            .expect("voices response");

        let captured = transport.take_get();
        assert_eq!(
            captured.url.split('?').next(),
            Some("https://api.elevenlabs.test/v2/voices")
        );
        assert_eq!(
            query_values(&captured.url, "next_page_token"),
            vec!["previous-token"]
        );
        assert_eq!(query_values(&captured.url, "page_size"), vec!["25"]);
        assert_eq!(query_values(&captured.url, "search"), vec!["rachel"]);
        assert_eq!(query_values(&captured.url, "sort"), vec!["created_at_unix"]);
        assert_eq!(query_values(&captured.url, "sort_direction"), vec!["desc"]);
        assert_eq!(query_values(&captured.url, "voice_type"), vec!["default"]);
        assert_eq!(query_values(&captured.url, "category"), vec!["premade"]);
        assert_eq!(
            query_values(&captured.url, "fine_tuning_state"),
            vec!["fine_tuned"]
        );
        assert_eq!(
            query_values(&captured.url, "collection_id"),
            vec!["collection-1"]
        );
        assert_eq!(
            query_values(&captured.url, "include_total_count"),
            vec!["true"]
        );
        assert_eq!(
            query_values(&captured.url, "voice_ids"),
            vec!["voice-1", "voice-2"]
        );
        assert_eq!(
            header_value(&captured.headers, XI_API_KEY),
            Some("test-key")
        );
        assert_eq!(
            header_value(&captured.headers, "x-global-header"),
            Some("global")
        );
        assert_eq!(
            header_value(&captured.headers, "x-request-header"),
            Some("request")
        );
        assert_eq!(
            header_value(&captured.headers, "x-shared"),
            Some("request-wins")
        );

        assert_eq!(response.has_more, true);
        assert_eq!(response.total_count, Some(1));
        assert_eq!(response.next_page_token.as_deref(), Some("next-token"));
        assert_eq!(response.extra.get("unknown_list"), Some(&json!("kept")));
        let voice = response.voices.first().expect("voice");
        assert_eq!(voice.voice_id, "voice-1");
        assert_eq!(voice.name.as_deref(), Some("Rachel"));
        assert_eq!(voice.category.as_deref(), Some("premade"));
        assert_eq!(voice.labels.get("accent"), Some(&json!("american")));
        assert_eq!(voice.description.as_deref(), Some("Narration voice"));
        assert_eq!(
            voice.preview_url.as_deref(),
            Some("https://cdn.elevenlabs.test/preview.mp3")
        );
        assert_eq!(
            voice.sharing.as_ref().and_then(|value| value.get("status")),
            Some(&json!("public"))
        );
        let language = voice
            .verified_languages
            .as_ref()
            .and_then(|languages| languages.first())
            .expect("verified language");
        assert_eq!(language.language.as_deref(), Some("en"));
        assert_eq!(language.accent.as_deref(), Some("american"));
        assert_eq!(language.locale.as_deref(), Some("en-US"));
        assert_eq!(
            language.preview_url.as_deref(),
            Some("https://cdn.elevenlabs.test/en.mp3")
        );
        assert_eq!(language.extra.get("future_language"), Some(&json!("kept")));
        assert_eq!(voice.created_at_unix, Some(1_700_000_000));
        assert_eq!(voice.extra.get("unknown_voice"), Some(&json!("kept")));
        let settings = voice.settings.as_ref().expect("settings");
        assert_eq!(settings.stability, Some(0.5));
        assert_eq!(settings.similarity_boost, Some(0.75));
        assert_eq!(settings.style, Some(0.1));
        assert_eq!(settings.use_speaker_boost, Some(true));
        assert_eq!(settings.speed, Some(1.0));
        assert_eq!(settings.extra.get("future_setting"), Some(&json!("kept")));
    }

    #[tokio::test]
    async fn voices_get_encodes_voice_id_path() {
        let transport = JsonGetTransport::new(json!({
            "voice_id": "voice/id with space",
            "name": "Custom"
        }));
        let config = ElevenLabsConfig::new("test-key")
            .with_base_url("https://api.elevenlabs.test/")
            .with_http_transport(Arc::new(transport.clone()));
        let voices = ElevenLabsVoices::new(config, reqwest::Client::new(), None);

        let response = voices
            .get("voice/id with space")
            .await
            .expect("voice response");

        let captured = transport.take_get();
        assert_eq!(
            captured.url,
            "https://api.elevenlabs.test/v1/voices/voice%2Fid%20with%20space"
        );
        assert_eq!(
            header_value(&captured.headers, XI_API_KEY),
            Some("test-key")
        );
        assert_eq!(response.voice_id, "voice/id with space");
        assert_eq!(response.name.as_deref(), Some("Custom"));
    }

    #[tokio::test]
    async fn voices_resources_use_explicit_api_key_override() {
        assert_eq!(ElevenLabsConfig::API_KEY_ENV, "ELEVENLABS_API_KEY");

        let transport = JsonGetTransport::new(json!({
            "voices": [],
            "has_more": false,
            "total_count": 0
        }));
        let config = ElevenLabsConfig::new("initial-key")
            .with_api_key("explicit-key")
            .with_base_url("https://api.elevenlabs.test")
            .with_http_transport(Arc::new(transport.clone()));
        let voices = ElevenLabsVoices::new(config, reqwest::Client::new(), None);

        voices
            .list(None)
            .await
            .expect("explicit key voices response");

        assert_eq!(
            header_value(&transport.take_get().headers, XI_API_KEY),
            Some("explicit-key")
        );
    }
}
