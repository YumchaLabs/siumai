//! ElevenLabs pronunciation dictionary resources.
//!
//! This module keeps pronunciation dictionary metadata and provider-owned mutations outside
//! Siumai's unified speech/transcription families.

use crate::error::LlmError;
use crate::provider_utils::url::join_url;
use crate::retry_api::RetryOptions;
use crate::types::HttpConfig;
use reqwest::Url;
use secrecy::ExposeSecret;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use super::config::ElevenLabsConfig;
use super::resource_http::{execute_get_json, execute_multipart_json, execute_post_json};

/// Provider-owned client for ElevenLabs pronunciation dictionary metadata resources.
#[derive(Clone)]
pub struct ElevenLabsPronunciationDictionaries {
    config: ElevenLabsConfig,
    http_client: reqwest::Client,
    retry_options: Option<RetryOptions>,
}

impl std::fmt::Debug for ElevenLabsPronunciationDictionaries {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ElevenLabsPronunciationDictionaries")
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

impl ElevenLabsPronunciationDictionaries {
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

    /// Create a pronunciation dictionary from inline rules using
    /// `POST /v1/pronunciation-dictionaries/add-from-rules`.
    pub async fn create_from_rules(
        &self,
        request: ElevenLabsCreatePronunciationDictionaryFromRulesRequest,
    ) -> Result<ElevenLabsPronunciationDictionaryCreateResponse, LlmError> {
        request.validate()?;
        let url = join_url(
            &self.base_url(),
            "v1/pronunciation-dictionaries/add-from-rules",
        );
        let body = request.body()?;
        execute_post_json(
            &self.config,
            self.http_client.clone(),
            self.retry_options.clone(),
            &url,
            body,
            request.http_config.as_ref(),
            "create pronunciation dictionary from rules",
        )
        .await
    }

    /// Create a pronunciation dictionary from a PLS file using
    /// `POST /v1/pronunciation-dictionaries/add-from-file`.
    pub async fn create_from_file(
        &self,
        request: ElevenLabsCreatePronunciationDictionaryFromFileRequest,
    ) -> Result<ElevenLabsPronunciationDictionaryCreateResponse, LlmError> {
        request.validate()?;
        let url = join_url(
            &self.base_url(),
            "v1/pronunciation-dictionaries/add-from-file",
        );
        let request_clone = request.clone();
        execute_multipart_json(
            &self.config,
            self.http_client.clone(),
            self.retry_options.clone(),
            &url,
            move || request_clone.build_form(),
            request.http_config.as_ref(),
            "create pronunciation dictionary from file",
        )
        .await
    }

    /// List pronunciation dictionary metadata using `GET /v1/pronunciation-dictionaries`.
    pub async fn list(
        &self,
        query: Option<ElevenLabsPronunciationDictionaryListQuery>,
    ) -> Result<ElevenLabsPronunciationDictionaryListResponse, LlmError> {
        let url = self.list_url(query.as_ref())?;
        let per_request_http_config = query.as_ref().and_then(|query| query.http_config.as_ref());
        execute_get_json(
            &self.config,
            self.http_client.clone(),
            self.retry_options.clone(),
            &url,
            per_request_http_config,
            "list pronunciation dictionaries",
        )
        .await
    }

    /// Retrieve pronunciation dictionary metadata and rules using
    /// `GET /v1/pronunciation-dictionaries/{pronunciation_dictionary_id}`.
    pub async fn get(
        &self,
        pronunciation_dictionary_id: impl AsRef<str>,
    ) -> Result<ElevenLabsPronunciationDictionary, LlmError> {
        self.get_with_http_config(pronunciation_dictionary_id, None)
            .await
    }

    /// Retrieve a pronunciation dictionary with per-request HTTP configuration.
    pub async fn get_with_http_config(
        &self,
        pronunciation_dictionary_id: impl AsRef<str>,
        http_config: Option<&HttpConfig>,
    ) -> Result<ElevenLabsPronunciationDictionary, LlmError> {
        let pronunciation_dictionary_id = pronunciation_dictionary_id.as_ref().trim();
        if pronunciation_dictionary_id.is_empty() {
            return Err(LlmError::InvalidInput(
                "ElevenLabs pronunciation_dictionary_id cannot be empty".to_string(),
            ));
        }

        let encoded = urlencoding::encode(pronunciation_dictionary_id);
        let url = join_url(
            &self.base_url(),
            &format!("v1/pronunciation-dictionaries/{encoded}"),
        );
        execute_get_json(
            &self.config,
            self.http_client.clone(),
            self.retry_options.clone(),
            &url,
            http_config,
            "get pronunciation dictionary",
        )
        .await
    }

    fn base_url(&self) -> String {
        self.config.base_url.trim_end_matches('/').to_string()
    }

    fn list_url(
        &self,
        query: Option<&ElevenLabsPronunciationDictionaryListQuery>,
    ) -> Result<String, LlmError> {
        let mut url = Url::parse(&join_url(&self.base_url(), "v1/pronunciation-dictionaries"))
            .map_err(|e| {
                LlmError::InvalidInput(format!(
                    "Invalid ElevenLabs pronunciation dictionaries URL: {e}"
                ))
            })?;

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

/// Query parameters for `GET /v1/pronunciation-dictionaries`.
#[derive(Debug, Clone, Default)]
pub struct ElevenLabsPronunciationDictionaryListQuery {
    pub cursor: Option<String>,
    pub page_size: Option<u32>,
    pub sort: Option<String>,
    pub sort_direction: Option<String>,
    pub http_config: Option<HttpConfig>,
}

impl ElevenLabsPronunciationDictionaryListQuery {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_cursor(mut self, value: impl Into<String>) -> Self {
        self.cursor = Some(value.into());
        self
    }

    pub fn with_page_size(mut self, value: u32) -> Self {
        self.page_size = Some(value);
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

    pub fn with_http_config(mut self, value: HttpConfig) -> Self {
        self.http_config = Some(value);
        self
    }

    fn validate(&self) -> Result<(), LlmError> {
        if let Some(page_size) = self.page_size {
            if page_size == 0 || page_size > 100 {
                return Err(LlmError::InvalidInput(
                    "ElevenLabs pronunciation dictionary page_size must be between 1 and 100"
                        .to_string(),
                ));
            }
        }
        Ok(())
    }

    fn to_query_pairs(&self) -> Vec<(String, String)> {
        let mut pairs = Vec::new();
        push_optional(&mut pairs, "cursor", self.cursor.as_deref());
        if let Some(page_size) = self.page_size {
            pairs.push(("page_size".to_string(), page_size.to_string()));
        }
        push_optional(&mut pairs, "sort", self.sort.as_deref());
        push_optional(&mut pairs, "sort_direction", self.sort_direction.as_deref());
        pairs
    }
}

fn push_optional(pairs: &mut Vec<(String, String)>, key: &str, value: Option<&str>) {
    if let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) {
        pairs.push((key.to_string(), value.to_string()));
    }
}

fn optional_trimmed(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

/// Request body for creating a pronunciation dictionary from inline rules.
#[derive(Debug, Clone)]
pub struct ElevenLabsCreatePronunciationDictionaryFromRulesRequest {
    pub rules: Vec<ElevenLabsPronunciationDictionaryRuleRequest>,
    pub name: String,
    pub description: Option<String>,
    pub workspace_access: Option<String>,
    pub http_config: Option<HttpConfig>,
}

impl ElevenLabsCreatePronunciationDictionaryFromRulesRequest {
    pub fn new(
        name: impl Into<String>,
        rules: Vec<ElevenLabsPronunciationDictionaryRuleRequest>,
    ) -> Self {
        Self {
            rules,
            name: name.into(),
            description: None,
            workspace_access: None,
            http_config: None,
        }
    }

    pub fn with_description(mut self, value: impl Into<String>) -> Self {
        self.description = Some(value.into());
        self
    }

    pub fn with_workspace_access(mut self, value: impl Into<String>) -> Self {
        self.workspace_access = Some(value.into());
        self
    }

    pub fn with_http_config(mut self, value: HttpConfig) -> Self {
        self.http_config = Some(value);
        self
    }

    fn validate(&self) -> Result<(), LlmError> {
        if self.name.trim().is_empty() {
            return Err(LlmError::InvalidInput(
                "ElevenLabs pronunciation dictionary name cannot be empty".to_string(),
            ));
        }

        for rule in &self.rules {
            rule.validate()?;
        }

        Ok(())
    }

    fn body(&self) -> Result<Value, LlmError> {
        #[derive(Serialize)]
        struct Body<'a> {
            rules: &'a [ElevenLabsPronunciationDictionaryRuleRequest],
            name: &'a str,
            #[serde(skip_serializing_if = "Option::is_none")]
            description: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            workspace_access: Option<&'a str>,
        }

        serde_json::to_value(Body {
            rules: &self.rules,
            name: self.name.trim(),
            description: optional_trimmed(self.description.as_deref()),
            workspace_access: optional_trimmed(self.workspace_access.as_deref()),
        })
        .map_err(|e| {
            LlmError::InvalidInput(format!(
                "Invalid ElevenLabs pronunciation dictionary create request: {e}"
            ))
        })
    }
}

/// Request body for creating a pronunciation dictionary from a caller-provided PLS file.
#[derive(Debug, Clone)]
pub struct ElevenLabsCreatePronunciationDictionaryFromFileRequest {
    pub name: String,
    pub file: Vec<u8>,
    pub filename: Option<String>,
    pub mime_type: Option<String>,
    pub description: Option<String>,
    pub workspace_access: Option<String>,
    pub http_config: Option<HttpConfig>,
}

impl ElevenLabsCreatePronunciationDictionaryFromFileRequest {
    pub fn new(name: impl Into<String>, file: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            file,
            filename: None,
            mime_type: None,
            description: None,
            workspace_access: None,
            http_config: None,
        }
    }

    pub fn with_filename(mut self, value: impl Into<String>) -> Self {
        self.filename = Some(value.into());
        self
    }

    pub fn with_mime_type(mut self, value: impl Into<String>) -> Self {
        self.mime_type = Some(value.into());
        self
    }

    pub fn with_description(mut self, value: impl Into<String>) -> Self {
        self.description = Some(value.into());
        self
    }

    pub fn with_workspace_access(mut self, value: impl Into<String>) -> Self {
        self.workspace_access = Some(value.into());
        self
    }

    pub fn with_http_config(mut self, value: HttpConfig) -> Self {
        self.http_config = Some(value);
        self
    }

    fn validate(&self) -> Result<(), LlmError> {
        if self.name.trim().is_empty() {
            return Err(LlmError::InvalidInput(
                "ElevenLabs pronunciation dictionary name cannot be empty".to_string(),
            ));
        }
        if self.file.is_empty() {
            return Err(LlmError::InvalidInput(
                "ElevenLabs pronunciation dictionary file cannot be empty".to_string(),
            ));
        }
        Ok(())
    }

    fn build_form(&self) -> Result<reqwest::multipart::Form, LlmError> {
        let mut file_part = reqwest::multipart::Part::bytes(self.file.clone());
        if let Some(filename) = optional_trimmed(self.filename.as_deref()) {
            file_part = file_part.file_name(filename.to_string());
        }
        if let Some(mime_type) = optional_trimmed(self.mime_type.as_deref()) {
            file_part = file_part.mime_str(mime_type).map_err(|e| {
                LlmError::InvalidInput(format!(
                    "Invalid ElevenLabs pronunciation dictionary file MIME type '{mime_type}': {e}"
                ))
            })?;
        }

        let mut form = reqwest::multipart::Form::new()
            .text("name", self.name.trim().to_string())
            .part("file", file_part);

        if let Some(description) = optional_trimmed(self.description.as_deref()) {
            form = form.text("description", description.to_string());
        }
        if let Some(workspace_access) = optional_trimmed(self.workspace_access.as_deref()) {
            form = form.text("workspace_access", workspace_access.to_string());
        }

        Ok(form)
    }
}

/// Alias or phoneme rule used when creating or mutating pronunciation dictionaries.
#[derive(Debug, Clone, Serialize)]
pub struct ElevenLabsPronunciationDictionaryRuleRequest {
    pub string_to_replace: String,
    #[serde(rename = "type")]
    pub rule_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub case_sensitive: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub word_boundaries: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alias: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phoneme: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alphabet: Option<String>,
}

impl ElevenLabsPronunciationDictionaryRuleRequest {
    pub fn alias(string_to_replace: impl Into<String>, alias: impl Into<String>) -> Self {
        Self {
            string_to_replace: string_to_replace.into(),
            rule_type: "alias".to_string(),
            case_sensitive: None,
            word_boundaries: None,
            alias: Some(alias.into()),
            phoneme: None,
            alphabet: None,
        }
    }

    pub fn phoneme(
        string_to_replace: impl Into<String>,
        phoneme: impl Into<String>,
        alphabet: impl Into<String>,
    ) -> Self {
        Self {
            string_to_replace: string_to_replace.into(),
            rule_type: "phoneme".to_string(),
            case_sensitive: None,
            word_boundaries: None,
            alias: None,
            phoneme: Some(phoneme.into()),
            alphabet: Some(alphabet.into()),
        }
    }

    pub fn with_case_sensitive(mut self, value: bool) -> Self {
        self.case_sensitive = Some(value);
        self
    }

    pub fn with_word_boundaries(mut self, value: bool) -> Self {
        self.word_boundaries = Some(value);
        self
    }

    fn validate(&self) -> Result<(), LlmError> {
        if self.string_to_replace.trim().is_empty() {
            return Err(LlmError::InvalidInput(
                "ElevenLabs pronunciation dictionary rule string_to_replace cannot be empty"
                    .to_string(),
            ));
        }

        match self.rule_type.as_str() {
            "alias" => {
                if optional_trimmed(self.alias.as_deref()).is_none() {
                    return Err(LlmError::InvalidInput(
                        "ElevenLabs pronunciation dictionary alias rule requires alias".to_string(),
                    ));
                }
            }
            "phoneme" => {
                if optional_trimmed(self.phoneme.as_deref()).is_none() {
                    return Err(LlmError::InvalidInput(
                        "ElevenLabs pronunciation dictionary phoneme rule requires phoneme"
                            .to_string(),
                    ));
                }
                if optional_trimmed(self.alphabet.as_deref()).is_none() {
                    return Err(LlmError::InvalidInput(
                        "ElevenLabs pronunciation dictionary phoneme rule requires alphabet"
                            .to_string(),
                    ));
                }
            }
            other => {
                return Err(LlmError::InvalidInput(format!(
                    "Unsupported ElevenLabs pronunciation dictionary rule type: {other}"
                )));
            }
        }

        Ok(())
    }
}

/// Response body for pronunciation dictionary create endpoints.
#[derive(Debug, Clone, Deserialize)]
pub struct ElevenLabsPronunciationDictionaryCreateResponse {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub created_by: Option<String>,
    #[serde(default)]
    pub creation_time_unix: Option<i64>,
    pub version_id: String,
    #[serde(default)]
    pub version_rules_num: Option<u32>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub permission_on_resource: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Response body for `GET /v1/pronunciation-dictionaries`.
#[derive(Debug, Clone, Deserialize)]
pub struct ElevenLabsPronunciationDictionaryListResponse {
    #[serde(default)]
    pub pronunciation_dictionaries: Vec<ElevenLabsPronunciationDictionary>,
    #[serde(default)]
    pub next_cursor: Option<String>,
    #[serde(default)]
    pub has_more: bool,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Pronunciation dictionary metadata returned by ElevenLabs.
#[derive(Debug, Clone, Deserialize)]
pub struct ElevenLabsPronunciationDictionary {
    pub id: String,
    #[serde(default)]
    pub latest_version_id: Option<String>,
    #[serde(default)]
    pub latest_version_rules_num: Option<u32>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub permission_on_resource: Option<String>,
    #[serde(default)]
    pub created_by: Option<String>,
    #[serde(default)]
    pub creation_time_unix: Option<i64>,
    #[serde(default)]
    pub archived_time_unix: Option<i64>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub rules: Option<Vec<ElevenLabsPronunciationDictionaryRule>>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Alias or phoneme rule returned by the dictionary detail endpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct ElevenLabsPronunciationDictionaryRule {
    #[serde(default)]
    pub string_to_replace: Option<String>,
    #[serde(default)]
    pub case_sensitive: Option<bool>,
    #[serde(default)]
    pub word_boundaries: Option<bool>,
    #[serde(rename = "type", default)]
    pub rule_type: Option<String>,
    #[serde(default)]
    pub alias: Option<String>,
    #[serde(default)]
    pub phoneme: Option<String>,
    #[serde(default)]
    pub alphabet: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::transport::{
        HttpTransport, HttpTransportGetRequest, HttpTransportMultipartRequest,
        HttpTransportRequest, HttpTransportResponse,
    };
    use async_trait::async_trait;
    use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
    use serde_json::json;
    use std::sync::{Arc, Mutex};

    use super::super::resource_http::XI_API_KEY;

    #[derive(Clone)]
    struct JsonGetTransport {
        response: Value,
        last_json: Arc<Mutex<Option<HttpTransportRequest>>>,
        last_get: Arc<Mutex<Option<HttpTransportGetRequest>>>,
        last_multipart: Arc<Mutex<Option<HttpTransportMultipartRequest>>>,
    }

    impl JsonGetTransport {
        fn new(response: Value) -> Self {
            Self {
                response,
                last_json: Arc::new(Mutex::new(None)),
                last_get: Arc::new(Mutex::new(None)),
                last_multipart: Arc::new(Mutex::new(None)),
            }
        }

        fn take_json(&self) -> HttpTransportRequest {
            self.last_json
                .lock()
                .expect("json transport lock")
                .take()
                .expect("captured json request")
        }

        fn take_get(&self) -> HttpTransportGetRequest {
            self.last_get
                .lock()
                .expect("get transport lock")
                .take()
                .expect("captured get request")
        }

        fn take_multipart(&self) -> HttpTransportMultipartRequest {
            self.last_multipart
                .lock()
                .expect("multipart transport lock")
                .take()
                .expect("captured multipart request")
        }
    }

    #[async_trait]
    impl HttpTransport for JsonGetTransport {
        async fn execute_json(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last_json.lock().expect("json transport lock") = Some(request);
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
            Ok(HttpTransportResponse {
                status: 200,
                headers,
                body: serde_json::to_vec(&self.response).expect("serialize response"),
            })
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
    async fn pronunciation_dictionaries_list_maps_query_headers_and_metadata() {
        let transport = JsonGetTransport::new(json!({
            "pronunciation_dictionaries": [
                {
                    "id": "dict-1",
                    "latest_version_id": "version-2",
                    "latest_version_rules_num": 2,
                    "name": "Product terms",
                    "permission_on_resource": "viewer",
                    "created_by": "user-1",
                    "creation_time_unix": 1_700_000_000,
                    "archived_time_unix": null,
                    "description": "Brand pronunciation",
                    "unknown_dict": "kept"
                }
            ],
            "next_cursor": "cursor-2",
            "has_more": true,
            "unknown_list": "kept"
        }));
        let mut request_http = HttpConfig::empty();
        request_http
            .headers
            .insert("x-request-header".to_string(), "request".to_string());

        let config = ElevenLabsConfig::new("test-key")
            .with_base_url("https://api.elevenlabs.test")
            .with_header("x-global-header", "global")
            .with_http_transport(Arc::new(transport.clone()));
        let dictionaries = ElevenLabsPronunciationDictionaries::new(
            config,
            reqwest::Client::new(),
            Some(RetryOptions::policy_default().with_max_attempts(1)),
        );

        let response = dictionaries
            .list(Some(
                ElevenLabsPronunciationDictionaryListQuery::new()
                    .with_cursor("cursor-1")
                    .with_page_size(30)
                    .with_sort("creation_time_unix")
                    .with_sort_direction("descending")
                    .with_http_config(request_http),
            ))
            .await
            .expect("pronunciation dictionary list response");

        let captured = transport.take_get();
        assert_eq!(
            captured.url.split('?').next(),
            Some("https://api.elevenlabs.test/v1/pronunciation-dictionaries")
        );
        assert_eq!(query_values(&captured.url, "cursor"), vec!["cursor-1"]);
        assert_eq!(query_values(&captured.url, "page_size"), vec!["30"]);
        assert_eq!(
            query_values(&captured.url, "sort"),
            vec!["creation_time_unix"]
        );
        assert_eq!(
            query_values(&captured.url, "sort_direction"),
            vec!["descending"]
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

        assert_eq!(response.has_more, true);
        assert_eq!(response.next_cursor.as_deref(), Some("cursor-2"));
        assert_eq!(response.extra.get("unknown_list"), Some(&json!("kept")));
        let dictionary = response
            .pronunciation_dictionaries
            .first()
            .expect("dictionary");
        assert_eq!(dictionary.id, "dict-1");
        assert_eq!(dictionary.latest_version_id.as_deref(), Some("version-2"));
        assert_eq!(dictionary.latest_version_rules_num, Some(2));
        assert_eq!(dictionary.name.as_deref(), Some("Product terms"));
        assert_eq!(dictionary.permission_on_resource.as_deref(), Some("viewer"));
        assert_eq!(dictionary.created_by.as_deref(), Some("user-1"));
        assert_eq!(dictionary.creation_time_unix, Some(1_700_000_000));
        assert_eq!(
            dictionary.description.as_deref(),
            Some("Brand pronunciation")
        );
        assert_eq!(dictionary.extra.get("unknown_dict"), Some(&json!("kept")));
    }

    #[tokio::test]
    async fn pronunciation_dictionaries_get_encodes_id_and_maps_rules() {
        let transport = JsonGetTransport::new(json!({
            "id": "dict/id with space",
            "latest_version_id": "version-3",
            "latest_version_rules_num": 2,
            "name": "Names",
            "permission_on_resource": "admin",
            "created_by": "user-1",
            "creation_time_unix": 1_700_000_000,
            "description": null,
            "rules": [
                {
                    "string_to_replace": "SQL",
                    "type": "alias",
                    "alias": "sequel",
                    "case_sensitive": false,
                    "word_boundaries": true,
                    "unknown_rule": "kept"
                },
                {
                    "string_to_replace": "Siumai",
                    "type": "phoneme",
                    "phoneme": "ˈsuːmaɪ",
                    "alphabet": "ipa"
                }
            ],
            "unknown_dict": "kept"
        }));
        let config = ElevenLabsConfig::new("test-key")
            .with_base_url("https://api.elevenlabs.test/")
            .with_http_transport(Arc::new(transport.clone()));
        let dictionaries =
            ElevenLabsPronunciationDictionaries::new(config, reqwest::Client::new(), None);

        let response = dictionaries
            .get("dict/id with space")
            .await
            .expect("pronunciation dictionary response");

        let captured = transport.take_get();
        assert_eq!(
            captured.url,
            "https://api.elevenlabs.test/v1/pronunciation-dictionaries/dict%2Fid%20with%20space"
        );
        assert_eq!(
            header_value(&captured.headers, XI_API_KEY),
            Some("test-key")
        );
        assert_eq!(response.id, "dict/id with space");
        assert_eq!(response.latest_version_id.as_deref(), Some("version-3"));
        assert_eq!(response.extra.get("unknown_dict"), Some(&json!("kept")));
        let rules = response.rules.expect("rules");
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].rule_type.as_deref(), Some("alias"));
        assert_eq!(rules[0].string_to_replace.as_deref(), Some("SQL"));
        assert_eq!(rules[0].alias.as_deref(), Some("sequel"));
        assert_eq!(rules[0].case_sensitive, Some(false));
        assert_eq!(rules[0].word_boundaries, Some(true));
        assert_eq!(rules[0].extra.get("unknown_rule"), Some(&json!("kept")));
        assert_eq!(rules[1].rule_type.as_deref(), Some("phoneme"));
        assert_eq!(rules[1].string_to_replace.as_deref(), Some("Siumai"));
        assert_eq!(rules[1].phoneme.as_deref(), Some("ˈsuːmaɪ"));
        assert_eq!(rules[1].alphabet.as_deref(), Some("ipa"));
    }

    #[tokio::test]
    async fn pronunciation_dictionaries_create_from_rules_posts_json_and_maps_response() {
        let transport = JsonGetTransport::new(json!({
            "id": "dict-1",
            "name": "Product terms",
            "created_by": "user-1",
            "creation_time_unix": 1_700_000_000,
            "version_id": "version-1",
            "version_rules_num": 2,
            "description": "Brand pronunciation",
            "permission_on_resource": "viewer",
            "unknown_create": "kept"
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
        let dictionaries = ElevenLabsPronunciationDictionaries::new(
            config,
            reqwest::Client::new(),
            Some(RetryOptions::policy_default().with_max_attempts(1)),
        );

        let response = dictionaries
            .create_from_rules(
                ElevenLabsCreatePronunciationDictionaryFromRulesRequest::new(
                    "Product terms",
                    vec![
                        ElevenLabsPronunciationDictionaryRuleRequest::alias("Siumai", "sue my")
                            .with_case_sensitive(false)
                            .with_word_boundaries(true),
                        ElevenLabsPronunciationDictionaryRuleRequest::phoneme(
                            "route", "rut", "ipa",
                        ),
                    ],
                )
                .with_description("Brand pronunciation")
                .with_workspace_access("viewer")
                .with_http_config(request_http),
            )
            .await
            .expect("pronunciation dictionary create response");

        let captured = transport.take_json();
        assert_eq!(
            captured.url,
            "https://api.elevenlabs.test/v1/pronunciation-dictionaries/add-from-rules"
        );
        assert_eq!(
            header_value(&captured.headers, XI_API_KEY),
            Some("test-key")
        );
        assert_eq!(
            header_value(&captured.headers, "content-type"),
            Some("application/json")
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
        assert_eq!(
            captured.body,
            json!({
                "rules": [
                    {
                        "string_to_replace": "Siumai",
                        "type": "alias",
                        "case_sensitive": false,
                        "word_boundaries": true,
                        "alias": "sue my"
                    },
                    {
                        "string_to_replace": "route",
                        "type": "phoneme",
                        "phoneme": "rut",
                        "alphabet": "ipa"
                    }
                ],
                "name": "Product terms",
                "description": "Brand pronunciation",
                "workspace_access": "viewer"
            })
        );

        assert_eq!(response.id, "dict-1");
        assert_eq!(response.name.as_deref(), Some("Product terms"));
        assert_eq!(response.created_by.as_deref(), Some("user-1"));
        assert_eq!(response.creation_time_unix, Some(1_700_000_000));
        assert_eq!(response.version_id, "version-1");
        assert_eq!(response.version_rules_num, Some(2));
        assert_eq!(response.description.as_deref(), Some("Brand pronunciation"));
        assert_eq!(response.permission_on_resource.as_deref(), Some("viewer"));
        assert_eq!(response.extra.get("unknown_create"), Some(&json!("kept")));
    }

    #[tokio::test]
    async fn pronunciation_dictionaries_create_from_file_posts_multipart_and_maps_response() {
        let transport = JsonGetTransport::new(json!({
            "id": "dict-2",
            "name": "File terms",
            "created_by": "user-2",
            "creation_time_unix": 1_700_000_001,
            "version_id": "version-2",
            "version_rules_num": 3,
            "description": "PLS pronunciation",
            "permission_on_resource": "editor",
            "unknown_create": "kept"
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
        let dictionaries = ElevenLabsPronunciationDictionaries::new(
            config,
            reqwest::Client::new(),
            Some(RetryOptions::policy_default().with_max_attempts(1)),
        );

        let response = dictionaries
            .create_from_file(
                ElevenLabsCreatePronunciationDictionaryFromFileRequest::new(
                    "File terms",
                    b"<lexicon><lexeme>route</lexeme></lexicon>".to_vec(),
                )
                .with_filename("terms.pls")
                .with_mime_type("application/pls+xml")
                .with_description("PLS pronunciation")
                .with_workspace_access("editor")
                .with_http_config(request_http),
            )
            .await
            .expect("pronunciation dictionary create-from-file response");

        let captured = transport.take_multipart();
        assert_eq!(
            captured.url,
            "https://api.elevenlabs.test/v1/pronunciation-dictionaries/add-from-file"
        );
        assert_eq!(
            header_value(&captured.headers, XI_API_KEY),
            Some("test-key")
        );
        assert!(
            header_value(&captured.headers, "content-type")
                .is_some_and(|value| value.starts_with("multipart/form-data; boundary="))
        );
        assert!(header_value(&captured.headers, "content-length").is_some());
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

        let body = String::from_utf8_lossy(&captured.body);
        assert!(body.contains("name=\"name\""));
        assert!(body.contains("File terms"));
        assert!(body.contains("name=\"description\""));
        assert!(body.contains("PLS pronunciation"));
        assert!(body.contains("name=\"workspace_access\""));
        assert!(body.contains("editor"));
        assert!(body.contains("name=\"file\"; filename=\"terms.pls\""));
        assert!(body.contains("Content-Type: application/pls+xml"));
        assert!(body.contains("<lexicon><lexeme>route</lexeme></lexicon>"));

        assert_eq!(response.id, "dict-2");
        assert_eq!(response.name.as_deref(), Some("File terms"));
        assert_eq!(response.created_by.as_deref(), Some("user-2"));
        assert_eq!(response.creation_time_unix, Some(1_700_000_001));
        assert_eq!(response.version_id, "version-2");
        assert_eq!(response.version_rules_num, Some(3));
        assert_eq!(response.description.as_deref(), Some("PLS pronunciation"));
        assert_eq!(response.permission_on_resource.as_deref(), Some("editor"));
        assert_eq!(response.extra.get("unknown_create"), Some(&json!("kept")));
    }
}
