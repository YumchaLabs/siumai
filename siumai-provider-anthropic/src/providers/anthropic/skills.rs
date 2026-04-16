//! Anthropic Skills API (provider-specific).
//!
//! This resource mirrors the AI SDK Anthropic `skills()` surface:
//! - `POST /v1/skills`
//! - `GET /v1/skills/{skill_id}/versions/{version}`

use crate::error::LlmError;
use crate::execution::executors::common::{
    HttpExecutionConfig, execute_get_request, execute_multipart_request,
};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::retry_api::RetryOptions;
use crate::traits::SkillsCapability;
use crate::types::{
    HttpConfig, ProviderReference, SkillFileContent, SkillUploadFile as SharedSkillUploadFile,
    SkillUploadRequest, SkillUploadResult as SharedSkillUploadResult, Warning,
};
use crate::utils::url::join_url;
use async_trait::async_trait;
use base64::{Engine, engine::general_purpose::STANDARD};
use reqwest::Client as HttpClient;
use secrecy::{ExposeSecret, SecretString};
use std::collections::HashMap;
use std::sync::Arc;

/// Provider-id keyed metadata map returned by `AnthropicSkills::upload`.
pub type AnthropicSkillProviderMetadata = HashMap<String, serde_json::Value>;

/// File content accepted by Anthropic skill uploads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnthropicSkillFileContent {
    /// Raw file bytes.
    Bytes(Vec<u8>),
    /// Base64-encoded file bytes.
    Base64(String),
}

impl AnthropicSkillFileContent {
    /// Create file content from raw bytes.
    pub fn bytes(data: Vec<u8>) -> Self {
        Self::Bytes(data)
    }

    /// Create file content from base64.
    pub fn base64(data: impl Into<String>) -> Self {
        Self::Base64(data.into())
    }

    fn into_bytes(self) -> Result<Vec<u8>, LlmError> {
        match self {
            Self::Bytes(data) => Ok(data),
            Self::Base64(data) => STANDARD.decode(data).map_err(|error| {
                LlmError::InvalidInput(format!("Invalid base64 skill file content: {error}"))
            }),
        }
    }
}

impl From<Vec<u8>> for AnthropicSkillFileContent {
    fn from(value: Vec<u8>) -> Self {
        Self::Bytes(value)
    }
}

impl From<&[u8]> for AnthropicSkillFileContent {
    fn from(value: &[u8]) -> Self {
        Self::Bytes(value.to_vec())
    }
}

/// One uploaded skill file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnthropicSkillFile {
    /// File path relative to the skill root.
    pub path: String,
    /// File content, as bytes or base64.
    pub content: AnthropicSkillFileContent,
}

impl AnthropicSkillFile {
    /// Create a skill file from a path and content.
    pub fn new(path: impl Into<String>, content: impl Into<AnthropicSkillFileContent>) -> Self {
        Self {
            path: path.into(),
            content: content.into(),
        }
    }

    /// Create a skill file from raw bytes.
    pub fn bytes(path: impl Into<String>, data: Vec<u8>) -> Self {
        Self::new(path, AnthropicSkillFileContent::Bytes(data))
    }

    /// Create a skill file from base64.
    pub fn base64(path: impl Into<String>, data: impl Into<String>) -> Self {
        Self::new(path, AnthropicSkillFileContent::Base64(data.into()))
    }
}

/// Canonical result returned by `AnthropicSkills::upload`.
#[derive(Debug, Clone, PartialEq)]
pub struct AnthropicSkillUploadResult {
    /// Provider-owned skill reference in stable AI SDK-style shape.
    pub provider_reference: ProviderReference,
    /// Optional human-readable title.
    pub display_title: Option<String>,
    /// Optional canonical skill name.
    pub name: Option<String>,
    /// Optional skill description.
    pub description: Option<String>,
    /// Optional latest version id.
    pub latest_version: Option<String>,
    /// Provider-owned metadata under the provider id root.
    pub provider_metadata: Option<AnthropicSkillProviderMetadata>,
    /// Non-fatal warnings emitted while uploading.
    pub warnings: Vec<Warning>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct AnthropicSkillResponse {
    id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    display_title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    latest_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    created_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    updated_at: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct AnthropicSkillVersionResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
}

/// Provider-scoped client for skill uploads.
#[derive(Clone)]
pub struct AnthropicSkills {
    pub(crate) api_key: SecretString,
    pub(crate) base_url: String,
    pub(crate) http_client: HttpClient,
    pub(crate) http_config: crate::types::HttpConfig,
    pub(crate) beta_features: Vec<String>,
    pub(crate) http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    pub(crate) http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    pub(crate) retry_options: Option<RetryOptions>,
}

impl AnthropicSkills {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        api_key: SecretString,
        base_url: String,
        http_client: HttpClient,
        http_config: crate::types::HttpConfig,
        beta_features: Vec<String>,
        http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
        http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
        retry_options: Option<RetryOptions>,
    ) -> Self {
        Self {
            api_key,
            base_url,
            http_client,
            http_config,
            beta_features,
            http_transport,
            http_interceptors,
            retry_options,
        }
    }

    /// Upload a new Anthropic skill from the provided files.
    pub async fn upload(
        &self,
        files: Vec<AnthropicSkillFile>,
        display_title: Option<String>,
        per_request_headers: Option<HashMap<String, String>>,
    ) -> Result<AnthropicSkillUploadResult, LlmError> {
        if files.is_empty() {
            return Err(LlmError::InvalidInput(
                "Anthropic skill uploads require at least one file.".to_string(),
            ));
        }

        let url = join_url(&self.base_url, "skills");
        let ctx = self.build_context();
        let http_config = self.build_http_config(ctx);
        let base_url = self.base_url.clone();

        let call = || {
            let http_config = http_config.clone();
            let url = url.clone();
            let base_url = base_url.clone();
            let headers = per_request_headers.clone();
            let files = files.clone();
            let display_title = display_title.clone();

            async move {
                let per_request_http_config = headers.as_ref().map(|headers| {
                    let mut config = crate::types::HttpConfig::empty();
                    config.headers = headers.clone();
                    config
                });

                let response = execute_multipart_request(
                    &http_config,
                    &url,
                    || build_upload_form(&files, display_title.as_deref()),
                    per_request_http_config.as_ref(),
                )
                .await?;

                let response: AnthropicSkillResponse = serde_json::from_value(response.json)
                    .map_err(|error| {
                        LlmError::ParseError(format!(
                            "Failed to parse Anthropic skill upload response: {error}"
                        ))
                    })?;

                let version_metadata = match response.latest_version.as_deref() {
                    Some(version) if !version.trim().is_empty() => Some(
                        fetch_version_metadata(
                            &http_config,
                            &base_url,
                            &response.id,
                            version,
                            per_request_http_config.as_ref(),
                        )
                        .await?,
                    ),
                    _ => None,
                };

                Ok(build_upload_result(response, version_metadata))
            }
        };

        crate::retry_api::maybe_retry(self.retry_options.clone(), call).await
    }

    fn build_context(&self) -> crate::core::ProviderContext {
        let mut headers = self.http_config.headers.clone();

        if !self.beta_features.is_empty() {
            let mut existing_values: Vec<String> = Vec::new();
            let keys: Vec<String> = headers
                .keys()
                .filter(|key| key.eq_ignore_ascii_case("anthropic-beta"))
                .cloned()
                .collect();
            for key in keys {
                if let Some(value) = headers.remove(&key) {
                    existing_values.push(value);
                }
            }

            let mut merged = String::new();
            if !existing_values.is_empty() {
                merged.push_str(&existing_values.join(","));
            }
            if !merged.is_empty() {
                merged.push(',');
            }
            merged.push_str(&self.beta_features.join(","));

            headers.insert("anthropic-beta".to_string(), merged);
        }

        crate::core::ProviderContext::new(
            "anthropic",
            self.base_url.clone(),
            Some(self.api_key.expose_secret().to_string()),
            headers,
        )
    }

    fn build_http_config(&self, ctx: crate::core::ProviderContext) -> HttpExecutionConfig {
        let mut wiring = crate::execution::wiring::HttpExecutionWiring::new(
            "anthropic",
            self.http_client.clone(),
            ctx,
        )
        .with_interceptors(self.http_interceptors.clone())
        .with_retry_options(self.retry_options.clone());

        if let Some(transport) = self.http_transport.clone() {
            wiring = wiring.with_transport(transport);
        }

        wiring.config(Arc::new(super::spec::AnthropicSpec::new()))
    }
}

fn build_upload_form(
    files: &[AnthropicSkillFile],
    display_title: Option<&str>,
) -> Result<reqwest::multipart::Form, LlmError> {
    let mut form = reqwest::multipart::Form::new();

    if let Some(display_title) = display_title.filter(|value| !value.trim().is_empty()) {
        form = form.text("display_title", display_title.to_string());
    }

    for file in files {
        let bytes = file.content.clone().into_bytes()?;
        let part = reqwest::multipart::Part::bytes(bytes).file_name(file.path.clone());
        form = form.part("files[]", part);
    }

    Ok(form)
}

async fn fetch_version_metadata(
    http_config: &HttpExecutionConfig,
    base_url: &str,
    skill_id: &str,
    version: &str,
    per_request_http_config: Option<&crate::types::HttpConfig>,
) -> Result<AnthropicSkillVersionResponse, LlmError> {
    let url = join_url(base_url, &format!("skills/{skill_id}/versions/{version}"));
    let response = execute_get_request(http_config, &url, per_request_http_config).await?;

    serde_json::from_value(response.json).map_err(|error| {
        LlmError::ParseError(format!(
            "Failed to parse Anthropic skill version response: {error}"
        ))
    })
}

fn build_upload_result(
    response: AnthropicSkillResponse,
    version_metadata: Option<AnthropicSkillVersionResponse>,
) -> AnthropicSkillUploadResult {
    let name = version_metadata
        .as_ref()
        .and_then(|metadata| metadata.name.clone())
        .or(response.name.clone());
    let description = version_metadata
        .as_ref()
        .and_then(|metadata| metadata.description.clone())
        .or(response.description.clone());

    let mut provider_metadata = HashMap::new();
    let mut anthropic = HashMap::new();
    if let Some(source) = response.source {
        anthropic.insert("source".to_string(), serde_json::Value::String(source));
    }
    if let Some(created_at) = response.created_at {
        anthropic.insert(
            "createdAt".to_string(),
            serde_json::Value::String(created_at),
        );
    }
    if let Some(updated_at) = response.updated_at {
        anthropic.insert(
            "updatedAt".to_string(),
            serde_json::Value::String(updated_at),
        );
    }
    if !anthropic.is_empty() {
        provider_metadata.insert(
            "anthropic".to_string(),
            serde_json::Value::Object(anthropic.into_iter().collect()),
        );
    }

    AnthropicSkillUploadResult {
        provider_reference: ProviderReference::single("anthropic", response.id),
        display_title: response.display_title,
        name,
        description,
        latest_version: response.latest_version,
        provider_metadata: (!provider_metadata.is_empty()).then_some(provider_metadata),
        warnings: Vec::new(),
    }
}

#[async_trait]
impl SkillsCapability for AnthropicSkills {
    async fn upload_skill(
        &self,
        request: SkillUploadRequest,
    ) -> Result<SharedSkillUploadResult, LlmError> {
        let SkillUploadRequest {
            files,
            display_title,
            provider_options: _provider_options,
            http_config,
        } = request;

        let headers = http_config
            .as_ref()
            .map(|config| config.headers.clone())
            .filter(|headers| !headers.is_empty());

        let mut result = self
            .upload(
                files
                    .into_iter()
                    .map(shared_skill_file_to_anthropic)
                    .collect(),
                display_title,
                headers,
            )
            .await
            .map(shared_skill_result_from_anthropic)?;

        if http_config
            .as_ref()
            .is_some_and(has_non_header_http_overrides)
        {
            result.warnings.push(Warning::compatibility(
                "httpConfig",
                Some("Anthropic skill uploads currently forward only per-request headers."),
            ));
        }

        Ok(result)
    }
}

#[async_trait]
impl SkillsCapability for super::AnthropicClient {
    async fn upload_skill(
        &self,
        request: SkillUploadRequest,
    ) -> Result<SharedSkillUploadResult, LlmError> {
        self.skills().upload_skill(request).await
    }
}

fn shared_skill_file_to_anthropic(file: SharedSkillUploadFile) -> AnthropicSkillFile {
    let content = match file.content {
        SkillFileContent::Bytes(data) => AnthropicSkillFileContent::Bytes(data),
        SkillFileContent::Base64(data) => AnthropicSkillFileContent::Base64(data),
    };

    AnthropicSkillFile::new(file.path, content)
}

fn shared_skill_result_from_anthropic(
    result: AnthropicSkillUploadResult,
) -> SharedSkillUploadResult {
    SharedSkillUploadResult {
        provider_reference: result.provider_reference,
        display_title: result.display_title,
        name: result.name,
        description: result.description,
        latest_version: result.latest_version,
        provider_metadata: result.provider_metadata,
        warnings: result.warnings,
    }
}

fn has_non_header_http_overrides(http_config: &HttpConfig) -> bool {
    http_config.timeout.is_some()
        || http_config.connect_timeout.is_some()
        || http_config.proxy.is_some()
        || http_config.user_agent.is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::transport::{
        HttpTransport, HttpTransportGetRequest, HttpTransportMultipartRequest,
        HttpTransportResponse,
    };
    use async_trait::async_trait;
    use reqwest::header::HeaderMap;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct CaptureTransport {
        multipart_requests: Arc<Mutex<Vec<HttpTransportMultipartRequest>>>,
        get_requests: Arc<Mutex<Vec<HttpTransportGetRequest>>>,
        responses: Arc<Mutex<VecDeque<HttpTransportResponse>>>,
    }

    impl CaptureTransport {
        fn new(responses: Vec<HttpTransportResponse>) -> Self {
            Self {
                multipart_requests: Arc::new(Mutex::new(Vec::new())),
                get_requests: Arc::new(Mutex::new(Vec::new())),
                responses: Arc::new(Mutex::new(responses.into_iter().collect())),
            }
        }

        fn take_multipart_requests(&self) -> Vec<HttpTransportMultipartRequest> {
            std::mem::take(&mut *self.multipart_requests.lock().expect("multipart lock"))
        }

        fn take_get_requests(&self) -> Vec<HttpTransportGetRequest> {
            std::mem::take(&mut *self.get_requests.lock().expect("get lock"))
        }
    }

    #[async_trait]
    impl HttpTransport for CaptureTransport {
        async fn execute_json(
            &self,
            _request: crate::execution::http::transport::HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "json transport should not be used in skills tests".to_string(),
            ))
        }

        async fn execute_multipart(
            &self,
            request: HttpTransportMultipartRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            self.multipart_requests
                .lock()
                .expect("multipart lock")
                .push(request);
            self.responses
                .lock()
                .expect("responses lock")
                .pop_front()
                .ok_or_else(|| LlmError::HttpError("missing multipart response".to_string()))
        }

        async fn execute_get(
            &self,
            request: HttpTransportGetRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            self.get_requests.lock().expect("get lock").push(request);
            self.responses
                .lock()
                .expect("responses lock")
                .pop_front()
                .ok_or_else(|| LlmError::HttpError("missing get response".to_string()))
        }
    }

    fn make_response(body: serde_json::Value) -> HttpTransportResponse {
        HttpTransportResponse {
            status: 200,
            headers: HeaderMap::new(),
            body: serde_json::to_vec(&body).expect("serialize response"),
        }
    }

    #[tokio::test]
    async fn upload_uses_multipart_and_fetches_version_metadata() {
        let transport = CaptureTransport::new(vec![
            make_response(serde_json::json!({
                "id": "skill_01Xud7kLMsjLfc7Aa6RvigZf",
                "display_title": "Test Capture Skill",
                "name": "server-name",
                "description": "server-description",
                "latest_version": "1772078378207930",
                "source": "custom",
                "created_at": "2026-02-26T03:59:39.314772Z",
                "updated_at": "2026-02-26T03:59:39.314772Z"
            })),
            make_response(serde_json::json!({
                "type": "skill_version",
                "skill_id": "skill_01Xud7kLMsjLfc7Aa6RvigZf",
                "name": "test-capture-skill",
                "description": "An updated test skill for fixture capture"
            })),
        ]);

        let skills = AnthropicSkills::new(
            SecretString::from("test-api-key".to_string()),
            "https://api.anthropic.com/v1".to_string(),
            reqwest::Client::new(),
            crate::types::HttpConfig::default(),
            vec!["skills-2025-10-02".to_string()],
            Some(Arc::new(transport.clone())),
            vec![],
            None,
        );

        let result = skills
            .upload(
                vec![AnthropicSkillFile::base64(
                    "index.ts",
                    STANDARD.encode(b"console.log('hello')"),
                )],
                Some("My Custom Title".to_string()),
                None,
            )
            .await
            .expect("upload result");

        assert_eq!(
            result.provider_reference.get("anthropic"),
            Some("skill_01Xud7kLMsjLfc7Aa6RvigZf")
        );
        assert_eq!(result.display_title.as_deref(), Some("Test Capture Skill"));
        assert_eq!(result.name.as_deref(), Some("test-capture-skill"));
        assert_eq!(
            result.description.as_deref(),
            Some("An updated test skill for fixture capture")
        );
        assert_eq!(result.latest_version.as_deref(), Some("1772078378207930"));
        assert_eq!(
            result
                .provider_metadata
                .as_ref()
                .and_then(|metadata| metadata.get("anthropic"))
                .and_then(|metadata| metadata.get("source")),
            Some(&serde_json::json!("custom"))
        );
        assert!(result.warnings.is_empty());

        let multipart_requests = transport.take_multipart_requests();
        assert_eq!(multipart_requests.len(), 1);
        assert_eq!(
            multipart_requests[0].url,
            "https://api.anthropic.com/v1/skills"
        );
        assert_eq!(
            multipart_requests[0]
                .headers
                .get("anthropic-beta")
                .and_then(|value| value.to_str().ok()),
            Some("skills-2025-10-02")
        );
        let multipart_body = String::from_utf8_lossy(&multipart_requests[0].body);
        assert!(multipart_body.contains("name=\"display_title\""));
        assert!(multipart_body.contains("My Custom Title"));
        assert!(multipart_body.contains("name=\"files[]\"; filename=\"index.ts\""));
        assert!(multipart_body.contains("console.log('hello')"));

        let get_requests = transport.take_get_requests();
        assert_eq!(get_requests.len(), 1);
        assert_eq!(
            get_requests[0].url,
            "https://api.anthropic.com/v1/skills/skill_01Xud7kLMsjLfc7Aa6RvigZf/versions/1772078378207930"
        );
        assert_eq!(
            get_requests[0]
                .headers
                .get("anthropic-beta")
                .and_then(|value| value.to_str().ok()),
            Some("skills-2025-10-02")
        );
    }

    #[test]
    fn build_context_merges_beta_features_into_anthropic_beta_header() {
        let mut cfg = crate::types::HttpConfig::default();
        cfg.headers
            .insert("Anthropic-Beta".to_string(), "existing".to_string());

        let skills = AnthropicSkills::new(
            SecretString::from("k".to_string()),
            "https://api.anthropic.com/v1".to_string(),
            reqwest::Client::new(),
            cfg,
            vec!["skills-2025-10-02".to_string()],
            None,
            vec![],
            None,
        );

        let ctx = skills.build_context();
        assert_eq!(
            ctx.http_extra_headers
                .get("anthropic-beta")
                .map(|value| value.as_str()),
            Some("existing,skills-2025-10-02")
        );
    }
}
