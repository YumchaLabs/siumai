//! `OpenAI` Skills API implementation.
//!
//! This resource mirrors the AI SDK OpenAI `skills()` surface:
//! - `POST /v1/skills`

use crate::error::LlmError;
use crate::execution::executors::common::{HttpExecutionConfig, execute_multipart_request};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::retry_api::RetryOptions;
use crate::traits::SkillsCapability;
use crate::types::{
    HttpConfig, ProviderMetadataMap, ProviderReference, SkillFileContent,
    SkillUploadFile as SharedSkillUploadFile, SkillUploadRequest,
    SkillUploadResult as SharedSkillUploadResult, Warning,
};
use crate::utils::url::join_url;
use async_trait::async_trait;
use base64::{Engine, engine::general_purpose::STANDARD};
use secrecy::ExposeSecret;
use std::collections::HashMap;
use std::sync::Arc;

use super::config::OpenAiConfig;

/// Provider-id keyed metadata map returned by `OpenAiSkills::upload`.
pub type OpenAiSkillProviderMetadata = ProviderMetadataMap;

/// File content accepted by OpenAI skill uploads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenAiSkillFileContent {
    /// Raw file bytes.
    Bytes(Vec<u8>),
    /// Base64-encoded file bytes.
    Base64(String),
}

impl OpenAiSkillFileContent {
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

impl From<Vec<u8>> for OpenAiSkillFileContent {
    fn from(value: Vec<u8>) -> Self {
        Self::Bytes(value)
    }
}

impl From<&[u8]> for OpenAiSkillFileContent {
    fn from(value: &[u8]) -> Self {
        Self::Bytes(value.to_vec())
    }
}

/// One uploaded skill file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAiSkillFile {
    /// File path relative to the skill root.
    pub path: String,
    /// File content.
    pub content: OpenAiSkillFileContent,
}

impl OpenAiSkillFile {
    /// Create a skill file from a path and content.
    pub fn new(path: impl Into<String>, content: impl Into<OpenAiSkillFileContent>) -> Self {
        Self {
            path: path.into(),
            content: content.into(),
        }
    }

    /// Create a skill file from raw bytes.
    pub fn bytes(path: impl Into<String>, data: Vec<u8>) -> Self {
        Self::new(path, OpenAiSkillFileContent::Bytes(data))
    }

    /// Create a skill file from base64.
    pub fn base64(path: impl Into<String>, data: impl Into<String>) -> Self {
        Self::new(path, OpenAiSkillFileContent::Base64(data.into()))
    }
}

/// Canonical result returned by `OpenAiSkills::upload`.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenAiSkillUploadResult {
    /// Provider-owned skill reference in stable AI SDK-style shape.
    pub provider_reference: ProviderReference,
    /// Optional canonical skill name.
    pub name: Option<String>,
    /// Optional skill description.
    pub description: Option<String>,
    /// Optional latest version id.
    pub latest_version: Option<String>,
    /// Provider-owned metadata under the provider id root.
    pub provider_metadata: Option<OpenAiSkillProviderMetadata>,
    /// Non-fatal warnings emitted while uploading.
    pub warnings: Vec<Warning>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct OpenAiSkillResponse {
    id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    default_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    latest_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    created_at: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    updated_at: Option<i64>,
}

/// Provider-scoped client for skill uploads.
#[derive(Clone)]
pub struct OpenAiSkills {
    config: OpenAiConfig,
    http_client: reqwest::Client,
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    retry_options: Option<RetryOptions>,
}

impl OpenAiSkills {
    /// Create a new `OpenAI` skills client.
    pub fn new(
        config: OpenAiConfig,
        http_client: reqwest::Client,
        http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
        retry_options: Option<RetryOptions>,
    ) -> Self {
        Self {
            config,
            http_client,
            http_interceptors,
            retry_options,
        }
    }

    /// Upload a new OpenAI skill from the provided files.
    pub async fn upload(
        &self,
        files: Vec<OpenAiSkillFile>,
        display_title: Option<String>,
        per_request_headers: Option<HashMap<String, String>>,
    ) -> Result<OpenAiSkillUploadResult, LlmError> {
        if files.is_empty() {
            return Err(LlmError::InvalidInput(
                "OpenAI skill uploads require at least one file.".to_string(),
            ));
        }

        let url = join_url(&self.config.base_url, "skills");
        let ctx = self.build_context();
        let http_config = self.build_http_config(ctx);

        let call = || {
            let http_config = http_config.clone();
            let url = url.clone();
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
                    || build_upload_form(&files),
                    per_request_http_config.as_ref(),
                )
                .await?;

                let response: OpenAiSkillResponse =
                    serde_json::from_value(response.json).map_err(|error| {
                        LlmError::ParseError(format!(
                            "Failed to parse OpenAI skill upload response: {error}"
                        ))
                    })?;

                let mut result = build_upload_result(response);
                if display_title.is_some() {
                    result
                        .warnings
                        .push(Warning::unsupported("displayTitle", None::<String>));
                }

                Ok(result)
            }
        };

        crate::retry_api::maybe_retry(self.retry_options.clone(), call).await
    }

    fn build_context(&self) -> crate::core::ProviderContext {
        crate::core::ProviderContext::new(
            "openai",
            self.config.base_url.clone(),
            Some(self.config.api_key.expose_secret().to_string()),
            self.config.http_config.headers.clone(),
        )
        .with_org_project(
            self.config.organization.clone(),
            self.config.project.clone(),
        )
    }

    fn build_http_config(&self, ctx: crate::core::ProviderContext) -> HttpExecutionConfig {
        let mut wiring = crate::execution::wiring::HttpExecutionWiring::new(
            "openai",
            self.http_client.clone(),
            ctx,
        )
        .with_interceptors(self.http_interceptors.clone())
        .with_retry_options(self.retry_options.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            wiring = wiring.with_transport(transport);
        }

        wiring.config(Arc::new(super::spec::OpenAiSpec::new()))
    }
}

fn build_upload_form(files: &[OpenAiSkillFile]) -> Result<reqwest::multipart::Form, LlmError> {
    let mut form = reqwest::multipart::Form::new();

    for file in files {
        let bytes = file.content.clone().into_bytes()?;
        let part = reqwest::multipart::Part::bytes(bytes).file_name(file.path.clone());
        form = form.part("files[]", part);
    }

    Ok(form)
}

fn build_upload_result(response: OpenAiSkillResponse) -> OpenAiSkillUploadResult {
    let mut provider_metadata = HashMap::new();
    let mut openai = HashMap::new();
    if let Some(default_version) = response.default_version {
        openai.insert(
            "defaultVersion".to_string(),
            serde_json::Value::String(default_version),
        );
    }
    if let Some(created_at) = response.created_at {
        openai.insert("createdAt".to_string(), serde_json::json!(created_at));
    }
    if let Some(updated_at) = response.updated_at {
        openai.insert("updatedAt".to_string(), serde_json::json!(updated_at));
    }
    if !openai.is_empty() {
        provider_metadata.insert(
            "openai".to_string(),
            serde_json::Value::Object(openai.into_iter().collect()),
        );
    }

    OpenAiSkillUploadResult {
        provider_reference: ProviderReference::single("openai", response.id),
        name: response.name,
        description: response.description,
        latest_version: response.latest_version,
        provider_metadata: (!provider_metadata.is_empty()).then_some(provider_metadata),
        warnings: Vec::new(),
    }
}

#[async_trait]
impl SkillsCapability for OpenAiSkills {
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
                files.into_iter().map(shared_skill_file_to_openai).collect(),
                display_title,
                headers,
            )
            .await
            .map(shared_skill_result_from_openai)?;

        if http_config
            .as_ref()
            .is_some_and(has_non_header_http_overrides)
        {
            result.warnings.push(Warning::compatibility(
                "httpConfig",
                Some("OpenAI skill uploads currently forward only per-request headers."),
            ));
        }

        Ok(result)
    }
}

#[async_trait]
impl SkillsCapability for super::OpenAiClient {
    async fn upload_skill(
        &self,
        request: SkillUploadRequest,
    ) -> Result<SharedSkillUploadResult, LlmError> {
        self.skills().upload_skill(request).await
    }
}

fn shared_skill_file_to_openai(file: SharedSkillUploadFile) -> OpenAiSkillFile {
    let content = match file.content {
        SkillFileContent::Bytes(data) => OpenAiSkillFileContent::Bytes(data),
        SkillFileContent::Base64(data) => OpenAiSkillFileContent::Base64(data),
    };

    OpenAiSkillFile::new(file.path, content)
}

fn shared_skill_result_from_openai(result: OpenAiSkillUploadResult) -> SharedSkillUploadResult {
    SharedSkillUploadResult {
        provider_reference: result.provider_reference,
        display_title: None,
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
        HttpTransport, HttpTransportMultipartRequest, HttpTransportRequest, HttpTransportResponse,
    };
    use async_trait::async_trait;
    use reqwest::header::HeaderMap;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    #[derive(Clone)]
    struct CaptureTransport {
        multipart_requests: Arc<Mutex<Vec<HttpTransportMultipartRequest>>>,
        responses: Arc<Mutex<VecDeque<HttpTransportResponse>>>,
    }

    impl CaptureTransport {
        fn new(responses: Vec<HttpTransportResponse>) -> Self {
            Self {
                multipart_requests: Arc::new(Mutex::new(Vec::new())),
                responses: Arc::new(Mutex::new(responses.into_iter().collect())),
            }
        }

        fn take_multipart_requests(&self) -> Vec<HttpTransportMultipartRequest> {
            std::mem::take(&mut *self.multipart_requests.lock().expect("multipart lock"))
        }
    }

    #[async_trait]
    impl HttpTransport for CaptureTransport {
        async fn execute_json(
            &self,
            _request: HttpTransportRequest,
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
    }

    fn make_response(body: serde_json::Value) -> HttpTransportResponse {
        HttpTransportResponse {
            status: 200,
            headers: HeaderMap::new(),
            body: serde_json::to_vec(&body).expect("serialize response"),
        }
    }

    #[tokio::test]
    async fn upload_uses_multipart_and_maps_result() {
        let transport = CaptureTransport::new(vec![make_response(serde_json::json!({
            "id": "skill_699fc58f408c8191825d8d06ae75fd5c06de7b381a5db7f5",
            "name": "test-capture-skill",
            "description": "A test skill for fixture capture",
            "default_version": "1",
            "latest_version": "1",
            "created_at": 1772078479
        }))]);

        let skills = OpenAiSkills::new(
            OpenAiConfig::new("test-api-key")
                .with_http_transport(Arc::new(transport.clone()))
                .with_base_url("https://api.openai.com/v1"),
            reqwest::Client::new(),
            vec![],
            None,
        );

        let result = skills
            .upload(
                vec![OpenAiSkillFile::base64(
                    "index.ts",
                    STANDARD.encode(b"console.log('hello')"),
                )],
                Some("My Skill".to_string()),
                None,
            )
            .await
            .expect("upload result");

        assert_eq!(
            result.provider_reference.get("openai"),
            Some("skill_699fc58f408c8191825d8d06ae75fd5c06de7b381a5db7f5")
        );
        assert_eq!(result.name.as_deref(), Some("test-capture-skill"));
        assert_eq!(
            result.description.as_deref(),
            Some("A test skill for fixture capture")
        );
        assert_eq!(result.latest_version.as_deref(), Some("1"));
        assert_eq!(
            result
                .provider_metadata
                .as_ref()
                .and_then(|metadata| metadata.get("openai"))
                .and_then(|metadata| metadata.get("defaultVersion")),
            Some(&serde_json::json!("1"))
        );
        assert_eq!(
            result.warnings,
            vec![Warning::unsupported("displayTitle", None::<String>)]
        );

        let multipart_requests = transport.take_multipart_requests();
        assert_eq!(multipart_requests.len(), 1);
        assert_eq!(
            multipart_requests[0].url,
            "https://api.openai.com/v1/skills"
        );
        assert!(
            multipart_requests[0]
                .headers
                .contains_key(reqwest::header::AUTHORIZATION),
            "expected authorization header on skills upload"
        );
        let multipart_body = String::from_utf8_lossy(&multipart_requests[0].body);
        assert!(multipart_body.contains("name=\"files[]\"; filename=\"index.ts\""));
        assert!(multipart_body.contains("console.log('hello')"));
    }
}
