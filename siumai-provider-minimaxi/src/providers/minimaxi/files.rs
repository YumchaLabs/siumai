//! MiniMaxi file management implementation.
//!
//! MiniMaxi exposes a dedicated file management API with endpoints under:
//! - `POST /v1/files/upload` (multipart)
//! - `GET /v1/files/list`
//! - `GET /v1/files/retrieve`
//! - `GET /v1/files/retrieve_content`
//! - `POST /v1/files/delete` (JSON body)

use async_trait::async_trait;

use crate::error::LlmError;
use crate::execution::executors::common::{
    HttpBody, HttpExecutionConfig, execute_get_request, execute_json_request,
    execute_multipart_request,
};
use crate::execution::executors::http_request::execute_get_binary;
use crate::execution::http::headers::HttpHeaderBuilder;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::wiring::HttpExecutionWiring;
use crate::retry_api::RetryOptions;
use crate::traits::{FileManagementCapability, ProviderCapabilities};
use crate::types::{
    FileDeleteResponse, FileListQuery, FileListResponse, FileObject, FileUploadRequest,
};

use super::config::MinimaxiConfig;
use super::utils::{build_context, resolve_api_root_base_url};

#[derive(Clone)]
struct MinimaxiFilesSpec;

impl crate::core::ProviderSpec for MinimaxiFilesSpec {
    fn id(&self) -> &'static str {
        "minimaxi"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_file_management()
    }

    fn build_headers(
        &self,
        ctx: &crate::core::ProviderContext,
    ) -> Result<reqwest::header::HeaderMap, LlmError> {
        let api_key = ctx
            .api_key
            .as_ref()
            .ok_or_else(|| LlmError::MissingApiKey("MiniMaxi API key not provided".into()))?;

        let mut builder = HttpHeaderBuilder::new()
            .with_bearer_auth(api_key)?
            .with_json_content_type();

        if let Some(org) = ctx.organization.as_deref() {
            builder = builder.with_header("OpenAI-Organization", org)?;
        }
        if let Some(proj) = ctx.project.as_deref() {
            builder = builder.with_header("OpenAI-Project", proj)?;
        }

        builder = builder.with_custom_headers(&ctx.http_extra_headers)?;
        Ok(builder.build())
    }
}

fn parse_file_id(file_id: &str) -> Result<i64, LlmError> {
    file_id.parse::<i64>().map_err(|e| {
        LlmError::InvalidParameter(format!("Invalid MiniMaxi file_id '{file_id}': {e}"))
    })
}

fn check_base_resp(provider: &str, json: &serde_json::Value) -> Result<(), LlmError> {
    let base = json.get("base_resp").ok_or_else(|| {
        LlmError::ParseError(format!("{provider}: missing 'base_resp' in response"))
    })?;

    let status_code = base
        .get("status_code")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| {
            LlmError::ParseError(format!("{provider}: missing 'base_resp.status_code'"))
        })?;
    let status_msg = base
        .get("status_msg")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown error");

    if status_code != 0 {
        return Err(LlmError::ProviderError {
            provider: provider.to_string(),
            message: status_msg.to_string(),
            error_code: Some(status_code.to_string()),
        });
    }
    Ok(())
}

fn map_file_object(provider: &str, raw: &serde_json::Value) -> Result<FileObject, LlmError> {
    let id = raw
        .get("file_id")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| LlmError::ParseError(format!("{provider}: missing 'file.file_id'")))?;

    let filename = raw
        .get("filename")
        .and_then(|v| v.as_str())
        .ok_or_else(|| LlmError::ParseError(format!("{provider}: missing 'file.filename'")))?;

    let bytes = raw
        .get("bytes")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| LlmError::ParseError(format!("{provider}: missing 'file.bytes'")))?;

    let created_at = raw
        .get("created_at")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| LlmError::ParseError(format!("{provider}: missing 'file.created_at'")))?;

    let purpose = raw
        .get("purpose")
        .and_then(|v| v.as_str())
        .ok_or_else(|| LlmError::ParseError(format!("{provider}: missing 'file.purpose'")))?;

    let mut metadata = std::collections::HashMap::new();
    if let Some(obj) = raw.as_object() {
        for (k, v) in obj {
            if matches!(
                k.as_str(),
                "file_id" | "bytes" | "created_at" | "filename" | "purpose"
            ) {
                continue;
            }
            metadata.insert(k.clone(), v.clone());
        }
    }

    Ok(FileObject {
        id: id.to_string(),
        filename: filename.to_string(),
        bytes: bytes.max(0) as u64,
        created_at: created_at.max(0) as u64,
        purpose: purpose.to_string(),
        // MiniMaxi does not expose a file status field; keep a stable placeholder.
        status: "available".to_string(),
        mime_type: None,
        metadata,
    })
}

/// MiniMaxi Files API client (extension capability).
///
/// Official docs:
/// - `https://platform.minimaxi.com/docs/api-reference/file-management-intro`
/// - OpenAPI: `https://platform.minimaxi.com/docs/api-reference/file/management/api/openapi.json`
#[derive(Clone)]
pub struct MinimaxiFiles {
    config: MinimaxiConfig,
    http_client: reqwest::Client,
    http_config: crate::types::HttpConfig,
    http_interceptors: Vec<std::sync::Arc<dyn HttpInterceptor>>,
    retry_options: Option<RetryOptions>,
    http_transport: Option<std::sync::Arc<dyn crate::execution::http::transport::HttpTransport>>,
}

impl MinimaxiFiles {
    pub fn new(
        config: MinimaxiConfig,
        http_client: reqwest::Client,
        http_config: crate::types::HttpConfig,
        http_interceptors: Vec<std::sync::Arc<dyn HttpInterceptor>>,
        retry_options: Option<RetryOptions>,
        http_transport: Option<
            std::sync::Arc<dyn crate::execution::http::transport::HttpTransport>,
        >,
    ) -> Self {
        Self {
            config,
            http_client,
            http_config,
            http_interceptors,
            retry_options,
            http_transport,
        }
    }

    fn build_http_config(&self) -> HttpExecutionConfig {
        let base_url = resolve_api_root_base_url(&self.config.base_url);
        let mut wiring = HttpExecutionWiring::new(
            "minimaxi",
            self.http_client.clone(),
            build_context(&self.config.api_key, &base_url, &self.http_config),
        )
        .with_interceptors(self.http_interceptors.clone())
        .with_retry_options(self.retry_options.clone());

        if let Some(transport) = self.http_transport.clone() {
            wiring = wiring.with_transport(transport);
        }

        wiring.config(std::sync::Arc::new(MinimaxiFilesSpec))
    }

    fn base_url(&self) -> String {
        resolve_api_root_base_url(&self.config.base_url)
            .trim_end_matches('/')
            .to_string()
    }

    fn validate_upload_request(&self, request: &FileUploadRequest) -> Result<(), LlmError> {
        // Docs: per-file limit 512MB.
        const MAX_BYTES: u64 = 512 * 1024 * 1024;
        if request.content.len() as u64 > MAX_BYTES {
            return Err(LlmError::InvalidInput(format!(
                "File size {} bytes exceeds maximum allowed size of {} bytes",
                request.content.len(),
                MAX_BYTES
            )));
        }

        if request.filename.is_empty() {
            return Err(LlmError::InvalidInput(
                "Filename cannot be empty".to_string(),
            ));
        }

        // MiniMaxi upload currently documents a restricted purpose set.
        let supported = ["voice_clone", "prompt_audio", "t2a_async_input"];
        if !supported.contains(&request.purpose.as_str()) {
            return Err(LlmError::InvalidInput(format!(
                "Unsupported MiniMaxi file purpose: {}. Supported purposes: {:?}",
                request.purpose, supported
            )));
        }

        Ok(())
    }

    fn split_file_id_and_purpose(file_id: &str) -> (String, Option<String>) {
        let trimmed = file_id.trim();
        if let Some((id, purpose)) = trimmed.split_once(':') {
            (id.trim().to_string(), Some(purpose.trim().to_string()))
        } else {
            (trimmed.to_string(), None)
        }
    }
}

#[async_trait]
impl FileManagementCapability for MinimaxiFiles {
    async fn upload_file(&self, request: FileUploadRequest) -> Result<FileObject, LlmError> {
        self.validate_upload_request(&request)?;

        let cfg = self.build_http_config();
        let url = format!("{}/v1/files/upload", self.base_url());

        let per_request_headers = request.http_config.as_ref().map(|hc| &hc.headers);
        let purpose = request.purpose.clone();
        let filename = request.filename.clone();
        let mime_type = request
            .mime_type
            .clone()
            .unwrap_or_else(|| crate::utils::guess_mime(Some(&request.content), Some(&filename)));
        let content = request.content.clone();

        let res = execute_multipart_request(
            &cfg,
            &url,
            move || {
                let part = reqwest::multipart::Part::bytes(content.clone())
                    .file_name(filename.clone())
                    .mime_str(&mime_type)
                    .map_err(|e| {
                        LlmError::InvalidParameter(format!("Invalid MIME type '{mime_type}': {e}"))
                    })?;
                Ok(reqwest::multipart::Form::new()
                    .text("purpose", purpose.clone())
                    .part("file", part))
            },
            per_request_headers,
        )
        .await?;

        check_base_resp("minimaxi", &res.json)?;
        let file = res
            .json
            .get("file")
            .ok_or_else(|| LlmError::ParseError("minimaxi: missing 'file' in response".into()))?;
        map_file_object("minimaxi", file)
    }

    async fn list_files(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError> {
        let purpose = query
            .as_ref()
            .and_then(|q| q.purpose.clone())
            .ok_or_else(|| {
                LlmError::InvalidParameter(
                    "MiniMaxi list_files requires FileListQuery.purpose".to_string(),
                )
            })?;

        let cfg = self.build_http_config();
        let url = format!(
            "{}/v1/files/list?purpose={}",
            self.base_url(),
            urlencoding::encode(&purpose)
        );

        let per_request_headers = query
            .as_ref()
            .and_then(|q| q.http_config.as_ref())
            .map(|hc| &hc.headers);

        let res = execute_get_request(&cfg, &url, per_request_headers).await?;
        check_base_resp("minimaxi", &res.json)?;

        let files = res
            .json
            .get("files")
            .and_then(|v| v.as_array())
            .ok_or_else(|| LlmError::ParseError("minimaxi: missing 'files' array".into()))?;

        let mut out = Vec::with_capacity(files.len());
        for f in files {
            out.push(map_file_object("minimaxi", f)?);
        }

        Ok(FileListResponse {
            files: out,
            has_more: false,
            next_cursor: None,
        })
    }

    async fn retrieve_file(&self, file_id: String) -> Result<FileObject, LlmError> {
        let (id_str, _) = Self::split_file_id_and_purpose(&file_id);
        let id = parse_file_id(&id_str)?;

        let cfg = self.build_http_config();
        let url = format!("{}/v1/files/retrieve?file_id={id}", self.base_url());
        let res = execute_get_request(&cfg, &url, None).await?;
        check_base_resp("minimaxi", &res.json)?;

        let file = res
            .json
            .get("file")
            .ok_or_else(|| LlmError::ParseError("minimaxi: missing 'file' in response".into()))?;
        map_file_object("minimaxi", file)
    }

    async fn delete_file(&self, file_id: String) -> Result<FileDeleteResponse, LlmError> {
        let (id_str, purpose_override) = Self::split_file_id_and_purpose(&file_id);
        let id = parse_file_id(&id_str)?;

        let purpose = if let Some(p) = purpose_override {
            p
        } else {
            let meta = self.retrieve_file(id_str.clone()).await?;
            meta.purpose
        };

        let cfg = self.build_http_config();
        let url = format!("{}/v1/files/delete", self.base_url());
        let body = serde_json::json!({
            "file_id": id,
            "purpose": purpose,
        });

        let res = execute_json_request(&cfg, &url, HttpBody::Json(body), None, false).await?;
        check_base_resp("minimaxi", &res.json)?;

        Ok(FileDeleteResponse {
            id: id_str,
            deleted: true,
        })
    }

    async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        let (id_str, _) = Self::split_file_id_and_purpose(&file_id);
        let id = parse_file_id(&id_str)?;

        let cfg = self.build_http_config();
        let url = format!("{}/v1/files/retrieve_content?file_id={id}", self.base_url());
        let res = execute_get_binary(&cfg, &url, None).await?;
        Ok(res.bytes)
    }
}
