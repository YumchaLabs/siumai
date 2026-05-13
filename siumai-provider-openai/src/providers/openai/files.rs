//! `OpenAI` Files API Implementation
//!
//! This module provides the `OpenAI` implementation of the `FileManagementCapability` trait,
//! including file upload, listing, retrieval, and deletion operations.

use async_trait::async_trait;
use secrecy::ExposeSecret;
// no extra imports

use crate::error::LlmError;
use crate::traits::FileManagementCapability;
use crate::types::{
    FileDeleteResponse, FileListQuery, FileListResponse, FileObject, FileUploadRequest,
};

use super::config::OpenAiConfig;

/// `OpenAI` file management capability implementation.
///
/// This struct provides the OpenAI-specific implementation of file management
/// operations using the `OpenAI` Files API.
///
/// # Supported Operations
/// - File upload with various purposes (assistants, fine-tune, batch, etc.)
/// - File listing with filtering and pagination
/// - File metadata retrieval
/// - File deletion
/// - File content download
///
/// # API Reference
/// <https://platform.openai.com/docs/api-reference/files>
#[derive(Clone)]
pub struct OpenAiFiles {
    /// `OpenAI` configuration
    config: OpenAiConfig,
    /// HTTP client
    http_client: reqwest::Client,
    /// HTTP interceptors to apply
    http_interceptors:
        Vec<std::sync::Arc<dyn crate::execution::http::interceptor::HttpInterceptor>>,
    /// Unified retry options
    retry_options: Option<crate::retry_api::RetryOptions>,
}

impl OpenAiFiles {
    /// Create a new `OpenAI` files instance.
    ///
    /// # Arguments
    /// * `config` - `OpenAI` configuration
    /// * `http_client` - HTTP client for making requests
    pub fn new(
        config: OpenAiConfig,
        http_client: reqwest::Client,
        http_interceptors: Vec<
            std::sync::Arc<dyn crate::execution::http::interceptor::HttpInterceptor>,
        >,
        retry_options: Option<crate::retry_api::RetryOptions>,
    ) -> Self {
        Self {
            config,
            http_client,
            http_interceptors,
            retry_options,
        }
    }

    /// Get supported file purposes.
    pub fn get_supported_purposes(&self) -> Vec<String> {
        vec![
            "assistants".to_string(),
            "batch".to_string(),
            "fine-tune".to_string(),
            "vision".to_string(),
        ]
    }

    /// Get maximum file size in bytes.
    pub const fn get_max_file_size(&self) -> u64 {
        512 * 1024 * 1024 // 512 MB
    }

    /// Get supported file formats.
    pub fn get_supported_formats(&self) -> Vec<String> {
        vec![
            // Text formats
            "txt".to_string(),
            "json".to_string(),
            "jsonl".to_string(),
            "csv".to_string(),
            "tsv".to_string(),
            // Document formats
            "pdf".to_string(),
            "docx".to_string(),
            // Image formats
            "png".to_string(),
            "jpg".to_string(),
            "jpeg".to_string(),
            "gif".to_string(),
            "webp".to_string(),
            // Audio formats
            "mp3".to_string(),
            "mp4".to_string(),
            "mpeg".to_string(),
            "mpga".to_string(),
            "m4a".to_string(),
            "wav".to_string(),
            "webm".to_string(),
        ]
    }

    /// Validate file upload request.
    fn validate_upload_request(&self, request: &FileUploadRequest) -> Result<(), LlmError> {
        // Best-effort validation (Vercel-aligned):
        // - Avoid hard-coding purpose allowlists, which can drift as OpenAI adds new purposes.
        // - Avoid hard-coding extension allowlists, which can be too restrictive for general file APIs.
        //
        // Let the provider return the authoritative error when inputs are invalid.

        // Validate effective purpose is non-empty. Provider options can override the legacy
        // top-level request field, matching the AI SDK OpenAI files option boundary.
        let provider_purpose = request
            .provider_options
            .get_object("openai")
            .and_then(|options| options.get("purpose"))
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|purpose| !purpose.is_empty());
        if provider_purpose.is_none() && request.purpose.trim().is_empty() {
            return Err(LlmError::InvalidInput(
                "File purpose cannot be empty".to_string(),
            ));
        }

        // Optional guardrail: reject extremely large payloads before hitting the network.
        if request.content.len() as u64 > self.get_max_file_size() {
            return Err(LlmError::InvalidInput(format!(
                "File size {} bytes exceeds maximum allowed size of {} bytes",
                request.content.len(),
                self.get_max_file_size()
            )));
        }

        Ok(())
    }

    fn merge_default_provider_options(&self, request: &mut FileUploadRequest) {
        let mut merged = self.config.provider_options_map.clone();
        merged.merge_overrides(std::mem::take(&mut request.provider_options));
        request.provider_options = merged;
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

    fn build_files_executor(
        &self,
    ) -> std::sync::Arc<crate::execution::executors::files::HttpFilesExecutor> {
        use crate::execution::executors::files::FilesExecutorBuilder;

        let ctx = self.build_context();
        let spec = std::sync::Arc::new(super::spec::OpenAiSpec::new());

        let mut builder = FilesExecutorBuilder::new("openai", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        builder.build()
    }
}

impl std::fmt::Debug for OpenAiFiles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiFiles")
            .field("base_url", &self.config.base_url)
            .field("has_interceptors", &(!self.http_interceptors.is_empty()))
            .field("has_retry", &self.retry_options.is_some())
            .finish()
    }
}

#[async_trait]
impl FileManagementCapability for OpenAiFiles {
    /// Upload a file to OpenAI's storage.
    async fn upload_file(&self, mut request: FileUploadRequest) -> Result<FileObject, LlmError> {
        self.merge_default_provider_options(&mut request);

        // Validate request
        self.validate_upload_request(&request)?;

        use crate::execution::executors::files::FilesExecutor;
        let exec = self.build_files_executor();
        FilesExecutor::upload(&*exec, request).await
    }

    /// List files with optional filtering.
    async fn list_files(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError> {
        use crate::execution::executors::files::FilesExecutor;
        let exec = self.build_files_executor();
        FilesExecutor::list(&*exec, query).await
    }

    /// Retrieve file metadata.
    async fn retrieve_file(&self, file_id: String) -> Result<FileObject, LlmError> {
        use crate::execution::executors::files::FilesExecutor;
        let exec = self.build_files_executor();
        FilesExecutor::retrieve(&*exec, file_id).await
    }

    /// Delete a file permanently.
    async fn delete_file(&self, file_id: String) -> Result<FileDeleteResponse, LlmError> {
        use crate::execution::executors::files::FilesExecutor;
        let exec = self.build_files_executor();
        FilesExecutor::delete(&*exec, file_id).await
    }

    /// Get file content as bytes.
    async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        use crate::execution::executors::files::FilesExecutor;
        let exec = self.build_files_executor();
        FilesExecutor::get_content(&*exec, file_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::transport::{
        HttpTransport, HttpTransportMultipartRequest, HttpTransportRequest, HttpTransportResponse,
    };
    use async_trait::async_trait;
    use reqwest::header::HeaderMap;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct CaptureTransport {
        multipart_requests: Arc<Mutex<Vec<HttpTransportMultipartRequest>>>,
        response: Arc<Mutex<Option<HttpTransportResponse>>>,
    }

    impl CaptureTransport {
        fn new(response: HttpTransportResponse) -> Self {
            Self {
                multipart_requests: Arc::new(Mutex::new(Vec::new())),
                response: Arc::new(Mutex::new(Some(response))),
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
                "json transport should not be used in OpenAI files tests".to_string(),
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
            self.response
                .lock()
                .expect("response lock")
                .take()
                .ok_or_else(|| LlmError::HttpError("missing multipart response".to_string()))
        }
    }

    fn make_file_response() -> HttpTransportResponse {
        HttpTransportResponse {
            status: 200,
            headers: HeaderMap::new(),
            body: serde_json::to_vec(&serde_json::json!({
                "id": "file-default-options",
                "object": "file",
                "bytes": 5,
                "created_at": 1710000000u64,
                "filename": "hello.txt",
                "purpose": "batch",
                "status": "uploaded",
                "status_details": null
            }))
            .expect("serialize response"),
        }
    }

    fn make_upload_request() -> FileUploadRequest {
        FileUploadRequest {
            content: b"hello".to_vec(),
            filename: Some("hello.txt".to_string()),
            mime_type: Some("text/plain".to_string()),
            purpose: String::new(),
            metadata: HashMap::new(),
            provider_options: Default::default(),
            http_config: None,
        }
    }

    #[tokio::test]
    async fn upload_merges_default_file_provider_options_into_multipart() {
        let transport = Arc::new(CaptureTransport::new(make_file_response()));
        let config = OpenAiConfig::new("test-api-key")
            .with_base_url("https://api.openai.test/v1")
            .with_provider_options(serde_json::json!({
                "purpose": "batch",
                "expiresAfter": 3600
            }))
            .with_http_transport(transport.clone());
        let files = OpenAiFiles::new(config, reqwest::Client::new(), Vec::new(), None);

        let result = files
            .upload_file(make_upload_request())
            .await
            .expect("upload result");
        assert_eq!(result.id, "file-default-options");

        let requests = transport.take_multipart_requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].url, "https://api.openai.test/v1/files");

        let body = String::from_utf8_lossy(&requests[0].body);
        assert!(body.contains("name=\"purpose\""));
        assert!(body.contains("\r\n\r\nbatch\r\n"));
        assert!(body.contains("name=\"expires_after\""));
        assert!(body.contains("\r\n\r\n3600\r\n"));
    }
}
