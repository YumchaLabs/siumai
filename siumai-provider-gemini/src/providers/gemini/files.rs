//! Gemini Files API Implementation
//!
//! This module provides the Gemini implementation of the `FileManagementCapability` trait,
//! including file upload, listing, retrieval, and deletion operations.

use async_trait::async_trait;
use reqwest::Client as HttpClient;
// no extra imports
use reqwest::StatusCode;
use secrecy::ExposeSecret;

use crate::error::LlmError;
use crate::traits::FileManagementCapability;
use crate::types::{
    FileDeleteResponse, FileListQuery, FileListResponse, FileObject, FileUploadRequest,
};

use super::types::GeminiConfig;

/// Gemini file management capability implementation.
///
/// This struct provides the Gemini-specific implementation of file management
/// operations using the Gemini Files API.
///
/// # Supported Operations
/// - File upload with metadata
/// - File listing with pagination
/// - File metadata retrieval
/// - File deletion
/// - File content download
///
/// # API Reference
/// <https://ai.google.dev/api/files>
#[derive(Clone)]
pub struct GeminiFiles {
    /// Gemini configuration
    config: GeminiConfig,
    /// HTTP client
    http_client: HttpClient,
    /// HTTP interceptors to apply
    http_interceptors:
        Vec<std::sync::Arc<dyn crate::execution::http::interceptor::HttpInterceptor>>,
    /// Unified retry options
    retry_options: Option<crate::retry_api::RetryOptions>,
}

impl GeminiFiles {
    /// Create a new Gemini files capability
    pub fn new(
        config: GeminiConfig,
        http_client: HttpClient,
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

    /// Validate file upload request
    fn validate_upload_request(&self, request: &FileUploadRequest) -> Result<(), LlmError> {
        if request.content.is_empty() {
            return Err(LlmError::InvalidInput(
                "File content cannot be empty".to_string(),
            ));
        }

        if request.filename.is_empty() {
            return Err(LlmError::InvalidInput(
                "Filename cannot be empty".to_string(),
            ));
        }

        // Check file size limits.
        //
        // Official Files API limits allow much larger files (up to 2 GB), but note that
        // `FileUploadRequest` is currently in-memory (`Vec<u8>`), so callers should avoid
        // very large uploads to prevent OOM.
        const MAX_FILE_SIZE: usize = 2 * 1024 * 1024 * 1024; // 2GB
        if request.content.len() > MAX_FILE_SIZE {
            return Err(LlmError::InvalidInput(format!(
                "File size {} bytes exceeds maximum allowed size of {} bytes",
                request.content.len(),
                MAX_FILE_SIZE
            )));
        }

        Ok(())
    }

    async fn build_files_executor(
        &self,
    ) -> std::sync::Arc<crate::execution::executors::files::HttpFilesExecutor> {
        use crate::execution::executors::files::FilesExecutorBuilder;

        let ctx = super::context::build_context(&self.config).await;
        let spec = std::sync::Arc::new(super::spec::GeminiSpecWithConfig::new(self.config.clone()));

        let mut builder = FilesExecutorBuilder::new("gemini", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone());

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        builder.build()
    }
}

impl std::fmt::Debug for GeminiFiles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiFiles")
            .field("base_url", &self.config.base_url)
            .field("has_interceptors", &(!self.http_interceptors.is_empty()))
            .field("has_retry", &self.retry_options.is_some())
            .finish()
    }
}

#[async_trait]
impl FileManagementCapability for GeminiFiles {
    /// Upload a file to Gemini's storage.
    async fn upload_file(&self, request: FileUploadRequest) -> Result<FileObject, LlmError> {
        // Validate request
        self.validate_upload_request(&request)?;

        use crate::execution::executors::files::FilesExecutor;
        let exec = self.build_files_executor().await;
        FilesExecutor::upload(&*exec, request).await
    }

    /// List files with optional filtering.
    async fn list_files(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError> {
        use crate::execution::executors::files::FilesExecutor;
        let exec = self.build_files_executor().await;
        FilesExecutor::list(&*exec, query).await
    }

    /// Retrieve file metadata.
    async fn retrieve_file(&self, file_id: String) -> Result<FileObject, LlmError> {
        use crate::execution::executors::files::FilesExecutor;
        let exec = self.build_files_executor().await;
        FilesExecutor::retrieve(&*exec, file_id).await
    }

    /// Delete a file permanently.
    async fn delete_file(&self, file_id: String) -> Result<FileDeleteResponse, LlmError> {
        use crate::execution::executors::files::FilesExecutor;
        let exec = self.build_files_executor().await;
        FilesExecutor::delete(&*exec, file_id).await
    }

    /// Get file content as bytes.
    async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        // The official Veo workflow returns a downloadable HTTPS URI (not a Files API resource).
        // Allow callers to pass that URI directly to `get_file_content` as a convenience.
        if file_id.starts_with("http://") || file_id.starts_with("https://") {
            return self.download_uri(file_id).await;
        }

        use crate::execution::executors::files::FilesExecutor;
        let exec = self.build_files_executor().await;
        FilesExecutor::get_content(&*exec, file_id).await
    }
}

impl GeminiFiles {
    /// Get file content as string.
    pub async fn get_file_content_as_string(&self, file_id: String) -> Result<String, LlmError> {
        let bytes = self.get_file_content(file_id).await?;
        String::from_utf8(bytes)
            .map_err(|e| LlmError::ParseError(format!("File content is not valid UTF-8: {e}")))
    }

    async fn download_uri(&self, uri: String) -> Result<Vec<u8>, LlmError> {
        use crate::core::ProviderSpec;

        fn is_google_host(host: &str) -> bool {
            let host = host.to_ascii_lowercase();
            host.ends_with("googleapis.com")
                || host.ends_with("googleusercontent.com")
                || host.ends_with("gstatic.com")
                || host.ends_with("google.com")
        }

        let mut url = reqwest::Url::parse(&uri)
            .map_err(|e| LlmError::InvalidInput(format!("Invalid download URI: {e}")))?;

        let ctx = super::context::build_context(&self.config).await;
        let headers = ProviderSpec::build_headers(&super::spec::GeminiSpec, &ctx)?;
        let api_key_present = !self.config.api_key.expose_secret().is_empty();
        let has_auth = api_key_present || headers.contains_key(reqwest::header::AUTHORIZATION);

        for _ in 0..10 {
            let mut req = self.http_client.get(url.clone());
            if url.host_str().is_some_and(is_google_host) && has_auth {
                req = req.headers(headers.clone());
            }
            let resp = req.send().await.map_err(|e| {
                LlmError::HttpError(format!("Failed to download Gemini file URI: {e}"))
            })?;

            if resp.status().is_success() {
                let bytes = resp.bytes().await.map_err(|e| {
                    LlmError::HttpError(format!("Failed to read Gemini file bytes: {e}"))
                })?;
                return Ok(bytes.to_vec());
            }

            if matches!(
                resp.status(),
                StatusCode::MOVED_PERMANENTLY
                    | StatusCode::FOUND
                    | StatusCode::SEE_OTHER
                    | StatusCode::TEMPORARY_REDIRECT
                    | StatusCode::PERMANENT_REDIRECT
            ) {
                let Some(location) = resp.headers().get(reqwest::header::LOCATION) else {
                    break;
                };
                let location = location.to_str().unwrap_or_default();
                let next = url
                    .join(location)
                    .or_else(|_| reqwest::Url::parse(location))
                    .map_err(|e| LlmError::HttpError(format!("Invalid redirect URL: {e}")))?;
                url = next;
                continue;
            }

            return Err(LlmError::HttpError(format!(
                "Download URI returned HTTP {}",
                resp.status()
            )));
        }

        Err(LlmError::HttpError(
            "Too many redirects while downloading Gemini URI".to_string(),
        ))
    }

    /// Check if a file exists.
    pub async fn file_exists(&self, file_id: String) -> bool {
        self.retrieve_file(file_id).await.is_ok()
    }

    /// Wait for file processing to complete.
    ///
    /// This method polls the file status until it's either active or failed.
    pub async fn wait_for_file_processing(
        &self,
        file_id: String,
        max_wait_seconds: u64,
    ) -> Result<FileObject, LlmError> {
        let start_time = std::time::Instant::now();
        let max_duration = std::time::Duration::from_secs(max_wait_seconds);

        loop {
            let file = self.retrieve_file(file_id.clone()).await?;

            match file.status.as_str() {
                "active" => return Ok(file),
                "failed" => {
                    return Err(LlmError::ProcessingError(
                        "File processing failed".to_string(),
                    ));
                }
                "processing" => {
                    // Continue waiting
                    if start_time.elapsed() >= max_duration {
                        return Err(LlmError::TimeoutError(format!(
                            "File processing timeout after {max_wait_seconds} seconds"
                        )));
                    }

                    // Wait before next check
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                }
                _ => {
                    return Err(LlmError::ProcessingError(format!(
                        "Unknown file status: {}",
                        file.status
                    )));
                }
            }
        }
    }
}
