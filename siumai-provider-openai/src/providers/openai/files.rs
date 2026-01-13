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

// Legacy typed response structs removed; transformer handles JSON parsing.

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

        // Validate filename
        if request.filename.is_empty() {
            return Err(LlmError::InvalidInput(
                "Filename cannot be empty".to_string(),
            ));
        }

        // Validate purpose is non-empty
        if request.purpose.trim().is_empty() {
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

    // Legacy direct HTTP helpers removed; requests are delegated to HttpFilesExecutor.
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
    async fn upload_file(&self, request: FileUploadRequest) -> Result<FileObject, LlmError> {
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
