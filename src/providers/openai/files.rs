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
#[derive(Debug, Clone)]
pub struct OpenAiFiles {
    /// `OpenAI` configuration
    config: OpenAiConfig,
    /// HTTP client
    http_client: reqwest::Client,
}

impl OpenAiFiles {
    /// Create a new `OpenAI` files instance.
    ///
    /// # Arguments
    /// * `config` - `OpenAI` configuration
    /// * `http_client` - HTTP client for making requests
    pub const fn new(config: OpenAiConfig, http_client: reqwest::Client) -> Self {
        Self {
            config,
            http_client,
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
        // Validate file size
        if request.content.len() as u64 > self.get_max_file_size() {
            return Err(LlmError::InvalidInput(format!(
                "File size {} bytes exceeds maximum allowed size of {} bytes",
                request.content.len(),
                self.get_max_file_size()
            )));
        }

        // Validate purpose
        if !self.get_supported_purposes().contains(&request.purpose) {
            return Err(LlmError::InvalidInput(format!(
                "Unsupported file purpose: {}. Supported purposes: {:?}",
                request.purpose,
                self.get_supported_purposes()
            )));
        }

        // Validate filename
        if request.filename.is_empty() {
            return Err(LlmError::InvalidInput(
                "Filename cannot be empty".to_string(),
            ));
        }

        // Validate file extension if provided
        if let Some(extension) = request.filename.split('.').next_back() {
            let supported_formats = self.get_supported_formats();
            if !supported_formats.contains(&extension.to_lowercase()) {
                return Err(LlmError::InvalidInput(format!(
                    "Unsupported file format: {extension}. Supported formats: {supported_formats:?}"
                )));
            }
        }

        Ok(())
    }

    // Legacy direct HTTP helpers removed; requests are delegated to HttpFilesExecutor.
}

#[async_trait]
impl FileManagementCapability for OpenAiFiles {
    /// Upload a file to OpenAI's storage.
    async fn upload_file(&self, request: FileUploadRequest) -> Result<FileObject, LlmError> {
        // Validate request
        self.validate_upload_request(&request)?;

        use crate::executors::files::{FilesExecutor, HttpFilesExecutor};
        let http = self.http_client.clone();
        let base = self.config.base_url.clone();
        let transformer = super::transformers::OpenAiFilesTransformer;
        let api_key = self.config.api_key.clone();
        let org = self.config.organization.clone();
        let proj = self.config.project.clone();
        let extra_headers = self.config.http_config.headers.clone();
        let headers_builder = move || {
            let mut headers = crate::utils::http_headers::ProviderHeaders::openai(
                api_key.expose_secret(),
                org.as_deref(),
                proj.as_deref(),
                &extra_headers,
            )?;
            crate::utils::http_headers::inject_tracing_headers(&mut headers);
            Ok(headers)
        };
        let exec = HttpFilesExecutor {
            provider_id: "openai".to_string(),
            http_client: http,
            transformer: std::sync::Arc::new(transformer),
            build_base_url: Box::new(move || base.clone()),
            build_headers: Box::new(headers_builder),
        };
        exec.upload(request).await
    }

    /// List files with optional filtering.
    async fn list_files(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError> {
        use crate::executors::files::{FilesExecutor, HttpFilesExecutor};
        let http = self.http_client.clone();
        let base = self.config.base_url.clone();
        let transformer = super::transformers::OpenAiFilesTransformer;
        let api_key = self.config.api_key.clone();
        let org = self.config.organization.clone();
        let proj = self.config.project.clone();
        let extra_headers = self.config.http_config.headers.clone();
        let headers_builder = move || {
            let mut headers = crate::utils::http_headers::ProviderHeaders::openai(
                api_key.expose_secret(),
                org.as_deref(),
                proj.as_deref(),
                &extra_headers,
            )?;
            crate::utils::http_headers::inject_tracing_headers(&mut headers);
            Ok(headers)
        };
        let exec = HttpFilesExecutor {
            provider_id: "openai".to_string(),
            http_client: http,
            transformer: std::sync::Arc::new(transformer),
            build_base_url: Box::new(move || base.clone()),
            build_headers: Box::new(headers_builder),
        };
        exec.list(query).await
    }

    /// Retrieve file metadata.
    async fn retrieve_file(&self, file_id: String) -> Result<FileObject, LlmError> {
        use crate::executors::files::{FilesExecutor, HttpFilesExecutor};
        let http = self.http_client.clone();
        let base = self.config.base_url.clone();
        let transformer = super::transformers::OpenAiFilesTransformer;
        let api_key = self.config.api_key.clone();
        let org = self.config.organization.clone();
        let proj = self.config.project.clone();
        let extra_headers = self.config.http_config.headers.clone();
        let headers_builder = move || {
            let mut headers = crate::utils::http_headers::ProviderHeaders::openai(
                api_key.expose_secret(),
                org.as_deref(),
                proj.as_deref(),
                &extra_headers,
            )?;
            crate::utils::http_headers::inject_tracing_headers(&mut headers);
            Ok(headers)
        };
        let exec = HttpFilesExecutor {
            provider_id: "openai".to_string(),
            http_client: http,
            transformer: std::sync::Arc::new(transformer),
            build_base_url: Box::new(move || base.clone()),
            build_headers: Box::new(headers_builder),
        };
        exec.retrieve(file_id).await
    }

    /// Delete a file permanently.
    async fn delete_file(&self, file_id: String) -> Result<FileDeleteResponse, LlmError> {
        use crate::executors::files::{FilesExecutor, HttpFilesExecutor};
        let http = self.http_client.clone();
        let base = self.config.base_url.clone();
        let transformer = super::transformers::OpenAiFilesTransformer;
        let api_key = self.config.api_key.clone();
        let org = self.config.organization.clone();
        let proj = self.config.project.clone();
        let extra_headers = self.config.http_config.headers.clone();
        let headers_builder = move || {
            let mut headers = crate::utils::http_headers::ProviderHeaders::openai(
                api_key.expose_secret(),
                org.as_deref(),
                proj.as_deref(),
                &extra_headers,
            )?;
            crate::utils::http_headers::inject_tracing_headers(&mut headers);
            Ok(headers)
        };
        let exec = HttpFilesExecutor {
            provider_id: "openai".to_string(),
            http_client: http,
            transformer: std::sync::Arc::new(transformer),
            build_base_url: Box::new(move || base.clone()),
            build_headers: Box::new(headers_builder),
        };
        exec.delete(file_id).await
    }

    /// Get file content as bytes.
    async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        use crate::executors::files::{FilesExecutor, HttpFilesExecutor};
        let http = self.http_client.clone();
        let base = self.config.base_url.clone();
        let transformer = super::transformers::OpenAiFilesTransformer;
        let api_key = self.config.api_key.clone();
        let org = self.config.organization.clone();
        let proj = self.config.project.clone();
        let extra_headers = self.config.http_config.headers.clone();
        let headers_builder = move || {
            let mut headers = crate::utils::http_headers::ProviderHeaders::openai(
                api_key.expose_secret(),
                org.as_deref(),
                proj.as_deref(),
                &extra_headers,
            )?;
            crate::utils::http_headers::inject_tracing_headers(&mut headers);
            Ok(headers)
        };
        let exec = HttpFilesExecutor {
            provider_id: "openai".to_string(),
            http_client: http,
            transformer: std::sync::Arc::new(transformer),
            build_base_url: Box::new(move || base.clone()),
            build_headers: Box::new(headers_builder),
        };
        exec.get_content(file_id).await
    }
}
