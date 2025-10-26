//! Gemini Files API Implementation
//!
//! This module provides the Gemini implementation of the `FileManagementCapability` trait,
//! including file upload, listing, retrieval, and deletion operations.

use async_trait::async_trait;
use reqwest::Client as HttpClient;
// no extra imports

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

        // Check file size limits (Gemini has specific limits)
        const MAX_FILE_SIZE: usize = 20 * 1024 * 1024; // 20MB for most files
        if request.content.len() > MAX_FILE_SIZE {
            return Err(LlmError::InvalidInput(format!(
                "File size {} bytes exceeds maximum allowed size of {} bytes",
                request.content.len(),
                MAX_FILE_SIZE
            )));
        }

        Ok(())
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

        use crate::execution::executors::files::{FilesExecutor, HttpFilesExecutor};
        use secrecy::ExposeSecret;

        let spec = std::sync::Arc::new(super::spec::GeminiSpec);
        let mut ctx = crate::core::ProviderContext::new(
            "gemini",
            self.config.base_url.clone(),
            Some(self.config.api_key.expose_secret().to_string()),
            self.config.http_config.headers.clone(),
        );

        // Handle token_provider if present
        if let Some(ref tp) = self.config.token_provider {
            if let Ok(tok) = tp.token().await {
                ctx.http_extra_headers
                    .insert("Authorization".to_string(), format!("Bearer {tok}"));
            }
        }

        let exec = HttpFilesExecutor {
            provider_id: "gemini".to_string(),
            http_client: self.http_client.clone(),
            transformer: std::sync::Arc::new(super::transformers::GeminiFilesTransformer {
                config: self.config.clone(),
            }),
            provider_spec: spec,
            provider_context: ctx,
            policy: crate::execution::ExecutionPolicy::new()
                .with_interceptors(self.http_interceptors.clone())
                .with_retry_options(self.retry_options.clone()),
        };
        exec.upload(request).await
    }

    /// List files with optional filtering.
    async fn list_files(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError> {
        use crate::execution::executors::files::{FilesExecutor, HttpFilesExecutor};
        use secrecy::ExposeSecret;

        let spec = std::sync::Arc::new(super::spec::GeminiSpec);
        let mut ctx = crate::core::ProviderContext::new(
            "gemini",
            self.config.base_url.clone(),
            Some(self.config.api_key.expose_secret().to_string()),
            self.config.http_config.headers.clone(),
        );

        // Handle token_provider if present
        if let Some(ref tp) = self.config.token_provider {
            if let Ok(tok) = tp.token().await {
                ctx.http_extra_headers
                    .insert("Authorization".to_string(), format!("Bearer {tok}"));
            }
        }

        let exec = HttpFilesExecutor {
            provider_id: "gemini".to_string(),
            http_client: self.http_client.clone(),
            transformer: std::sync::Arc::new(super::transformers::GeminiFilesTransformer {
                config: self.config.clone(),
            }),
            provider_spec: spec,
            provider_context: ctx,
            policy: crate::execution::ExecutionPolicy::new()
                .with_interceptors(self.http_interceptors.clone())
                .with_retry_options(self.retry_options.clone()),
        };
        exec.list(query).await
    }

    /// Retrieve file metadata.
    async fn retrieve_file(&self, file_id: String) -> Result<FileObject, LlmError> {
        use crate::execution::executors::files::{FilesExecutor, HttpFilesExecutor};
        use secrecy::ExposeSecret;

        let spec = std::sync::Arc::new(super::spec::GeminiSpec);
        let mut ctx = crate::core::ProviderContext::new(
            "gemini",
            self.config.base_url.clone(),
            Some(self.config.api_key.expose_secret().to_string()),
            self.config.http_config.headers.clone(),
        );

        // Handle token_provider if present
        if let Some(ref tp) = self.config.token_provider {
            if let Ok(tok) = tp.token().await {
                ctx.http_extra_headers
                    .insert("Authorization".to_string(), format!("Bearer {tok}"));
            }
        }

        let exec = HttpFilesExecutor {
            provider_id: "gemini".to_string(),
            http_client: self.http_client.clone(),
            transformer: std::sync::Arc::new(super::transformers::GeminiFilesTransformer {
                config: self.config.clone(),
            }),
            provider_spec: spec,
            provider_context: ctx,
            policy: crate::execution::ExecutionPolicy::new()
                .with_interceptors(self.http_interceptors.clone())
                .with_retry_options(self.retry_options.clone()),
        };
        exec.retrieve(file_id).await
    }

    /// Delete a file permanently.
    async fn delete_file(&self, file_id: String) -> Result<FileDeleteResponse, LlmError> {
        use crate::execution::executors::files::{FilesExecutor, HttpFilesExecutor};
        use secrecy::ExposeSecret;

        let spec = std::sync::Arc::new(super::spec::GeminiSpec);
        let mut ctx = crate::core::ProviderContext::new(
            "gemini",
            self.config.base_url.clone(),
            Some(self.config.api_key.expose_secret().to_string()),
            self.config.http_config.headers.clone(),
        );

        // Handle token_provider if present
        if let Some(ref tp) = self.config.token_provider {
            if let Ok(tok) = tp.token().await {
                ctx.http_extra_headers
                    .insert("Authorization".to_string(), format!("Bearer {tok}"));
            }
        }

        let exec = HttpFilesExecutor {
            provider_id: "gemini".to_string(),
            http_client: self.http_client.clone(),
            transformer: std::sync::Arc::new(super::transformers::GeminiFilesTransformer {
                config: self.config.clone(),
            }),
            provider_spec: spec,
            provider_context: ctx,
            policy: crate::execution::ExecutionPolicy::new()
                .with_interceptors(self.http_interceptors.clone())
                .with_retry_options(self.retry_options.clone()),
        };
        exec.delete(file_id).await
    }

    /// Get file content as bytes.
    async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        use crate::execution::executors::files::{FilesExecutor, HttpFilesExecutor};
        use secrecy::ExposeSecret;

        let spec = std::sync::Arc::new(super::spec::GeminiSpec);
        let mut ctx = crate::core::ProviderContext::new(
            "gemini",
            self.config.base_url.clone(),
            Some(self.config.api_key.expose_secret().to_string()),
            self.config.http_config.headers.clone(),
        );

        // Handle token_provider if present
        if let Some(ref tp) = self.config.token_provider {
            if let Ok(tok) = tp.token().await {
                ctx.http_extra_headers
                    .insert("Authorization".to_string(), format!("Bearer {tok}"));
            }
        }

        let exec = HttpFilesExecutor {
            provider_id: "gemini".to_string(),
            http_client: self.http_client.clone(),
            transformer: std::sync::Arc::new(super::transformers::GeminiFilesTransformer {
                config: self.config.clone(),
            }),
            provider_spec: spec,
            provider_context: ctx,
            policy: crate::execution::ExecutionPolicy::new()
                .with_interceptors(self.http_interceptors.clone())
                .with_retry_options(self.retry_options.clone()),
        };
        exec.get_content(file_id).await
    }
}

impl GeminiFiles {
    /// Get file content as string.
    pub async fn get_file_content_as_string(&self, file_id: String) -> Result<String, LlmError> {
        let bytes = self.get_file_content(file_id).await?;
        String::from_utf8(bytes)
            .map_err(|e| LlmError::ParseError(format!("File content is not valid UTF-8: {e}")))
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
