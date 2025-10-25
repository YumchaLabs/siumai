//! Files executor traits

use crate::error::LlmError;
use crate::transformers::files::{FilesHttpBody, FilesTransformer};
use crate::types::{
    FileDeleteResponse, FileListQuery, FileListResponse, FileObject, FileUploadRequest,
};
use std::sync::Arc;

#[async_trait::async_trait]
pub trait FilesExecutor: Send + Sync {
    async fn upload(&self, req: FileUploadRequest) -> Result<FileObject, LlmError>;
    async fn list(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError>;
    async fn retrieve(&self, file_id: String) -> Result<FileObject, LlmError>;
    async fn delete(&self, file_id: String) -> Result<FileDeleteResponse, LlmError>;
    async fn get_content(&self, file_id: String) -> Result<Vec<u8>, LlmError>;
}

/// Generic HTTP-based Files executor
pub struct HttpFilesExecutor {
    pub provider_id: String,
    pub http_client: reqwest::Client,
    pub transformer: Arc<dyn FilesTransformer>,
    pub provider_spec: Arc<dyn crate::core::ProviderSpec>,
    pub provider_context: crate::core::ProviderContext,
    /// HTTP interceptors for request/response observation and modification
    pub interceptors: Vec<Arc<dyn crate::utils::http_interceptor::HttpInterceptor>>,
    /// Optional retry options for controlling retry behavior (including 401 retry)
    /// If None, uses default behavior (401 retry enabled)
    pub retry_options: Option<crate::retry_api::RetryOptions>,
}

#[async_trait::async_trait]
impl FilesExecutor for HttpFilesExecutor {
    async fn upload(&self, req: FileUploadRequest) -> Result<FileObject, LlmError> {
        // Capability guard
        let caps = self.provider_spec.capabilities();
        if !caps.supports("file_management") {
            return Err(LlmError::UnsupportedOperation(
                "File management is not supported by this provider".to_string(),
            ));
        }
        // 1. Get URL
        let base_url = self.provider_spec.files_base_url(&self.provider_context);
        let url = format!("{}{}", base_url, self.transformer.upload_endpoint(&req));

        // 2. Build execution config for common HTTP layer
        let config = crate::executors::common::HttpExecutionConfig {
            provider_id: self.provider_id.clone(),
            http_client: self.http_client.clone(),
            provider_spec: self.provider_spec.clone(),
            provider_context: self.provider_context.clone(),
            interceptors: self.interceptors.clone(),
            retry_options: self.retry_options.clone(),
        };

        // 3. Transform request and execute based on body type
        let per_request_headers = req.http_config.as_ref().map(|hc| &hc.headers);
        let body = self.transformer.build_upload_body(&req)?;
        let result = match body {
            FilesHttpBody::Json(json) => {
                // Use JSON request path
                crate::executors::common::execute_json_request(
                    &config,
                    &url,
                    crate::executors::common::HttpBody::Json(json),
                    per_request_headers,
                    false, // stream = false
                )
                .await?
            }
            FilesHttpBody::Multipart(_) => {
                // Use multipart request path
                let req_clone = req.clone();
                crate::executors::common::execute_multipart_request(
                    &config,
                    &url,
                    || {
                        self.transformer
                            .build_upload_body(&req_clone)
                            .and_then(|body| match body {
                                FilesHttpBody::Multipart(form) => Ok(form),
                                _ => Err(LlmError::InvalidParameter(
                                    "Expected multipart body".into(),
                                )),
                            })
                    },
                    per_request_headers,
                )
                .await?
            }
        };

        // 4. Transform response
        self.transformer.transform_file_object(&result.json)
    }

    async fn list(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError> {
        let caps = self.provider_spec.capabilities();
        if !caps.supports("file_management") {
            return Err(LlmError::UnsupportedOperation(
                "File listing is not supported by this provider".to_string(),
            ));
        }
        // 1. Get URL from transformer
        let endpoint = self.transformer.list_endpoint(&query);
        let base_url = self.provider_spec.files_base_url(&self.provider_context);
        let url = format!("{}{}", base_url, endpoint);

        // 2. Build execution config for common HTTP layer
        let config = crate::executors::common::HttpExecutionConfig {
            provider_id: self.provider_id.clone(),
            http_client: self.http_client.clone(),
            provider_spec: self.provider_spec.clone(),
            provider_context: self.provider_context.clone(),
            interceptors: self.interceptors.clone(),
            retry_options: self.retry_options.clone(),
        };

        // 3. Extract per-request headers from query
        let per_request_headers = query
            .as_ref()
            .and_then(|q| q.http_config.as_ref())
            .map(|hc| &hc.headers);

        // 4. Execute GET request using common HTTP layer
        let result =
            crate::executors::common::execute_get_request(&config, &url, per_request_headers)
                .await?;

        // 5. Transform response
        self.transformer.transform_list_response(&result.json)
    }

    async fn retrieve(&self, file_id: String) -> Result<FileObject, LlmError> {
        let caps = self.provider_spec.capabilities();
        if !caps.supports("file_management") {
            return Err(LlmError::UnsupportedOperation(
                "File retrieve is not supported by this provider".to_string(),
            ));
        }
        // 1. Get URL from transformer
        let endpoint = self.transformer.retrieve_endpoint(&file_id);
        let base_url = self.provider_spec.files_base_url(&self.provider_context);
        let url = format!("{}{}", base_url, endpoint);

        // 2. Build execution config for common HTTP layer
        let config = crate::executors::common::HttpExecutionConfig {
            provider_id: self.provider_id.clone(),
            http_client: self.http_client.clone(),
            provider_spec: self.provider_spec.clone(),
            provider_context: self.provider_context.clone(),
            interceptors: self.interceptors.clone(),
            retry_options: self.retry_options.clone(),
        };

        // 3. Execute GET request using common HTTP layer
        let result = crate::executors::common::execute_get_request(
            &config, &url, None, // No per-request headers for retrieve
        )
        .await?;

        // 4. Transform response
        self.transformer.transform_file_object(&result.json)
    }

    async fn delete(&self, file_id: String) -> Result<FileDeleteResponse, LlmError> {
        let caps = self.provider_spec.capabilities();
        if !caps.supports("file_management") {
            return Err(LlmError::UnsupportedOperation(
                "File delete is not supported by this provider".to_string(),
            ));
        }
        // 1. Get URL from transformer
        let endpoint = self.transformer.delete_endpoint(&file_id);
        let base_url = self.provider_spec.files_base_url(&self.provider_context);
        let url = format!("{}{}", base_url, endpoint);

        // 2. Build execution config for common HTTP layer
        let config = crate::executors::common::HttpExecutionConfig {
            provider_id: self.provider_id.clone(),
            http_client: self.http_client.clone(),
            provider_spec: self.provider_spec.clone(),
            provider_context: self.provider_context.clone(),
            interceptors: self.interceptors.clone(),
            retry_options: self.retry_options.clone(),
        };

        // 3. Execute DELETE request using common HTTP layer
        let _result = crate::executors::common::execute_delete_request(
            &config, &url, None, // No per-request headers for delete
        )
        .await?;

        // 4. Return success response
        // Some providers may return an empty body or a small JSON; we just acknowledge success
        let id = file_id.trim_start_matches("files/").to_string();
        Ok(FileDeleteResponse { id, deleted: true })
    }

    async fn get_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        let caps = self.provider_spec.capabilities();
        if !caps.supports("file_management") {
            return Err(LlmError::UnsupportedOperation(
                "File content download is not supported by this provider".to_string(),
            ));
        }
        // 1. Determine URL (prefer API endpoint if provided; otherwise fall back to URL from file object)
        let mut maybe_endpoint = self.transformer.content_endpoint(&file_id);
        let url = if let Some(ep) = maybe_endpoint.take() {
            let base_url = self.provider_spec.files_base_url(&self.provider_context);
            format!("{}{}", base_url, ep)
        } else {
            let file = self.retrieve(file_id.clone()).await?;
            self.transformer
                .content_url_from_file_object(&file)
                .ok_or_else(|| {
                    LlmError::UnsupportedOperation("File download URI not available".to_string())
                })?
        };

        // 2. Build execution config for common HTTP layer
        let config = crate::executors::common::HttpExecutionConfig {
            provider_id: self.provider_id.clone(),
            http_client: self.http_client.clone(),
            provider_spec: self.provider_spec.clone(),
            provider_context: self.provider_context.clone(),
            interceptors: self.interceptors.clone(),
            retry_options: self.retry_options.clone(),
        };

        // 3. Execute GET request for binary content using common HTTP layer
        let result = crate::executors::common::execute_get_binary(
            &config, &url, None, // No per-request headers for get_content
        )
        .await?;

        // 4. Return binary content
        Ok(result.bytes)
    }
}
