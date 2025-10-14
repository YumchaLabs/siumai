//! Files executor traits

use crate::error::LlmError;
use crate::transformers::files::{FilesHttpBody, FilesTransformer};
use crate::types::{
    FileDeleteResponse, FileListQuery, FileListResponse, FileObject, FileUploadRequest,
};
use reqwest::header::HeaderMap;
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
    pub build_base_url: Box<dyn Fn() -> String + Send + Sync>,
    pub build_headers: Box<dyn Fn() -> Result<HeaderMap, LlmError> + Send + Sync>,
}

#[async_trait::async_trait]
impl FilesExecutor for HttpFilesExecutor {
    async fn upload(&self, req: FileUploadRequest) -> Result<FileObject, LlmError> {
        let body = self.transformer.build_upload_body(&req)?;
        let url = format!(
            "{}{}",
            (self.build_base_url)(),
            self.transformer.upload_endpoint(&req)
        );
        let headers = (self.build_headers)()?;
        let builder = self.http_client.post(url).headers(headers);
        let resp = match body {
            FilesHttpBody::Json(json) => builder.json(&json).send().await,
            FilesHttpBody::Multipart(form) => builder.multipart(form).send().await,
        }
        .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: text,
                details: None,
            });
        }
        let text = resp
            .text()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
        let json: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| LlmError::ParseError(e.to_string()))?;
        self.transformer.transform_file_object(&json)
    }

    async fn list(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError> {
        let endpoint = self.transformer.list_endpoint(&query);
        let url = format!("{}{}", (self.build_base_url)(), endpoint);
        let headers = (self.build_headers)()?;
        let resp = self
            .http_client
            .get(url)
            .headers(headers)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: text,
                details: None,
            });
        }
        let text = resp
            .text()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
        let json: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| LlmError::ParseError(e.to_string()))?;
        self.transformer.transform_list_response(&json)
    }

    async fn retrieve(&self, file_id: String) -> Result<FileObject, LlmError> {
        let endpoint = self.transformer.retrieve_endpoint(&file_id);
        let url = format!("{}{}", (self.build_base_url)(), endpoint);
        let headers = (self.build_headers)()?;
        let resp = self
            .http_client
            .get(url)
            .headers(headers)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: text,
                details: None,
            });
        }
        let text = resp
            .text()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
        let json: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| LlmError::ParseError(e.to_string()))?;
        self.transformer.transform_file_object(&json)
    }

    async fn delete(&self, file_id: String) -> Result<FileDeleteResponse, LlmError> {
        let endpoint = self.transformer.delete_endpoint(&file_id);
        let url = format!("{}{}", (self.build_base_url)(), endpoint);
        let headers = (self.build_headers)()?;
        let resp = self
            .http_client
            .delete(url)
            .headers(headers)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: text,
                details: None,
            });
        }
        // Some providers may return an empty body or a small JSON; we just acknowledge success
        let id = file_id.trim_start_matches("files/").to_string();
        Ok(FileDeleteResponse { id, deleted: true })
    }

    async fn get_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        // Prefer API endpoint if provided; otherwise fall back to URL from file object
        let mut maybe_endpoint = self.transformer.content_endpoint(&file_id);
        let (url, headers, use_absolute) = if let Some(ep) = maybe_endpoint.take() {
            let url = format!("{}{}", (self.build_base_url)(), ep);
            let headers = (self.build_headers)()?;
            (url, headers, false)
        } else {
            let file = self.retrieve(file_id.clone()).await?;
            let content_url = self
                .transformer
                .content_url_from_file_object(&file)
                .ok_or_else(|| {
                    LlmError::UnsupportedOperation("File download URI not available".to_string())
                })?;
            let headers = (self.build_headers)()?;
            (content_url, headers, true)
        };
        let req = self.http_client.get(url);
        let resp = if use_absolute {
            // absolute URL may require only auth header
            req.headers(headers).send().await
        } else {
            req.headers(headers).send().await
        }
        .map_err(|e| LlmError::HttpError(e.to_string()))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: text,
                details: None,
            });
        }
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| LlmError::HttpError(format!("Failed to read response body: {e}")))?;
        Ok(bytes.to_vec())
    }
}
