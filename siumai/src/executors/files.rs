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
    pub build_headers: Box<
        dyn (Fn() -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<HeaderMap, LlmError>> + Send>,
            >) + Send
            + Sync,
    >,
}

#[async_trait::async_trait]
impl FilesExecutor for HttpFilesExecutor {
    async fn upload(&self, req: FileUploadRequest) -> Result<FileObject, LlmError> {
        let url = format!(
            "{}{}",
            (self.build_base_url)(),
            self.transformer.upload_endpoint(&req)
        );

        // First attempt
        let body1 = self.transformer.build_upload_body(&req)?;
        let headers1 = (self.build_headers)().await?;
        let builder1 = self.http_client.post(&url).headers(headers1);
        let mut resp = match body1 {
            FilesHttpBody::Json(json) => builder1.json(&json).send().await,
            FilesHttpBody::Multipart(form) => builder1.multipart(form).send().await,
        }
        .map_err(|e| LlmError::HttpError(e.to_string()))?;
        if !resp.status().is_success() {
            let status = resp.status();
            if status.as_u16() == 401 {
                // Rebuild headers and body, then retry once
                let body2 = self.transformer.build_upload_body(&req)?;
                let headers2 = (self.build_headers)().await?;
                let builder2 = self.http_client.post(&url).headers(headers2);
                resp = match body2 {
                    FilesHttpBody::Json(json) => builder2.json(&json).send().await,
                    FilesHttpBody::Multipart(form) => builder2.multipart(form).send().await,
                }
                .map_err(|e| LlmError::HttpError(e.to_string()))?;
            } else {
                let headers = resp.headers().clone();
                let text = resp.text().await.unwrap_or_default();
                return Err(crate::retry_api::classify_http_error(
                    &self.provider_id,
                    status.as_u16(),
                    &text,
                    &headers,
                    None,
                ));
            }
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
        let headers = (self.build_headers)().await?;
        let resp_first = self
            .http_client
            .get(&url)
            .headers(headers)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
        let resp = if !resp_first.status().is_success() {
            let status = resp_first.status();
            if status.as_u16() == 401 {
                let headers = (self.build_headers)().await?;
                self.http_client
                    .get(&url)
                    .headers(headers)
                    .send()
                    .await
                    .map_err(|e| LlmError::HttpError(e.to_string()))?
            } else {
                let headers = resp_first.headers().clone();
                let text = resp_first.text().await.unwrap_or_default();
                return Err(crate::retry_api::classify_http_error(
                    &self.provider_id,
                    status.as_u16(),
                    &text,
                    &headers,
                    None,
                ));
            }
        } else {
            resp_first
        };
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
        let headers = (self.build_headers)().await?;
        let resp_first = self
            .http_client
            .get(&url)
            .headers(headers)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
        let resp = if !resp_first.status().is_success() {
            let status = resp_first.status();
            if status.as_u16() == 401 {
                let headers = (self.build_headers)().await?;
                self.http_client
                    .get(&url)
                    .headers(headers)
                    .send()
                    .await
                    .map_err(|e| LlmError::HttpError(e.to_string()))?
            } else {
                let headers = resp_first.headers().clone();
                let text = resp_first.text().await.unwrap_or_default();
                return Err(crate::retry_api::classify_http_error(
                    &self.provider_id,
                    status.as_u16(),
                    &text,
                    &headers,
                    None,
                ));
            }
        } else {
            resp_first
        };
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
        let headers = (self.build_headers)().await?;
        let resp = self
            .http_client
            .delete(url.clone())
            .headers(headers)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
        if !resp.status().is_success() {
            let status = resp.status();
            if status.as_u16() == 401 {
                let headers = (self.build_headers)().await?;
                let resp2 = self
                    .http_client
                    .delete(url)
                    .headers(headers)
                    .send()
                    .await
                    .map_err(|e| LlmError::HttpError(e.to_string()))?;
                if !resp2.status().is_success() {
                    let status2 = resp2.status();
                    let headers2 = resp2.headers().clone();
                    let text2 = resp2.text().await.unwrap_or_default();
                    return Err(crate::retry_api::classify_http_error(
                        &self.provider_id,
                        status2.as_u16(),
                        &text2,
                        &headers2,
                        None,
                    ));
                }
            } else {
                let headers = resp.headers().clone();
                let text = resp.text().await.unwrap_or_default();
                return Err(crate::retry_api::classify_http_error(
                    &self.provider_id,
                    status.as_u16(),
                    &text,
                    &headers,
                    None,
                ));
            }
        }
        // Some providers may return an empty body or a small JSON; we just acknowledge success
        let id = file_id.trim_start_matches("files/").to_string();
        Ok(FileDeleteResponse { id, deleted: true })
    }

    async fn get_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        // Prefer API endpoint if provided; otherwise fall back to URL from file object
        let mut maybe_endpoint = self.transformer.content_endpoint(&file_id);
        let (url, headers, _use_absolute) = if let Some(ep) = maybe_endpoint.take() {
            let url = format!("{}{}", (self.build_base_url)(), ep);
            let headers = (self.build_headers)().await?;
            (url, headers, false)
        } else {
            let file = self.retrieve(file_id.clone()).await?;
            let content_url = self
                .transformer
                .content_url_from_file_object(&file)
                .ok_or_else(|| {
                    LlmError::UnsupportedOperation("File download URI not available".to_string())
                })?;
            let headers = (self.build_headers)().await?;
            (content_url, headers, true)
        };
        let req = self.http_client.get(url.clone());
        let resp_first = req
            .headers(headers.clone())
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
        let resp = if !resp_first.status().is_success() {
            let status = resp_first.status();
            if status.as_u16() == 401 {
                let headers2 = (self.build_headers)().await?;
                self.http_client
                    .get(url)
                    .headers(headers2)
                    .send()
                    .await
                    .map_err(|e| LlmError::HttpError(e.to_string()))?
            } else {
                let headers = resp_first.headers().clone();
                let text = resp_first.text().await.unwrap_or_default();
                return Err(crate::retry_api::classify_http_error(
                    &self.provider_id,
                    status.as_u16(),
                    &text,
                    &headers,
                    None,
                ));
            }
        } else {
            resp_first
        };
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| LlmError::HttpError(format!("Failed to read response body: {e}")))?;
        Ok(bytes.to_vec())
    }
}
