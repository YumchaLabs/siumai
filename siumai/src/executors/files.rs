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
    pub provider_spec: Arc<dyn crate::provider_core::ProviderSpec>,
    pub provider_context: crate::provider_core::ProviderContext,
}

#[async_trait::async_trait]
impl FilesExecutor for HttpFilesExecutor {
    async fn upload(&self, req: FileUploadRequest) -> Result<FileObject, LlmError> {
        let base_url = self.provider_spec.files_base_url(&self.provider_context);
        let url = format!("{}{}", base_url, self.transformer.upload_endpoint(&req));

        // First attempt
        let body1 = self.transformer.build_upload_body(&req)?;
        let mut headers1 = self.provider_spec.build_headers(&self.provider_context)?;
        crate::utils::http_headers::inject_tracing_headers(&mut headers1);
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
                let mut headers2 = self.provider_spec.build_headers(&self.provider_context)?;
                crate::utils::http_headers::inject_tracing_headers(&mut headers2);
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

        // Use parse_json_with_repair for automatic JSON repair when enabled
        let json: serde_json::Value = crate::streaming::parse_json_with_repair(&text)
            .map_err(|e| LlmError::ParseError(e.to_string()))?;
        self.transformer.transform_file_object(&json)
    }

    async fn list(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError> {
        let endpoint = self.transformer.list_endpoint(&query);
        let base_url = self.provider_spec.files_base_url(&self.provider_context);
        let url = format!("{}{}", base_url, endpoint);

        let mut headers = self.provider_spec.build_headers(&self.provider_context)?;
        crate::utils::http_headers::inject_tracing_headers(&mut headers);
        let resp_first = self
            .http_client
            .get(&url)
            .headers(headers.clone())
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
        let resp = if !resp_first.status().is_success() {
            let status = resp_first.status();
            if status.as_u16() == 401 {
                let mut headers = self.provider_spec.build_headers(&self.provider_context)?;
                crate::utils::http_headers::inject_tracing_headers(&mut headers);
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

        // Use parse_json_with_repair for automatic JSON repair when enabled
        let json: serde_json::Value = crate::streaming::parse_json_with_repair(&text)
            .map_err(|e| LlmError::ParseError(e.to_string()))?;
        self.transformer.transform_list_response(&json)
    }

    async fn retrieve(&self, file_id: String) -> Result<FileObject, LlmError> {
        let endpoint = self.transformer.retrieve_endpoint(&file_id);
        let base_url = self.provider_spec.files_base_url(&self.provider_context);
        let url = format!("{}{}", base_url, endpoint);

        let mut headers = self.provider_spec.build_headers(&self.provider_context)?;
        crate::utils::http_headers::inject_tracing_headers(&mut headers);
        let resp_first = self
            .http_client
            .get(&url)
            .headers(headers.clone())
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
        let resp = if !resp_first.status().is_success() {
            let status = resp_first.status();
            if status.as_u16() == 401 {
                let mut headers = self.provider_spec.build_headers(&self.provider_context)?;
                crate::utils::http_headers::inject_tracing_headers(&mut headers);
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

        // Use parse_json_with_repair for automatic JSON repair when enabled
        let json: serde_json::Value = crate::streaming::parse_json_with_repair(&text)
            .map_err(|e| LlmError::ParseError(e.to_string()))?;
        self.transformer.transform_file_object(&json)
    }

    async fn delete(&self, file_id: String) -> Result<FileDeleteResponse, LlmError> {
        let endpoint = self.transformer.delete_endpoint(&file_id);
        let base_url = self.provider_spec.files_base_url(&self.provider_context);
        let url = format!("{}{}", base_url, endpoint);

        let mut headers = self.provider_spec.build_headers(&self.provider_context)?;
        crate::utils::http_headers::inject_tracing_headers(&mut headers);
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
                let mut headers = self.provider_spec.build_headers(&self.provider_context)?;
                crate::utils::http_headers::inject_tracing_headers(&mut headers);
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
            let base_url = self.provider_spec.files_base_url(&self.provider_context);
            let url = format!("{}{}", base_url, ep);
            let mut headers = self.provider_spec.build_headers(&self.provider_context)?;
            crate::utils::http_headers::inject_tracing_headers(&mut headers);
            (url, headers, false)
        } else {
            let file = self.retrieve(file_id.clone()).await?;
            let content_url = self
                .transformer
                .content_url_from_file_object(&file)
                .ok_or_else(|| {
                    LlmError::UnsupportedOperation("File download URI not available".to_string())
                })?;
            let mut headers = self.provider_spec.build_headers(&self.provider_context)?;
            crate::utils::http_headers::inject_tracing_headers(&mut headers);
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
                let mut headers2 = self.provider_spec.build_headers(&self.provider_context)?;
                crate::utils::http_headers::inject_tracing_headers(&mut headers2);
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
