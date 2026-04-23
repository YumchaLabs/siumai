//! Anthropic Files API (provider-specific).
//!
//! This module remains a provider-owned helper surface, but now reuses the shared file-management
//! request/response structs so upload paths and provider-specific file management stay on one
//! contract.
//!
//! Notes:
//! - File IDs (e.g. `file_...`) can be produced by server-side tools such as code execution.
//! - Some environments require enabling the `files-api-2025-04-14` beta via `anthropic-beta`.

use crate::error::LlmError;
use crate::execution::executors::common::{
    HttpBinaryResult, HttpExecutionConfig, execute_delete_request, execute_get_binary,
    execute_get_request, execute_multipart_request,
};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::retry_api::RetryOptions;
use crate::traits::FileManagementCapability;
use crate::types::{
    FileDeleteResponse, FileListQuery, FileListResponse, FileObject, FileUploadRequest, HttpConfig,
};
use crate::utils::url::join_url;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use reqwest::Client as HttpClient;
use secrecy::{ExposeSecret, SecretString};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, serde::Deserialize)]
struct AnthropicFileResponse {
    id: Option<String>,
    #[serde(rename = "type")]
    r#type: Option<String>,
    #[serde(flatten)]
    extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct AnthropicListFilesResponseBody {
    #[serde(default)]
    data: Vec<AnthropicFileResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    first_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_id: Option<String>,
    #[serde(default)]
    has_more: bool,
    #[serde(flatten)]
    _extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct AnthropicFileDeleteResponseBody {
    id: Option<String>,
    deleted: Option<bool>,
    #[serde(flatten)]
    extra: HashMap<String, serde_json::Value>,
}

/// Provider-scoped client for file operations.
#[derive(Clone)]
pub struct AnthropicFiles {
    pub(crate) api_key: SecretString,
    pub(crate) base_url: String,
    pub(crate) http_client: HttpClient,
    pub(crate) http_config: crate::types::HttpConfig,
    pub(crate) beta_features: Vec<String>,
    pub(crate) http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    pub(crate) http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    pub(crate) retry_options: Option<RetryOptions>,
}

impl AnthropicFiles {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        api_key: SecretString,
        base_url: String,
        http_client: HttpClient,
        http_config: crate::types::HttpConfig,
        beta_features: Vec<String>,
        http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
        http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
        retry_options: Option<RetryOptions>,
    ) -> Self {
        Self {
            api_key,
            base_url,
            http_client,
            http_config,
            beta_features,
            http_transport,
            http_interceptors,
            retry_options,
        }
    }

    /// Upload a file via multipart form data using the shared file request shape.
    pub async fn upload_file(&self, request: FileUploadRequest) -> Result<FileObject, LlmError> {
        let FileUploadRequest {
            content,
            filename,
            mime_type,
            purpose,
            metadata: _metadata,
            provider_options: _provider_options,
            http_config,
        } = request;

        let response = self
            .upload_response(
                filename,
                content,
                mime_type,
                (!purpose.trim().is_empty()).then_some(purpose.clone()),
                per_request_headers(http_config.as_ref()),
            )
            .await?;

        anthropic_file_to_file_object(response, Some(purpose.as_str()))
    }

    /// List files using the shared list query shape.
    pub async fn list_files(
        &self,
        query: Option<FileListQuery>,
    ) -> Result<FileListResponse, LlmError> {
        let (query, http_config) = match query {
            Some(query) => {
                let params = anthropic_query_map(&query);
                (Some(params), query.http_config)
            }
            None => (None, None),
        };

        let response = self
            .list_response(query, per_request_headers(http_config.as_ref()))
            .await?;

        anthropic_list_files_response(response)
    }

    /// Retrieve a file metadata object.
    pub async fn retrieve_file(&self, file_id: String) -> Result<FileObject, LlmError> {
        let response = self.retrieve_response(file_id, None).await?;
        anthropic_file_to_file_object(response, None)
    }

    /// Delete a file.
    pub async fn delete_file(&self, file_id: String) -> Result<FileDeleteResponse, LlmError> {
        let response = self.delete_response(file_id, None).await?;
        anthropic_file_delete_response(response)
    }

    /// Download file content as bytes.
    pub async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        let response = self.get_content_response(file_id, None).await?;
        Ok(response.bytes)
    }

    async fn upload_response(
        &self,
        filename: Option<String>,
        bytes: Vec<u8>,
        mime_type: Option<String>,
        purpose: Option<String>,
        per_request_headers: Option<HashMap<String, String>>,
    ) -> Result<AnthropicFileResponse, LlmError> {
        let url = join_url(&self.base_url, "files");
        let ctx = self.build_context();
        let http_config = self.build_http_config(ctx);

        let call = || {
            let http_config = http_config.clone();
            let url = url.clone();
            let headers = per_request_headers.clone();
            let filename = filename.clone();
            let bytes = bytes.clone();
            let mime_type = mime_type.clone();
            let purpose = purpose.clone();

            async move {
                let per_request_http_config = headers.as_ref().map(|h| {
                    let mut cfg = crate::types::HttpConfig::empty();
                    cfg.headers = h.clone();
                    cfg
                });
                let res = execute_multipart_request(
                    &http_config,
                    &url,
                    || {
                        let mut form = reqwest::multipart::Form::new();

                        let mut part = reqwest::multipart::Part::bytes(bytes.clone());
                        if let Some(filename) = filename.clone() {
                            part = part.file_name(filename);
                        }
                        if let Some(mt) = mime_type.as_deref() {
                            part = part.mime_str(mt).map_err(|e| {
                                LlmError::InvalidParameter(format!("Invalid MIME type '{mt}': {e}"))
                            })?;
                        }
                        form = form.part("file", part);

                        if let Some(purpose) = purpose.as_deref() {
                            form = form.text("purpose", purpose.to_string());
                        }

                        Ok(form)
                    },
                    per_request_http_config.as_ref(),
                )
                .await?;

                serde_json::from_value(res.json).map_err(|e| {
                    LlmError::ParseError(format!("Failed to parse Anthropic upload response: {e}"))
                })
            }
        };

        crate::retry_api::maybe_retry(self.retry_options.clone(), call).await
    }

    async fn list_response(
        &self,
        query: Option<HashMap<String, String>>,
        per_request_headers: Option<HashMap<String, String>>,
    ) -> Result<AnthropicListFilesResponseBody, LlmError> {
        let mut url = reqwest::Url::parse(&join_url(&self.base_url, "files"))
            .map_err(|e| LlmError::InvalidInput(format!("Invalid files URL: {e}")))?;
        if let Some(q) = &query {
            url.query_pairs_mut().extend_pairs(q.iter());
        }

        let ctx = self.build_context();
        let http_config = self.build_http_config(ctx);
        let call = || {
            let http_config = http_config.clone();
            let url = url.to_string();
            let headers = per_request_headers.clone();
            async move {
                let per_request_http_config = headers.as_ref().map(|h| {
                    let mut cfg = crate::types::HttpConfig::empty();
                    cfg.headers = h.clone();
                    cfg
                });
                let res = execute_get_request(&http_config, &url, per_request_http_config.as_ref())
                    .await?;
                serde_json::from_value(res.json).map_err(|e| {
                    LlmError::ParseError(format!(
                        "Failed to parse Anthropic list files response: {e}"
                    ))
                })
            }
        };

        crate::retry_api::maybe_retry(self.retry_options.clone(), call).await
    }

    async fn retrieve_response(
        &self,
        file_id: String,
        per_request_headers: Option<HashMap<String, String>>,
    ) -> Result<AnthropicFileResponse, LlmError> {
        let url = join_url(&self.base_url, &format!("files/{file_id}"));
        let ctx = self.build_context();
        let http_config = self.build_http_config(ctx);

        let call = || {
            let http_config = http_config.clone();
            let url = url.clone();
            let headers = per_request_headers.clone();
            async move {
                let per_request_http_config = headers.as_ref().map(|h| {
                    let mut cfg = crate::types::HttpConfig::empty();
                    cfg.headers = h.clone();
                    cfg
                });
                let res = execute_get_request(&http_config, &url, per_request_http_config.as_ref())
                    .await?;
                serde_json::from_value(res.json).map_err(|e| {
                    LlmError::ParseError(format!(
                        "Failed to parse Anthropic retrieve file response: {e}"
                    ))
                })
            }
        };

        crate::retry_api::maybe_retry(self.retry_options.clone(), call).await
    }

    async fn delete_response(
        &self,
        file_id: String,
        per_request_headers: Option<HashMap<String, String>>,
    ) -> Result<AnthropicFileDeleteResponseBody, LlmError> {
        let url = join_url(&self.base_url, &format!("files/{file_id}"));
        let ctx = self.build_context();
        let http_config = self.build_http_config(ctx);

        let call = || {
            let http_config = http_config.clone();
            let url = url.clone();
            let headers = per_request_headers.clone();
            async move {
                let per_request_http_config = headers.as_ref().map(|h| {
                    let mut cfg = crate::types::HttpConfig::empty();
                    cfg.headers = h.clone();
                    cfg
                });
                let res =
                    execute_delete_request(&http_config, &url, per_request_http_config.as_ref())
                        .await?;
                serde_json::from_value(res.json).map_err(|e| {
                    LlmError::ParseError(format!(
                        "Failed to parse Anthropic delete file response: {e}"
                    ))
                })
            }
        };

        crate::retry_api::maybe_retry(self.retry_options.clone(), call).await
    }

    async fn get_content_response(
        &self,
        file_id: String,
        per_request_headers: Option<HashMap<String, String>>,
    ) -> Result<HttpBinaryResult, LlmError> {
        let url = join_url(&self.base_url, &format!("files/{file_id}/content"));
        let ctx = self.build_context();
        let http_config = self.build_http_config(ctx);

        let call = || {
            let http_config = http_config.clone();
            let url = url.clone();
            let headers = per_request_headers.clone();
            async move {
                let per_request_http_config = headers.as_ref().map(|h| {
                    let mut cfg = crate::types::HttpConfig::empty();
                    cfg.headers = h.clone();
                    cfg
                });
                execute_get_binary(&http_config, &url, per_request_http_config.as_ref()).await
            }
        };

        crate::retry_api::maybe_retry(self.retry_options.clone(), call).await
    }

    fn build_context(&self) -> crate::core::ProviderContext {
        let mut headers = self.http_config.headers.clone();

        if !self.beta_features.is_empty() {
            let mut existing_values: Vec<String> = Vec::new();
            let keys: Vec<String> = headers
                .keys()
                .filter(|k| k.eq_ignore_ascii_case("anthropic-beta"))
                .cloned()
                .collect();
            for k in keys {
                if let Some(v) = headers.remove(&k) {
                    existing_values.push(v);
                }
            }

            let mut merged = String::new();
            if !existing_values.is_empty() {
                merged.push_str(&existing_values.join(","));
            }
            if !merged.is_empty() {
                merged.push(',');
            }
            merged.push_str(&self.beta_features.join(","));

            headers.insert("anthropic-beta".to_string(), merged);
        }

        crate::core::ProviderContext::new(
            "anthropic",
            self.base_url.clone(),
            Some(self.api_key.expose_secret().to_string()),
            headers,
        )
    }

    fn build_http_config(&self, ctx: crate::core::ProviderContext) -> HttpExecutionConfig {
        let mut wiring = crate::execution::wiring::HttpExecutionWiring::new(
            "anthropic",
            self.http_client.clone(),
            ctx,
        )
        .with_interceptors(self.http_interceptors.clone())
        .with_retry_options(self.retry_options.clone());

        if let Some(transport) = self.http_transport.clone() {
            wiring = wiring.with_transport(transport);
        }

        wiring.config(Arc::new(super::spec::AnthropicSpec::new()))
    }
}

#[async_trait]
impl FileManagementCapability for AnthropicFiles {
    async fn upload_file(&self, request: FileUploadRequest) -> Result<FileObject, LlmError> {
        AnthropicFiles::upload_file(self, request).await
    }

    async fn list_files(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError> {
        AnthropicFiles::list_files(self, query).await
    }

    async fn retrieve_file(&self, file_id: String) -> Result<FileObject, LlmError> {
        AnthropicFiles::retrieve_file(self, file_id).await
    }

    async fn delete_file(&self, file_id: String) -> Result<FileDeleteResponse, LlmError> {
        AnthropicFiles::delete_file(self, file_id).await
    }

    async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        AnthropicFiles::get_file_content(self, file_id).await
    }
}

#[async_trait]
impl FileManagementCapability for super::AnthropicClient {
    async fn upload_file(&self, request: FileUploadRequest) -> Result<FileObject, LlmError> {
        self.files().upload_file(request).await
    }

    async fn list_files(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError> {
        self.files().list_files(query).await
    }

    async fn retrieve_file(&self, file_id: String) -> Result<FileObject, LlmError> {
        self.files().retrieve_file(file_id).await
    }

    async fn delete_file(&self, file_id: String) -> Result<FileDeleteResponse, LlmError> {
        self.files().delete_file(file_id).await
    }

    async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        self.files().get_file_content(file_id).await
    }
}

fn per_request_headers(http_config: Option<&HttpConfig>) -> Option<HashMap<String, String>> {
    http_config
        .map(|config| config.headers.clone())
        .filter(|headers| !headers.is_empty())
}

fn anthropic_query_map(query: &FileListQuery) -> HashMap<String, String> {
    let mut params = HashMap::new();

    if let Some(purpose) = query
        .purpose
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        params.insert("purpose".to_string(), purpose.to_string());
    }

    if let Some(limit) = query.limit {
        params.insert("limit".to_string(), limit.to_string());
    }

    if let Some(after) = query
        .after
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        params.insert("after".to_string(), after.to_string());
    }

    if let Some(order) = query
        .order
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        params.insert("order".to_string(), order.to_string());
    }

    params
}

fn anthropic_file_to_file_object(
    file: AnthropicFileResponse,
    fallback_purpose: Option<&str>,
) -> Result<FileObject, LlmError> {
    let file_id = required_non_empty_id(file.id, "Anthropic file response missing file id")?;
    let filename = extra_non_empty_string(&file.extra, "filename");
    let mime_type = extra_non_empty_string(&file.extra, "mime_type");
    let bytes = file
        .extra
        .get("size_bytes")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_default();
    let created_at = file
        .extra
        .get("created_at")
        .and_then(serde_json::Value::as_str)
        .and_then(parse_anthropic_timestamp)
        .unwrap_or_default();
    let purpose = extra_non_empty_string(&file.extra, "purpose")
        .or_else(|| fallback_purpose.and_then(|value| non_empty_string(value.to_string())))
        .unwrap_or_default();
    let status = extra_non_empty_string(&file.extra, "status").unwrap_or_default();

    Ok(FileObject {
        id: file_id,
        filename,
        bytes,
        created_at,
        purpose,
        status,
        mime_type,
        metadata: anthropic_file_metadata(file.r#type.as_deref(), &file.extra),
    })
}

fn anthropic_list_files_response(
    response: AnthropicListFilesResponseBody,
) -> Result<FileListResponse, LlmError> {
    let files = response
        .data
        .into_iter()
        .map(|file| anthropic_file_to_file_object(file, None))
        .collect::<Result<Vec<_>, _>>()?;

    let next_cursor = if response.has_more {
        response.last_id.or(response.first_id)
    } else {
        None
    };

    Ok(FileListResponse {
        files,
        has_more: response.has_more,
        next_cursor,
    })
}

fn anthropic_file_delete_response(
    response: AnthropicFileDeleteResponseBody,
) -> Result<FileDeleteResponse, LlmError> {
    let file_id = required_non_empty_id(
        response.id,
        "Anthropic file delete response missing file id",
    )?;

    let deleted = response.deleted.unwrap_or(false)
        || response
            .extra
            .get("deleted")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);

    Ok(FileDeleteResponse {
        id: file_id,
        deleted,
    })
}

fn anthropic_file_metadata(
    file_type: Option<&str>,
    extra: &HashMap<String, serde_json::Value>,
) -> HashMap<String, serde_json::Value> {
    let mut metadata = HashMap::new();

    if let Some(file_type) = file_type.filter(|value| !value.trim().is_empty()) {
        metadata.insert(
            "type".to_string(),
            serde_json::Value::String(file_type.to_string()),
        );
    }

    for (key, value) in extra {
        let normalized_key = match key.as_str() {
            "mime_type" => "mimeType",
            "size_bytes" => "sizeBytes",
            "created_at" => "createdAt",
            other => other,
        };

        metadata.insert(normalized_key.to_string(), value.clone());
    }

    metadata
}

fn extra_non_empty_string(extra: &HashMap<String, serde_json::Value>, key: &str) -> Option<String> {
    extra
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned)
        .and_then(non_empty_string)
}

fn required_non_empty_id(id: Option<String>, message: &str) -> Result<String, LlmError> {
    id.and_then(non_empty_string)
        .ok_or_else(|| LlmError::ParseError(message.to_string()))
}

fn non_empty_string(value: String) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else if trimmed.len() == value.len() {
        Some(value)
    } else {
        Some(trimmed.to_string())
    }
}

fn parse_anthropic_timestamp(value: &str) -> Option<u64> {
    value
        .parse::<DateTime<Utc>>()
        .ok()
        .map(|dt| dt.timestamp() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_context_merges_beta_features_into_anthropic_beta_header() {
        let mut cfg = crate::types::HttpConfig::default();
        cfg.headers
            .insert("Anthropic-Beta".to_string(), "a,b".to_string());

        let files = AnthropicFiles::new(
            SecretString::from("k".to_string()),
            "https://api.anthropic.com/v1".to_string(),
            reqwest::Client::new(),
            cfg,
            vec!["b".to_string(), "c".to_string()],
            None,
            vec![],
            None,
        );

        let ctx = files.build_context();
        assert_eq!(
            ctx.http_extra_headers
                .get("anthropic-beta")
                .map(|s| s.as_str()),
            Some("a,b,b,c")
        );
    }

    #[test]
    fn content_url_includes_files_content_suffix() {
        let files = AnthropicFiles::new(
            SecretString::from("k".to_string()),
            "https://api.anthropic.com/v1".to_string(),
            reqwest::Client::new(),
            crate::types::HttpConfig::default(),
            vec![],
            None,
            vec![],
            None,
        );

        let url = join_url(&files.base_url, "files/file_123/content");
        assert!(url.ends_with("/v1/files/file_123/content"));
    }
}
