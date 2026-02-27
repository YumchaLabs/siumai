//! Anthropic Files API (provider-specific).
//!
//! This module is intentionally implemented as a provider-only helper (not part of the unified
//! `FileManagementCapability` surface) to stay aligned with the Vercel AI SDK scope.
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
use crate::utils::url::join_url;
use reqwest::Client as HttpClient;
use secrecy::{ExposeSecret, SecretString};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct AnthropicFile {
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub r#type: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct AnthropicListFilesResponse {
    #[serde(default)]
    pub data: Vec<AnthropicFile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_id: Option<String>,
    #[serde(default)]
    pub has_more: bool,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct AnthropicFileDeleteResponse {
    pub id: Option<String>,
    pub deleted: Option<bool>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
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

    /// Upload a file via multipart form data.
    pub async fn upload(
        &self,
        filename: String,
        bytes: Vec<u8>,
        mime_type: Option<String>,
        purpose: Option<String>,
        per_request_headers: Option<HashMap<String, String>>,
    ) -> Result<AnthropicFile, LlmError> {
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
                let res = execute_multipart_request(
                    &http_config,
                    &url,
                    || {
                        let mut form = reqwest::multipart::Form::new();

                        let mut part = reqwest::multipart::Part::bytes(bytes.clone())
                            .file_name(filename.clone());
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
                    headers.as_ref(),
                )
                .await?;

                serde_json::from_value(res.json).map_err(|e| {
                    LlmError::ParseError(format!("Failed to parse Anthropic upload response: {e}"))
                })
            }
        };

        crate::retry_api::maybe_retry(self.retry_options.clone(), call).await
    }

    /// List files (query params are provider-defined; pass through as raw key/value pairs).
    pub async fn list(
        &self,
        query: Option<HashMap<String, String>>,
        per_request_headers: Option<HashMap<String, String>>,
    ) -> Result<AnthropicListFilesResponse, LlmError> {
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
                let res = execute_get_request(&http_config, &url, headers.as_ref()).await?;
                serde_json::from_value(res.json).map_err(|e| {
                    LlmError::ParseError(format!(
                        "Failed to parse Anthropic list files response: {e}"
                    ))
                })
            }
        };

        crate::retry_api::maybe_retry(self.retry_options.clone(), call).await
    }

    /// Retrieve a file metadata object.
    pub async fn retrieve(
        &self,
        file_id: String,
        per_request_headers: Option<HashMap<String, String>>,
    ) -> Result<AnthropicFile, LlmError> {
        let url = join_url(&self.base_url, &format!("files/{file_id}"));
        let ctx = self.build_context();
        let http_config = self.build_http_config(ctx);

        let call = || {
            let http_config = http_config.clone();
            let url = url.clone();
            let headers = per_request_headers.clone();
            async move {
                let res = execute_get_request(&http_config, &url, headers.as_ref()).await?;
                serde_json::from_value(res.json).map_err(|e| {
                    LlmError::ParseError(format!(
                        "Failed to parse Anthropic retrieve file response: {e}"
                    ))
                })
            }
        };

        crate::retry_api::maybe_retry(self.retry_options.clone(), call).await
    }

    /// Delete a file.
    pub async fn delete(
        &self,
        file_id: String,
        per_request_headers: Option<HashMap<String, String>>,
    ) -> Result<AnthropicFileDeleteResponse, LlmError> {
        let url = join_url(&self.base_url, &format!("files/{file_id}"));
        let ctx = self.build_context();
        let http_config = self.build_http_config(ctx);

        let call = || {
            let http_config = http_config.clone();
            let url = url.clone();
            let headers = per_request_headers.clone();
            async move {
                let res = execute_delete_request(&http_config, &url, headers.as_ref()).await?;
                serde_json::from_value(res.json).map_err(|e| {
                    LlmError::ParseError(format!(
                        "Failed to parse Anthropic delete file response: {e}"
                    ))
                })
            }
        };

        crate::retry_api::maybe_retry(self.retry_options.clone(), call).await
    }

    /// Download file content as bytes.
    pub async fn get_content(
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
            async move { execute_get_binary(&http_config, &url, headers.as_ref()).await }
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
