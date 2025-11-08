//! Gemini File Search Stores API (provider-specific)
//!
//! Minimal client for managing File Search Stores and long-running operations.
//! This is intentionally provider-specific and not exposed via unified traits.
//!
//! API shape (v1beta):
//! - Create store: POST /fileSearchStores { displayName }
//! - Import file: POST /{store}:importFile { fileName }
//! - Get operation: GET /operations/{operation}
//!
//! Notes:
//! - For direct upload to a store, Gemini also supports `uploadToFileSearchStore`,
//!   which can be added later. The MVP focuses on create/import/poll flow.

use crate::error::LlmError;
use crate::execution::http::headers::ProviderHeaders;
use crate::execution::http::interceptor::{HttpInterceptor, HttpRequestContext};
use crate::retry_api::RetryOptions;
use crate::utils::url::{join_url, join_url_segments};
use reqwest::Client as HttpClient;
use reqwest::header::CONTENT_TYPE;
use secrecy::ExposeSecret;

use super::types::{
    FileSearchOperation, FileSearchStore, FileSearchStoresList, FileSearchUploadConfig,
    GeminiConfig,
};

/// Provider-scoped client for File Search Stores
#[derive(Clone)]
pub struct GeminiFileSearchStores {
    pub(crate) config: GeminiConfig,
    pub(crate) http_client: HttpClient,
    pub(crate) http_interceptors: Vec<std::sync::Arc<dyn HttpInterceptor>>,
    pub(crate) retry_options: Option<RetryOptions>,
}

impl GeminiFileSearchStores {
    pub fn new(
        config: GeminiConfig,
        http_client: HttpClient,
        http_interceptors: Vec<std::sync::Arc<dyn HttpInterceptor>>,
        retry_options: Option<RetryOptions>,
    ) -> Self {
        Self {
            config,
            http_client,
            http_interceptors,
            retry_options,
        }
    }

    /// Create a File Search Store
    ///
    /// If `display_name` is provided, it will be set on the store.
    pub async fn create_store(
        &self,
        display_name: Option<String>,
    ) -> Result<FileSearchStore, LlmError> {
        let base = self.config.base_url.clone();
        let url = join_url(&base, "fileSearchStores");
        let mut body = serde_json::json!({});
        if let Some(name) = display_name {
            body["displayName"] = serde_json::json!(name);
        }
        let headers = self.build_headers().await?;

        let ctx = HttpRequestContext {
            provider_id: "gemini".to_string(),
            url: url.clone(),
            stream: false,
        };

        self.send_json::<FileSearchStore>(&ctx, url, headers, body, reqwest::Method::POST)
            .await
    }

    /// Directly upload bytes into a File Search Store (multipart upload)
    pub async fn upload_to_file_search_store(
        &self,
        store_name: String,
        content: Vec<u8>,
        filename: String,
        mime_type: Option<String>,
        display_name: Option<String>,
        config: Option<FileSearchUploadConfig>,
    ) -> Result<FileSearchOperation, LlmError> {
        // Compute upload base by inserting '/upload' before version segment
        let base = self.config.base_url.clone();
        let upload_base = if base.contains("/v1beta") {
            base.replacen("/v1beta", "/upload/v1beta", 1)
        } else if base.contains("/v1/") {
            base.replacen("/v1/", "/upload/v1/", 1)
        } else if base.ends_with("/v1") {
            base.replacen("/v1", "/upload/v1", 1)
        } else {
            // Fallback: append /upload
            format!("{}/upload", base.trim_end_matches('/'))
        };
        // URL format: /upload/v1beta/{fileSearchStoreName}:uploadToFileSearchStore
        let endpoint = format!(
            "{}:uploadToFileSearchStore",
            store_name.trim_end_matches('/')
        );
        let mut url = join_url_segments(&[&upload_base, &endpoint]);
        // Google style upload uses uploadType=multipart
        url.push_str("?uploadType=multipart");

        // Auth headers: build gemini headers then remove Content-Type (reqwest sets multipart boundary)
        let mut headers = self.build_headers().await?;
        headers.remove(CONTENT_TYPE);

        let ctx = HttpRequestContext {
            provider_id: "gemini".to_string(),
            url: url.clone(),
            stream: false,
        };

        // Build form builder closure (multipart forms cannot be cloned, so we need a builder)
        let content_clone = content.clone();
        let filename_clone = filename.clone();
        let mime_type_clone = mime_type.clone();
        let display_name_clone = display_name.clone();
        let config_clone = config.clone();

        let build_form = move || -> Result<reqwest::multipart::Form, LlmError> {
            // Build metadata JSON - only include display_name and chunking_config at top level
            let mut metadata = serde_json::Map::new();
            if let Some(name) = display_name_clone.clone() {
                metadata.insert("display_name".to_string(), serde_json::json!(name));
            }
            if let Some(cfg) = config_clone.clone() {
                // Extract chunking_config from FileSearchUploadConfig
                if let Some(chunking) = cfg.chunking_config {
                    metadata.insert(
                        "chunking_config".to_string(),
                        serde_json::to_value(&chunking)
                            .map_err(|e| LlmError::JsonError(e.to_string()))?,
                    );
                }
            }

            // Google multipart upload format: metadata part + file part
            let metadata_json =
                serde_json::to_string(&metadata).map_err(|e| LlmError::JsonError(e.to_string()))?;
            let metadata_part = reqwest::multipart::Part::text(metadata_json)
                .mime_str("application/json")
                .map_err(|e| LlmError::HttpError(format!("Invalid MIME type: {e}")))?;

            let detected = mime_type_clone.clone().unwrap_or_else(|| {
                crate::utils::guess_mime(Some(&content_clone), Some(&filename_clone))
            });
            let file_part = reqwest::multipart::Part::bytes(content_clone.clone())
                .file_name(filename_clone.clone())
                .mime_str(&detected)
                .map_err(|e| LlmError::HttpError(format!("Invalid MIME type: {e}")))?;

            let form = reqwest::multipart::Form::new()
                .part("metadata", metadata_part)
                .part("file", file_part);
            Ok(form)
        };

        self.send_multipart_with_builder(&ctx, url, headers, build_form)
            .await
    }

    /// Import a previously uploaded file (by Files API) into a store
    ///
    /// `store_name` is the full resource name returned by `create_store` (for example,
    /// "fileSearchStores/abc123"). `file_name` is the name returned by Files API upload.
    pub async fn import_file(
        &self,
        store_name: String,
        file_name: String,
        config: Option<FileSearchUploadConfig>,
    ) -> Result<FileSearchOperation, LlmError> {
        // POST /{store}:importFile { fileName }
        let base = self.config.base_url.clone();
        let endpoint = format!("{}:importFile", store_name.trim_end_matches('/'));
        let url = join_url_segments(&[&base, &endpoint]);
        let mut body = serde_json::json!({ "fileName": file_name });
        if let Some(cfg) = config {
            if let Ok(v) = serde_json::to_value(cfg) {
                body["config"] = v;
            }
        }
        let headers = self.build_headers().await?;

        let ctx = HttpRequestContext {
            provider_id: "gemini".to_string(),
            url: url.clone(),
            stream: false,
        };

        self.send_json::<FileSearchOperation>(&ctx, url, headers, body, reqwest::Method::POST)
            .await
    }

    /// Get a File Search Store by name
    pub async fn get_store(&self, store_name: String) -> Result<FileSearchStore, LlmError> {
        let base = self.config.base_url.clone();
        let url = join_url_segments(&[&base, &store_name]);
        let headers = self.build_headers().await?;
        let ctx = HttpRequestContext {
            provider_id: "gemini".to_string(),
            url: url.clone(),
            stream: false,
        };
        self.send_get_json::<FileSearchStore>(&ctx, url, headers)
            .await
    }

    /// List File Search Stores (optionally paginated)
    pub async fn list_stores(
        &self,
        page_size: Option<u32>,
        page_token: Option<String>,
    ) -> Result<FileSearchStoresList, LlmError> {
        let base = self.config.base_url.clone();
        let mut url = join_url(&base, "fileSearchStores");
        let mut query: Vec<(String, String)> = Vec::new();
        if let Some(ps) = page_size {
            query.push(("pageSize".to_string(), ps.to_string()));
        }
        if let Some(pt) = page_token {
            query.push(("pageToken".to_string(), pt));
        }
        if !query.is_empty() {
            let qs = query
                .iter()
                .map(|(k, v)| format!("{}={}", urlencoding::encode(k), urlencoding::encode(v)))
                .collect::<Vec<_>>()
                .join("&");
            url = format!("{}?{}", url, qs);
        }
        let headers = self.build_headers().await?;
        let ctx = HttpRequestContext {
            provider_id: "gemini".to_string(),
            url: url.clone(),
            stream: false,
        };
        self.send_get_json::<FileSearchStoresList>(&ctx, url, headers)
            .await
    }

    /// Delete a File Search Store
    pub async fn delete_store(&self, store_name: String) -> Result<(), LlmError> {
        let base = self.config.base_url.clone();
        let url = join_url_segments(&[&base, &store_name]);
        let headers = self.build_headers().await?;
        let ctx = HttpRequestContext {
            provider_id: "gemini".to_string(),
            url: url.clone(),
            stream: false,
        };
        self.send_delete(&ctx, url, headers).await
    }

    /// Get long-running operation status
    pub async fn get_operation(
        &self,
        operation_name: String,
    ) -> Result<FileSearchOperation, LlmError> {
        // GET /{operation_name}
        // operation_name is the full path like "fileSearchStores/.../operations/..."
        let base = self.config.base_url.clone();
        let url = join_url(&base, &operation_name);
        let headers = self.build_headers().await?;
        let ctx = HttpRequestContext {
            provider_id: "gemini".to_string(),
            url: url.clone(),
            stream: false,
        };
        self.send_get_json::<FileSearchOperation>(&ctx, url, headers)
            .await
    }

    /// Wait for operation completion with a simple polling strategy.
    pub async fn wait_operation(
        &self,
        name: String,
        max_wait_seconds: u64,
    ) -> Result<FileSearchOperation, LlmError> {
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(max_wait_seconds);
        loop {
            let op = self.get_operation(name.clone()).await?;
            if op.done.unwrap_or(false) {
                return Ok(op);
            }
            if start.elapsed() >= timeout {
                return Err(LlmError::TimeoutError(format!(
                    "Operation {name} timed out after {max_wait_seconds}s"
                )));
            }
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }
    }

    async fn build_headers(&self) -> Result<reqwest::header::HeaderMap, LlmError> {
        let mut extra = self.config.http_config.headers.clone();
        // If a token provider exists, prefer Bearer auth over API key
        if let Some(ref tp) = self.config.token_provider {
            if let Ok(tok) = tp.token().await {
                extra.insert("Authorization".to_string(), format!("Bearer {tok}"));
            }
        }
        let api_key = self.config.api_key.expose_secret();
        ProviderHeaders::gemini(api_key, &extra)
    }

    async fn send_json<T: serde::de::DeserializeOwned + Send>(
        &self,
        ctx: &HttpRequestContext,
        url: String,
        headers: reqwest::header::HeaderMap,
        body: serde_json::Value,
        method: reqwest::Method,
    ) -> Result<T, LlmError> {
        // Build request
        let mut rb = self
            .http_client
            .request(method, &url)
            .headers(headers.clone())
            .json(&body);
        // Interceptors (before send)
        for itc in &self.http_interceptors {
            rb = itc.on_before_send(ctx, rb, &body, &headers)?;
        }

        let send = || {
            let rb = rb.try_clone().expect("reqwest::RequestBuilder clone");
            let ctx = ctx.clone();
            async move {
                let resp = rb
                    .send()
                    .await
                    .map_err(|e| LlmError::HttpError(e.to_string()))?;
                let status = resp.status();
                if !status.is_success() {
                    let code = status.as_u16();
                    let text = resp.text().await.unwrap_or_default();
                    let msg = if text.is_empty() {
                        format!("HTTP {}", code)
                    } else {
                        text
                    };
                    return Err(LlmError::api_error(code, msg));
                }
                // Notify interceptors on success
                for itc in &self.http_interceptors {
                    itc.on_response(&ctx, &resp).ok();
                }
                let json: serde_json::Value = resp
                    .json()
                    .await
                    .map_err(|e| LlmError::JsonError(e.to_string()))?;
                serde_json::from_value::<T>(json).map_err(|e| LlmError::ParseError(e.to_string()))
            }
        };

        if let Some(opts) = &self.retry_options {
            crate::retry_api::retry_with(send, opts.clone()).await
        } else {
            send().await
        }
    }

    async fn send_get_json<T: serde::de::DeserializeOwned + Send>(
        &self,
        ctx: &HttpRequestContext,
        url: String,
        headers: reqwest::header::HeaderMap,
    ) -> Result<T, LlmError> {
        // Build request
        let mut rb = self.http_client.get(&url).headers(headers.clone());
        for itc in &self.http_interceptors {
            rb = itc.on_before_send(ctx, rb, &serde_json::json!({}), &headers)?;
        }

        let send = || {
            let rb = rb.try_clone().expect("reqwest::RequestBuilder clone");
            let ctx = ctx.clone();
            async move {
                let resp = rb
                    .send()
                    .await
                    .map_err(|e| LlmError::HttpError(e.to_string()))?;
                let status = resp.status();
                if !status.is_success() {
                    let code = status.as_u16();
                    let text = resp.text().await.unwrap_or_default();
                    let msg = if text.is_empty() {
                        format!("HTTP {}", code)
                    } else {
                        text
                    };
                    return Err(LlmError::api_error(code, msg));
                }
                for itc in &self.http_interceptors {
                    itc.on_response(&ctx, &resp).ok();
                }
                let json: serde_json::Value = resp
                    .json()
                    .await
                    .map_err(|e| LlmError::JsonError(e.to_string()))?;
                serde_json::from_value::<T>(json).map_err(|e| LlmError::ParseError(e.to_string()))
            }
        };

        if let Some(opts) = &self.retry_options {
            crate::retry_api::retry_with(send, opts.clone()).await
        } else {
            send().await
        }
    }

    async fn send_delete(
        &self,
        ctx: &HttpRequestContext,
        url: String,
        headers: reqwest::header::HeaderMap,
    ) -> Result<(), LlmError> {
        let mut rb = self.http_client.delete(&url).headers(headers.clone());
        for itc in &self.http_interceptors {
            rb = itc.on_before_send(ctx, rb, &serde_json::json!({}), &headers)?;
        }
        let send = || {
            let rb = rb.try_clone().expect("reqwest::RequestBuilder clone");
            async move {
                let resp = rb
                    .send()
                    .await
                    .map_err(|e| LlmError::HttpError(e.to_string()))?;
                if !resp.status().is_success() {
                    let code = resp.status().as_u16();
                    let text = resp.text().await.unwrap_or_default();
                    let msg = if text.is_empty() {
                        format!("HTTP {}", code)
                    } else {
                        text
                    };
                    return Err(LlmError::api_error(code, msg));
                }
                Ok(())
            }
        };
        if let Some(opts) = &self.retry_options {
            crate::retry_api::retry_with(send, opts.clone()).await
        } else {
            send().await
        }
    }

    async fn send_multipart_with_builder<T, F>(
        &self,
        ctx: &HttpRequestContext,
        url: String,
        headers: reqwest::header::HeaderMap,
        build_form: F,
    ) -> Result<T, LlmError>
    where
        T: serde::de::DeserializeOwned + Send,
        F: Fn() -> Result<reqwest::multipart::Form, LlmError> + Send + Sync,
    {
        // Build form once for initial request
        let form = build_form()?;
        let mut rb = self
            .http_client
            .post(&url)
            .headers(headers.clone())
            .multipart(form);
        for itc in &self.http_interceptors {
            rb = itc.on_before_send(ctx, rb, &serde_json::json!({}), &headers)?;
        }

        // Send request (no retry for multipart to avoid complexity)
        // If retry is needed, caller should handle it at a higher level
        let resp = rb
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
        let status = resp.status();
        if !status.is_success() {
            let code = status.as_u16();
            let text = resp.text().await.unwrap_or_default();
            let msg = if text.is_empty() {
                format!("HTTP {}", code)
            } else {
                text
            };
            return Err(LlmError::api_error(code, msg));
        }
        for itc in &self.http_interceptors {
            itc.on_response(ctx, &resp).ok();
        }
        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| LlmError::JsonError(e.to_string()))?;
        serde_json::from_value::<T>(json).map_err(|e| LlmError::ParseError(e.to_string()))
    }
}
