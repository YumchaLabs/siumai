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
use crate::execution::executors::common::{
    HttpBody, HttpExecutionConfig, execute_delete_request, execute_get_request,
    execute_json_request, execute_multipart_request,
};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::retry_api::RetryOptions;
use crate::utils::url::{join_url, join_url_segments};
use reqwest::Client as HttpClient;
use std::sync::Arc;

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

        let ctx = self.build_context().await;
        let config = self.build_http_config(ctx);
        let call = || {
            let config = config.clone();
            let url = url.clone();
            let body = body.clone();
            async move {
                let result =
                    execute_json_request(&config, &url, HttpBody::Json(body), None, false).await?;
                parse_json::<FileSearchStore>(result.json, "create_store")
            }
        };
        self.retry(call).await
    }

    /// Directly upload bytes into a File Search Store (multipart upload)
    pub async fn upload_to_file_search_store(
        &self,
        store_name: String,
        content: Vec<u8>,
        filename: String,
        mime_type: Option<String>,
        display_name: Option<String>,
        upload_config: Option<FileSearchUploadConfig>,
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

        let ctx = self.build_context().await;
        let http_config = self.build_http_config(ctx);

        // Build form builder closure (multipart forms cannot be cloned, so we need a builder)
        let content_clone = content.clone();
        let filename_clone = filename.clone();
        let mime_type_clone = mime_type.clone();
        let display_name_clone = display_name.clone();
        let upload_config_clone = upload_config.clone();

        let build_form = move || -> Result<reqwest::multipart::Form, LlmError> {
            // Build metadata JSON - only include display_name and chunking_config at top level
            let mut metadata = serde_json::Map::new();
            if let Some(name) = display_name_clone.clone() {
                metadata.insert("display_name".to_string(), serde_json::json!(name));
            }
            if let Some(cfg) = upload_config_clone.clone() {
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
                .map_err(|e| {
                    LlmError::InternalError(format!(
                        "Invalid hardcoded MIME type 'application/json': {e}"
                    ))
                })?;

            let detected = mime_type_clone.clone().unwrap_or_else(|| {
                crate::utils::guess_mime(Some(&content_clone), Some(&filename_clone))
            });
            let file_part = reqwest::multipart::Part::bytes(content_clone.clone())
                .file_name(filename_clone.clone())
                .mime_str(&detected)
                .map_err(|e| {
                    LlmError::InvalidParameter(format!("Invalid MIME type '{detected}': {e}"))
                })?;

            let form = reqwest::multipart::Form::new()
                .part("metadata", metadata_part)
                .part("file", file_part);
            Ok(form)
        };

        let result = execute_multipart_request(&http_config, &url, build_form, None).await?;
        parse_json::<FileSearchOperation>(result.json, "upload_to_file_search_store")
    }

    /// Import a previously uploaded file (by Files API) into a store
    ///
    /// `store_name` is the full resource name returned by `create_store` (for example,
    /// "fileSearchStores/abc123"). `file_name` is the name returned by Files API upload.
    pub async fn import_file(
        &self,
        store_name: String,
        file_name: String,
        upload_config: Option<FileSearchUploadConfig>,
    ) -> Result<FileSearchOperation, LlmError> {
        // POST /{store}:importFile { fileName }
        let base = self.config.base_url.clone();
        let endpoint = format!("{}:importFile", store_name.trim_end_matches('/'));
        let url = join_url_segments(&[&base, &endpoint]);
        let mut body = serde_json::json!({ "fileName": file_name });
        if let Some(cfg) = upload_config
            && let Ok(v) = serde_json::to_value(cfg)
        {
            body["config"] = v;
        }

        let ctx = self.build_context().await;
        let config = self.build_http_config(ctx);
        let call = || {
            let config = config.clone();
            let url = url.clone();
            let body = body.clone();
            async move {
                let result =
                    execute_json_request(&config, &url, HttpBody::Json(body), None, false).await?;
                parse_json::<FileSearchOperation>(result.json, "import_file")
            }
        };
        self.retry(call).await
    }

    /// Get a File Search Store by name
    pub async fn get_store(&self, store_name: String) -> Result<FileSearchStore, LlmError> {
        let base = self.config.base_url.clone();
        let url = join_url_segments(&[&base, &store_name]);
        let ctx = self.build_context().await;
        let config = self.build_http_config(ctx);
        let call = || {
            let config = config.clone();
            let url = url.clone();
            async move {
                let result = execute_get_request(&config, &url, None).await?;
                parse_json::<FileSearchStore>(result.json, "get_store")
            }
        };
        self.retry(call).await
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

        let ctx = self.build_context().await;
        let config = self.build_http_config(ctx);
        let call = || {
            let config = config.clone();
            let url = url.clone();
            async move {
                let result = execute_get_request(&config, &url, None).await?;
                parse_json::<FileSearchStoresList>(result.json, "list_stores")
            }
        };
        self.retry(call).await
    }

    /// Delete a File Search Store
    pub async fn delete_store(&self, store_name: String) -> Result<(), LlmError> {
        let base = self.config.base_url.clone();
        let url = join_url_segments(&[&base, &store_name]);
        let ctx = self.build_context().await;
        let config = self.build_http_config(ctx);
        let call = || {
            let config = config.clone();
            let url = url.clone();
            async move {
                let _ = execute_delete_request(&config, &url, None).await?;
                Ok(())
            }
        };
        self.retry(call).await
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
        let ctx = self.build_context().await;
        let config = self.build_http_config(ctx);
        let call = || {
            let config = config.clone();
            let url = url.clone();
            async move {
                let result = execute_get_request(&config, &url, None).await?;
                parse_json::<FileSearchOperation>(result.json, "get_operation")
            }
        };
        self.retry(call).await
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

    async fn build_context(&self) -> crate::core::ProviderContext {
        super::context::build_context(&self.config).await
    }

    fn build_http_config(&self, ctx: crate::core::ProviderContext) -> HttpExecutionConfig {
        let mut wiring = crate::execution::wiring::HttpExecutionWiring::new(
            "gemini",
            self.http_client.clone(),
            ctx,
        )
        .with_interceptors(self.http_interceptors.clone())
        .with_retry_options(self.retry_options.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            wiring = wiring.with_transport(transport);
        }

        wiring.config(Arc::new(crate::providers::gemini::spec::GeminiSpec))
    }

    async fn retry<T, F, Fut>(&self, call: F) -> Result<T, LlmError>
    where
        T: Send,
        F: Fn() -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<T, LlmError>> + Send,
    {
        crate::retry_api::maybe_retry(self.retry_options.clone(), call).await
    }
}

fn parse_json<T: serde::de::DeserializeOwned>(
    value: serde_json::Value,
    what: &str,
) -> Result<T, LlmError> {
    serde_json::from_value::<T>(value).map_err(|e| {
        LlmError::ParseError(format!(
            "Failed to parse Gemini file_search_stores {what} response: {e}"
        ))
    })
}
