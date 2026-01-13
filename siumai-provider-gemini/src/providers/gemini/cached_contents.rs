//! Gemini Cached Contents API (provider-specific)
//!
//! This module implements CRUD helpers for `/cachedContents` and related endpoints.

use crate::error::LlmError;
use crate::execution::executors::common::{
    HttpBody, HttpExecutionConfig, execute_delete_request, execute_get_request,
    execute_json_request, execute_patch_json_request,
};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::retry_api::RetryOptions;
use crate::utils::url::join_url;
use reqwest::Client as HttpClient;
use std::collections::HashMap;
use std::sync::Arc;

use super::types::GeminiConfig;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GeminiCachedContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GeminiListCachedContentsResponse {
    #[serde(default, rename = "cachedContents")]
    pub cached_contents: Vec<GeminiCachedContent>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "nextPageToken")]
    pub next_page_token: Option<String>,
}

/// Provider-scoped client for cached contents.
#[derive(Clone)]
pub struct GeminiCachedContents {
    pub(crate) config: GeminiConfig,
    pub(crate) http_client: HttpClient,
    pub(crate) http_interceptors: Vec<std::sync::Arc<dyn HttpInterceptor>>,
    pub(crate) retry_options: Option<RetryOptions>,
}

impl GeminiCachedContents {
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

    /// Create a cached content entry.
    ///
    /// The request body must match the Gemini `CachedContent` schema.
    pub async fn create(
        &self,
        cached_content: serde_json::Value,
    ) -> Result<GeminiCachedContent, LlmError> {
        let url = join_url(&self.config.base_url, "cachedContents");
        let ctx = self.build_context().await;
        let http_config = self.build_http_config(ctx);
        let call = || {
            let http_config = http_config.clone();
            let url = url.clone();
            let body = cached_content.clone();
            async move {
                let result =
                    execute_json_request(&http_config, &url, HttpBody::Json(body), None, false)
                        .await?;
                parse_json::<GeminiCachedContent>(result.json, "create")
            }
        };
        self.retry(call).await
    }

    /// List cached content entries.
    pub async fn list(
        &self,
        page_size: Option<i32>,
        page_token: Option<String>,
    ) -> Result<GeminiListCachedContentsResponse, LlmError> {
        let mut url = join_url(&self.config.base_url, "cachedContents");
        let mut qs = Vec::new();
        if let Some(ps) = page_size {
            qs.push(format!("pageSize={ps}"));
        }
        if let Some(pt) = page_token {
            qs.push(format!("pageToken={}", urlencoding::encode(&pt)));
        }
        if !qs.is_empty() {
            url.push('?');
            url.push_str(&qs.join("&"));
        }

        let ctx = self.build_context().await;
        let http_config = self.build_http_config(ctx);
        let call = || {
            let http_config = http_config.clone();
            let url = url.clone();
            async move {
                let result = execute_get_request(&http_config, &url, None).await?;
                parse_json::<GeminiListCachedContentsResponse>(result.json, "list")
            }
        };
        self.retry(call).await
    }

    /// Get a cached content entry by id.
    pub async fn get(&self, id: String) -> Result<GeminiCachedContent, LlmError> {
        let url = join_url(&self.config.base_url, &format!("cachedContents/{id}"));
        let ctx = self.build_context().await;
        let http_config = self.build_http_config(ctx);
        let call = || {
            let http_config = http_config.clone();
            let url = url.clone();
            async move {
                let result = execute_get_request(&http_config, &url, None).await?;
                parse_json::<GeminiCachedContent>(result.json, "get")
            }
        };
        self.retry(call).await
    }

    /// Delete a cached content entry by id.
    pub async fn delete(&self, id: String) -> Result<(), LlmError> {
        let url = join_url(&self.config.base_url, &format!("cachedContents/{id}"));
        let ctx = self.build_context().await;
        let http_config = self.build_http_config(ctx);
        let call = || {
            let http_config = http_config.clone();
            let url = url.clone();
            async move {
                let _ = execute_delete_request(&http_config, &url, None).await?;
                Ok(())
            }
        };
        self.retry(call).await
    }

    /// Update a cached content entry (PATCH).
    ///
    /// Note: the API only allows updating expiration fields.
    pub async fn update(
        &self,
        id: String,
        cached_content: serde_json::Value,
        update_mask: Option<String>,
    ) -> Result<GeminiCachedContent, LlmError> {
        let mut url = join_url(&self.config.base_url, &format!("cachedContents/{id}"));
        if let Some(mask) = update_mask {
            url.push_str(&format!("?updateMask={}", urlencoding::encode(&mask)));
        }

        let ctx = self.build_context().await;
        let http_config = self.build_http_config(ctx);
        let call = || {
            let http_config = http_config.clone();
            let url = url.clone();
            let body = cached_content.clone();
            async move {
                let result = execute_patch_json_request(&http_config, &url, body, None).await?;
                parse_json::<GeminiCachedContent>(result.json, "update")
            }
        };
        self.retry(call).await
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
            "Failed to parse Gemini cached_contents {what} response: {e}"
        ))
    })
}
