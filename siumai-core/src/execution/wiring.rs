//! Execution wiring helpers.
//!
//! This module provides small, provider-agnostic helpers to reduce boilerplate when
//! assembling `HttpExecutionConfig` for executor/common HTTP helpers.

use crate::core::{ProviderContext, ProviderSpec};
use crate::execution::executors::common::HttpExecutionConfig;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::retry_api::RetryOptions;
use std::sync::Arc;

/// A reusable bundle of HTTP execution wiring for a single provider client instance.
///
/// This is intentionally minimal and focuses on the common `HttpExecutionConfig` inputs:
/// - `reqwest::Client`
/// - `ProviderContext` (base_url/api_key/custom headers)
/// - retry + interceptors
#[derive(Clone)]
pub struct HttpExecutionWiring {
    pub provider_id: String,
    pub http_client: reqwest::Client,
    pub provider_context: ProviderContext,
    pub interceptors: Vec<Arc<dyn HttpInterceptor>>,
    pub retry_options: Option<RetryOptions>,
}

impl std::fmt::Debug for HttpExecutionWiring {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpExecutionWiring")
            .field("provider_id", &self.provider_id)
            .field("base_url", &self.provider_context.base_url)
            .field("has_api_key", &self.provider_context.api_key.is_some())
            .field("interceptors", &self.interceptors.len())
            .field("has_retry", &self.retry_options.is_some())
            .finish()
    }
}

impl HttpExecutionWiring {
    pub fn new(
        provider_id: impl Into<String>,
        http_client: reqwest::Client,
        provider_context: ProviderContext,
    ) -> Self {
        Self {
            provider_id: provider_id.into(),
            http_client,
            provider_context,
            interceptors: Vec::new(),
            retry_options: None,
        }
    }

    pub fn with_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.interceptors = interceptors;
        self
    }

    pub fn with_retry_options(mut self, retry_options: Option<RetryOptions>) -> Self {
        self.retry_options = retry_options;
        self
    }

    /// Build an `HttpExecutionConfig` for common HTTP helpers (JSON/multipart/bytes GET).
    pub fn config(&self, provider_spec: Arc<dyn ProviderSpec>) -> HttpExecutionConfig {
        HttpExecutionConfig {
            provider_id: self.provider_id.clone(),
            http_client: self.http_client.clone(),
            provider_spec,
            provider_context: self.provider_context.clone(),
            interceptors: self.interceptors.clone(),
            retry_options: self.retry_options.clone(),
        }
    }

    /// Return a copy of this wiring with an updated base URL.
    pub fn with_base_url(&self, base_url: impl Into<String>) -> Self {
        let mut out = self.clone();
        out.provider_context.base_url = base_url.into();
        out
    }
}
