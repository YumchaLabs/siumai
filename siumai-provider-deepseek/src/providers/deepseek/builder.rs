//! `DeepSeek` Builder Implementation
//!
//! Thin wrapper around the OpenAI-compatible vendor builder:
//! `LlmBuilder::new().openai().compatible("deepseek")`.

use crate::builder::BuilderBase;
use crate::error::LlmError;
use crate::retry_api::RetryOptions;
use siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleBuilder;
use std::collections::HashMap;
use std::sync::Arc;

/// `DeepSeek` client builder.
///
/// This builder intentionally reuses the OpenAI-compatible vendor implementation.
#[derive(Clone)]
pub struct DeepSeekBuilder {
    inner: OpenAiCompatibleBuilder,
}

impl DeepSeekBuilder {
    pub fn new(base: BuilderBase) -> Self {
        Self {
            inner: OpenAiCompatibleBuilder::new(base, "deepseek"),
        }
    }

    pub fn api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.inner = self.inner.api_key(api_key);
        self
    }

    pub fn base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.inner = self.inner.base_url(base_url);
        self
    }

    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.inner = self.inner.model(model);
        self
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.inner = self.inner.temperature(temperature);
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.inner = self.inner.max_tokens(max_tokens);
        self
    }

    pub fn top_p(mut self, top_p: f64) -> Self {
        self.inner = self.inner.top_p(top_p);
        self
    }

    pub fn stop_sequences(mut self, stop: Vec<String>) -> Self {
        self.inner = self.inner.stop(stop);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.inner = self.inner.seed(seed);
        self
    }

    /// Enable DeepSeek reasoning output (mapped to `enable_reasoning` in request body).
    pub fn reasoning(mut self, enable: bool) -> Self {
        self.inner = self.inner.reasoning(enable);
        self
    }

    /// Set reasoning budget (DeepSeek stores this as a provider option; also enables reasoning).
    pub fn reasoning_budget(mut self, budget: i32) -> Self {
        self.inner = self.inner.reasoning_budget(budget);
        self
    }

    pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
        self.inner = self.inner.timeout(timeout);
        self
    }

    pub fn connect_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.inner = self.inner.connect_timeout(timeout);
        self
    }

    pub fn http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.inner = self.inner.http_stream_disable_compression(disable);
        self
    }

    pub fn with_http_config(mut self, config: crate::types::HttpConfig) -> Self {
        self.inner = self.inner.with_http_config(config);
        self
    }

    pub fn custom_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.inner = self.inner.custom_headers(headers);
        self
    }

    pub fn header<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.inner = self.inner.header(key, value);
        self
    }

    pub fn with_retry(mut self, options: RetryOptions) -> Self {
        self.inner = self.inner.with_retry(options);
        self
    }

    pub fn with_http_interceptor(
        mut self,
        interceptor: Arc<dyn crate::execution::http::interceptor::HttpInterceptor>,
    ) -> Self {
        self.inner = self.inner.with_http_interceptor(interceptor);
        self
    }

    pub fn http_debug(mut self, enabled: bool) -> Self {
        self.inner = self.inner.http_debug(enabled);
        self
    }

    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.inner = self.inner.with_http_client(client);
        self
    }

    pub fn with_http_transport(
        mut self,
        transport: Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.inner = self.inner.with_http_transport(transport);
        self
    }

    /// Alias for `with_http_transport(...)` (Vercel-aligned: `fetch`).
    pub fn fetch(
        self,
        transport: Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.with_http_transport(transport)
    }

    pub async fn build(self) -> Result<super::DeepSeekClient, LlmError> {
        self.inner.build().await
    }
}
