//! `Groq` Builder Implementation
//!
//! Thin wrapper around the OpenAI-compatible vendor builder:
//! `LlmBuilder::new().openai().compatible("groq")`.

use crate::builder::BuilderBase;
use crate::error::LlmError;
use crate::retry_api::RetryOptions;
use siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleBuilder;
use std::collections::HashMap;
use std::sync::Arc;

/// `Groq` client builder.
///
/// This builder intentionally reuses the OpenAI-compatible vendor implementation.
#[derive(Clone)]
pub struct GroqBuilder {
    inner: OpenAiCompatibleBuilder,
}

impl GroqBuilder {
    pub fn new(base: BuilderBase) -> Self {
        Self {
            inner: OpenAiCompatibleBuilder::new(base, "groq"),
        }
    }

    pub fn api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.inner = self.inner.api_key(api_key);
        self
    }

    pub fn base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        let custom = base_url.into();
        let path = custom.splitn(4, '/').nth(3).unwrap_or("");
        if path.is_empty() {
            self.inner = self
                .inner
                .base_url(format!("{}/openai/v1", custom.trim_end_matches('/')));
        } else {
            self.inner = self.inner.base_url(custom);
        }
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

    pub async fn build(self) -> Result<super::GroqClient, LlmError> {
        let inner = self.inner.build().await?;
        Ok(super::GroqClient::new(inner))
    }
}
