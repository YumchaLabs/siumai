//! MiniMaxi Builder Implementation
//!
//! Builder pattern implementation for creating MiniMaxi clients.

use std::time::Duration;

use crate::LlmBuilder;
use crate::core::builder_core::ProviderCore;
use crate::error::LlmError;
use crate::retry_api::RetryOptions;

use super::client::MinimaxiClient;
use super::config::MinimaxiConfig;

/// MiniMaxi client builder
///
/// Retry: call `.with_retry(RetryOptions::backoff())` to enable unified retry
/// for chat operations.
#[derive(Clone)]
pub struct MinimaxiBuilder {
    /// Core provider configuration (composition)
    pub(crate) core: ProviderCore,
    pub(crate) config: MinimaxiConfig,
}

impl MinimaxiBuilder {
    /// Create a new MiniMaxi builder
    pub fn new(base: LlmBuilder) -> Self {
        Self {
            core: ProviderCore::new(base),
            config: MinimaxiConfig::default(),
        }
    }

    /// Set the API key
    pub fn api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.config.api_key = api_key.into();
        self
    }

    /// Set the base URL
    pub fn base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.config.base_url = base_url.into();
        self
    }

    /// Set the model
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.config.common_params.model = model.into();
        self
    }

    /// Set request timeout
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.core = self.core.timeout(timeout);
        self
    }

    /// Set connection timeout
    pub fn connect_timeout(mut self, connect_timeout: Duration) -> Self {
        self.core = self.core.connect_timeout(connect_timeout);
        self
    }

    /// Control whether to disable compression for streaming (SSE) requests.
    pub fn http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.core = self.core.http_stream_disable_compression(disable);
        self
    }

    // === Tracing Configuration ===

    /// Set custom tracing configuration
    pub fn tracing(mut self, config: crate::observability::tracing::TracingConfig) -> Self {
        self.core = self.core.tracing(config);
        self
    }

    /// Enable debug tracing (development-friendly configuration)
    pub fn debug_tracing(mut self) -> Self {
        self.core = self.core.debug_tracing();
        self
    }

    /// Enable minimal tracing (info level, LLM only)
    pub fn minimal_tracing(mut self) -> Self {
        self.core = self.core.minimal_tracing();
        self
    }

    /// Enable production-ready JSON tracing
    pub fn json_tracing(mut self) -> Self {
        self.core = self.core.json_tracing();
        self
    }

    /// Enable pretty-printed formatting for JSON bodies and headers in tracing
    pub fn pretty_json(mut self, pretty: bool) -> Self {
        self.core = self.core.pretty_json(pretty);
        self
    }

    /// Control masking of sensitive values (API keys, tokens) in tracing logs
    pub fn mask_sensitive_values(mut self, mask: bool) -> Self {
        self.core = self.core.mask_sensitive_values(mask);
        self
    }

    // === Retry Configuration ===

    /// Enable unified retry for chat operations
    pub fn with_retry(mut self, retry_options: RetryOptions) -> Self {
        self.core = self.core.with_retry(retry_options);
        self
    }

    // === HTTP Interceptors ===

    /// Add an HTTP interceptor
    pub fn with_http_interceptor(
        mut self,
        interceptor: std::sync::Arc<dyn crate::execution::http::interceptor::HttpInterceptor>,
    ) -> Self {
        self.core = self.core.with_http_interceptor(interceptor);
        self
    }

    /// Build the MiniMaxi client
    pub async fn build(self) -> Result<MinimaxiClient, LlmError> {
        // Validate configuration
        self.config.validate()?;

        // Get API key from config or environment
        let api_key = if self.config.api_key.is_empty() {
            std::env::var("MINIMAXI_API_KEY").map_err(|_| {
                LlmError::ConfigurationError(
                    "MiniMaxi API key not provided. Set MINIMAXI_API_KEY environment variable or use .api_key()".to_string(),
                )
            })?
        } else {
            self.config.api_key.clone()
        };

        // Update config with API key
        let mut config = self.config;
        config.api_key = api_key;

        // Build HTTP client from core
        let http_client = self.core.build_http_client()?;

        // Create client
        let mut client = MinimaxiClient::new(config, http_client);

        // Apply tracing configuration
        if let Some(tracing_config) = self.core.tracing_config {
            client = client.with_tracing(tracing_config);
        }

        // Apply retry options
        if let Some(retry_options) = self.core.retry_options {
            client = client.with_retry(retry_options);
        }

        // Apply HTTP interceptors
        if !self.core.http_interceptors.is_empty() {
            client = client.with_interceptors(self.core.http_interceptors);
        }

        Ok(client)
    }
}
