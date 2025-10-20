//! `xAI` Builder Implementation
//!
//! Provides a builder pattern for creating `xAI` clients.

use crate::LlmBuilder;
use crate::error::LlmError;
use crate::retry_api::RetryOptions;
use crate::types::{CommonParams, HttpConfig, WebSearchConfig};
use crate::utils::http_interceptor::{HttpInterceptor, LoggingInterceptor};
use std::sync::Arc;

use super::client::XaiClient;
use super::config::XaiConfig;

/// `xAI` Client Builder
///
/// Retry: call `.with_retry(RetryOptions::backoff())` to enable unified retry
/// for chat operations.
#[derive(Clone)]
pub struct XaiBuilder {
    config: XaiConfig,
    tracing_config: Option<crate::tracing::TracingConfig>,
    retry_options: Option<RetryOptions>,
    /// Optional HTTP interceptors applied to chat requests
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Enable lightweight HTTP debug logging interceptor
    http_debug: bool,
}

impl XaiBuilder {
    /// Create a new `xAI` builder
    pub fn new() -> Self {
        Self {
            config: XaiConfig::default(),
            tracing_config: None,
            retry_options: None,
            http_interceptors: Vec::new(),
            http_debug: false,
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

    /// Set the temperature
    pub const fn temperature(mut self, temperature: f32) -> Self {
        self.config.common_params.temperature = Some(temperature);
        self
    }

    /// Set the maximum tokens
    pub const fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.config.common_params.max_tokens = Some(max_tokens);
        self
    }

    /// Set the top-p value
    pub const fn top_p(mut self, top_p: f32) -> Self {
        self.config.common_params.top_p = Some(top_p);
        self
    }

    /// Set stop sequences
    pub fn stop_sequences(mut self, stop: Vec<String>) -> Self {
        self.config.common_params.stop_sequences = Some(stop);
        self
    }

    /// Set the seed
    pub const fn seed(mut self, seed: u64) -> Self {
        self.config.common_params.seed = Some(seed);
        self
    }

    /// Set common parameters
    pub fn common_params(mut self, params: CommonParams) -> Self {
        self.config.common_params = params;
        self
    }

    /// Set HTTP configuration
    pub fn http_config(mut self, config: HttpConfig) -> Self {
        self.config.http_config = config;
        self
    }

    /// Control whether to disable compression for streaming (SSE) requests.
    pub fn http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.config.http_config.stream_disable_compression = disable;
        self
    }

    /// Set web search configuration
    pub fn web_search_config(mut self, config: WebSearchConfig) -> Self {
        self.config.web_search_config = config;
        self
    }

    /// Enable web search with default settings
    pub const fn enable_web_search(mut self) -> Self {
        self.config.web_search_config.enabled = true;
        self
    }

    /// Set the entire configuration
    pub fn config(mut self, config: XaiConfig) -> Self {
        self.config = config;
        self
    }

    // === Tracing Configuration ===

    /// Set custom tracing configuration
    pub fn tracing(mut self, config: crate::tracing::TracingConfig) -> Self {
        self.tracing_config = Some(config);
        self
    }

    /// Enable debug tracing (development-friendly configuration)
    pub fn debug_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::development())
    }

    /// Enable minimal tracing (info level, LLM only)
    pub fn minimal_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::minimal())
    }

    /// Enable production-ready JSON tracing
    pub fn json_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::json_production())
    }

    /// Enable pretty-printed formatting for JSON bodies and headers in tracing
    pub fn pretty_json(mut self, pretty: bool) -> Self {
        let config = self
            .tracing_config
            .take()
            .unwrap_or_else(crate::tracing::TracingConfig::development)
            .with_pretty_json(pretty);
        self.tracing_config = Some(config);
        self
    }

    /// Control masking of sensitive values (API keys, tokens) in tracing logs
    pub fn mask_sensitive_values(mut self, mask: bool) -> Self {
        let config = self
            .tracing_config
            .take()
            .unwrap_or_else(crate::tracing::TracingConfig::development)
            .with_mask_sensitive_values(mask);
        self.tracing_config = Some(config);
        self
    }

    /// Set unified retry options for chat operations
    pub fn with_retry(mut self, options: RetryOptions) -> Self {
        self.retry_options = Some(options);
        self
    }

    /// Add a custom HTTP interceptor (builder collects and installs them on build).
    pub fn with_http_interceptor(mut self, interceptor: Arc<dyn HttpInterceptor>) -> Self {
        self.http_interceptors.push(interceptor);
        self
    }

    /// Enable a built-in logging interceptor for HTTP debugging (no sensitive data).
    pub fn http_debug(mut self, enabled: bool) -> Self {
        self.http_debug = enabled;
        self
    }

    /// Build the `xAI` client
    pub async fn build(self) -> Result<XaiClient, LlmError> {
        // Validate configuration
        self.config
            .validate()
            .map_err(|e| LlmError::InvalidInput(format!("Invalid xAI configuration: {e}")))?;

        // Note: Tracing initialization has been moved to siumai-extras.
        // Users should initialize tracing manually using siumai_extras::telemetry
        // or tracing_subscriber directly before creating the client.

        // Set default model if not specified
        let mut config = self.config;
        if config.common_params.model.is_empty() {
            config.common_params.model = crate::providers::xai::models::popular::LATEST.to_string();
        }

        let mut client = XaiClient::new(config).await?;
        client.set_tracing_config(self.tracing_config);
        client.set_retry_options(self.retry_options.clone());

        // Install interceptors
        let mut interceptors = self.http_interceptors;
        if self.http_debug {
            interceptors.push(Arc::new(LoggingInterceptor));
        }
        if !interceptors.is_empty() {
            client = client.with_http_interceptors(interceptors);
        }

        Ok(client)
    }

    /// Build the `xAI` client with a custom HTTP client
    pub async fn build_with_client(
        self,
        http_client: reqwest::Client,
    ) -> Result<XaiClient, LlmError> {
        // Validate configuration
        self.config
            .validate()
            .map_err(|e| LlmError::InvalidInput(format!("Invalid xAI configuration: {e}")))?;

        // Note: Tracing initialization has been moved to siumai-extras.
        // Users should initialize tracing manually using siumai_extras::telemetry
        // or tracing_subscriber directly before creating the client.

        // Set default model if not specified
        let mut config = self.config;
        if config.common_params.model.is_empty() {
            config.common_params.model = crate::providers::xai::models::popular::LATEST.to_string();
        }

        let mut client = XaiClient::with_http_client(config, http_client).await?;
        client.set_tracing_config(self.tracing_config);
        client.set_retry_options(self.retry_options.clone());

        // Install interceptors
        let mut interceptors = self.http_interceptors;
        if self.http_debug {
            interceptors.push(Arc::new(LoggingInterceptor));
        }
        if !interceptors.is_empty() {
            client = client.with_http_interceptors(interceptors);
        }

        Ok(client)
    }
}

/// Wrapper for xAI builder that supports HTTP client inheritance
#[cfg(feature = "xai")]
pub struct XaiBuilderWrapper {
    pub(crate) base: LlmBuilder,
    xai_builder: crate::providers::xai::XaiBuilder,
}

#[cfg(feature = "xai")]
impl XaiBuilderWrapper {
    pub fn new(base: LlmBuilder) -> Self {
        // Inherit interceptors/debug from unified builder
        let mut inner = crate::providers::xai::XaiBuilder::new();
        for it in &base.http_interceptors {
            inner = inner.with_http_interceptor(it.clone());
        }
        inner = inner.http_debug(base.http_debug);
        Self {
            base,
            xai_builder: inner,
        }
    }

    /// Set the API key
    pub fn api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.xai_builder = self.xai_builder.api_key(api_key);
        self
    }

    /// Set the base URL
    pub fn base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.xai_builder = self.xai_builder.base_url(base_url);
        self
    }

    /// Set the model
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.xai_builder = self.xai_builder.model(model);
        self
    }

    /// Set the temperature
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.xai_builder = self.xai_builder.temperature(temperature);
        self
    }

    /// Set the maximum number of tokens
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.xai_builder = self.xai_builder.max_tokens(max_tokens);
        self
    }

    /// Set the top-p value
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.xai_builder = self.xai_builder.top_p(top_p);
        self
    }

    /// Set the stop sequences
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.xai_builder = self.xai_builder.stop_sequences(sequences);
        self
    }

    /// Set the random seed
    pub fn seed(mut self, seed: u64) -> Self {
        self.xai_builder = self.xai_builder.seed(seed);
        self
    }

    /// Enable tracing
    pub fn tracing(mut self, config: crate::tracing::TracingConfig) -> Self {
        self.xai_builder = self.xai_builder.tracing(config);
        self
    }

    /// Enable debug tracing
    pub fn debug_tracing(mut self) -> Self {
        self.xai_builder = self.xai_builder.debug_tracing();
        self
    }

    /// Enable minimal tracing
    pub fn minimal_tracing(mut self) -> Self {
        self.xai_builder = self.xai_builder.minimal_tracing();
        self
    }

    /// Enable JSON tracing
    pub fn json_tracing(mut self) -> Self {
        self.xai_builder = self.xai_builder.json_tracing();
        self
    }

    /// Build the xAI client
    pub async fn build(self) -> Result<crate::providers::xai::XaiClient, LlmError> {
        // Build HTTP client from base configuration
        let http_client = self.base.build_http_client()?;

        // Use the build_with_client method to pass the custom HTTP client
        self.xai_builder.build_with_client(http_client).await
    }
}

impl Default for XaiBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let builder = XaiBuilder::new();
        assert_eq!(builder.config.base_url, "https://api.x.ai/v1");
        assert!(builder.config.api_key.is_empty());
    }

    #[test]
    fn test_builder_configuration() {
        let builder = XaiBuilder::new()
            .api_key("test-key")
            .model("grok-3-latest")
            .temperature(0.7)
            .max_tokens(1000);

        assert_eq!(builder.config.api_key, "test-key");
        assert_eq!(builder.config.common_params.model, "grok-3-latest");
        assert_eq!(builder.config.common_params.temperature, Some(0.7));
        assert_eq!(builder.config.common_params.max_tokens, Some(1000));
    }

    #[tokio::test]
    async fn test_builder_validation() {
        let builder = XaiBuilder::new();

        // Should fail without API key
        let result = builder.build().await;
        assert!(result.is_err());

        // Should succeed with API key
        let builder = XaiBuilder::new().api_key("test-key");
        let result = builder.build().await;
        assert!(result.is_ok());
    }
}
