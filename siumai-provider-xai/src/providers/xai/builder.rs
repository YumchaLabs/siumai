//! `xAI` Builder Implementation
//!
//! Provides a builder pattern for creating `xAI` clients.

use crate::builder::{BuilderBase, ProviderCore};
use crate::error::LlmError;
use crate::retry_api::RetryOptions;

use super::client::XaiClient;
use super::config::XaiConfig;

/// `xAI` Client Builder
///
/// Retry: call `.with_retry(RetryOptions::backoff())` to enable unified retry
/// for chat operations.
#[derive(Clone)]
pub struct XaiBuilder {
    /// Core provider configuration (composition)
    pub(crate) core: ProviderCore,
    config: XaiConfig,
}

impl XaiBuilder {
    /// Create a new `xAI` builder
    pub fn new(base: BuilderBase) -> Self {
        Self {
            core: ProviderCore::new(base),
            config: XaiConfig::default(),
        }
    }

    /// Set the API key
    pub fn api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        use secrecy::SecretString;
        self.config.api_key = SecretString::from(api_key.into());
        self
    }

    /// Set the base URL
    pub fn base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        let custom = base_url.into();
        let path = custom.splitn(4, '/').nth(3).unwrap_or("");
        if path.is_empty() {
            // If no path provided, append xAI's default prefix
            self.config.base_url = format!("{}/{}", custom.trim_end_matches('/'), "v1");
        } else {
            self.config.base_url = custom;
        }
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

    // ========================================================================
    // Common Configuration Methods (delegated to ProviderCore)
    // ========================================================================

    // === HTTP Basic Configuration ===

    /// Set request timeout
    pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
        self.core = self.core.timeout(timeout);
        self
    }

    /// Set connection timeout
    pub fn connect_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.core = self.core.connect_timeout(timeout);
        self
    }

    /// Set custom HTTP client
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.core = self.core.with_http_client(client);
        self
    }

    // === HTTP Advanced Configuration ===

    /// Control whether to disable compression for streaming (SSE) requests.
    pub fn http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.core = self.core.http_stream_disable_compression(disable);
        self
    }

    /// Add a custom HTTP interceptor (builder collects and installs them on build).
    pub fn with_http_interceptor(
        mut self,
        interceptor: std::sync::Arc<dyn crate::execution::http::interceptor::HttpInterceptor>,
    ) -> Self {
        self.core = self.core.with_http_interceptor(interceptor);
        self
    }

    /// Enable a built-in logging interceptor for HTTP debugging (no sensitive data).
    pub fn http_debug(mut self, enabled: bool) -> Self {
        self.core = self.core.http_debug(enabled);
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

    /// Set unified retry options for chat operations
    pub fn with_retry(mut self, options: RetryOptions) -> Self {
        self.core = self.core.with_retry(options);
        self
    }

    // ========================================================================
    // Provider-Specific Configuration
    // ========================================================================

    /// Build the `xAI` client
    pub async fn build(self) -> Result<XaiClient, LlmError> {
        // Step 4: Build HTTP client from core
        let http_client = self.core.build_http_client()?;

        // Delegate to build_with_client
        self.build_with_client(http_client).await
    }

    /// Build the `xAI` client with a custom HTTP client
    pub async fn build_with_client(
        self,
        http_client: reqwest::Client,
    ) -> Result<XaiClient, LlmError> {
        // Step 1: Get API key (priority: parameter/config > environment variable)
        use secrecy::{ExposeSecret, SecretString};
        let mut config = self.config;
        if config.api_key.expose_secret().is_empty()
            && let Ok(k) = std::env::var("XAI_API_KEY")
        {
            config.api_key = SecretString::from(k);
        }

        // Step 2: Get base URL (from parameter or default in config)
        // Note: Base URL is already set in XaiConfig

        // Note: Tracing initialization has been moved to siumai-extras.
        // Users should initialize tracing manually using siumai_extras::telemetry
        // or tracing_subscriber directly before creating the client.

        // Step 3: Build configuration
        config.validate()?;
        if config.common_params.model.is_empty() {
            config.common_params.model = crate::providers::xai::models::popular::LATEST.to_string();
        }

        // Apply HTTP config from ProviderCore (headers/proxy/stream_disable_compression/etc.).
        // This ensures `http_stream_disable_compression` affects streaming requests.
        config.http_config = self.core.http_config.clone();

        let model_id = config.common_params.model.clone();

        // Step 5: Create client instance
        let mut client = XaiClient::with_http_client(config, http_client).await?;

        // Step 6: Apply tracing and retry configuration from core
        if let Some(ref tracing_config) = self.core.tracing_config {
            client.set_tracing_config(Some(tracing_config.clone()));
        }
        if let Some(ref retry_options) = self.core.retry_options {
            client.set_retry_options(Some(retry_options.clone()));
        }

        // Step 7: Install HTTP interceptors
        let interceptors = self.core.get_http_interceptors();
        if !interceptors.is_empty() {
            client = client.with_http_interceptors(interceptors);
        }

        // Step 8: Install automatic middlewares
        let middlewares = self.core.get_auto_middlewares("xai", &model_id);
        if !middlewares.is_empty() {
            client = client.with_model_middlewares(middlewares);
        }

        Ok(client)
    }
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        use secrecy::ExposeSecret;
        let builder = XaiBuilder::new(crate::builder::BuilderBase::default());
        assert_eq!(builder.config.base_url, "https://api.x.ai/v1");
        assert!(builder.config.api_key.expose_secret().is_empty());
    }

    #[test]
    fn test_builder_configuration() {
        use secrecy::ExposeSecret;
        let builder = XaiBuilder::new(crate::builder::BuilderBase::default())
            .api_key("test-key")
            .model("grok-3-latest")
            .temperature(0.7)
            .max_tokens(1000);

        assert_eq!(builder.config.api_key.expose_secret(), "test-key");
        assert_eq!(builder.config.common_params.model, "grok-3-latest");
        assert_eq!(builder.config.common_params.temperature, Some(0.7));
        assert_eq!(builder.config.common_params.max_tokens, Some(1000));
    }

    #[tokio::test]
    async fn test_builder_validation() {
        // Temporarily remove API key from environment
        let original_key = std::env::var("XAI_API_KEY").ok();
        unsafe {
            std::env::remove_var("XAI_API_KEY");
        }

        let builder = XaiBuilder::new(crate::builder::BuilderBase::default());

        // Should fail without API key
        let result = builder.build().await;

        // Restore original key if it existed
        if let Some(key) = original_key.clone() {
            unsafe {
                std::env::set_var("XAI_API_KEY", key);
            }
        }

        assert!(result.is_err());

        // Should succeed with API key
        let builder = XaiBuilder::new(crate::builder::BuilderBase::default()).api_key("test-key");
        let result = builder.build().await;
        assert!(result.is_ok());

        // Restore original key if it existed
        if let Some(key) = original_key {
            unsafe {
                std::env::set_var("XAI_API_KEY", key);
            }
        }
    }
}
