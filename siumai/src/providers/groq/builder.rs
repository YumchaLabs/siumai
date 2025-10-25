//! `Groq` Builder Implementation
//!
//! Builder pattern implementation for creating Groq clients.

use std::time::Duration;

use crate::LlmBuilder;
use crate::core::builder_core::ProviderCore;
use crate::error::LlmError;
use crate::retry_api::RetryOptions;

use super::client::GroqClient;
use super::config::GroqConfig;

/// `Groq` client builder
///
/// Retry: call `.with_retry(RetryOptions::backoff())` to enable unified retry
/// for chat operations.
#[derive(Clone)]
pub struct GroqBuilder {
    /// Core provider configuration (composition)
    pub(crate) core: ProviderCore,
    pub(crate) config: GroqConfig,
}

impl GroqBuilder {
    /// Create a new `Groq` builder
    pub fn new(base: LlmBuilder) -> Self {
        Self {
            core: ProviderCore::new(base),
            config: GroqConfig::default(),
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
        self.config.base_url = base_url.into();
        self
    }

    /// Set the model
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.config.common_params.model = model.into();
        self
    }

    /// Set the temperature
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.config.common_params.temperature = Some(temperature);
        self
    }

    /// Set the maximum tokens
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.config.common_params.max_tokens = Some(max_tokens);
        self
    }

    /// Set the top_p parameter
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.config.common_params.top_p = Some(top_p);
        self
    }

    /// Set stop sequences
    pub fn stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.config.common_params.stop_sequences = Some(stop_sequences);
        self
    }

    /// Set the seed
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.common_params.seed = Some(seed);
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
        self.config.http_config.stream_disable_compression = disable;
        self
    }

    // === Tracing Configuration ===

    /// Set custom tracing configuration
    pub fn tracing(mut self, config: crate::tracing::TracingConfig) -> Self {
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

    /// Set unified retry options for chat operations
    pub fn with_retry(mut self, options: RetryOptions) -> Self {
        self.core = self.core.with_retry(options);
        self
    }

    /// Add a custom HTTP interceptor (builder collects and installs them on build).
    pub fn with_http_interceptor(
        mut self,
        interceptor: std::sync::Arc<dyn crate::utils::http_interceptor::HttpInterceptor>,
    ) -> Self {
        self.core = self.core.with_http_interceptor(interceptor);
        self
    }

    /// Enable a built-in logging interceptor for HTTP debugging (no sensitive data).
    pub fn http_debug(mut self, enabled: bool) -> Self {
        self.core = self.core.http_debug(enabled);
        self
    }

    /// Set custom HTTP client
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.core = self.core.with_http_client(client);
        self
    }

    /// Build the `Groq` client
    pub async fn build(mut self) -> Result<GroqClient, LlmError> {
        use secrecy::{ExposeSecret, SecretString};
        // Step 1: Get API key (priority: parameter > environment variable)
        if self.config.api_key.expose_secret().is_empty()
            && let Ok(api_key) = std::env::var("GROQ_API_KEY")
        {
            self.config.api_key = SecretString::from(api_key);
        }

        // Step 2: Get base URL (from parameter or default in config)
        // Note: Base URL is already set in GroqConfig

        // Note: Tracing initialization has been moved to siumai-extras.
        // Users should initialize tracing manually using siumai_extras::telemetry
        // or tracing_subscriber directly before creating the client.

        // Step 3: Build configuration
        self.config.validate()?;
        let model_id = self.config.common_params.model.clone();

        // Step 4: Build HTTP client from core
        let http_client = self.core.build_http_client()?;

        // Step 5: Create client instance
        let mut client = GroqClient::new(self.config, http_client);

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
        let middlewares = self.core.get_auto_middlewares("groq", &model_id);
        if !middlewares.is_empty() {
            client = client.with_model_middlewares(middlewares);
        }

        Ok(client)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_groq_builder() {
        use secrecy::ExposeSecret;
        let builder = GroqBuilder::new(LlmBuilder::new())
            .api_key("test-key")
            .model("llama-3.3-70b-versatile")
            .temperature(0.7)
            .max_tokens(1000)
            .timeout(Duration::from_secs(30));

        // Access config field directly
        assert_eq!(builder.config.api_key.expose_secret(), "test-key");
        assert_eq!(
            builder.config.common_params.model,
            "llama-3.3-70b-versatile"
        );
        assert_eq!(builder.config.common_params.temperature, Some(0.7));
        assert_eq!(builder.config.common_params.max_tokens, Some(1000));
        assert_eq!(
            builder.core.http_config.timeout,
            Some(Duration::from_secs(30))
        );
    }

    #[test]
    fn test_groq_builder_validation() {
        let builder = GroqBuilder::new(LlmBuilder::new())
            .api_key("") // Empty API key should fail validation
            .model(crate::providers::groq::models::popular::FLAGSHIP);

        // This should fail during build due to empty API key
        assert!(builder.config.validate().is_err());
    }
}
