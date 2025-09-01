//! `Groq` Builder Implementation
//!
//! Builder pattern implementation for creating Groq clients.

use std::time::Duration;

use crate::LlmBuilder;
use crate::error::LlmError;
use crate::types::HttpConfig;

use super::client::GroqClient;
use super::config::GroqConfig;

/// `Groq` client builder
#[derive(Debug, Clone)]
pub struct GroqBuilder {
    config: GroqConfig,
    tracing_config: Option<crate::tracing::TracingConfig>,
}

impl GroqBuilder {
    /// Create a new `Groq` builder
    pub fn new() -> Self {
        Self {
            config: GroqConfig::default(),
            tracing_config: None,
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
        self.config.http_config.timeout = Some(timeout);
        self
    }

    /// Set connection timeout
    pub fn connect_timeout(mut self, connect_timeout: Duration) -> Self {
        self.config.http_config.connect_timeout = Some(connect_timeout);
        self
    }

    /// Add a custom header
    pub fn header<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.config
            .http_config
            .headers
            .insert(key.into(), value.into());
        self
    }

    /// Set proxy URL
    pub fn proxy<S: Into<String>>(mut self, proxy: S) -> Self {
        self.config.http_config.proxy = Some(proxy.into());
        self
    }

    /// Set user agent
    pub fn user_agent<S: Into<String>>(mut self, user_agent: S) -> Self {
        self.config.http_config.user_agent = Some(user_agent.into());
        self
    }

    /// Add a built-in tool
    pub fn tool(mut self, tool: crate::types::Tool) -> Self {
        self.config.built_in_tools.push(tool);
        self
    }

    /// Add multiple built-in tools
    pub fn tools(mut self, tools: Vec<crate::types::Tool>) -> Self {
        self.config.built_in_tools.extend(tools);
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

    /// Build the `Groq` client
    pub async fn build(mut self) -> Result<GroqClient, LlmError> {
        // Try to get API key from environment if not set
        if self.config.api_key.is_empty()
            && let Ok(api_key) = std::env::var("GROQ_API_KEY")
        {
            self.config.api_key = api_key;
        }

        // Validate configuration
        self.config.validate()?;

        // Initialize tracing if configured
        let _tracing_guard = if let Some(ref tracing_config) = self.tracing_config {
            Some(crate::tracing::init_tracing(tracing_config.clone())?)
        } else {
            None
        };

        // Create HTTP client
        let mut client_builder = reqwest::Client::builder();

        // Set timeouts
        if let Some(timeout) = self.config.http_config.timeout {
            client_builder = client_builder.timeout(timeout);
        }
        if let Some(connect_timeout) = self.config.http_config.connect_timeout {
            client_builder = client_builder.connect_timeout(connect_timeout);
        }

        // Set proxy
        if let Some(proxy_url) = &self.config.http_config.proxy {
            let proxy = reqwest::Proxy::all(proxy_url)
                .map_err(|e| LlmError::ConfigurationError(format!("Invalid proxy URL: {e}")))?;
            client_builder = client_builder.proxy(proxy);
        }

        // Set user agent
        if let Some(user_agent) = &self.config.http_config.user_agent {
            client_builder = client_builder.user_agent(user_agent);
        }

        let http_client = client_builder.build().map_err(|e| {
            LlmError::ConfigurationError(format!("Failed to create HTTP client: {e}"))
        })?;

        let mut client = GroqClient::new(self.config, http_client);
        client.set_tracing_guard(_tracing_guard);
        client.set_tracing_config(self.tracing_config);

        Ok(client)
    }

    /// Get the current configuration (for inspection)
    pub fn config(&self) -> &GroqConfig {
        &self.config
    }

    /// Set the entire HTTP configuration
    pub fn http_config(mut self, http_config: HttpConfig) -> Self {
        self.config.http_config = http_config;
        self
    }

    /// Set the entire configuration
    pub fn with_config(mut self, config: GroqConfig) -> Self {
        self.config = config;
        self
    }
}

impl Default for GroqBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Wrapper for Groq builder that supports HTTP client inheritance
#[cfg(feature = "groq")]
pub struct GroqBuilderWrapper {
    pub(crate) base: LlmBuilder,
    groq_builder: crate::providers::groq::GroqBuilder,
}

#[cfg(feature = "groq")]
impl GroqBuilderWrapper {
    pub fn new(base: LlmBuilder) -> Self {
        Self {
            base,
            groq_builder: crate::providers::groq::GroqBuilder::new(),
        }
    }

    /// Set the API key
    pub fn api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.groq_builder = self.groq_builder.api_key(api_key);
        self
    }

    /// Set the base URL
    pub fn base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.groq_builder = self.groq_builder.base_url(base_url);
        self
    }

    /// Set the model
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.groq_builder = self.groq_builder.model(model);
        self
    }

    /// Set the temperature
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.groq_builder = self.groq_builder.temperature(temperature);
        self
    }

    /// Set the maximum number of tokens
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.groq_builder = self.groq_builder.max_tokens(max_tokens);
        self
    }

    /// Set the top-p value
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.groq_builder = self.groq_builder.top_p(top_p);
        self
    }

    /// Set the stop sequences
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.groq_builder = self.groq_builder.stop_sequences(sequences);
        self
    }

    /// Set the random seed
    pub fn seed(mut self, seed: u64) -> Self {
        self.groq_builder = self.groq_builder.seed(seed);
        self
    }

    /// Add a built-in tool
    pub fn tool(mut self, tool: crate::types::Tool) -> Self {
        self.groq_builder = self.groq_builder.tool(tool);
        self
    }

    /// Add multiple built-in tools
    pub fn tools(mut self, tools: Vec<crate::types::Tool>) -> Self {
        self.groq_builder = self.groq_builder.tools(tools);
        self
    }

    /// Enable tracing
    pub fn tracing(mut self, config: crate::tracing::TracingConfig) -> Self {
        self.groq_builder = self.groq_builder.tracing(config);
        self
    }

    /// Enable debug tracing
    pub fn debug_tracing(mut self) -> Self {
        self.groq_builder = self.groq_builder.debug_tracing();
        self
    }

    /// Enable minimal tracing
    pub fn minimal_tracing(mut self) -> Self {
        self.groq_builder = self.groq_builder.minimal_tracing();
        self
    }

    /// Enable JSON tracing
    pub fn json_tracing(mut self) -> Self {
        self.groq_builder = self.groq_builder.json_tracing();
        self
    }

    /// Build the Groq client
    pub async fn build(self) -> Result<crate::providers::groq::GroqClient, LlmError> {
        // Apply all HTTP configuration from base LlmBuilder to Groq builder
        let mut groq_builder = self.groq_builder;

        // Apply timeout settings
        if let Some(timeout) = self.base.timeout {
            groq_builder = groq_builder.timeout(timeout);
        }
        if let Some(connect_timeout) = self.base.connect_timeout {
            groq_builder = groq_builder.connect_timeout(connect_timeout);
        }

        // Apply proxy settings
        if let Some(proxy) = &self.base.proxy {
            groq_builder = groq_builder.proxy(proxy);
        }

        // Apply user agent
        if let Some(user_agent) = &self.base.user_agent {
            groq_builder = groq_builder.user_agent(user_agent);
        }

        // Apply default headers
        for (key, value) in &self.base.default_headers {
            groq_builder = groq_builder.header(key, value);
        }

        groq_builder.build().await
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_groq_builder() {
        let builder = GroqBuilder::new()
            .api_key("test-key")
            .model("llama-3.3-70b-versatile")
            .temperature(0.7)
            .max_tokens(1000)
            .timeout(Duration::from_secs(30));

        let config = builder.config();
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.common_params.model, "llama-3.3-70b-versatile");
        assert_eq!(config.common_params.temperature, Some(0.7));
        assert_eq!(config.common_params.max_tokens, Some(1000));
        assert_eq!(config.http_config.timeout, Some(Duration::from_secs(30)));
    }

    #[test]
    fn test_groq_builder_default() {
        let builder = GroqBuilder::default();
        let config = builder.config();
        assert_eq!(config.base_url, GroqConfig::DEFAULT_BASE_URL);
        assert_eq!(config.common_params.model, GroqConfig::default_model());
    }

    #[test]
    fn test_groq_builder_validation() {
        let builder = GroqBuilder::new()
            .api_key("") // Empty API key should fail validation
            .model(crate::providers::groq::models::popular::FLAGSHIP);

        // This should fail during build due to empty API key
        assert!(builder.config.validate().is_err());
    }

    #[test]
    fn test_groq_builder_tools() {
        use crate::types::{Tool, ToolFunction};

        let tool = Tool {
            r#type: "function".to_string(),
            function: ToolFunction {
                name: "test_function".to_string(),
                description: "A test function".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            },
        };

        let builder = GroqBuilder::new().api_key("test-key").tool(tool.clone());

        let config = builder.config();
        assert_eq!(config.built_in_tools.len(), 1);
        assert_eq!(config.built_in_tools[0].function.name, "test_function");
    }

    #[test]
    fn test_groq_builder_headers() {
        let builder = GroqBuilder::new()
            .header("X-Custom-Header", "custom-value")
            .header("X-Another-Header", "another-value");

        let config = builder.config();
        assert_eq!(
            config.http_config.headers.get("X-Custom-Header"),
            Some(&"custom-value".to_string())
        );
        assert_eq!(
            config.http_config.headers.get("X-Another-Header"),
            Some(&"another-value".to_string())
        );
    }
}
