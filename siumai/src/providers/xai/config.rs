//! `xAI` Configuration
//!
//! This module provides configuration structures for the `xAI` provider.

use std::collections::HashMap;

use crate::types::{CommonParams, HttpConfig, WebSearchConfig};

/// `xAI` provider configuration.
///
/// This structure holds all the configuration needed to create and use
/// an `xAI` client, including authentication, API settings, and parameters.
///
/// # Example
/// ```rust
/// use siumai::providers::xai::XaiConfig;
///
/// let config = XaiConfig {
///     api_key: "your-api-key".to_string(),
///     base_url: "https://api.x.ai/v1".to_string(),
///     common_params: Default::default(),
///     http_config: Default::default(),
///     web_search_config: Default::default(),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct XaiConfig {
    /// `xAI` API key
    pub api_key: String,

    /// Base URL for the `xAI` API
    pub base_url: String,

    /// Common parameters shared across providers
    pub common_params: CommonParams,

    /// HTTP configuration
    pub http_config: HttpConfig,

    /// Web search configuration
    pub web_search_config: WebSearchConfig,
}

impl XaiConfig {
    /// Create a new `xAI` configuration with the given API key.
    ///
    /// # Arguments
    /// * `api_key` - The `xAI` API key
    ///
    /// # Returns
    /// A new configuration with default settings
    pub fn new<S: Into<String>>(api_key: S) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.x.ai/v1".to_string(),
            common_params: CommonParams::default(),
            http_config: HttpConfig::default(),
            web_search_config: WebSearchConfig::default(),
        }
    }

    /// Set the base URL for the `xAI` API.
    ///
    /// # Arguments
    /// * `url` - The base URL
    pub fn with_base_url<S: Into<String>>(mut self, url: S) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set the model name.
    ///
    /// # Arguments
    /// * `model` - The model name
    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.common_params.model = model.into();
        self
    }

    /// Set the temperature.
    ///
    /// # Arguments
    /// * `temperature` - The temperature value
    pub const fn with_temperature(mut self, temperature: f32) -> Self {
        self.common_params.temperature = Some(temperature);
        self
    }

    /// Set the maximum tokens.
    ///
    /// # Arguments
    /// * `max_tokens` - The maximum number of tokens
    pub const fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.common_params.max_tokens = Some(max_tokens);
        self
    }

    /// Enable web search functionality.
    ///
    /// # Arguments
    /// * `config` - Optional web search configuration
    pub fn with_web_search(mut self, config: Option<WebSearchConfig>) -> Self {
        self.web_search_config = config.unwrap_or_else(|| WebSearchConfig {
            enabled: true,
            ..Default::default()
        });
        self
    }

    /// Enable web search with default settings.
    pub const fn enable_web_search(mut self) -> Self {
        self.web_search_config.enabled = true;
        self
    }

    /// Get the authorization header value.
    ///
    /// # Returns
    /// The authorization header value for API requests
    pub fn auth_header(&self) -> String {
        format!("Bearer {}", self.api_key)
    }

    /// Get all HTTP headers needed for `xAI` API requests.
    ///
    /// # Returns
    /// `HashMap` of header names to values
    pub fn get_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::new();

        // Authorization header
        headers.insert("Authorization".to_string(), self.auth_header());

        // Content-Type header
        headers.insert("Content-Type".to_string(), "application/json".to_string());

        headers
    }

    /// Validate the configuration.
    ///
    /// # Returns
    /// Result indicating whether the configuration is valid
    pub fn validate(&self) -> Result<(), String> {
        if self.api_key.is_empty() {
            return Err("API key cannot be empty".to_string());
        }

        if self.base_url.is_empty() {
            return Err("Base URL cannot be empty".to_string());
        }

        if !self.base_url.starts_with("http://") && !self.base_url.starts_with("https://") {
            return Err("Base URL must start with http:// or https://".to_string());
        }

        // Validate common parameters
        if let Some(temp) = self.common_params.temperature
            && !(0.0..=2.0).contains(&temp)
        {
            return Err("Temperature must be between 0.0 and 2.0".to_string());
        }

        if let Some(top_p) = self.common_params.top_p
            && !(0.0..=1.0).contains(&top_p)
        {
            return Err("Top-p must be between 0.0 and 1.0".to_string());
        }

        Ok(())
    }
}

impl Default for XaiConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://api.x.ai/v1".to_string(),
            common_params: CommonParams::default(),
            http_config: HttpConfig::default(),
            web_search_config: WebSearchConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = XaiConfig::new("test-key");
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.base_url, "https://api.x.ai/v1");
    }

    #[test]
    fn test_config_validation() {
        let mut config = XaiConfig::new("test-key");
        assert!(config.validate().is_ok());

        config.api_key = String::new();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_headers() {
        let config = XaiConfig::new("test-key");

        let headers = config.get_headers();
        assert_eq!(
            headers.get("Authorization"),
            Some(&"Bearer test-key".to_string())
        );
        assert_eq!(
            headers.get("Content-Type"),
            Some(&"application/json".to_string())
        );
    }
}
