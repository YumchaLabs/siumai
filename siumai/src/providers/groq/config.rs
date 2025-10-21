//! `Groq` Configuration
//!
//! Configuration structures and validation for the Groq provider.

use secrecy::SecretString;

use crate::error::LlmError;
use crate::types::{CommonParams, HttpConfig, WebSearchConfig};

/// `Groq` Configuration
#[derive(Debug, Clone)]
pub struct GroqConfig {
    /// API key for authentication (securely stored)
    pub api_key: SecretString,
    /// Base URL for the Groq API
    pub base_url: String,
    /// Common parameters
    pub common_params: CommonParams,
    /// HTTP configuration
    pub http_config: HttpConfig,
    /// Web search configuration
    pub web_search_config: WebSearchConfig,
}

impl GroqConfig {
    /// Default Groq API base URL
    pub const DEFAULT_BASE_URL: &'static str = "https://api.groq.com/openai/v1";

    /// Create a new `Groq` configuration
    pub fn new<S: Into<String>>(api_key: S) -> Self {
        Self {
            api_key: SecretString::from(api_key.into()),
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            common_params: CommonParams::default(),
            http_config: HttpConfig::default(),
            web_search_config: WebSearchConfig::default(),
        }
    }

    /// Set the base URL
    pub fn with_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Set the model
    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.common_params.model = model.into();
        self
    }

    /// Set the temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.common_params.temperature = Some(temperature);
        self
    }

    /// Set the maximum tokens
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.common_params.max_tokens = Some(max_tokens);
        self
    }

    /// Set the top_p parameter
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.common_params.top_p = Some(top_p);
        self
    }

    /// Set stop sequences
    pub fn with_stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.common_params.stop_sequences = Some(stop_sequences);
        self
    }

    /// Set the seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.common_params.seed = Some(seed);
        self
    }

    /// Set web search configuration
    pub fn with_web_search_config(mut self, web_search_config: WebSearchConfig) -> Self {
        self.web_search_config = web_search_config;
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), LlmError> {
        use secrecy::ExposeSecret;
        if self.api_key.expose_secret().is_empty() {
            return Err(LlmError::ConfigurationError(
                "API key cannot be empty".to_string(),
            ));
        }

        if self.base_url.is_empty() {
            return Err(LlmError::ConfigurationError(
                "Base URL cannot be empty".to_string(),
            ));
        }

        if self.common_params.model.is_empty() {
            return Err(LlmError::ConfigurationError(
                "Model cannot be empty".to_string(),
            ));
        }

        // Validate temperature range (relaxed validation - only check for negative values)
        if let Some(temp) = self.common_params.temperature
            && temp < 0.0
        {
            return Err(LlmError::ConfigurationError(
                "Temperature cannot be negative".to_string(),
            ));
        }

        // Validate top_p range
        if let Some(top_p) = self.common_params.top_p
            && !(0.0..=1.0).contains(&top_p)
        {
            return Err(LlmError::ConfigurationError(
                "top_p must be between 0.0 and 1.0".to_string(),
            ));
        }

        // Validate max_tokens
        if let Some(max_tokens) = self.common_params.max_tokens
            && max_tokens == 0
        {
            return Err(LlmError::ConfigurationError(
                "max_tokens must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }

    /// Get supported models for Groq
    pub fn supported_models() -> Vec<&'static str> {
        crate::providers::groq::models::all_models()
    }

    /// Check if a model is supported
    pub fn is_model_supported(model: &str) -> bool {
        Self::supported_models().contains(&model)
    }

    /// Get default model
    pub fn default_model() -> &'static str {
        crate::providers::groq::models::popular::FLAGSHIP
    }
}

impl Default for GroqConfig {
    fn default() -> Self {
        Self::new("").with_model(Self::default_model())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_groq_config_creation() {
        use secrecy::ExposeSecret;
        let config = GroqConfig::new("test-api-key")
            .with_model("llama-3.3-70b-versatile")
            .with_temperature(0.7)
            .with_max_tokens(1000);

        assert_eq!(config.api_key.expose_secret(), "test-api-key");
        assert_eq!(config.common_params.model, "llama-3.3-70b-versatile");
        assert_eq!(config.common_params.temperature, Some(0.7));
        assert_eq!(config.common_params.max_tokens, Some(1000));
        assert_eq!(config.base_url, GroqConfig::DEFAULT_BASE_URL);
    }

    #[test]
    fn test_groq_config_validation() {
        // Valid configuration
        let valid_config = GroqConfig::new("test-api-key")
            .with_model("llama-3.3-70b-versatile")
            .with_temperature(0.7);
        assert!(valid_config.validate().is_ok());

        // High temperature (now allowed with relaxed validation)
        let high_temp_config = GroqConfig::new("test-api-key")
            .with_model("llama-3.3-70b-versatile")
            .with_temperature(3.0);
        assert!(high_temp_config.validate().is_ok());

        // Negative temperature (still invalid)
        let invalid_temp_config = GroqConfig::new("test-api-key")
            .with_model("llama-3.3-70b-versatile")
            .with_temperature(-1.0);
        assert!(invalid_temp_config.validate().is_err());

        // Empty API key
        let empty_key_config =
            GroqConfig::new("").with_model(crate::providers::groq::models::popular::FLAGSHIP);
        assert!(empty_key_config.validate().is_err());
    }

    #[test]
    fn test_supported_models() {
        let models = GroqConfig::supported_models();
        assert!(models.contains(&crate::providers::groq::models::popular::FLAGSHIP));
        assert!(models.contains(&crate::providers::groq::models::popular::SPEECH_TO_TEXT));

        assert!(GroqConfig::is_model_supported(
            crate::providers::groq::models::popular::FLAGSHIP
        ));
        assert!(!GroqConfig::is_model_supported("non-existent-model"));
    }
}
