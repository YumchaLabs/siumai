//! `Groq` Configuration
//!
//! Configuration structures and validation for the Groq provider.

use secrecy::SecretString;
use std::sync::Arc;

use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::http::transport::HttpTransport;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::provider_options::{
    GroqOptions, GroqReasoningEffort, GroqReasoningFormat, GroqServiceTier,
};
use crate::types::{CommonParams, HttpConfig};

/// `Groq` Configuration
#[derive(Clone)]
pub struct GroqConfig {
    /// API key for authentication (securely stored)
    pub api_key: SecretString,
    /// Base URL for the Groq API
    pub base_url: String,
    /// Common parameters
    pub common_params: CommonParams,
    /// HTTP configuration
    pub http_config: HttpConfig,
    /// Optional custom HTTP transport.
    pub http_transport: Option<Arc<dyn HttpTransport>>,
    /// Optional HTTP interceptors applied to all requests built from this config.
    pub http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional model-level middlewares applied before provider mapping (chat only).
    pub model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    /// Provider-specific request defaults merged into chat requests.
    pub provider_specific_config: std::collections::HashMap<String, serde_json::Value>,
}

impl std::fmt::Debug for GroqConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use secrecy::ExposeSecret;
        let mut ds = f.debug_struct("GroqConfig");
        ds.field("base_url", &self.base_url)
            .field("common_params", &self.common_params)
            .field("http_config", &self.http_config);

        if !self.api_key.expose_secret().is_empty() {
            ds.field("has_api_key", &true);
        }
        if self.http_transport.is_some() {
            ds.field("has_http_transport", &true);
        }
        if !self.provider_specific_config.is_empty() {
            ds.field("provider_specific_config", &self.provider_specific_config);
        }

        ds.finish()
    }
}

impl GroqConfig {
    /// Default Groq API base URL
    pub const DEFAULT_BASE_URL: &'static str = "https://api.groq.com/openai/v1";

    pub(crate) fn normalize_base_url(base_url: &str) -> String {
        let trimmed = base_url.trim_end_matches('/');
        let path = trimmed.splitn(4, '/').nth(3).unwrap_or("");

        if path.is_empty() {
            format!("{trimmed}/openai/v1")
        } else {
            trimmed.to_string()
        }
    }

    /// Create a new `Groq` configuration
    pub fn new<S: Into<String>>(api_key: S) -> Self {
        Self {
            api_key: SecretString::from(api_key.into()),
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            common_params: CommonParams::default(),
            http_config: HttpConfig::default(),
            http_transport: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
            provider_specific_config: std::collections::HashMap::new(),
        }
    }

    /// Set the base URL
    pub fn with_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        let base_url = base_url.into();
        self.base_url = Self::normalize_base_url(&base_url);
        self
    }

    /// Set the model
    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.common_params.model = model.into();
        self
    }

    /// Set the temperature
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.common_params.temperature = Some(temperature);
        self
    }

    /// Set the maximum tokens
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.common_params.max_tokens = Some(max_tokens);
        self
    }

    /// Set the top_p parameter
    pub fn with_top_p(mut self, top_p: f64) -> Self {
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

    /// Set HTTP configuration.
    pub fn with_http_config(mut self, http_config: HttpConfig) -> Self {
        self.http_config = http_config;
        self
    }

    /// Replace custom HTTP headers.
    pub fn with_headers(mut self, headers: std::collections::HashMap<String, String>) -> Self {
        self.http_config.headers = headers;
        self
    }

    /// Add a single custom HTTP header.
    pub fn with_header<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.http_config.headers.insert(key.into(), value.into());
        self
    }

    /// Install HTTP interceptors for requests created by clients built from this config.
    pub fn with_http_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors;
        self
    }

    /// Set a custom HTTP transport.
    pub fn with_http_transport(mut self, transport: Arc<dyn HttpTransport>) -> Self {
        self.http_transport = Some(transport);
        self
    }

    /// Install model-level middlewares for chat requests created by clients built from this config.
    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.model_middlewares = middlewares;
        self
    }

    /// Merge provider-specific request defaults into Groq chat requests.
    pub fn with_provider_specific_config(
        mut self,
        params: std::collections::HashMap<String, serde_json::Value>,
    ) -> Self {
        self.provider_specific_config
            .extend(params.into_iter().filter(|(_, value)| !value.is_null()));
        self
    }

    /// Add a single provider-specific request default.
    pub fn with_provider_specific_param(
        mut self,
        key: impl Into<String>,
        value: serde_json::Value,
    ) -> Self {
        self.provider_specific_config.insert(key.into(), value);
        self
    }

    /// Merge typed Groq options into config defaults.
    pub fn with_groq_options(mut self, options: GroqOptions) -> Self {
        if let Ok(serde_json::Value::Object(obj)) = serde_json::to_value(options) {
            self.provider_specific_config
                .extend(obj.into_iter().filter(|(_, value)| !value.is_null()));
        }
        self
    }

    /// Enable or disable logprobs by default.
    pub fn with_logprobs(self, enabled: bool) -> Self {
        self.with_groq_options(GroqOptions::new().with_logprobs(enabled))
    }

    /// Set top-logprobs by default.
    pub fn with_top_logprobs(self, count: u32) -> Self {
        self.with_groq_options(GroqOptions::new().with_top_logprobs(count))
    }

    /// Set Groq service tier by default.
    pub fn with_service_tier(self, tier: GroqServiceTier) -> Self {
        self.with_groq_options(GroqOptions::new().with_service_tier(tier))
    }

    /// Set Groq reasoning effort by default.
    pub fn with_reasoning_effort(self, effort: GroqReasoningEffort) -> Self {
        self.with_groq_options(GroqOptions::new().with_reasoning_effort(effort))
    }

    /// Set Groq reasoning format by default.
    pub fn with_reasoning_format(self, format: GroqReasoningFormat) -> Self {
        self.with_groq_options(GroqOptions::new().with_reasoning_format(format))
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

        if !self.provider_specific_config.is_empty() {
            let params = serde_json::to_value(&self.provider_specific_config).map_err(|e| {
                LlmError::ConfigurationError(format!(
                    "Failed to serialize Groq provider defaults for validation: {e}"
                ))
            })?;
            crate::providers::groq::utils::validate_groq_params(&params)?;
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
    use crate::provider_options::{GroqReasoningEffort, GroqReasoningFormat, GroqServiceTier};

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

    #[test]
    fn test_groq_config_normalizes_root_base_url() {
        let config = GroqConfig::new("test-api-key")
            .with_base_url("https://example.com")
            .with_model("playai-tts");

        assert_eq!(config.base_url, "https://example.com/openai/v1");
    }

    #[test]
    fn test_groq_config_merges_typed_provider_defaults() {
        let config = GroqConfig::new("test-api-key")
            .with_model("llama-3.3-70b-versatile")
            .with_logprobs(true)
            .with_top_logprobs(2)
            .with_service_tier(GroqServiceTier::Flex)
            .with_reasoning_effort(GroqReasoningEffort::Default)
            .with_reasoning_format(GroqReasoningFormat::Parsed);

        assert_eq!(
            config.provider_specific_config.get("logprobs"),
            Some(&serde_json::json!(true))
        );
        assert_eq!(
            config.provider_specific_config.get("top_logprobs"),
            Some(&serde_json::json!(2))
        );
        assert_eq!(
            config.provider_specific_config.get("service_tier"),
            Some(&serde_json::json!("flex"))
        );
        assert_eq!(
            config.provider_specific_config.get("reasoning_effort"),
            Some(&serde_json::json!("default"))
        );
        assert_eq!(
            config.provider_specific_config.get("reasoning_format"),
            Some(&serde_json::json!("parsed"))
        );
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_groq_config_rejects_invalid_provider_defaults() {
        let config = GroqConfig::new("test-api-key")
            .with_model("llama-3.3-70b-versatile")
            .with_provider_specific_param("service_tier", serde_json::json!("invalid"));

        assert!(config.validate().is_err());
    }
}
