//! `OpenAI` Configuration
//!
//! This module provides configuration structures for the `OpenAI` provider.

use secrecy::{ExposeSecret, SecretString};
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::http::transport::HttpTransport;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
pub use crate::params::OpenAiParams;
use crate::types::{CommonParams, HttpConfig, ProviderOptionsMap};

// Re-export legacy OpenAI parameter structs/types for backwards compatibility.
pub use crate::params::openai::{FunctionChoice, OpenAiParamsBuilder, ResponseFormat, ToolChoice};

/// `OpenAI` provider configuration.
///
/// This structure holds all the configuration needed to create and use
/// an `OpenAI` client, including authentication, API settings, and parameters.
///
/// # Example
/// ```rust
/// use siumai::providers::openai::OpenAiConfig;
/// use secrecy::SecretString;
///
/// let config = OpenAiConfig {
///     api_key: SecretString::from("your-api-key"),
///     base_url: "https://api.openai.com/v1".to_string(),
///     organization: Some("org-123".to_string()),
///     project: None,
///     common_params: Default::default(),
///     openai_params: Default::default(),
///     provider_options_map: Default::default(),
///     http_config: Default::default(),
///     http_transport: None,
///     http_interceptors: Vec::new(),
///     model_middlewares: Vec::new(),
/// };
/// ```
#[derive(Clone)]
pub struct OpenAiConfig {
    /// `OpenAI` API key (securely stored)
    pub api_key: SecretString,

    /// Base URL for the `OpenAI` API
    pub base_url: String,

    /// Optional organization ID
    pub organization: Option<String>,

    /// Optional project ID
    pub project: Option<String>,

    /// Common parameters shared across providers
    pub common_params: CommonParams,

    /// OpenAI-specific parameters
    pub openai_params: OpenAiParams,

    /// Default provider options (open map, Vercel-aligned).
    ///
    /// These defaults are merged into each request's `provider_options_map` with
    /// "request overrides defaults" semantics.
    pub provider_options_map: ProviderOptionsMap,

    /// HTTP configuration
    pub http_config: HttpConfig,

    /// Optional custom HTTP transport (Vercel-style "custom fetch" parity).
    pub http_transport: Option<Arc<dyn HttpTransport>>,

    /// Optional HTTP interceptors applied to all requests built from this config.
    pub http_interceptors: Vec<Arc<dyn HttpInterceptor>>,

    /// Optional model-level middlewares applied before provider mapping (chat only).
    pub model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
}

impl std::fmt::Debug for OpenAiConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("OpenAiConfig");
        ds.field("base_url", &self.base_url)
            .field("common_params", &self.common_params)
            .field("http_config", &self.http_config);

        if !self.api_key.expose_secret().is_empty() {
            ds.field("has_api_key", &true);
        }
        if self.organization.is_some() {
            ds.field("has_organization", &true);
        }
        if self.project.is_some() {
            ds.field("has_project", &true);
        }
        if self.http_transport.is_some() {
            ds.field("has_http_transport", &true);
        }

        ds.finish()
    }
}

impl OpenAiConfig {
    fn default_provider_options_map() -> ProviderOptionsMap {
        let mut provider_options_map = ProviderOptionsMap::default();
        provider_options_map.insert(
            "openai",
            serde_json::json!({
                "responsesApi": { "enabled": true }
            }),
        );
        provider_options_map
    }

    /// Create a new `OpenAI` configuration with the given API key.
    ///
    /// # Arguments
    /// * `api_key` - The `OpenAI` API key
    ///
    /// # Returns
    /// A new configuration with default settings
    pub fn new<S: Into<String>>(api_key: S) -> Self {
        Self {
            api_key: SecretString::from(api_key.into()),
            base_url: "https://api.openai.com/v1".to_string(),
            organization: None,
            project: None,
            common_params: CommonParams::default(),
            openai_params: OpenAiParams::default(),
            // Align config-first construction with the builder and Vercel AI SDK:
            // chat text generation defaults to the Responses API unless explicitly disabled.
            provider_options_map: Self::default_provider_options_map(),
            http_config: HttpConfig::default(),
            http_transport: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
        }
    }

    /// Set the base URL for the `OpenAI` API.
    ///
    /// # Arguments
    /// * `url` - The base URL
    pub fn with_base_url<S: Into<String>>(mut self, url: S) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set the organization ID.
    ///
    /// # Arguments
    /// * `org` - The organization ID
    pub fn with_organization<S: Into<String>>(mut self, org: S) -> Self {
        self.organization = Some(org.into());
        self
    }

    /// Set the project ID.
    ///
    /// # Arguments
    /// * `project` - The project ID
    pub fn with_project<S: Into<String>>(mut self, project: S) -> Self {
        self.project = Some(project.into());
        self
    }

    /// Set a custom HTTP transport (Vercel-style "custom fetch" parity).
    pub fn with_http_transport(mut self, transport: Arc<dyn HttpTransport>) -> Self {
        self.http_transport = Some(transport);
        self
    }

    /// Set request timeout on the canonical config-first HTTP surface.
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http_config.timeout = Some(timeout);
        self
    }

    /// Set connection timeout on the canonical config-first HTTP surface.
    pub fn with_connect_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http_config.connect_timeout = Some(timeout);
        self
    }

    /// Control whether streaming requests disable compression.
    pub fn with_http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.http_config.stream_disable_compression = disable;
        self
    }

    /// Install HTTP interceptors for requests created by clients built from this config.
    pub fn with_http_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors;
        self
    }

    /// Append a single HTTP interceptor on the canonical config-first HTTP surface.
    pub fn with_http_interceptor(mut self, interceptor: Arc<dyn HttpInterceptor>) -> Self {
        self.http_interceptors.push(interceptor);
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

    /// Set the HTTP configuration for this client.
    ///
    /// This config is used when building the internal `reqwest::Client` via
    /// `OpenAiClient::from_config(...)`.
    pub fn with_http_config(mut self, http_config: HttpConfig) -> Self {
        self.http_config = http_config;
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
    pub const fn with_temperature(mut self, temperature: f64) -> Self {
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

    /// Set the top-p sampling value.
    pub const fn with_top_p(mut self, top_p: f64) -> Self {
        self.common_params.top_p = Some(top_p);
        self
    }

    /// Set the stop sequences.
    pub fn with_stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.common_params.stop_sequences = Some(sequences);
        self
    }

    /// Set the deterministic seed.
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.common_params.seed = Some(seed);
        self
    }

    /// Replace the full legacy OpenAI parameter block.
    pub fn with_openai_params(mut self, params: OpenAiParams) -> Self {
        self.openai_params = params;
        self
    }

    /// Set the response format.
    pub fn with_response_format(mut self, format: ResponseFormat) -> Self {
        self.openai_params.response_format = Some(format);
        self
    }

    /// Set the tool choice strategy.
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.openai_params.tool_choice = Some(choice);
        self
    }

    /// Set the frequency penalty.
    pub const fn with_frequency_penalty(mut self, penalty: f32) -> Self {
        self.openai_params.frequency_penalty = Some(penalty);
        self
    }

    /// Set the presence penalty.
    pub const fn with_presence_penalty(mut self, penalty: f32) -> Self {
        self.openai_params.presence_penalty = Some(penalty);
        self
    }

    /// Set whether parallel tool calls are allowed.
    pub const fn with_parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.openai_params.parallel_tool_calls = Some(enabled);
        self
    }

    /// Merge default OpenAI provider options.
    pub fn with_provider_options(mut self, options: serde_json::Value) -> Self {
        let mut overrides = ProviderOptionsMap::new();
        overrides.insert("openai", options);
        self.provider_options_map.merge_overrides(overrides);
        self
    }

    /// Merge typed default OpenAI provider options.
    pub fn with_openai_options(
        self,
        options: crate::provider_options::openai::OpenAiOptions,
    ) -> Self {
        self.with_provider_options(
            serde_json::to_value(options).expect("OpenAI options should serialize"),
        )
    }

    /// Replace the full provider options map.
    pub fn with_provider_options_map(mut self, map: ProviderOptionsMap) -> Self {
        self.provider_options_map = map;
        self
    }

    /// Configure whether Responses API is enabled by default.
    pub fn with_use_responses_api(mut self, enabled: bool) -> Self {
        self = self.with_provider_options(serde_json::json!({
            "responsesApi": { "enabled": enabled }
        }));
        self
    }

    /// Configure the default previous response id for Responses API chaining.
    pub fn with_responses_previous_response_id<S: Into<String>>(mut self, id: S) -> Self {
        self = self.with_provider_options(serde_json::json!({
            "responsesApi": { "previousResponseId": id.into() }
        }));
        self
    }

    /// Get the authorization header value.
    ///
    /// # Returns
    /// The authorization header value for API requests
    pub fn auth_header(&self) -> String {
        format!("Bearer {}", self.api_key.expose_secret())
    }

    /// Get the organization header if set.
    ///
    /// # Returns
    /// Optional organization header value
    pub fn organization_header(&self) -> Option<String> {
        self.organization.clone()
    }

    /// Get the project header if set.
    ///
    /// # Returns
    /// Optional project header value
    pub fn project_header(&self) -> Option<String> {
        self.project.clone()
    }

    /// Get all HTTP headers needed for `OpenAI` API requests.
    ///
    /// # Returns
    /// `HashMap` of header names to values
    pub fn get_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::new();

        // Authorization header
        headers.insert("Authorization".to_string(), self.auth_header());

        // Content-Type header
        headers.insert("Content-Type".to_string(), "application/json".to_string());

        // Organization header
        if let Some(org) = &self.organization {
            headers.insert("OpenAI-Organization".to_string(), org.clone());
        }

        // Project header
        if let Some(project) = &self.project {
            headers.insert("OpenAI-Project".to_string(), project.clone());
        }

        headers
    }

    /// Validate the configuration.
    ///
    /// # Returns
    /// Result indicating whether the configuration is valid
    pub fn validate(&self) -> Result<(), LlmError> {
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

        if !self.base_url.starts_with("http://") && !self.base_url.starts_with("https://") {
            return Err(LlmError::ConfigurationError(
                "Base URL must start with http:// or https://".to_string(),
            ));
        }

        // Validate common parameters
        if let Some(temp) = self.common_params.temperature
            && !(0.0..=2.0).contains(&temp)
        {
            return Err(LlmError::ConfigurationError(
                "Temperature must be between 0.0 and 2.0".to_string(),
            ));
        }

        if let Some(top_p) = self.common_params.top_p
            && !(0.0..=1.0).contains(&top_p)
        {
            return Err(LlmError::ConfigurationError(
                "Top-p must be between 0.0 and 1.0".to_string(),
            ));
        }

        // Validate OpenAI-specific parameters
        if let Some(freq_penalty) = self.openai_params.frequency_penalty
            && !(-2.0..=2.0).contains(&freq_penalty)
        {
            return Err(LlmError::ConfigurationError(
                "Frequency penalty must be between -2.0 and 2.0".to_string(),
            ));
        }

        if let Some(pres_penalty) = self.openai_params.presence_penalty
            && !(-2.0..=2.0).contains(&pres_penalty)
        {
            return Err(LlmError::ConfigurationError(
                "Presence penalty must be between -2.0 and 2.0".to_string(),
            ));
        }

        Ok(())
    }
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        Self {
            api_key: SecretString::from(String::new()),
            base_url: "https://api.openai.com/v1".to_string(),
            organization: None,
            project: None,
            common_params: CommonParams::default(),
            openai_params: OpenAiParams::default(),
            provider_options_map: Self::default_provider_options_map(),
            http_config: HttpConfig::default(),
            http_transport: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::Arc, time::Duration};

    #[test]
    fn test_config_creation() {
        let config = OpenAiConfig::new("test-key");
        assert_eq!(config.api_key.expose_secret(), "test-key");
        assert_eq!(config.base_url, "https://api.openai.com/v1");
        assert_eq!(
            config.provider_options_map.get("openai"),
            Some(&serde_json::json!({
                "responsesApi": { "enabled": true }
            }))
        );
    }

    #[test]
    fn test_config_validation() {
        let mut config = OpenAiConfig::new("test-key");
        assert!(config.validate().is_ok());

        config.api_key = SecretString::from(String::new());
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_headers() {
        let config = OpenAiConfig::new("test-key")
            .with_organization("org-123")
            .with_project("proj-456");

        let headers = config.get_headers();
        assert_eq!(
            headers.get("Authorization"),
            Some(&"Bearer test-key".to_string())
        );
        assert_eq!(
            headers.get("OpenAI-Organization"),
            Some(&"org-123".to_string())
        );
        assert_eq!(headers.get("OpenAI-Project"), Some(&"proj-456".to_string()));
    }

    #[test]
    fn test_openai_specific_fluent_setters() {
        let config = OpenAiConfig::new("test-key")
            .with_model("gpt-4o-mini")
            .with_top_p(0.8)
            .with_stop_sequences(vec!["END".to_string()])
            .with_seed(7)
            .with_response_format(ResponseFormat::JsonObject)
            .with_tool_choice(ToolChoice::String("required".to_string()))
            .with_frequency_penalty(0.5)
            .with_presence_penalty(-0.25)
            .with_parallel_tool_calls(true)
            .with_use_responses_api(true)
            .with_responses_previous_response_id("resp_123")
            .with_provider_options(serde_json::json!({ "custom": { "enabled": true } }));

        assert_eq!(config.common_params.model, "gpt-4o-mini");
        assert_eq!(config.common_params.top_p, Some(0.8));
        assert_eq!(
            config.common_params.stop_sequences.as_deref(),
            Some(&["END".to_string()][..])
        );
        assert_eq!(config.common_params.seed, Some(7));
        assert!(matches!(
            config.openai_params.response_format,
            Some(ResponseFormat::JsonObject)
        ));
        assert!(matches!(
            config.openai_params.tool_choice,
            Some(ToolChoice::String(ref choice)) if choice == "required"
        ));
        assert_eq!(config.openai_params.frequency_penalty, Some(0.5));
        assert_eq!(config.openai_params.presence_penalty, Some(-0.25));
        assert_eq!(config.openai_params.parallel_tool_calls, Some(true));

        let openai_options = config
            .provider_options_map
            .get("openai")
            .expect("openai provider options");
        assert_eq!(
            openai_options["responsesApi"]["enabled"],
            serde_json::json!(true)
        );
        assert_eq!(
            openai_options["responsesApi"]["previousResponseId"],
            serde_json::json!("resp_123")
        );
        assert_eq!(openai_options["custom"]["enabled"], serde_json::json!(true));
    }

    #[test]
    fn test_openai_http_convenience_helpers() {
        let config = OpenAiConfig::new("test-key")
            .with_timeout(Duration::from_secs(12))
            .with_connect_timeout(Duration::from_secs(3))
            .with_http_stream_disable_compression(true)
            .with_http_interceptor(Arc::new(
                crate::execution::http::interceptor::LoggingInterceptor,
            ));

        assert_eq!(config.http_config.timeout, Some(Duration::from_secs(12)));
        assert_eq!(
            config.http_config.connect_timeout,
            Some(Duration::from_secs(3))
        );
        assert!(config.http_config.stream_disable_compression);
        assert_eq!(config.http_interceptors.len(), 1);
    }
}
