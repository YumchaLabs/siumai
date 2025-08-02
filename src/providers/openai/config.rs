//! `OpenAI` Configuration
//!
//! This module provides configuration structures for the `OpenAI` provider.

use std::collections::HashMap;
use secrecy::{ExposeSecret, SecretString};

use crate::params::OpenAiParams;
use crate::types::{CommonParams, HttpConfig, WebSearchConfig};

/// `OpenAI` provider configuration.
///
/// This structure holds all the configuration needed to create and use
/// an `OpenAI` client, including authentication, API settings, and parameters.
///
/// # Example
/// ```rust
/// use siumai::providers::openai::OpenAiConfig;
///
/// let config = OpenAiConfig {
///     api_key: SecretString::from("your-api-key"),
///     base_url: "https://api.openai.com/v1".to_string(),
///     organization: Some("org-123".to_string()),
///     project: None,
///     common_params: Default::default(),
///     openai_params: Default::default(),
///     http_config: Default::default(),
///     web_search_config: Default::default(),
///     use_responses_api: false,
///     previous_response_id: None,
///     built_in_tools: Vec::new(),
/// };
/// ```
#[derive(Debug, Clone)]
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

    /// HTTP configuration
    pub http_config: HttpConfig,

    /// Web search configuration
    pub web_search_config: WebSearchConfig,

    /// Whether to use Responses API instead of Chat Completions API
    pub use_responses_api: bool,

    /// Previous response ID for chaining responses (Responses API only)
    pub previous_response_id: Option<String>,

    /// Built-in tools for Responses API
    pub built_in_tools: Vec<crate::types::OpenAiBuiltInTool>,
}

impl OpenAiConfig {
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
            http_config: HttpConfig::default(),
            web_search_config: WebSearchConfig::default(),
            use_responses_api: false,
            previous_response_id: None,
            built_in_tools: Vec::new(),
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

    /// Enable Responses API instead of Chat Completions API.
    pub const fn with_responses_api(mut self, use_responses: bool) -> Self {
        self.use_responses_api = use_responses;
        self
    }

    /// Set previous response ID for chaining responses.
    pub fn with_previous_response_id<S: Into<String>>(mut self, response_id: S) -> Self {
        self.previous_response_id = Some(response_id.into());
        self
    }

    /// Add a built-in tool for Responses API.
    pub fn with_built_in_tool(mut self, tool: crate::types::OpenAiBuiltInTool) -> Self {
        self.built_in_tools.push(tool);
        self
    }

    /// Add multiple built-in tools for Responses API.
    pub fn with_built_in_tools(mut self, tools: Vec<crate::types::OpenAiBuiltInTool>) -> Self {
        self.built_in_tools.extend(tools);
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
    pub fn validate(&self) -> Result<(), String> {
        if self.api_key.expose_secret().is_empty() {
            return Err("API key cannot be empty".to_string());
        }

        if self.base_url.is_empty() {
            return Err("Base URL cannot be empty".to_string());
        }

        if !self.base_url.starts_with("http://") && !self.base_url.starts_with("https://") {
            return Err("Base URL must start with http:// or https://".to_string());
        }

        // Validate common parameters
        if let Some(temp) = self.common_params.temperature {
            if !(0.0..=2.0).contains(&temp) {
                return Err("Temperature must be between 0.0 and 2.0".to_string());
            }
        }

        if let Some(top_p) = self.common_params.top_p {
            if !(0.0..=1.0).contains(&top_p) {
                return Err("Top-p must be between 0.0 and 1.0".to_string());
            }
        }

        // Validate OpenAI-specific parameters
        if let Some(freq_penalty) = self.openai_params.frequency_penalty {
            if !(-2.0..=2.0).contains(&freq_penalty) {
                return Err("Frequency penalty must be between -2.0 and 2.0".to_string());
            }
        }

        if let Some(pres_penalty) = self.openai_params.presence_penalty {
            if !(-2.0..=2.0).contains(&pres_penalty) {
                return Err("Presence penalty must be between -2.0 and 2.0".to_string());
            }
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
            http_config: HttpConfig::default(),
            web_search_config: WebSearchConfig::default(),
            use_responses_api: false,
            previous_response_id: None,
            built_in_tools: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = OpenAiConfig::new("test-key");
        assert_eq!(config.api_key.expose_secret(), "test-key");
        assert_eq!(config.base_url, "https://api.openai.com/v1");
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
}
