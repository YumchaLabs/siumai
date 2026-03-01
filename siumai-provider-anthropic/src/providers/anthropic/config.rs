//! Anthropic configuration helpers.

use crate::error::LlmError;
use crate::execution::http::transport::HttpTransport;
use crate::types::{CommonParams, HttpConfig};
use secrecy::{ExposeSecret, SecretString};
use std::sync::Arc;

// Legacy params re-export (kept for backwards compatibility).
pub use crate::params::AnthropicParams;
pub use crate::params::anthropic::CacheControl;

/// Anthropic provider configuration (provider layer).
///
/// This configuration is intended for **config-first** construction:
/// `AnthropicClient::from_config(AnthropicConfig)`.
#[derive(Clone)]
pub struct AnthropicConfig {
    /// API key (securely stored).
    pub api_key: SecretString,
    /// Base URL for the Anthropic API.
    pub base_url: String,
    /// Common parameters shared across providers.
    pub common_params: CommonParams,
    /// Legacy Anthropic parameters (client-level defaults).
    pub anthropic_params: AnthropicParams,
    /// HTTP configuration.
    pub http_config: HttpConfig,
    /// Optional custom HTTP transport (Vercel-style "custom fetch" parity).
    pub http_transport: Option<Arc<dyn HttpTransport>>,
}

impl std::fmt::Debug for AnthropicConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("AnthropicConfig");
        ds.field("base_url", &self.base_url)
            .field("common_params", &self.common_params)
            .field("http_config", &self.http_config);

        if !self.api_key.expose_secret().is_empty() {
            ds.field("has_api_key", &true);
        }
        if self.http_transport.is_some() {
            ds.field("has_http_transport", &true);
        }

        ds.finish()
    }
}

impl AnthropicConfig {
    /// Create a new Anthropic configuration with the given API key.
    pub fn new<S: Into<String>>(api_key: S) -> Self {
        Self {
            api_key: SecretString::from(api_key.into()),
            base_url: "https://api.anthropic.com".to_string(),
            common_params: CommonParams::default(),
            anthropic_params: AnthropicParams::default(),
            http_config: HttpConfig::default(),
            http_transport: None,
        }
    }

    /// Create config from `ANTHROPIC_API_KEY`.
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| LlmError::MissingApiKey("Anthropic API key not provided".to_string()))?;
        Ok(Self::new(api_key))
    }

    pub fn with_base_url<S: Into<String>>(mut self, url: S) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_http_transport(mut self, transport: Arc<dyn HttpTransport>) -> Self {
        self.http_transport = Some(transport);
        self
    }

    pub fn with_http_config(mut self, http: HttpConfig) -> Self {
        self.http_config = http;
        self
    }

    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.common_params.model = model.into();
        self
    }

    pub const fn with_temperature(mut self, temperature: f64) -> Self {
        self.common_params.temperature = Some(temperature);
        self
    }

    pub const fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.common_params.max_tokens = Some(max_tokens);
        self
    }
}
