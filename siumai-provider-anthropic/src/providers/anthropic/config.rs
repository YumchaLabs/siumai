//! Anthropic configuration helpers.

use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::http::transport::HttpTransport;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
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
    /// Optional HTTP interceptors applied to all requests built from this config.
    pub http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional model-level middlewares applied before provider mapping (chat only).
    pub model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
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
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
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

    /// Install HTTP interceptors for requests created by clients built from this config.
    pub fn with_http_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors;
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

    pub const fn with_top_p(mut self, top_p: f64) -> Self {
        self.common_params.top_p = Some(top_p);
        self
    }

    pub fn with_anthropic_params(mut self, params: AnthropicParams) -> Self {
        self.anthropic_params = params;
        self
    }

    pub fn with_cache_control(mut self, cache: CacheControl) -> Self {
        self.anthropic_params.cache_control = Some(cache);
        self
    }

    pub const fn with_thinking_budget(mut self, budget: u32) -> Self {
        self.anthropic_params.thinking_budget = Some(budget);
        self
    }

    pub const fn with_thinking_enabled(mut self) -> Self {
        self.anthropic_params.thinking_budget = Some(10000);
        self
    }

    pub const fn with_thinking_mode(mut self, budget_tokens: Option<u32>) -> Self {
        self.anthropic_params.thinking_budget = budget_tokens;
        self
    }

    pub fn with_system_message<S: Into<String>>(mut self, system: S) -> Self {
        self.anthropic_params.system = Some(system.into());
        self
    }

    pub fn with_metadata(mut self, metadata: std::collections::HashMap<String, String>) -> Self {
        self.anthropic_params.metadata = Some(metadata);
        self
    }

    pub fn add_metadata<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<String>,
    {
        self.anthropic_params
            .metadata
            .get_or_insert_with(std::collections::HashMap::new)
            .insert(key.into(), value.into());
        self
    }

    pub const fn with_stream(mut self, enabled: bool) -> Self {
        self.anthropic_params.stream = Some(enabled);
        self
    }

    pub fn with_beta_features(mut self, features: Vec<String>) -> Self {
        self.anthropic_params.beta_features = Some(features);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn anthropic_config_provider_specific_fluent_setters() {
        let mut metadata = HashMap::new();
        metadata.insert("team".to_string(), "core".to_string());

        let config = AnthropicConfig::new("test-key")
            .with_model("claude-3-7-sonnet-latest")
            .with_top_p(0.9)
            .with_cache_control(CacheControl::ephemeral())
            .with_thinking_budget(2048)
            .with_system_message("You are helpful")
            .with_metadata(metadata)
            .add_metadata("feature", "config-first")
            .with_stream(true)
            .with_beta_features(vec!["prompt-caching-2024-07-31".to_string()]);

        assert_eq!(config.common_params.model, "claude-3-7-sonnet-latest");
        assert_eq!(config.common_params.top_p, Some(0.9));
        assert_eq!(
            config
                .anthropic_params
                .cache_control
                .as_ref()
                .map(|c| c.r#type.as_str()),
            Some("ephemeral")
        );
        assert_eq!(config.anthropic_params.thinking_budget, Some(2048));
        assert_eq!(
            config.anthropic_params.system.as_deref(),
            Some("You are helpful")
        );
        assert_eq!(
            config
                .anthropic_params
                .metadata
                .as_ref()
                .and_then(|m| m.get("team"))
                .map(String::as_str),
            Some("core")
        );
        assert_eq!(
            config
                .anthropic_params
                .metadata
                .as_ref()
                .and_then(|m| m.get("feature"))
                .map(String::as_str),
            Some("config-first")
        );
        assert_eq!(config.anthropic_params.stream, Some(true));
        assert_eq!(
            config.anthropic_params.beta_features.as_deref(),
            Some(&["prompt-caching-2024-07-31".to_string()][..])
        );
    }
}
