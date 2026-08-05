//! MiniMax Configuration
//!
//! Configuration structures for MiniMax API client.

use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::provider_options::{MinimaxOptions, MinimaxServiceTier, MinimaxThinking};
use crate::types::{CommonParams, HttpConfig};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// MiniMax API configuration
#[derive(Clone, Serialize, Deserialize)]
pub struct MinimaxConfig {
    /// API key for authentication
    pub api_key: String,
    /// Base URL for MiniMax API
    pub base_url: String,
    /// Chat protocol endpoint used by this client.
    pub chat_endpoint: super::models::MinimaxChatEndpoint,
    /// Common parameters (model, temperature, etc.)
    pub common_params: CommonParams,
    /// HTTP configuration
    #[serde(default)]
    pub http_config: HttpConfig,
    /// Optional custom HTTP transport (Vercel-style "custom fetch" parity).
    #[serde(skip)]
    pub http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    /// Optional HTTP interceptors applied to all requests built from this config.
    #[serde(skip)]
    pub http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional model-level middlewares applied before provider mapping (chat only).
    #[serde(skip)]
    pub model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    /// Default provider-owned request options merged before request-local overrides.
    #[serde(default)]
    pub default_provider_options_map: crate::types::ProviderOptionsMap,
}

impl std::fmt::Debug for MinimaxConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MinimaxConfig")
            .field("base_url", &self.base_url)
            .field("chat_endpoint", &self.chat_endpoint)
            .field("common_params", &self.common_params)
            .field("http_config", &self.http_config)
            .field("has_api_key", &(!self.api_key.is_empty()))
            .field("has_http_transport", &self.http_transport.is_some())
            .field(
                "default_provider_options_map",
                &self.default_provider_options_map,
            )
            .finish()
    }
}

impl MinimaxConfig {
    /// Default MiniMax API root.
    pub const DEFAULT_BASE_URL: &'static str = "https://api.minimax.io";

    /// OpenAI-compatible base URL for audio, image, video, and music APIs
    pub const OPENAI_BASE_URL: &'static str = "https://api.minimax.io/v1";

    /// Default current text model.
    pub const DEFAULT_MODEL: &'static str = super::models::CHAT;

    /// Create a new MiniMax configuration
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            chat_endpoint: super::models::MinimaxChatEndpoint::AnthropicMessages,
            common_params: CommonParams {
                model: Self::DEFAULT_MODEL.to_string(),
                ..Default::default()
            },
            http_config: crate::defaults::http::config_default(),
            http_transport: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
            default_provider_options_map: crate::types::ProviderOptionsMap::default(),
        }
    }

    /// Set the base URL
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Select the chat protocol endpoint.
    pub fn with_chat_endpoint(mut self, endpoint: super::models::MinimaxChatEndpoint) -> Self {
        self.chat_endpoint = endpoint;
        self
    }

    /// Set the default model
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.common_params.model = model.into();
        self
    }

    /// Set the HTTP configuration.
    pub fn with_http_config(mut self, http_config: HttpConfig) -> Self {
        self.http_config = http_config;
        self
    }

    /// Set a custom HTTP transport (Vercel-style "custom fetch" parity).
    pub fn with_http_transport(
        mut self,
        transport: Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.http_transport = Some(transport);
        self
    }

    /// Set request timeout on the canonical config-first HTTP surface.
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http_config.timeout = Some(timeout);
        self
    }

    /// Set connection timeout on the canonical config-first HTTP surface.
    pub fn with_connect_timeout(mut self, connect_timeout: std::time::Duration) -> Self {
        self.http_config.connect_timeout = Some(connect_timeout);
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

    /// Merge provider default options into this config.
    pub fn with_provider_options_map(
        mut self,
        provider_options_map: crate::types::ProviderOptionsMap,
    ) -> Self {
        self.default_provider_options_map
            .merge_overrides(provider_options_map);
        self
    }

    /// Merge MiniMax-specific default chat options into this config.
    pub fn with_minimax_options(mut self, options: MinimaxOptions) -> Self {
        let value = serde_json::to_value(options).expect("MiniMax options should serialize");
        match (
            self.default_provider_options_map.get("minimax").cloned(),
            value,
        ) {
            (Some(serde_json::Value::Object(mut base)), serde_json::Value::Object(extra)) => {
                for (key, value) in extra {
                    base.insert(key, value);
                }
                self.default_provider_options_map
                    .insert("minimax", serde_json::Value::Object(base));
            }
            (_, value) => {
                self.default_provider_options_map.insert("minimax", value);
            }
        }
        self
    }

    /// Configure MiniMax thinking defaults.
    pub fn with_thinking(self, thinking: MinimaxThinking) -> Self {
        self.with_minimax_options(MinimaxOptions::new().with_thinking(thinking))
    }

    /// Configure the request admission tier.
    pub fn with_service_tier(self, service_tier: MinimaxServiceTier) -> Self {
        self.with_minimax_options(MinimaxOptions::new().with_service_tier(service_tier))
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), LlmError> {
        if self.api_key.is_empty() {
            return Err(LlmError::ConfigurationError(
                "MiniMax API key cannot be empty".to_string(),
            ));
        }

        if self.base_url.is_empty() {
            return Err(LlmError::ConfigurationError(
                "MiniMax base URL cannot be empty".to_string(),
            ));
        }

        if !self.base_url.starts_with("http://") && !self.base_url.starts_with("https://") {
            return Err(LlmError::ConfigurationError(
                "MiniMax base URL must start with http:// or https://".to_string(),
            ));
        }

        if let Some(profile) = super::models::chat_profile(&self.common_params.model)
            && !profile.supports_endpoint(self.chat_endpoint)
        {
            return Err(LlmError::ConfigurationError(format!(
                "MiniMax model '{}' is not available through the {:?} endpoint",
                self.common_params.model, self.chat_endpoint
            )));
        }

        Ok(())
    }
}

impl Default for MinimaxConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            chat_endpoint: super::models::MinimaxChatEndpoint::AnthropicMessages,
            common_params: CommonParams {
                model: Self::DEFAULT_MODEL.to_string(),
                ..Default::default()
            },
            http_config: crate::defaults::http::config_default(),
            http_transport: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
            default_provider_options_map: crate::types::ProviderOptionsMap::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::Arc, time::Duration};

    #[test]
    fn minimax_config_merges_default_provider_options() {
        let config = MinimaxConfig::new("test-key")
            .with_thinking(MinimaxThinking::Adaptive)
            .with_service_tier(MinimaxServiceTier::Priority);

        let value = config
            .default_provider_options_map
            .get("minimax")
            .expect("minimax defaults");

        assert_eq!(value["thinking"], serde_json::json!({ "type": "adaptive" }));
        assert_eq!(value["service_tier"], serde_json::json!("priority"));
    }

    #[test]
    fn minimax_config_http_convenience_helpers() {
        let config = MinimaxConfig::new("test-key")
            .with_timeout(Duration::from_secs(10))
            .with_connect_timeout(Duration::from_secs(3))
            .with_http_stream_disable_compression(true)
            .with_http_interceptor(Arc::new(
                crate::execution::http::interceptor::LoggingInterceptor,
            ));

        assert_eq!(config.http_config.timeout, Some(Duration::from_secs(10)));
        assert_eq!(
            config.http_config.connect_timeout,
            Some(Duration::from_secs(3))
        );
        assert!(config.http_config.stream_disable_compression);
        assert_eq!(config.http_interceptors.len(), 1);
    }
}
