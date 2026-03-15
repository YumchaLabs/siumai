//! MiniMaxi Configuration
//!
//! Configuration structures for MiniMaxi API client.

use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::provider_options::{MinimaxiOptions, MinimaxiThinkingModeConfig};
use crate::types::{CommonParams, HttpConfig};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// MiniMaxi API configuration
#[derive(Clone, Serialize, Deserialize)]
pub struct MinimaxiConfig {
    /// API key for authentication
    pub api_key: String,
    /// Base URL for MiniMaxi API
    pub base_url: String,
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

impl std::fmt::Debug for MinimaxiConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MinimaxiConfig")
            .field("base_url", &self.base_url)
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

impl MinimaxiConfig {
    /// Default base URL for MiniMaxi API (Anthropic-compatible endpoint for chat)
    pub const DEFAULT_BASE_URL: &'static str = "https://api.minimaxi.com/anthropic";

    /// OpenAI-compatible base URL for audio, image, video, and music APIs
    pub const OPENAI_BASE_URL: &'static str = "https://api.minimaxi.com/v1";

    /// Default model (M2 text model)
    pub const DEFAULT_MODEL: &'static str = "MiniMax-M2";

    /// Create a new MiniMaxi configuration
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            common_params: CommonParams {
                model: Self::DEFAULT_MODEL.to_string(),
                ..Default::default()
            },
            http_config: HttpConfig::default(),
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

    /// Merge provider default options into this config.
    pub fn with_provider_options_map(
        mut self,
        provider_options_map: crate::types::ProviderOptionsMap,
    ) -> Self {
        self.default_provider_options_map
            .merge_overrides(provider_options_map);
        self
    }

    /// Merge MiniMaxi-specific default chat options into this config.
    pub fn with_minimaxi_options(mut self, options: MinimaxiOptions) -> Self {
        let value = serde_json::to_value(options).expect("MiniMaxi options should serialize");
        match (
            self.default_provider_options_map.get("minimaxi").cloned(),
            value,
        ) {
            (Some(serde_json::Value::Object(mut base)), serde_json::Value::Object(extra)) => {
                for (key, value) in extra {
                    base.insert(key, value);
                }
                self.default_provider_options_map
                    .insert("minimaxi", serde_json::Value::Object(base));
            }
            (_, value) => {
                self.default_provider_options_map.insert("minimaxi", value);
            }
        }
        self
    }

    /// Configure MiniMaxi thinking mode defaults.
    pub fn with_thinking_mode(self, config: MinimaxiThinkingModeConfig) -> Self {
        self.with_minimaxi_options(MinimaxiOptions::new().with_thinking_mode(config))
    }

    /// Configure MiniMaxi reasoning enablement defaults.
    pub fn with_reasoning_enabled(self, enabled: bool) -> Self {
        self.with_minimaxi_options(MinimaxiOptions::new().with_reasoning_enabled(enabled))
    }

    /// Configure MiniMaxi reasoning budget defaults.
    pub fn with_reasoning_budget(self, budget: u32) -> Self {
        self.with_minimaxi_options(MinimaxiOptions::new().with_reasoning_budget(budget))
    }

    /// Configure MiniMaxi JSON-object structured output defaults.
    pub fn with_json_object(self) -> Self {
        self.with_minimaxi_options(MinimaxiOptions::new().with_json_object())
    }

    /// Configure MiniMaxi JSON-schema structured output defaults.
    pub fn with_json_schema(
        self,
        name: impl Into<String>,
        schema: serde_json::Value,
        strict: bool,
    ) -> Self {
        self.with_minimaxi_options(MinimaxiOptions::new().with_json_schema(name, schema, strict))
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), LlmError> {
        if self.api_key.is_empty() {
            return Err(LlmError::ConfigurationError(
                "MiniMaxi API key cannot be empty".to_string(),
            ));
        }

        if self.base_url.is_empty() {
            return Err(LlmError::ConfigurationError(
                "MiniMaxi base URL cannot be empty".to_string(),
            ));
        }

        if !self.base_url.starts_with("http://") && !self.base_url.starts_with("https://") {
            return Err(LlmError::ConfigurationError(
                "MiniMaxi base URL must start with http:// or https://".to_string(),
            ));
        }

        Ok(())
    }
}

impl Default for MinimaxiConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            common_params: CommonParams {
                model: Self::DEFAULT_MODEL.to_string(),
                ..Default::default()
            },
            http_config: HttpConfig::default(),
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

    #[test]
    fn minimaxi_config_merges_default_provider_options() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"]
        });
        let config = MinimaxiConfig::new("test-key")
            .with_reasoning_budget(2048)
            .with_json_schema("response", schema.clone(), true);

        let value = config
            .default_provider_options_map
            .get("minimaxi")
            .expect("minimaxi defaults");

        assert_eq!(value["thinking_mode"]["enabled"], serde_json::json!(true));
        assert_eq!(
            value["thinking_mode"]["thinking_budget"],
            serde_json::json!(2048)
        );
        assert_eq!(value["response_format"]["JsonSchema"]["schema"], schema);
    }
}
