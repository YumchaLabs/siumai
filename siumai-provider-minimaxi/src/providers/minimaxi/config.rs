//! MiniMaxi Configuration
//!
//! Configuration structures for MiniMaxi API client.

use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
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
}

impl std::fmt::Debug for MinimaxiConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MinimaxiConfig")
            .field("base_url", &self.base_url)
            .field("common_params", &self.common_params)
            .field("http_config", &self.http_config)
            .field("has_api_key", &(!self.api_key.is_empty()))
            .field("has_http_transport", &self.http_transport.is_some())
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
        }
    }
}
