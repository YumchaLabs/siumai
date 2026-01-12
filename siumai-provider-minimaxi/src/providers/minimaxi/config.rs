//! MiniMaxi Configuration
//!
//! Configuration structures for MiniMaxi API client.

use crate::error::LlmError;
use crate::types::CommonParams;
use serde::{Deserialize, Serialize};

/// MiniMaxi API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimaxiConfig {
    /// API key for authentication
    pub api_key: String,
    /// Base URL for MiniMaxi API
    pub base_url: String,
    /// Common parameters (model, temperature, etc.)
    pub common_params: CommonParams,
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
        }
    }
}
