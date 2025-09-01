//! OpenRouter Provider Adapter
//!
//! This module provides the adapter for OpenRouter's OpenAI-compatible API.
//! OpenRouter provides access to multiple AI models through a unified API.
//!
//! # Features
//! - Access to multiple AI providers (OpenAI, Anthropic, Google, Meta, etc.)
//! - Standard OpenAI API compatibility
//! - Model routing and load balancing
//! - Unified pricing and billing
//!
//! # Usage
//! ```rust,no_run
//! use siumai::prelude::*;
//! use siumai::providers::openai_compatible::providers::openrouter;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = LlmBuilder::new()
//!         .openrouter()
//!         .api_key("your-api-key")
//!         .model(openrouter::openai::GPT_4O)
//!         .build()
//!         .await?;
//!
//!     // For reasoning models
//!     let reasoner = LlmBuilder::new()
//!         .openrouter()
//!         .api_key("your-api-key")
//!         .model(openrouter::openai::O1)
//!         .build()
//!         .await?;
//!
//!     Ok(())
//! }
//! ```

use crate::error::LlmError;
use crate::providers::openai_compatible::adapter::ProviderAdapter;
use crate::providers::openai_compatible::types::{FieldMappings, ModelConfig, RequestType};
use crate::traits::ProviderCapabilities;
use serde_json::Value;

/// OpenRouter model constants (re-export from models.rs)
pub use crate::providers::openai_compatible::providers::models::openrouter::*;

/// OpenRouter adapter configuration
#[derive(Debug, Clone, Default)]
pub struct OpenRouterConfig {
    /// Whether to enable reasoning output for supported models
    pub enable_reasoning: Option<bool>,
}

/// OpenRouter adapter
///
/// Handles OpenRouter-specific adaptations. OpenRouter is fully OpenAI-compatible
/// and routes requests to various underlying providers.
#[derive(Debug, Clone)]
pub struct OpenRouterAdapter {
    /// Configuration for reasoning capabilities
    pub config: OpenRouterConfig,
}

impl Default for OpenRouterAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenRouterAdapter {
    /// Create a new OpenRouter adapter with default configuration
    pub fn new() -> Self {
        Self {
            config: OpenRouterConfig::default(),
        }
    }

    /// Create adapter with reasoning explicitly enabled
    pub fn with_reasoning_enabled() -> Self {
        Self {
            config: OpenRouterConfig {
                enable_reasoning: Some(true),
            },
        }
    }

    /// Create adapter with reasoning explicitly disabled
    pub fn with_reasoning_disabled() -> Self {
        Self {
            config: OpenRouterConfig {
                enable_reasoning: Some(false),
            },
        }
    }

    /// Check if a model supports reasoning
    fn supports_reasoning(model: &str) -> bool {
        // OpenRouter reasoning models include OpenAI's o1 series and other reasoning models
        model.contains("o1")
            || model.contains("reasoning")
            || model.contains("deepseek-r")
            || model.contains("qwq")
            || model.contains("thinking")
    }

    /// Get model-specific configuration
    fn get_model_config_for_model(&self, model: &str) -> ModelConfig {
        if Self::supports_reasoning(model) {
            ModelConfig {
                supports_thinking: true,
                max_tokens: Some(8192),
                ..Default::default()
            }
        } else {
            ModelConfig::standard_chat()
        }
    }
}

impl ProviderAdapter for OpenRouterAdapter {
    fn provider_id(&self) -> &'static str {
        "openrouter"
    }

    fn transform_request_params(
        &self,
        _params: &mut Value,
        model: &str,
        _request_type: RequestType,
    ) -> Result<(), LlmError> {
        // OpenRouter API is fully OpenAI-compatible, no parameter transformation needed
        // Reasoning capabilities are handled by the underlying models automatically

        // For reasoning models, OpenRouter passes through the requests to the underlying providers
        if Self::supports_reasoning(model) {
            // OpenRouter handles reasoning models transparently
            // No special parameters needed in the request
        }

        Ok(())
    }

    fn get_field_mappings(&self, _model: &str) -> FieldMappings {
        // OpenRouter uses standard OpenAI field names
        FieldMappings::default()
    }

    fn get_model_config(&self, model: &str) -> ModelConfig {
        self.get_model_config_for_model(model)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_embedding()
            .with_custom_feature("reasoning", true)
            .with_custom_feature("multi_provider", true)
            .with_custom_feature("model_routing", true)
    }

    fn base_url(&self) -> &str {
        "https://openrouter.ai/api/v1"
    }

    fn validate_model(&self, model: &str) -> Result<(), LlmError> {
        if model.is_empty() {
            return Err(LlmError::ConfigurationError(
                "Model name cannot be empty".to_string(),
            ));
        }

        // OpenRouter supports a wide variety of models from different providers
        // We'll allow any non-empty model name since the list is constantly changing
        Ok(())
    }

    fn clone_adapter(&self) -> Box<dyn ProviderAdapter> {
        Box::new(self.clone())
    }
}

/// Builder for OpenRouter client configuration
#[derive(Debug, Clone)]
pub struct OpenRouterBuilder {
    api_key: String,
    base_url: Option<String>,
    model: Option<String>,
    enable_reasoning: Option<bool>,
}

impl Default for OpenRouterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenRouterBuilder {
    /// Create a new OpenRouter builder
    pub fn new() -> Self {
        Self {
            api_key: String::new(),
            base_url: None,
            model: None,
            enable_reasoning: None,
        }
    }

    /// Set the API key
    pub fn api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = api_key.into();
        self
    }

    /// Set a custom base URL (optional)
    pub fn base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Set the model to use
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Enable reasoning output for supported models
    pub fn enable_reasoning(mut self, enable: bool) -> Self {
        self.enable_reasoning = Some(enable);
        self
    }

    /// Build the OpenRouter client
    pub async fn build(
        self,
    ) -> Result<crate::providers::openai_compatible::OpenAiCompatibleClient, LlmError> {
        if self.api_key.is_empty() {
            return Err(LlmError::ConfigurationError(
                "API key is required".to_string(),
            ));
        }

        let adapter = if let Some(enable) = self.enable_reasoning {
            if enable {
                OpenRouterAdapter::with_reasoning_enabled()
            } else {
                OpenRouterAdapter::with_reasoning_disabled()
            }
        } else {
            OpenRouterAdapter::new()
        };

        let base_url = self
            .base_url
            .unwrap_or_else(|| adapter.base_url().to_string());

        let mut config = crate::providers::openai_compatible::OpenAiCompatibleConfig::new(
            "openrouter",
            &self.api_key,
            &base_url,
            std::sync::Arc::new(adapter),
        );

        if let Some(model) = self.model {
            config = config.with_model(&model);
        }

        crate::providers::openai_compatible::OpenAiCompatibleClient::new(config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::openai_compatible::types::RequestType;
    use serde_json::json;

    #[test]
    fn test_openrouter_adapter_creation() {
        let adapter = OpenRouterAdapter::new();
        assert_eq!(adapter.provider_id(), "openrouter");
        assert_eq!(adapter.base_url(), "https://openrouter.ai/api/v1");
    }

    #[test]
    fn test_openrouter_adapter_with_reasoning() {
        let adapter = OpenRouterAdapter::with_reasoning_enabled();
        assert_eq!(adapter.provider_id(), "openrouter");
        assert!(adapter.config.enable_reasoning.unwrap_or(false));
    }

    #[test]
    fn test_openrouter_model_validation() {
        let adapter = OpenRouterAdapter::new();

        // Valid models
        assert!(adapter.validate_model("openai/gpt-4").is_ok());
        assert!(adapter.validate_model("anthropic/claude-3-sonnet").is_ok());
        assert!(adapter.validate_model("google/gemini-pro").is_ok());
        assert!(adapter.validate_model("meta-llama/llama-3-70b").is_ok());

        // Invalid models
        assert!(adapter.validate_model("").is_err());
    }

    #[test]
    fn test_openrouter_capabilities() {
        let adapter = OpenRouterAdapter::new();
        let capabilities = adapter.capabilities();

        assert!(capabilities.supports("chat"));
        assert!(capabilities.supports("streaming"));
        assert!(capabilities.supports("tools"));
        assert!(capabilities.supports("vision"));
        assert!(capabilities.supports("embedding"));
        assert!(capabilities.supports("reasoning"));
        assert!(capabilities.supports("multi_provider"));
        assert!(capabilities.supports("model_routing"));
    }

    #[test]
    fn test_openrouter_request_transformation() {
        let adapter = OpenRouterAdapter::new();
        let mut params = json!({
            "model": "openai/gpt-4",
            "messages": [],
            "temperature": 0.7
        });

        // Should not modify parameters for OpenRouter (fully OpenAI compatible)
        let original_params = params.clone();
        adapter
            .transform_request_params(&mut params, "openai/gpt-4", RequestType::Chat)
            .unwrap();

        assert_eq!(params, original_params);
    }

    #[test]
    fn test_openrouter_model_config() {
        let adapter = OpenRouterAdapter::new();

        // Reasoning model should support thinking
        let o1_config = adapter.get_model_config("openai/o1");
        assert!(o1_config.supports_thinking);
        assert_eq!(o1_config.max_tokens, Some(8192));

        // Regular model should use standard config
        let gpt4_config = adapter.get_model_config("openai/gpt-4");
        assert!(!gpt4_config.supports_thinking);
    }

    #[test]
    fn test_openrouter_field_mappings() {
        let adapter = OpenRouterAdapter::new();
        let mappings = adapter.get_field_mappings("openai/gpt-4");

        // OpenRouter uses standard OpenAI field names
        assert_eq!(mappings.content_field, "content");
        assert_eq!(mappings.role_field, "role");
        assert_eq!(mappings.tool_calls_field, "tool_calls");
    }

    #[test]
    fn test_supports_reasoning() {
        assert!(OpenRouterAdapter::supports_reasoning("openai/o1"));
        assert!(OpenRouterAdapter::supports_reasoning("openai/o1-mini"));
        assert!(OpenRouterAdapter::supports_reasoning(
            "deepseek/deepseek-r1"
        ));
        assert!(OpenRouterAdapter::supports_reasoning("qwen/qwq-32b"));
        assert!(!OpenRouterAdapter::supports_reasoning("openai/gpt-4"));
        assert!(!OpenRouterAdapter::supports_reasoning(
            "anthropic/claude-3-sonnet"
        ));
    }
}
