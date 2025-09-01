//! DeepSeek Provider Adapter
//!
//! This module provides the adapter for DeepSeek's OpenAI-compatible API.
//! DeepSeek supports both standard chat models and reasoning models with thinking capabilities.
//!
//! # Features
//! - Standard chat completion with `deepseek-chat`
//! - Reasoning capabilities with `deepseek-reasoner`
//! - Automatic handling of `reasoning_content` field for thinking output
//! - Full OpenAI API compatibility
//!
//! # Usage
//! ```rust,no_run
//! use siumai::prelude::*;
//! use siumai::providers::openai_compatible::providers::deepseek;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = LlmBuilder::new()
//!         .deepseek()
//!         .api_key("your-api-key")
//!         .model(deepseek::CHAT)
//!         .build()
//!         .await?;
//!
//!     // For reasoning model
//!     let reasoner = LlmBuilder::new()
//!         .deepseek()
//!         .api_key("your-api-key")
//!         .model(deepseek::REASONER)
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

/// DeepSeek model constants
pub mod models {
    /// DeepSeek Chat model - general purpose conversational AI
    pub const CHAT: &str = "deepseek-chat";

    /// DeepSeek Reasoner model - advanced reasoning with thinking capabilities
    pub const REASONER: &str = "deepseek-reasoner";
}

// Re-export for convenience
pub use models::*;

/// DeepSeek adapter configuration
#[derive(Debug, Clone, Default)]
pub struct DeepSeekConfig {
    /// Whether to enable reasoning output for supported models
    pub enable_reasoning: Option<bool>,
}

/// DeepSeek adapter
///
/// Handles DeepSeek-specific adaptations, particularly for reasoning models
/// which provide thinking content in the `reasoning_content` field.
#[derive(Debug, Clone)]
pub struct DeepSeekAdapter {
    /// Configuration for reasoning capabilities
    pub config: DeepSeekConfig,
}

impl Default for DeepSeekAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl DeepSeekAdapter {
    /// Create a new DeepSeek adapter with default configuration
    pub fn new() -> Self {
        Self {
            config: DeepSeekConfig::default(),
        }
    }

    /// Create adapter with reasoning explicitly enabled
    pub fn with_reasoning_enabled() -> Self {
        Self {
            config: DeepSeekConfig {
                enable_reasoning: Some(true),
            },
        }
    }

    /// Create adapter with reasoning explicitly disabled
    pub fn with_reasoning_disabled() -> Self {
        Self {
            config: DeepSeekConfig {
                enable_reasoning: Some(false),
            },
        }
    }

    /// Check if a model supports reasoning
    fn supports_reasoning(model: &str) -> bool {
        model == REASONER || model.contains("reasoner")
    }

    /// Get model-specific configuration
    fn get_model_config_for_model(&self, model: &str) -> ModelConfig {
        if Self::supports_reasoning(model) {
            ModelConfig {
                supports_thinking: true,
                // Don't set max_tokens - let DeepSeek use its defaults for reasoning models
                ..Default::default()
            }
        } else {
            ModelConfig::standard_chat()
        }
    }
}

impl ProviderAdapter for DeepSeekAdapter {
    fn provider_id(&self) -> &'static str {
        "deepseek"
    }

    fn transform_request_params(
        &self,
        params: &mut Value,
        model: &str,
        _request_type: RequestType,
    ) -> Result<(), LlmError> {
        // DeepSeek API is fully OpenAI-compatible, but reasoning models have special requirements

        if Self::supports_reasoning(model) {
            // For reasoning models (deepseek-reasoner):
            // 1. Clean up messages to remove reasoning_content from context
            if let Some(messages) = params.get_mut("messages") {
                if let Some(messages_array) = messages.as_array_mut() {
                    for message in messages_array.iter_mut() {
                        if let Some(message_obj) = message.as_object_mut() {
                            // Remove reasoning_content from assistant messages in context
                            // as per DeepSeek documentation: "reasoning_content should not be included in next round"
                            if message_obj.get("role").and_then(|r| r.as_str()) == Some("assistant") {
                                message_obj.remove("reasoning_content");
                            }
                        }
                    }
                }
            }

            // 2. Remove parameters that might interfere with reasoning content
            // Based on testing, DeepSeek reasoning models are sensitive to extra parameters
            let potentially_interfering_params = [
                "temperature", "top_p", "presence_penalty", "frequency_penalty",
                "logprobs", "top_logprobs", "max_tokens"
            ];

            for param in &potentially_interfering_params {
                if params.get(param).is_some() {
                    params.as_object_mut().unwrap().remove(*param);
                }
            }

            // 3. Don't set any default parameters for reasoning models
            // Let DeepSeek use its own defaults to ensure reasoning_content works

            // 4. DeepSeek reasoning models support reasoning_content in streaming mode
            // No special handling needed - the streaming implementation will handle reasoning_content correctly
        }

        Ok(())
    }

    fn get_field_mappings(&self, model: &str) -> FieldMappings {
        // DeepSeek uses standard OpenAI field names
        // For reasoning models, reasoning_content is an additional field alongside content
        if Self::supports_reasoning(model) {
            FieldMappings {
                thinking_fields: vec!["reasoning_content", "thinking"],
                content_field: "content",
                tool_calls_field: "tool_calls",
                role_field: "role",
            }
        } else {
            FieldMappings::default()
        }
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
            .with_custom_feature("reasoning", true)
            .with_custom_feature("thinking", true)
    }

    fn validate_model(&self, model: &str) -> Result<(), LlmError> {
        if model.is_empty() {
            return Err(LlmError::ConfigurationError(
                "Model name cannot be empty".to_string(),
            ));
        }

        // Validate known DeepSeek models
        match model {
            CHAT | REASONER => Ok(()),
            _ if model.starts_with("deepseek-") => Ok(()), // Allow other deepseek models
            _ => Err(LlmError::ConfigurationError(format!(
                "Unknown DeepSeek model: {}. Supported models: {}, {}",
                model, CHAT, REASONER
            ))),
        }
    }

    fn base_url(&self) -> &str {
        "https://api.deepseek.com/v1"
    }

    fn clone_adapter(&self) -> Box<dyn ProviderAdapter> {
        Box::new(self.clone())
    }
}

/// Builder for DeepSeek client configuration
#[derive(Debug, Clone)]
pub struct DeepSeekBuilder {
    api_key: String,
    base_url: Option<String>,
    model: Option<String>,
    enable_reasoning: Option<bool>,
}

impl Default for DeepSeekBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DeepSeekBuilder {
    /// Create a new DeepSeek builder
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

    /// Build the DeepSeek client
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
                DeepSeekAdapter::with_reasoning_enabled()
            } else {
                DeepSeekAdapter::with_reasoning_disabled()
            }
        } else {
            DeepSeekAdapter::new()
        };

        let base_url = self
            .base_url
            .unwrap_or_else(|| adapter.base_url().to_string());

        let mut config = crate::providers::openai_compatible::OpenAiCompatibleConfig::new(
            "deepseek",
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
    fn test_deepseek_adapter_creation() {
        let adapter = DeepSeekAdapter::new();
        assert_eq!(adapter.provider_id(), "deepseek");
        assert_eq!(adapter.base_url(), "https://api.deepseek.com/v1");
    }

    #[test]
    fn test_deepseek_adapter_with_reasoning() {
        let adapter = DeepSeekAdapter::with_reasoning_enabled();
        assert_eq!(adapter.provider_id(), "deepseek");
        assert!(adapter.config.enable_reasoning.unwrap_or(false));
    }

    #[test]
    fn test_deepseek_model_validation() {
        let adapter = DeepSeekAdapter::new();

        // Valid models
        assert!(adapter.validate_model(CHAT).is_ok());
        assert!(adapter.validate_model(REASONER).is_ok());
        assert!(adapter.validate_model("deepseek-custom").is_ok());

        // Invalid models
        assert!(adapter.validate_model("").is_err());
        assert!(adapter.validate_model("gpt-4").is_err());
    }

    #[test]
    fn test_deepseek_capabilities() {
        let adapter = DeepSeekAdapter::new();
        let capabilities = adapter.capabilities();

        assert!(capabilities.supports("chat"));
        assert!(capabilities.supports("streaming"));
        assert!(capabilities.supports("tools"));
        assert!(capabilities.supports("vision"));
        assert!(capabilities.supports("reasoning"));
        assert!(capabilities.supports("thinking"));
    }

    #[test]
    fn test_deepseek_request_transformation() {
        let adapter = DeepSeekAdapter::new();
        let mut params = json!({
            "model": REASONER,
            "messages": [],
            "temperature": 0.7
        });

        // Should not modify parameters for DeepSeek (fully OpenAI compatible)
        let original_params = params.clone();
        adapter
            .transform_request_params(&mut params, REASONER, RequestType::Chat)
            .unwrap();

        assert_eq!(params, original_params);
    }

    #[test]
    fn test_deepseek_model_config() {
        let adapter = DeepSeekAdapter::new();

        // Reasoning model should support thinking
        let reasoner_config = adapter.get_model_config(REASONER);
        assert!(reasoner_config.supports_thinking);
        assert_eq!(reasoner_config.max_tokens, Some(8192));

        // Chat model should use standard config
        let chat_config = adapter.get_model_config(CHAT);
        assert!(!chat_config.supports_thinking);
    }

    #[test]
    fn test_deepseek_field_mappings() {
        let adapter = DeepSeekAdapter::new();
        let mappings = adapter.get_field_mappings(REASONER);

        // DeepSeek uses standard OpenAI field names
        assert_eq!(mappings.content_field, "content");
        assert_eq!(mappings.role_field, "role");
        assert_eq!(mappings.tool_calls_field, "tool_calls");
    }

    #[test]
    fn test_supports_reasoning() {
        assert!(DeepSeekAdapter::supports_reasoning(REASONER));
        assert!(DeepSeekAdapter::supports_reasoning("deepseek-reasoner-v2"));
        assert!(!DeepSeekAdapter::supports_reasoning(CHAT));
        assert!(!DeepSeekAdapter::supports_reasoning("gpt-4"));
    }
}
