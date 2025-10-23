//! OpenAI-Compatible Provider Implementation
//!
//! This module provides a unified provider implementation for OpenAI-compatible APIs.
//! It uses the Provider-Model architecture to support multiple endpoints (chat, embedding, image, rerank).
//!
//! # Supported Providers
//!
//! - DeepSeek
//! - SiliconFlow (including rerank support)
//! - OpenRouter
//! - Together
//! - Fireworks
//! - Groq
//! - And other OpenAI-compatible providers
//!
//! # Example
//!
//! ```rust,no_run
//! use siumai::prelude::*;
//! use siumai::providers::openai_compatible::{OpenAiCompatibleProvider, OpenAiCompatibleConfig};
//! use siumai::registry::get_provider_adapter;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Get SiliconFlow adapter from registry
//!     let adapter = get_provider_adapter("siliconflow")?;
//!     
//!     // Create configuration
//!     let config = OpenAiCompatibleConfig::new(
//!         "siliconflow",
//!         "your-api-key",
//!         "https://api.siliconflow.cn/v1",
//!         adapter,
//!     );
//!     
//!     // Create provider
//!     let provider = OpenAiCompatibleProvider::new(config);
//!     
//!     // Create chat model
//!     let chat_model = provider.chat("Qwen/Qwen2.5-7B-Instruct")?;
//!     
//!     // Create executor
//!     let http_client = reqwest::Client::new();
//!     let executor = chat_model.create_executor(
//!         http_client,
//!         vec![],  // interceptors
//!         vec![],  // middlewares
//!         None,    // retry_options
//!     );
//!     
//!     Ok(())
//! }
//! ```

use crate::error::LlmError;
use crate::provider_model::{ChatModel, EmbeddingModel, ImageModel, Provider, RerankModel};
use crate::providers::openai_compatible::config::get_provider_config;
use crate::providers::openai_compatible::openai_config::OpenAiCompatibleConfig;
use std::sync::Arc;

use super::model_impls::{
    OpenAiCompatibleChatModel, OpenAiCompatibleEmbeddingModel, OpenAiCompatibleImageModel,
    OpenAiCompatibleRerankModel,
};

/// OpenAI-Compatible Provider
///
/// Factory for creating models for OpenAI-compatible providers.
/// Supports multiple providers through configuration and adapters.
#[derive(Clone)]
pub struct OpenAiCompatibleProvider {
    /// Provider configuration
    config: OpenAiCompatibleConfig,
    /// Provider ID (deepseek, siliconflow, openrouter, etc.)
    provider_id: String,
}

impl OpenAiCompatibleProvider {
    /// Create a new OpenAI-Compatible Provider
    ///
    /// # Arguments
    /// * `config` - Provider configuration with adapter
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::providers::openai_compatible::{OpenAiCompatibleProvider, OpenAiCompatibleConfig};
    /// use siumai::registry::get_provider_adapter;
    ///
    /// let adapter = get_provider_adapter("siliconflow").unwrap();
    /// let config = OpenAiCompatibleConfig::new(
    ///     "siliconflow",
    ///     "your-api-key",
    ///     "https://api.siliconflow.cn/v1",
    ///     adapter,
    /// );
    /// let provider = OpenAiCompatibleProvider::new(config);
    /// ```
    pub fn new(config: OpenAiCompatibleConfig) -> Self {
        let provider_id = config.provider_id.clone();
        Self {
            config,
            provider_id,
        }
    }

    /// Get the provider configuration from the registry
    ///
    /// Returns the provider configuration if available in the built-in registry.
    pub fn provider_config(
        &self,
    ) -> Option<crate::providers::openai_compatible::registry::ProviderConfig> {
        get_provider_config(&self.provider_id)
    }

    /// Check if this provider supports a specific capability
    ///
    /// # Arguments
    /// * `capability` - The capability to check (e.g., "chat", "embedding", "image", "rerank")
    ///
    /// # Returns
    /// `true` if the provider supports the capability, `false` otherwise
    pub fn supports_capability(&self, capability: &str) -> bool {
        if let Some(config) = self.provider_config() {
            config.capabilities.contains(&capability.to_string())
        } else {
            // If not in registry, assume basic capabilities
            matches!(capability, "chat" | "embedding")
        }
    }
}

impl Provider for OpenAiCompatibleProvider {
    fn id(&self) -> &str {
        &self.provider_id
    }

    fn chat(&self, model: &str) -> Result<Box<dyn ChatModel>, LlmError> {
        Ok(Box::new(OpenAiCompatibleChatModel::new(
            self.config.clone(),
            model.to_string(),
        )))
    }

    fn embedding(&self, model: &str) -> Result<Box<dyn EmbeddingModel>, LlmError> {
        Ok(Box::new(OpenAiCompatibleEmbeddingModel::new(
            self.config.clone(),
            model.to_string(),
        )))
    }

    fn image(&self, model: &str) -> Result<Box<dyn ImageModel>, LlmError> {
        Ok(Box::new(OpenAiCompatibleImageModel::new(
            self.config.clone(),
            model.to_string(),
        )))
    }

    fn rerank(&self, model: &str) -> Result<Box<dyn RerankModel>, LlmError> {
        // Check if provider supports rerank
        if !self.supports_capability("rerank") {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not support rerank",
                self.provider_id
            )));
        }

        Ok(Box::new(OpenAiCompatibleRerankModel::new(
            self.config.clone(),
            model.to_string(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        // This test requires the registry to be initialized
        // We'll skip it for now since it depends on runtime state
    }

    #[test]
    fn test_provider_id() {
        // Create a minimal config for testing
        use crate::providers::openai_compatible::adapter::ProviderAdapter;
        use crate::providers::openai_compatible::types::{FieldMappings, ModelConfig, RequestType};
        use crate::traits::ProviderCapabilities;

        #[derive(Debug)]
        struct TestAdapter;
        impl ProviderAdapter for TestAdapter {
            fn provider_id(&self) -> &'static str {
                "test"
            }
            fn transform_request_params(
                &self,
                _params: &mut serde_json::Value,
                _model: &str,
                _request_type: RequestType,
            ) -> Result<(), LlmError> {
                Ok(())
            }
            fn get_field_mappings(&self, _model: &str) -> FieldMappings {
                FieldMappings::default()
            }
            fn get_model_config(&self, _model: &str) -> ModelConfig {
                ModelConfig::default()
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::new().with_chat().with_embedding()
            }
            fn base_url(&self) -> &str {
                "https://api.test.com/v1"
            }
            fn clone_adapter(&self) -> Box<dyn ProviderAdapter> {
                Box::new(TestAdapter)
            }
        }

        let config = OpenAiCompatibleConfig::new(
            "test",
            "test-key",
            "https://api.test.com/v1",
            Arc::new(TestAdapter),
        );

        let provider = OpenAiCompatibleProvider::new(config);
        assert_eq!(provider.id(), "test");
    }

    #[test]
    fn test_supports_capability() {
        use crate::providers::openai_compatible::adapter::ProviderAdapter;
        use crate::providers::openai_compatible::types::{FieldMappings, ModelConfig, RequestType};
        use crate::traits::ProviderCapabilities;

        #[derive(Debug)]
        struct TestAdapter;
        impl ProviderAdapter for TestAdapter {
            fn provider_id(&self) -> &'static str {
                "test"
            }
            fn transform_request_params(
                &self,
                _params: &mut serde_json::Value,
                _model: &str,
                _request_type: RequestType,
            ) -> Result<(), LlmError> {
                Ok(())
            }
            fn get_field_mappings(&self, _model: &str) -> FieldMappings {
                FieldMappings::default()
            }
            fn get_model_config(&self, _model: &str) -> ModelConfig {
                ModelConfig::default()
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::new().with_chat().with_embedding()
            }
            fn base_url(&self) -> &str {
                "https://api.test.com/v1"
            }
            fn clone_adapter(&self) -> Box<dyn ProviderAdapter> {
                Box::new(TestAdapter)
            }
        }

        let config = OpenAiCompatibleConfig::new(
            "test",
            "test-key",
            "https://api.test.com/v1",
            Arc::new(TestAdapter),
        );

        let provider = OpenAiCompatibleProvider::new(config);

        // Basic capabilities should be supported by default
        assert!(provider.supports_capability("chat"));
        assert!(provider.supports_capability("embedding"));
    }
}
