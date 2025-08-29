//! Provider Adapter System
//!
//! This module defines the core adapter trait for OpenAI-compatible providers.
//! It's inspired by Cherry Studio's RequestTransformer and ResponseChunkTransformer patterns.

use super::types::{FieldMappings, ModelConfig, ProviderCapabilities, RequestType};
use crate::error::LlmError;

/// Provider adapter trait
///
/// This trait defines the interface for adapting different OpenAI-compatible providers
/// to handle their specific request/response formats and parameter requirements.
///
/// Inspired by Cherry Studio's transformer patterns:
/// - RequestTransformer: handles request parameter transformation
/// - ResponseChunkTransformer: handles response format adaptation
pub trait ProviderAdapter: Send + Sync + std::fmt::Debug {
    /// Provider identifier
    fn provider_id(&self) -> &'static str;

    /// Transform request parameters based on provider and model specifics
    ///
    /// This is similar to Cherry Studio's RequestTransformer.transform method.
    /// It allows providers to modify request parameters based on their specific requirements.
    ///
    /// # Arguments
    /// * `params` - The request parameters as JSON value
    /// * `model` - The model name being used
    /// * `request_type` - The type of request (Chat, Embedding, etc.)
    ///
    /// # Example
    /// ```rust,ignore
    /// // SiliconFlow DeepSeek models need parameter mapping
    /// if model.contains("deepseek") && request_type == RequestType::Chat {
    ///     if let Some(thinking_budget) = params.get("thinking_budget") {
    ///         params["reasoning_effort"] = thinking_budget.clone();
    ///         params.as_object_mut().unwrap().remove("thinking_budget");
    ///     }
    /// }
    /// ```
    fn transform_request_params(
        &self,
        params: &mut serde_json::Value,
        model: &str,
        request_type: RequestType,
    ) -> Result<(), LlmError>;

    /// Get field mappings for response parsing
    ///
    /// This is similar to Cherry Studio's ResponseChunkTransformer logic.
    /// Different providers may use different field names for the same concepts.
    ///
    /// # Arguments
    /// * `model` - The model name being used
    ///
    /// # Returns
    /// Field mappings that specify which fields to look for in responses
    fn get_field_mappings(&self, model: &str) -> FieldMappings;

    /// Get model-specific configuration
    ///
    /// This handles model-specific behaviors like Cherry Studio's model checks
    /// (e.g., Qwen reasoning models requiring streaming).
    ///
    /// # Arguments
    /// * `model` - The model name being used
    ///
    /// # Returns
    /// Configuration specific to this model
    fn get_model_config(&self, model: &str) -> ModelConfig;

    /// Get custom HTTP headers for this provider
    ///
    /// Some providers require specific headers beyond the standard Authorization header.
    ///
    /// # Returns
    /// Additional headers to include in requests
    fn custom_headers(&self) -> reqwest::header::HeaderMap {
        reqwest::header::HeaderMap::new()
    }

    /// Get provider capabilities
    ///
    /// This defines what features this provider supports.
    ///
    /// # Returns
    /// Capabilities supported by this provider
    fn capabilities(&self) -> ProviderCapabilities;

    /// Validate model compatibility
    ///
    /// Check if a given model is supported by this provider.
    ///
    /// # Arguments
    /// * `model` - The model name to validate
    ///
    /// # Returns
    /// Ok(()) if the model is supported, Err otherwise
    fn validate_model(&self, model: &str) -> Result<(), LlmError> {
        // Default implementation accepts all models
        let _ = model;
        Ok(())
    }

    /// Get the base URL for this provider
    ///
    /// # Returns
    /// The base URL for API requests
    fn base_url(&self) -> &str;

    /// Clone the adapter
    ///
    /// This is needed because we store adapters in configurations.
    fn clone_adapter(&self) -> Box<dyn ProviderAdapter>;
}

/// Helper trait for cloning boxed adapters
impl Clone for Box<dyn ProviderAdapter> {
    fn clone(&self) -> Self {
        self.clone_adapter()
    }
}

/// Adapter registry for managing different provider adapters
#[derive(Debug, Default)]
pub struct AdapterRegistry {
    adapters: std::collections::HashMap<String, Box<dyn ProviderAdapter>>,
}

impl AdapterRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an adapter
    pub fn register(&mut self, adapter: Box<dyn ProviderAdapter>) {
        let provider_id = adapter.provider_id().to_string();
        self.adapters.insert(provider_id, adapter);
    }

    /// Get an adapter by provider ID
    pub fn get_adapter(&self, provider_id: &str) -> Option<&dyn ProviderAdapter> {
        self.adapters.get(provider_id).map(|a| a.as_ref())
    }

    /// List all registered provider IDs
    pub fn list_providers(&self) -> Vec<String> {
        self.adapters.keys().cloned().collect()
    }

    /// Check if a provider is registered
    pub fn has_provider(&self, provider_id: &str) -> bool {
        self.adapters.contains_key(provider_id)
    }
}

impl Clone for AdapterRegistry {
    fn clone(&self) -> Self {
        let mut registry = Self::new();
        for (id, adapter) in &self.adapters {
            registry.adapters.insert(id.clone(), adapter.clone());
        }
        registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::openai_compatible::types::*;

    #[derive(Debug, Clone)]
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
            ProviderCapabilities::default()
        }

        fn base_url(&self) -> &str {
            "https://api.test.com/v1"
        }

        fn clone_adapter(&self) -> Box<dyn ProviderAdapter> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn test_adapter_registry() {
        let mut registry = AdapterRegistry::new();
        assert_eq!(registry.list_providers().len(), 0);

        registry.register(Box::new(TestAdapter));
        assert_eq!(registry.list_providers().len(), 1);
        assert!(registry.has_provider("test"));

        let adapter = registry.get_adapter("test").unwrap();
        assert_eq!(adapter.provider_id(), "test");
    }

    #[test]
    fn test_adapter_clone() {
        let adapter: Box<dyn ProviderAdapter> = Box::new(TestAdapter);
        let cloned = adapter.clone();
        assert_eq!(adapter.provider_id(), cloned.provider_id());
    }
}
