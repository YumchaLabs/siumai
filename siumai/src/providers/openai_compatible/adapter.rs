//! Enhanced Provider Adapter System
//!
//! This module defines the core adapter trait for OpenAI-compatible providers.
//! It's inspired by Cherry Studio's RequestTransformer and ResponseChunkTransformer patterns
//! and fully integrates with our existing traits and HTTP configuration system.

use super::types::{FieldAccessor, FieldMappings, JsonFieldAccessor, ModelConfig, RequestType};
use crate::error::LlmError;
use crate::traits::ProviderCapabilities;
use crate::types::HttpConfig;
use std::collections::HashMap;

/// Enhanced Provider Compatibility Configuration
///
/// This struct defines compatibility flags for different OpenAI API features,
/// similar to Cherry Studio's provider configuration system.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ProviderCompatibility {
    /// Whether the provider supports array content format in messages
    pub supports_array_content: bool,
    /// Whether the provider supports stream_options parameter
    pub supports_stream_options: bool,
    /// Whether the provider supports developer role in messages
    pub supports_developer_role: bool,
    /// Whether the provider supports enable_thinking parameter
    pub supports_enable_thinking: bool,
    /// Whether the provider supports service_tier parameter
    pub supports_service_tier: bool,
    /// Whether the provider forces streaming for certain models
    pub force_streaming_models: Vec<String>,
    /// Custom compatibility flags
    pub custom_flags: HashMap<String, bool>,
}

impl ProviderCompatibility {
    /// Create compatibility config for standard OpenAI API
    pub fn openai_standard() -> Self {
        Self {
            supports_array_content: true,
            supports_stream_options: true,
            supports_developer_role: true,
            supports_enable_thinking: true,
            supports_service_tier: true,
            force_streaming_models: vec![],
            custom_flags: HashMap::new(),
        }
    }

    /// Create compatibility config for DeepSeek
    pub fn deepseek() -> Self {
        Self {
            supports_array_content: false, // DeepSeek doesn't support array content
            supports_stream_options: true,
            supports_developer_role: true,
            supports_enable_thinking: false, // Uses reasoning_content instead
            supports_service_tier: false,
            force_streaming_models: vec!["deepseek-reasoner".to_string()],
            custom_flags: HashMap::new(),
        }
    }

    /// Create compatibility config for providers with limited support
    pub fn limited_compatibility() -> Self {
        Self {
            supports_array_content: false,
            supports_stream_options: false,
            supports_developer_role: false,
            supports_enable_thinking: false,
            supports_service_tier: false,
            force_streaming_models: vec![],
            custom_flags: HashMap::new(),
        }
    }
}

/// Enhanced Provider adapter trait
///
/// This trait defines the interface for adapting different OpenAI-compatible providers
/// to handle their specific request/response formats and parameter requirements.
///
/// Inspired by Cherry Studio's transformer patterns and fully integrated with our
/// existing traits system including ProviderCapabilities and HttpConfig.
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

    /// Get field accessor for dynamic field extraction
    ///
    /// This provides a configurable way to extract fields from JSON responses,
    /// similar to Cherry Studio's response transformation system.
    ///
    /// # Returns
    /// A field accessor that can extract values from JSON using field paths
    fn get_field_accessor(&self) -> Box<dyn FieldAccessor> {
        Box::new(JsonFieldAccessor)
    }

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

    /// Get provider capabilities (integrates with existing traits system)
    ///
    /// This defines what features this provider supports, using our existing
    /// ProviderCapabilities struct from traits.rs.
    ///
    /// # Returns
    /// Capabilities supported by this provider
    fn capabilities(&self) -> ProviderCapabilities;

    /// Get provider compatibility configuration
    ///
    /// This defines OpenAI API compatibility flags, similar to Cherry Studio's
    /// provider configuration system.
    ///
    /// # Returns
    /// Compatibility configuration for this provider
    fn compatibility(&self) -> ProviderCompatibility {
        ProviderCompatibility::openai_standard()
    }

    /// Apply HTTP configuration to the adapter
    ///
    /// This allows the adapter to customize HTTP settings based on provider requirements.
    /// Integrates with our existing HttpConfig system from types/common.rs.
    ///
    /// # Arguments
    /// * `http_config` - The base HTTP configuration to modify
    ///
    /// # Returns
    /// Modified HTTP configuration
    fn apply_http_config(&self, http_config: HttpConfig) -> HttpConfig {
        // Default implementation: no modifications
        // Providers can override to add custom headers, timeouts, etc.
        http_config
    }

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

    /// Check if provider supports image generation
    fn supports_image_generation(&self) -> bool {
        false
    }

    /// Transform image generation request parameters
    fn transform_image_request(
        &self,
        _request: &mut crate::types::ImageGenerationRequest,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    /// Get supported image sizes
    fn get_supported_image_sizes(&self) -> Vec<String> {
        vec!["1024x1024".to_string()]
    }

    /// Get supported image formats
    fn get_supported_image_formats(&self) -> Vec<String> {
        vec!["url".to_string()]
    }

    /// Check if provider supports image editing
    fn supports_image_editing(&self) -> bool {
        false
    }

    /// Check if provider supports image variations
    fn supports_image_variations(&self) -> bool {
        false
    }

    /// Get API route for a given request type
    ///
    /// Default mappings follow common OpenAI-compatible conventions and can be
    /// overridden by providers with divergent paths (e.g., rerank endpoints).
    fn route_for(&self, req: super::types::RequestType) -> &'static str {
        match req {
            super::types::RequestType::Chat => "chat/completions",
            super::types::RequestType::Embedding => "embeddings",
            super::types::RequestType::ImageGeneration => "images/generations",
            super::types::RequestType::Rerank => "rerank",
        }
    }
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
