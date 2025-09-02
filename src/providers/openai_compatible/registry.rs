//! OpenAI-Compatible Provider Registry
//!
//! This module provides a configuration-driven registry for OpenAI-compatible providers,
//! inspired by Cherry Studio's design. Instead of creating separate builders for each
//! provider, we use a unified configuration system.

use super::adapter::ProviderAdapter;
use super::types::{FieldAccessor, FieldMappings, JsonFieldAccessor};
use crate::error::LlmError;
use crate::traits::ProviderCapabilities;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Provider configuration entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Provider identifier
    pub id: String,
    /// Display name
    pub name: String,
    /// Base URL for API
    pub base_url: String,
    /// Field mappings for response parsing
    pub field_mappings: ProviderFieldMappings,
    /// Supported capabilities
    pub capabilities: Vec<String>,
    /// Default model (optional)
    pub default_model: Option<String>,
    /// Whether this provider supports reasoning/thinking
    pub supports_reasoning: bool,
}

/// Field mappings configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderFieldMappings {
    /// Fields that contain thinking/reasoning content (in priority order)
    pub thinking_fields: Vec<String>,
    /// Field that contains regular content
    pub content_field: String,
    /// Field that contains tool calls
    pub tool_calls_field: String,
    /// Field that contains role information
    pub role_field: String,
}

impl Default for ProviderFieldMappings {
    fn default() -> Self {
        Self {
            thinking_fields: vec!["thinking".to_string()],
            content_field: "content".to_string(),
            tool_calls_field: "tool_calls".to_string(),
            role_field: "role".to_string(),
        }
    }
}

/// Generic adapter that uses configuration
#[derive(Debug, Clone)]
pub struct ConfigurableAdapter {
    config: ProviderConfig,
}

impl ConfigurableAdapter {
    pub fn new(config: ProviderConfig) -> Self {
        Self { config }
    }
}

impl ProviderAdapter for ConfigurableAdapter {
    fn provider_id(&self) -> &'static str {
        // Note: This is a limitation of the current trait design
        // In practice, we might want to change the trait to return &str
        Box::leak(self.config.id.clone().into_boxed_str())
    }

    fn transform_request_params(
        &self,
        _params: &mut serde_json::Value,
        _model: &str,
        _request_type: super::types::RequestType,
    ) -> Result<(), LlmError> {
        // Most OpenAI-compatible providers don't need parameter transformation
        Ok(())
    }

    fn get_field_mappings(&self, _model: &str) -> FieldMappings {
        let config_mappings = &self.config.field_mappings;
        FieldMappings {
            thinking_fields: config_mappings
                .thinking_fields
                .iter()
                .map(|s| Box::leak(s.clone().into_boxed_str()) as &'static str)
                .collect(),
            content_field: Box::leak(config_mappings.content_field.clone().into_boxed_str()),
            tool_calls_field: Box::leak(config_mappings.tool_calls_field.clone().into_boxed_str()),
            role_field: Box::leak(config_mappings.role_field.clone().into_boxed_str()),
        }
    }

    fn get_model_config(&self, _model: &str) -> super::types::ModelConfig {
        super::types::ModelConfig {
            supports_thinking: self.config.supports_reasoning,
            ..Default::default()
        }
    }

    fn get_field_accessor(&self) -> Box<dyn FieldAccessor> {
        Box::new(JsonFieldAccessor::default())
    }

    fn capabilities(&self) -> ProviderCapabilities {
        let mut caps = ProviderCapabilities::new().with_chat().with_streaming();

        if self.config.capabilities.contains(&"tools".to_string()) {
            caps = caps.with_tools();
        }
        if self.config.capabilities.contains(&"vision".to_string()) {
            caps = caps.with_vision();
        }
        if self.config.supports_reasoning {
            caps = caps.with_custom_feature("reasoning", true);
        }

        caps
    }

    fn base_url(&self) -> &str {
        &self.config.base_url
    }

    fn clone_adapter(&self) -> Box<dyn ProviderAdapter> {
        Box::new(self.clone())
    }
}

/// Provider registry with built-in configurations
pub struct ProviderRegistry {
    providers: HashMap<String, ProviderConfig>,
}

impl ProviderRegistry {
    /// Create a new registry with built-in providers
    pub fn new() -> Self {
        let mut registry = Self {
            providers: HashMap::new(),
        };

        // Register built-in providers
        registry.register_builtin_providers();
        registry
    }

    /// Register built-in providers (inspired by Cherry Studio's config)
    fn register_builtin_providers(&mut self) {
        // DeepSeek
        self.providers.insert(
            "deepseek".to_string(),
            ProviderConfig {
                id: "deepseek".to_string(),
                name: "DeepSeek".to_string(),
                base_url: "https://api.deepseek.com/v1".to_string(),
                field_mappings: ProviderFieldMappings {
                    thinking_fields: vec!["reasoning_content".to_string(), "thinking".to_string()],
                    ..Default::default()
                },
                capabilities: vec!["tools".to_string(), "vision".to_string()],
                default_model: Some("deepseek-chat".to_string()),
                supports_reasoning: true,
            },
        );

        // SiliconFlow
        self.providers.insert(
            "siliconflow".to_string(),
            ProviderConfig {
                id: "siliconflow".to_string(),
                name: "SiliconFlow".to_string(),
                base_url: "https://api.siliconflow.cn/v1".to_string(),
                field_mappings: ProviderFieldMappings {
                    thinking_fields: vec!["reasoning_content".to_string(), "thinking".to_string()],
                    ..Default::default()
                },
                capabilities: vec!["tools".to_string(), "vision".to_string()],
                default_model: Some("deepseek-chat".to_string()),
                supports_reasoning: true,
            },
        );

        // OpenRouter
        self.providers.insert(
            "openrouter".to_string(),
            ProviderConfig {
                id: "openrouter".to_string(),
                name: "OpenRouter".to_string(),
                base_url: "https://openrouter.ai/api/v1".to_string(),
                field_mappings: ProviderFieldMappings::default(),
                capabilities: vec!["tools".to_string(), "vision".to_string()],
                default_model: None,
                supports_reasoning: false,
            },
        );

        // Add more providers here...
    }

    /// Get provider configuration by ID
    pub fn get_provider(&self, id: &str) -> Option<&ProviderConfig> {
        self.providers.get(id)
    }

    /// Create adapter for provider
    pub fn create_adapter(&self, provider_id: &str) -> Result<Arc<dyn ProviderAdapter>, LlmError> {
        let config = self.get_provider(provider_id).ok_or_else(|| {
            LlmError::ConfigurationError(format!("Unknown provider: {}", provider_id))
        })?;

        Ok(Arc::new(ConfigurableAdapter::new(config.clone())))
    }

    /// Register a custom provider
    pub fn register_provider(&mut self, config: ProviderConfig) {
        self.providers.insert(config.id.clone(), config);
    }

    /// List all available providers
    pub fn list_providers(&self) -> Vec<&str> {
        self.providers.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// Global provider registry instance
lazy_static::lazy_static! {
    pub static ref PROVIDER_REGISTRY: std::sync::Mutex<ProviderRegistry> =
        std::sync::Mutex::new(ProviderRegistry::new());
}

/// Convenience function to get an adapter for a provider
pub fn get_provider_adapter(provider_id: &str) -> Result<Arc<dyn ProviderAdapter>, LlmError> {
    PROVIDER_REGISTRY
        .lock()
        .map_err(|_| LlmError::ConfigurationError("Failed to lock provider registry".to_string()))?
        .create_adapter(provider_id)
}
