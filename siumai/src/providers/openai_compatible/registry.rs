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
        Box::new(JsonFieldAccessor)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        let mut caps = ProviderCapabilities::new().with_chat().with_streaming();

        if self.config.capabilities.contains(&"tools".to_string()) {
            caps = caps.with_tools();
        }
        if self.config.capabilities.contains(&"vision".to_string()) {
            caps = caps.with_vision();
        }
        if self.config.capabilities.contains(&"embedding".to_string()) {
            caps = caps.with_embedding();
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

    /// Check if provider supports image generation
    fn supports_image_generation(&self) -> bool {
        self.config
            .capabilities
            .contains(&"image_generation".to_string())
    }

    /// Transform image generation request parameters
    fn transform_image_request(
        &self,
        _request: &mut crate::types::ImageGenerationRequest,
    ) -> Result<(), LlmError> {
        // Most OpenAI-compatible providers use standard format
        Ok(())
    }

    /// Get supported image sizes
    fn get_supported_image_sizes(&self) -> Vec<String> {
        // Standard sizes supported by most providers
        vec![
            "256x256".to_string(),
            "512x512".to_string(),
            "1024x1024".to_string(),
            "1024x1792".to_string(),
            "1792x1024".to_string(),
        ]
    }

    /// Get supported image formats
    fn get_supported_image_formats(&self) -> Vec<String> {
        vec!["url".to_string(), "b64_json".to_string()]
    }

    /// Check if provider supports image editing
    fn supports_image_editing(&self) -> bool {
        // Most OpenAI-compatible providers support basic image editing
        self.supports_image_generation()
    }

    /// Check if provider supports image variations
    fn supports_image_variations(&self) -> bool {
        // Most OpenAI-compatible providers support image variations
        self.supports_image_generation()
    }
}
