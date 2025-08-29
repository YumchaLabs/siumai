//! Shared types for OpenAI-compatible providers

use std::collections::HashMap;

/// Request type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestType {
    Chat,
    Embedding,
    Rerank,
    ImageGeneration,
}

/// Field mappings configuration for different providers
///
/// This allows different providers to use different field names for the same concepts.
/// For example, SiliconFlow's DeepSeek models use "reasoning_content" instead of "thinking".
#[derive(Debug, Clone)]
pub struct FieldMappings {
    /// Fields that contain thinking/reasoning content (in priority order)
    pub thinking_fields: Vec<&'static str>,
    /// Field that contains regular content
    pub content_field: &'static str,
    /// Field that contains tool calls
    pub tool_calls_field: &'static str,
    /// Field that contains role information
    pub role_field: &'static str,
}

impl Default for FieldMappings {
    fn default() -> Self {
        Self {
            thinking_fields: vec!["thinking"],
            content_field: "content",
            tool_calls_field: "tool_calls",
            role_field: "role",
        }
    }
}

impl FieldMappings {
    /// Create field mappings for DeepSeek models (supports reasoning_content)
    pub fn deepseek() -> Self {
        Self {
            thinking_fields: vec!["reasoning_content", "thinking"],
            content_field: "content",
            tool_calls_field: "tool_calls",
            role_field: "role",
        }
    }

    /// Create field mappings for standard OpenAI format
    pub fn standard() -> Self {
        Self::default()
    }
}

/// Model-specific configuration
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Whether this model requires streaming output
    pub force_streaming: bool,
    /// Whether this model supports thinking/reasoning
    pub supports_thinking: bool,
    /// Whether this model supports tool calls
    pub supports_tools: bool,
    /// Whether this model supports vision
    pub supports_vision: bool,
    /// Special parameter mappings for this model
    pub parameter_mappings: HashMap<String, String>,
    /// Maximum tokens for this model
    pub max_tokens: Option<u32>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            force_streaming: false,
            supports_thinking: false,
            supports_tools: true,
            supports_vision: false,
            parameter_mappings: HashMap::new(),
            max_tokens: None,
        }
    }
}

impl ModelConfig {
    /// Create configuration for DeepSeek models
    pub fn deepseek() -> Self {
        let mut parameter_mappings = std::collections::HashMap::new();
        parameter_mappings.insert(
            "thinking_budget".to_string(),
            "reasoning_effort".to_string(),
        );

        Self {
            supports_thinking: true,
            max_tokens: Some(8192),
            parameter_mappings,
            ..Default::default()
        }
    }

    /// Create configuration for Qwen reasoning models
    pub fn qwen_reasoning() -> Self {
        Self {
            force_streaming: true, // Qwen reasoning models only support streaming
            supports_thinking: true,
            ..Default::default()
        }
    }

    /// Create configuration for standard chat models
    pub fn standard_chat() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_mappings_default() {
        let mappings = FieldMappings::default();
        assert_eq!(mappings.thinking_fields, vec!["thinking"]);
        assert_eq!(mappings.content_field, "content");
    }

    #[test]
    fn test_field_mappings_deepseek() {
        let mappings = FieldMappings::deepseek();
        assert_eq!(
            mappings.thinking_fields,
            vec!["reasoning_content", "thinking"]
        );
        assert_eq!(mappings.content_field, "content");
    }

    #[test]
    fn test_model_config_deepseek() {
        let config = ModelConfig::deepseek();
        assert!(config.supports_thinking);
        assert_eq!(config.max_tokens, Some(8192));
        assert!(config.parameter_mappings.contains_key("thinking_budget"));
    }
}
