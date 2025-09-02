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

/// Dynamic field accessor for flexible response parsing
///
/// This provides a more configurable approach similar to Cherry Studio's
/// response transformation system.
pub trait FieldAccessor {
    /// Extract a field value from a JSON object using a field path
    fn get_field_value(&self, json: &serde_json::Value, field_path: &str) -> Option<String>;

    /// Extract thinking content using configured field mappings
    fn extract_thinking_content(
        &self,
        json: &serde_json::Value,
        mappings: &FieldMappings,
    ) -> Option<String>;

    /// Extract regular content
    fn extract_content(&self, json: &serde_json::Value, mappings: &FieldMappings)
    -> Option<String>;
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

/// Advanced field accessor with full JSON path support
#[derive(Debug, Clone, Default)]
pub struct JsonFieldAccessor;

impl JsonFieldAccessor {
    /// Get nested field value with support for complex paths
    fn get_nested_value<'a>(
        &self,
        json: &'a serde_json::Value,
        path: &str,
    ) -> Option<&'a serde_json::Value> {
        let parts: Vec<&str> = path.split('.').collect();
        let mut current = json;

        for part in parts {
            if part.is_empty() {
                continue;
            }

            // Handle array indices like [0] or just 0
            if let Ok(index) = part.parse::<usize>() {
                current = current.get(index)?;
            } else if part.starts_with('[') && part.ends_with(']') {
                let index_str = &part[1..part.len() - 1];
                let index = index_str.parse::<usize>().ok()?;
                current = current.get(index)?;
            } else {
                current = current.get(part)?;
            }
        }

        Some(current)
    }

    /// Extract string value from various JSON types
    fn extract_string_value(&self, value: &serde_json::Value) -> Option<String> {
        match value {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Number(n) => Some(n.to_string()),
            serde_json::Value::Bool(b) => Some(b.to_string()),
            _ => None,
        }
    }
}

impl FieldAccessor for JsonFieldAccessor {
    fn get_field_value(&self, json: &serde_json::Value, field_path: &str) -> Option<String> {
        self.get_nested_value(json, field_path)
            .and_then(|v| self.extract_string_value(v))
    }

    fn extract_thinking_content(
        &self,
        json: &serde_json::Value,
        mappings: &FieldMappings,
    ) -> Option<String> {
        // Try each thinking field in priority order
        for field_name in &mappings.thinking_fields {
            // Define possible paths for thinking content
            let paths = vec![
                format!("choices.0.delta.{}", field_name),
                format!("choices.0.message.{}", field_name),
                format!("delta.{}", field_name),
                format!("message.{}", field_name),
                field_name.to_string(), // Direct field access
            ];

            for path in paths {
                if let Some(value) = self.get_field_value(json, &path) {
                    if !value.trim().is_empty() {
                        return Some(value);
                    }
                }
            }
        }

        None
    }

    fn extract_content(
        &self,
        json: &serde_json::Value,
        mappings: &FieldMappings,
    ) -> Option<String> {
        let content_field = mappings.content_field;

        // Define possible paths for content
        let paths = vec![
            format!("choices.0.delta.{}", content_field),
            format!("choices.0.message.{}", content_field),
            format!("delta.{}", content_field),
            format!("message.{}", content_field),
            content_field.to_string(), // Direct field access
        ];

        for path in paths {
            if let Some(value) = self.get_field_value(json, &path) {
                if !value.trim().is_empty() {
                    return Some(value);
                }
            }
        }

        None
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
