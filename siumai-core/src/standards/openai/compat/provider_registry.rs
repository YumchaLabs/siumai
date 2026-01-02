//! Configuration-driven adapter helpers for OpenAI-compatible providers.
//!
//! This module is intentionally provider-agnostic: it lives in `siumai-core`
//! and can be reused by any crate that needs a lightweight, configuration-based
//! `ProviderAdapter` implementation.

use super::adapter::ProviderAdapter;
use super::types::{FieldAccessor, FieldMappings, JsonFieldAccessor, RequestType};
use crate::error::LlmError;
use crate::traits::ProviderCapabilities;
use serde::{Deserialize, Serialize};

/// Provider configuration entry (best-effort hints, not exhaustive).
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
    /// Environment variable name to read the API key from (optional).
    ///
    /// When not set, callers typically fall back to `${PROVIDER_ID}_API_KEY`.
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Additional environment variable names to try (fallbacks).
    #[serde(default)]
    pub api_key_env_aliases: Vec<String>,
}

/// Field mappings configuration (string-based, suitable for deserialization).
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

/// Generic adapter that uses configuration.
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
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Owned(self.config.id.clone())
    }

    fn transform_request_params(
        &self,
        params: &mut serde_json::Value,
        _model: &str,
        _request_type: RequestType,
    ) -> Result<(), LlmError> {
        // Most OpenAI-compatible providers don't need parameter transformation.
        //
        // For a small set of vendors, we centralize well-known OpenAI-compat quirks here
        // to keep provider crates thin and avoid copy/paste drift.
        match self.config.id.as_str() {
            // Groq and xAI (Grok) are OpenAI-compatible but have a few request differences:
            // - "developer" role is not supported (treat as "system")
            // - "stream_options" is not supported (omit it)
            // - "max_completion_tokens" is not supported (use "max_tokens")
            "groq" | "xai" => {
                if let Some(msgs) = params.get_mut("messages").and_then(|v| v.as_array_mut()) {
                    for m in msgs {
                        if m.get("role").and_then(|v| v.as_str()) == Some("developer") {
                            m["role"] = serde_json::Value::String("system".to_string());
                        }
                    }
                }

                if let Some(obj) = params.as_object_mut() {
                    obj.remove("stream_options");
                    if let Some(v) = obj.remove("max_completion_tokens") {
                        obj.entry("max_tokens".to_string()).or_insert(v);
                    }
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn get_field_mappings(&self, _model: &str) -> FieldMappings {
        let config_mappings = &self.config.field_mappings;
        FieldMappings {
            thinking_fields: config_mappings
                .thinking_fields
                .iter()
                .map(|s| std::borrow::Cow::Owned(s.clone()))
                .collect(),
            content_field: std::borrow::Cow::Owned(config_mappings.content_field.clone()),
            tool_calls_field: std::borrow::Cow::Owned(config_mappings.tool_calls_field.clone()),
            role_field: std::borrow::Cow::Owned(config_mappings.role_field.clone()),
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
        if self.config.capabilities.contains(&"rerank".to_string()) {
            caps = caps.with_rerank();
        }
        if self
            .config
            .capabilities
            .contains(&"image_generation".to_string())
        {
            caps = caps.with_image_generation();
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

    fn supports_image_generation(&self) -> bool {
        self.config
            .capabilities
            .contains(&"image_generation".to_string())
    }

    fn transform_image_request(
        &self,
        _request: &mut crate::types::ImageGenerationRequest,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    fn get_supported_image_sizes(&self) -> Vec<String> {
        vec![
            "256x256".to_string(),
            "512x512".to_string(),
            "1024x1024".to_string(),
            "1024x1792".to_string(),
            "1792x1024".to_string(),
        ]
    }

    fn get_supported_image_formats(&self) -> Vec<String> {
        vec!["url".to_string(), "b64_json".to_string()]
    }

    fn supports_image_editing(&self) -> bool {
        self.supports_image_generation()
    }

    fn supports_image_variations(&self) -> bool {
        self.supports_image_generation()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configurable_adapter_capabilities_include_image_generation_and_rerank() {
        let cfg = ProviderConfig {
            id: "test".to_string(),
            name: "Test".to_string(),
            base_url: "https://example.invalid/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "tools".to_string(),
                "vision".to_string(),
                "embedding".to_string(),
                "rerank".to_string(),
                "image_generation".to_string(),
            ],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        };

        let adapter = ConfigurableAdapter::new(cfg);
        let caps = adapter.capabilities();
        assert!(caps.supports("tools"));
        assert!(caps.supports("vision"));
        assert!(caps.supports("embedding"));
        assert!(caps.supports("rerank"));
        assert!(caps.supports("image_generation"));
        assert!(caps.supports("reasoning"));
        assert!(adapter.supports_image_generation());
    }

    #[test]
    fn configurable_adapter_transforms_groq_compat_quirks() {
        let cfg = ProviderConfig {
            id: "groq".to_string(),
            name: "Groq".to_string(),
            base_url: "https://api.groq.com/openai/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["chat".to_string(), "streaming".to_string()],
            default_model: Some("llama-3.3-70b-versatile".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        };
        let adapter = ConfigurableAdapter::new(cfg);

        let mut params = serde_json::json!({
            "model": "llama-3.3-70b-versatile",
            "messages": [
                { "role": "developer", "content": "dev" },
                { "role": "user", "content": "hi" }
            ],
            "stream_options": { "include_usage": true },
            "max_completion_tokens": 123
        });

        adapter
            .transform_request_params(&mut params, "llama-3.3-70b-versatile", RequestType::Chat)
            .expect("transform ok");

        assert_eq!(params["messages"][0]["role"], "system");
        assert!(params.get("stream_options").is_none());
        assert_eq!(params["max_tokens"], 123);
    }
}
