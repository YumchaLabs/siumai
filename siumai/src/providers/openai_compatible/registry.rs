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
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Owned(self.config.id.clone())
    }

    fn transform_request_params(
        &self,
        params: &mut serde_json::Value,
        _model: &str,
        request_type: super::types::RequestType,
    ) -> Result<(), LlmError> {
        // Most OpenAI-compatible providers don't need parameter transformation.
        // Doubao is a special case: it uses a nested `thinking` object instead of
        // a flat `enable_thinking` flag. We convert the unified flag here.
        if matches!(request_type, super::types::RequestType::Chat)
            && self.config.id.eq_ignore_ascii_case("doubao")
        {
            if let Some(obj) = params.as_object_mut() {
                if let Some(enable_val) = obj.remove("enable_thinking") {
                    if let Some(flag) = enable_val.as_bool() {
                        // Do not override an explicit thinking block set by the user.
                        if !obj.contains_key("thinking") {
                            let thinking = if flag {
                                serde_json::json!({ "type": "enabled" })
                            } else {
                                serde_json::json!({ "type": "disabled" })
                            };
                            obj.insert("thinking".to_string(), thinking);
                        }
                    }
                }
            }
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

    fn get_model_config(&self, model: &str) -> super::types::ModelConfig {
        use crate::providers::openai_compatible::types::ModelConfig;

        // Provider-level flag: if the provider does not support reasoning at all,
        // fall back to a standard chat config.
        if !self.config.supports_reasoning {
            return ModelConfig::standard_chat();
        }

        let provider_id = self.config.id.to_lowercase();
        let model_id = model.to_lowercase();

        // DeepSeek native provider
        if provider_id == "deepseek" {
            // DeepSeek v3 / R1 models expose reasoning_content and support thinking.
            if model_id.contains("deepseek-v3")
                || model_id.contains("deepseek-r1")
                || model_id.contains("deepseek-chat")
                || model_id.contains("deepseek-reasoner")
            {
                return ModelConfig::deepseek();
            }
        }

        // SiliconFlow hosting DeepSeek / Qwen models
        if provider_id == "siliconflow" {
            // DeepSeek models on SiliconFlow
            if model_id.contains("deepseek-v3")
                || model_id.contains("deepseek-r1")
                || model_id.contains("deepseek-v2.5")
                || model_id.contains("deepseek-vl2")
            {
                return ModelConfig::deepseek();
            }

            // Qwen reasoning-style models hosted on SiliconFlow.
            // Heuristic: treat Qwen3/QwQ/QVQ families as supporting thinking.
            if model_id.contains("qwen3")
                || model_id.contains("qwq")
                || model_id.contains("qvq")
                || model_id.contains("qwen2.5")
            {
                return ModelConfig::qwen_reasoning();
            }
        }

        // Qwen DashScope provider (compatible-mode)
        if provider_id == "qwen" {
            // Align with Qwen docs / Vercel examples:
            // - "qwen-plus", "qwen-max", "qwen3-max" 等模型具备推理能力
            // - QwQ / QVQ 系列是专门的推理模型
            if model_id.starts_with("qwen-plus")
                || model_id.starts_with("qwen-max")
                || model_id.starts_with("qwen3-max")
                || model_id.contains("qwq")
                || model_id.contains("qvq")
            {
                return ModelConfig::qwen_reasoning();
            }
        }

        // Default: provider-level reasoning support without model-specific tweaks
        ModelConfig {
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
            caps = caps.with_custom_feature("rerank", true);
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
