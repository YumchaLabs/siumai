//! SiliconFlow Provider Adapter
//!
//! This module implements the adapter for SiliconFlow, handling their specific
//! response formats and parameter requirements, especially for DeepSeek models.

use crate::error::LlmError;
use crate::providers::openai_compatible::{
    adapter::ProviderAdapter,
    types::{FieldMappings, ModelConfig, RequestType},
};
use crate::traits::ProviderCapabilities;

/// SiliconFlow adapter
///
/// Handles SiliconFlow-specific adaptations, particularly for DeepSeek models
/// which use "reasoning_content" instead of "thinking" for reasoning output.
#[derive(Debug, Clone)]
pub struct SiliconFlowAdapter;

impl SiliconFlowAdapter {
    /// Check if a model is a DeepSeek hybrid inference model
    /// Based on Cherry Studio's isDeepSeekHybridInferenceModel function
    fn is_deepseek_hybrid_inference_model(model: &str) -> bool {
        let model_lower = model.to_lowercase();
        // Match DeepSeek V3.1 and related models
        let deepseek_v3_regex = regex::Regex::new(r"deepseek-v3(?:\.1|-1-\d+)?").unwrap();
        deepseek_v3_regex.is_match(&model_lower) || model_lower.contains("deepseek-chat-v3.1")
    }
}

impl ProviderAdapter for SiliconFlowAdapter {
    fn provider_id(&self) -> &'static str {
        "siliconflow"
    }

    fn transform_request_params(
        &self,
        params: &mut serde_json::Value,
        _model: &str,
        request_type: RequestType,
    ) -> Result<(), LlmError> {
        match request_type {
            RequestType::Chat => {
                // No special chat parameter transformation needed for SiliconFlow
                // The thinking functionality is handled at the library level
                // through ChatResponse.thinking field
            }
            RequestType::ImageGeneration => {
                // SiliconFlow image generation parameter mappings
                if let Some(n_value) = params.get("n") {
                    params["batch_size"] = n_value.clone();
                    if let Some(obj) = params.as_object_mut() {
                        obj.remove("n");
                    }
                }
                if let Some(size_value) = params.get("size") {
                    params["image_size"] = size_value.clone();
                    if let Some(obj) = params.as_object_mut() {
                        obj.remove("size");
                    }
                }
            }
            RequestType::Rerank => {
                // SiliconFlow rerank uses different parameter names
                if let Some(top_k) = params.get("top_k") {
                    params["top_n"] = top_k.clone();
                    if let Some(obj) = params.as_object_mut() {
                        obj.remove("top_k");
                    }
                }
            }
            RequestType::Embedding => {
                // Standard embedding parameters work fine
            }
        }

        Ok(())
    }

    fn get_field_mappings(&self, model: &str) -> FieldMappings {
        if Self::is_deepseek_hybrid_inference_model(model) {
            // DeepSeek models use reasoning_content field
            FieldMappings::deepseek()
        } else {
            // Other models use standard fields
            FieldMappings::standard()
        }
    }

    fn get_model_config(&self, model: &str) -> ModelConfig {
        if Self::is_deepseek_hybrid_inference_model(model) {
            ModelConfig::deepseek()
        } else if model.contains("qwen") && (model.contains("reasoning") || model.contains("qwq")) {
            ModelConfig::qwen_reasoning()
        } else {
            ModelConfig::standard_chat()
        }
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_embedding()
            .with_custom_feature("thinking", true)
            .with_custom_feature("rerank", true)
            .with_custom_feature("image_generation", true)
    }

    fn base_url(&self) -> &str {
        "https://api.siliconflow.cn"
    }

    fn validate_model(&self, model: &str) -> Result<(), LlmError> {
        // SiliconFlow supports a wide range of models
        // We could add specific validation here if needed
        if model.is_empty() {
            return Err(LlmError::ConfigurationError(
                "Model name cannot be empty".to_string(),
            ));
        }
        Ok(())
    }

    fn clone_adapter(&self) -> Box<dyn ProviderAdapter> {
        Box::new(self.clone())
    }

    fn supports_image_generation(&self) -> bool {
        true // SiliconFlow supports image generation
    }

    fn get_supported_image_sizes(&self) -> Vec<String> {
        vec![
            "512x512".to_string(),
            "768x768".to_string(),
            "1024x1024".to_string(),
            "1152x896".to_string(),
            "896x1152".to_string(),
            "1216x832".to_string(),
            "832x1216".to_string(),
            "1344x768".to_string(),
            "768x1344".to_string(),
            "1536x640".to_string(),
            "640x1536".to_string(),
        ]
    }

    fn get_supported_image_formats(&self) -> Vec<String> {
        vec!["url".to_string(), "b64_json".to_string()]
    }
}

/// SiliconFlow builder for creating clients
pub struct SiliconFlowBuilder {
    api_key: String,
    model: Option<String>,
    base_url: Option<String>,
}

impl SiliconFlowBuilder {
    /// Create a new SiliconFlow builder
    pub fn new() -> Self {
        Self {
            api_key: String::new(),
            model: None,
            base_url: None,
        }
    }

    /// Set the API key
    pub fn api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = api_key.into();
        self
    }

    /// Set the model
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set a custom base URL (optional)
    pub fn base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Build the SiliconFlow client
    pub async fn build(
        self,
    ) -> Result<crate::providers::openai_compatible::OpenAiCompatibleClient, LlmError> {
        if self.api_key.is_empty() {
            return Err(LlmError::ConfigurationError(
                "API key is required".to_string(),
            ));
        }

        let adapter = SiliconFlowAdapter;
        let base_url = self
            .base_url
            .unwrap_or_else(|| adapter.base_url().to_string());

        let config = crate::providers::openai_compatible::OpenAiCompatibleConfig::new(
            adapter.provider_id(),
            &self.api_key,
            &base_url,
            Box::new(adapter),
        );

        let config = if let Some(model) = self.model {
            config.with_model(&model)
        } else {
            config
        };

        crate::providers::openai_compatible::OpenAiCompatibleClient::new(config).await
    }
}

impl Default for SiliconFlowBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Re-export model constants from the existing models module
pub use super::super::providers::models::siliconflow::*;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_siliconflow_adapter_basic() {
        let adapter = SiliconFlowAdapter;
        assert_eq!(adapter.provider_id(), "siliconflow");
        assert_eq!(adapter.base_url(), "https://api.siliconflow.cn/v1");
    }

    #[test]
    fn test_deepseek_field_mappings() {
        let adapter = SiliconFlowAdapter;
        let mappings = adapter.get_field_mappings("deepseek-v3.1");
        assert_eq!(
            mappings.thinking_fields,
            vec!["reasoning_content", "thinking"]
        );
    }

    #[test]
    fn test_standard_field_mappings() {
        let adapter = SiliconFlowAdapter;
        let mappings = adapter.get_field_mappings("qwen-turbo");
        assert_eq!(mappings.thinking_fields, vec!["thinking"]);
    }

    #[test]
    fn test_chat_parameter_transform() {
        let adapter = SiliconFlowAdapter;
        let mut params = json!({
            "model": "deepseek-v3.1",
            "messages": [],
            "temperature": 0.7
        });

        adapter
            .transform_request_params(&mut params, "deepseek-v3.1", RequestType::Chat)
            .unwrap();

        // Chat parameters should remain unchanged
        assert_eq!(
            params.get("temperature").unwrap(),
            &serde_json::Value::Number(serde_json::Number::from_f64(0.7).unwrap())
        );
        assert_eq!(params.get("model").unwrap(), "deepseek-v3.1");
    }

    #[test]
    fn test_deepseek_v3_model_detection() {
        assert!(SiliconFlowAdapter::is_deepseek_hybrid_inference_model(
            "deepseek-v3"
        ));
        assert!(SiliconFlowAdapter::is_deepseek_hybrid_inference_model(
            "deepseek-v3.1"
        ));
        assert!(SiliconFlowAdapter::is_deepseek_hybrid_inference_model(
            "deepseek-chat-v3.1"
        ));
        assert!(SiliconFlowAdapter::is_deepseek_hybrid_inference_model(
            "DeepSeek-V3.1"
        ));
        assert!(!SiliconFlowAdapter::is_deepseek_hybrid_inference_model(
            "deepseek-chat"
        ));
        assert!(!SiliconFlowAdapter::is_deepseek_hybrid_inference_model(
            "qwen-turbo"
        ));
    }

    #[test]
    fn test_image_generation_parameter_transform() {
        let adapter = SiliconFlowAdapter;
        let mut params = json!({
            "model": "stable-diffusion",
            "prompt": "test",
            "n": 2,
            "size": "512x512"
        });

        adapter
            .transform_request_params(
                &mut params,
                "stable-diffusion",
                RequestType::ImageGeneration,
            )
            .unwrap();

        assert!(params.get("batch_size").is_some());
        assert!(params.get("image_size").is_some());
        assert!(params.get("n").is_none());
        assert!(params.get("size").is_none());
    }

    #[test]
    fn test_builder() {
        let builder = SiliconFlowBuilder::new()
            .api_key("test-key")
            .model("deepseek-chat");

        assert_eq!(builder.api_key, "test-key");
        assert_eq!(builder.model, Some("deepseek-chat".to_string()));
    }

    #[test]
    fn test_model_validation() {
        let adapter = SiliconFlowAdapter;
        assert!(adapter.validate_model("deepseek-chat").is_ok());
        assert!(adapter.validate_model("").is_err());
    }
}
