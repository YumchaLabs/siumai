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
use serde::{Deserialize, Serialize};

/// SiliconFlow-specific thinking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiliconFlowThinkingConfig {
    /// Enable thinking mode
    pub enable_thinking: bool,
    /// Thinking budget in tokens
    pub thinking_budget: u32,
}

impl SiliconFlowThinkingConfig {
    /// Create a new thinking configuration
    pub fn new(enable_thinking: bool, thinking_budget: u32) -> Self {
        Self {
            enable_thinking,
            thinking_budget: thinking_budget.clamp(128, 32768),
        }
    }

    /// Create enabled thinking configuration with default budget
    pub fn enabled() -> Self {
        Self::new(true, 4096)
    }

    /// Create enabled thinking configuration with custom budget
    pub fn with_budget(budget: u32) -> Self {
        Self::new(true, budget)
    }

    /// Create disabled thinking configuration
    pub fn disabled() -> Self {
        Self::new(false, 0)
    }
}

/// SiliconFlow adapter
///
/// Handles SiliconFlow-specific adaptations, particularly for DeepSeek models
/// which use "reasoning_content" instead of "thinking" for reasoning output.
#[derive(Debug, Clone)]
pub struct SiliconFlowAdapter {
    /// Thinking configuration for supported models
    pub thinking_config: Option<SiliconFlowThinkingConfig>,
}

impl Default for SiliconFlowAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl SiliconFlowAdapter {
    /// Create a new SiliconFlow adapter with default settings
    pub fn new() -> Self {
        Self {
            thinking_config: None,
        }
    }

    /// Create a new SiliconFlow adapter with thinking configuration
    pub fn with_thinking_config(thinking_config: SiliconFlowThinkingConfig) -> Self {
        Self {
            thinking_config: Some(thinking_config),
        }
    }

    /// Create a new SiliconFlow adapter with thinking enabled
    pub fn with_thinking_enabled(budget: Option<u32>) -> Self {
        let config = if let Some(budget) = budget {
            SiliconFlowThinkingConfig::with_budget(budget)
        } else {
            SiliconFlowThinkingConfig::enabled()
        };
        Self::with_thinking_config(config)
    }

    /// Create a new SiliconFlow adapter with thinking disabled
    pub fn with_thinking_disabled() -> Self {
        Self::with_thinking_config(SiliconFlowThinkingConfig::disabled())
    }

    /// Check if a model is a DeepSeek hybrid inference model
    /// Based on Cherry Studio's isDeepSeekHybridInferenceModel function
    fn is_deepseek_hybrid_inference_model(model: &str) -> bool {
        let model_lower = model.to_lowercase();
        // Match DeepSeek V3.1 and related models
        let deepseek_v3_regex = regex::Regex::new(r"deepseek-v3(?:\.1|-1-\d+)?").unwrap();
        deepseek_v3_regex.is_match(&model_lower) || model_lower.contains("deepseek-chat-v3.1")
    }

    /// Check if a model supports thinking parameters (enable_thinking, thinking_budget)
    /// Based on SiliconFlow documentation
    fn supports_thinking_parameters(&self, model: &str) -> bool {
        let model_lower = model.to_lowercase();

        // Debug: print model name for troubleshooting (only in debug builds)
        #[cfg(debug_assertions)]
        eprintln!(
            "🔍 Checking thinking support for model: '{}' (lowercase: '{}')",
            model, model_lower
        );

        // DeepSeek models (confirmed to support enable_thinking)
        if model_lower.contains("deepseek-v3.1") || model_lower.contains("deepseek-v3-1") {
            return true;
        }

        // DeepSeek R1 models (reasoning models)
        if model_lower.contains("deepseek-r1") {
            return true;
        }

        // Qwen 3 models (based on Cherry Studio's logic)
        // Note: 'instruct' and 'thinking' models do NOT support enable_thinking parameter
        if model_lower.contains("qwen3") {
            // Exclude instruct and thinking models (they don't support enable_thinking)
            if model_lower.contains("instruct") || model_lower.contains("thinking") {
                return false;
            }

            // Check for base Qwen 3 model patterns (without instruct/thinking suffixes)
            let qwen3_patterns = [
                "qwen3-8b",
                "qwen3-14b",
                "qwen3-32b",
                "qwen3-30b-a3b",
                "qwen3-235b-a22b",
            ];

            for &pattern in qwen3_patterns.iter() {
                if model_lower.contains(pattern) {
                    return true;
                }
            }
        }

        // Hunyuan models
        if model_lower.contains("hunyuan-a13b") {
            return true;
        }

        // GLM models
        if model_lower.contains("glm-4.5v") {
            return true;
        }

        // Be conservative - only enable for models we're confident about
        false
    }

    /// Get default thinking budget based on model type
    /// Following Cherry Studio's approach of different budgets for different models
    fn get_default_thinking_budget(&self, model: &str) -> u32 {
        let model_lower = model.to_lowercase();

        // Large models get more thinking budget
        if model_lower.contains("235b") || model_lower.contains("480b") {
            8192 // Large models: 8K tokens for complex reasoning
        } else if model_lower.contains("72b") || model_lower.contains("70b") {
            6144 // Medium-large models: 6K tokens
        } else if model_lower.contains("32b") || model_lower.contains("30b") {
            4096 // Medium models: 4K tokens (default)
        } else if model_lower.contains("14b") || model_lower.contains("8b") {
            2048 // Smaller models: 2K tokens
        } else if model_lower.contains("7b") {
            1024 // Small models: 1K tokens
        } else {
            4096 // Default: 4K tokens
        }
    }
}

impl ProviderAdapter for SiliconFlowAdapter {
    fn provider_id(&self) -> &'static str {
        "siliconflow"
    }

    fn transform_request_params(
        &self,
        params: &mut serde_json::Value,
        model: &str,
        request_type: RequestType,
    ) -> Result<(), LlmError> {
        match request_type {
            RequestType::Chat => {
                // Add SiliconFlow-specific thinking parameters for supported models
                if self.supports_thinking_parameters(model) {
                    // Use thinking configuration if provided, otherwise use defaults for thinking-capable models
                    let default_config = SiliconFlowThinkingConfig {
                        enable_thinking: true,
                        thinking_budget: self.get_default_thinking_budget(model),
                    };
                    let thinking_config = self.thinking_config.as_ref().unwrap_or(&default_config);

                    // Set enable_thinking parameter if not already set
                    if !params
                        .as_object()
                        .unwrap_or(&serde_json::Map::new())
                        .contains_key("enable_thinking")
                    {
                        params["enable_thinking"] =
                            serde_json::Value::Bool(thinking_config.enable_thinking);
                    }

                    // Set thinking_budget parameter if not already set and thinking is enabled
                    if thinking_config.enable_thinking
                        && !params
                            .as_object()
                            .unwrap_or(&serde_json::Map::new())
                            .contains_key("thinking_budget")
                    {
                        params["thinking_budget"] = serde_json::Value::Number(
                            serde_json::Number::from(thinking_config.thinking_budget),
                        );
                    }
                }
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
        "https://api.siliconflow.cn/v1"
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
    enable_thinking: Option<bool>,
    thinking_budget: Option<u32>,
}

impl SiliconFlowBuilder {
    /// Create a new SiliconFlow builder
    pub fn new() -> Self {
        Self {
            api_key: String::new(),
            model: None,
            base_url: None,
            enable_thinking: None,
            thinking_budget: None,
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

    /// Enable or disable thinking mode for supported models
    ///
    /// When enabled, models that support thinking (like DeepSeek V3.1, Qwen 3, etc.)
    /// will include their reasoning process in the response.
    ///
    /// # Arguments
    /// * `enable` - Whether to enable thinking mode
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::LlmBuilder;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = LlmBuilder::new()
    ///         .siliconflow()
    ///         .api_key("your-api-key")
    ///         .model("deepseek-ai/DeepSeek-V3.1")
    ///         .with_thinking(true)
    ///         .build()
    ///         .await?;
    /// #   Ok(())
    /// # }
    /// ```
    pub fn with_thinking(mut self, enable: bool) -> Self {
        self.enable_thinking = Some(enable);
        self
    }

    /// Set the thinking budget (maximum tokens for reasoning)
    ///
    /// This controls how many tokens the model can use for its internal reasoning process.
    /// Higher values allow for more detailed reasoning but consume more tokens.
    ///
    /// # Arguments
    /// * `budget` - Number of tokens (128-32768, default varies by model size)
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::LlmBuilder;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = LlmBuilder::new()
    ///         .siliconflow()
    ///         .api_key("your-api-key")
    ///         .model("deepseek-ai/DeepSeek-V3.1")
    ///         .with_thinking_budget(8192)  // 8K tokens for complex reasoning
    ///         .build()
    ///         .await?;
    /// #   Ok(())
    /// # }
    /// ```
    pub fn with_thinking_budget(mut self, budget: u32) -> Self {
        // Clamp to SiliconFlow's supported range
        let clamped_budget = budget.clamp(128, 32768);
        self.thinking_budget = Some(clamped_budget);
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

        let adapter = if let Some(enable) = self.enable_thinking {
            if enable {
                SiliconFlowAdapter::with_thinking_enabled(self.thinking_budget)
            } else {
                SiliconFlowAdapter::with_thinking_disabled()
            }
        } else if let Some(budget) = self.thinking_budget {
            SiliconFlowAdapter::with_thinking_enabled(Some(budget))
        } else {
            SiliconFlowAdapter::new()
        };
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
        let adapter = SiliconFlowAdapter::new();
        assert_eq!(adapter.provider_id(), "siliconflow");
        assert_eq!(adapter.base_url(), "https://api.siliconflow.cn/v1");
    }

    #[test]
    fn test_deepseek_field_mappings() {
        let adapter = SiliconFlowAdapter::new();
        let mappings = adapter.get_field_mappings("deepseek-v3.1");
        assert_eq!(
            mappings.thinking_fields,
            vec!["reasoning_content", "thinking"]
        );
    }

    #[test]
    fn test_standard_field_mappings() {
        let adapter = SiliconFlowAdapter::new();
        let mappings = adapter.get_field_mappings("qwen-turbo");
        assert_eq!(mappings.thinking_fields, vec!["thinking"]);
    }

    #[test]
    fn test_chat_parameter_transform() {
        let adapter = SiliconFlowAdapter::new();
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
        let adapter = SiliconFlowAdapter::new();
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
        let adapter = SiliconFlowAdapter::new();
        assert!(adapter.validate_model("deepseek-chat").is_ok());
        assert!(adapter.validate_model("").is_err());
    }

    #[test]
    fn test_thinking_settings() {
        // Test default settings
        let default_adapter = SiliconFlowAdapter::new();
        assert!(default_adapter.thinking_config.is_none());

        // Test enabled thinking
        let enabled_adapter = SiliconFlowAdapter::with_thinking_enabled(Some(2048));
        let config = enabled_adapter.thinking_config.unwrap();
        assert!(config.enable_thinking);
        assert_eq!(config.thinking_budget, 2048);

        // Test disabled thinking
        let disabled_adapter = SiliconFlowAdapter::with_thinking_disabled();
        let config = disabled_adapter.thinking_config.unwrap();
        assert!(!config.enable_thinking);
    }

    #[test]
    fn test_thinking_budget_calculation() {
        let adapter = SiliconFlowAdapter::new();

        // Test different model sizes
        assert_eq!(
            adapter.get_default_thinking_budget("qwen/qwen3-235b-a22b"),
            8192
        );
        assert_eq!(adapter.get_default_thinking_budget("qwen/qwen3-32b"), 4096);
        assert_eq!(adapter.get_default_thinking_budget("qwen/qwen3-7b"), 1024);
        assert_eq!(adapter.get_default_thinking_budget("unknown-model"), 4096);
    }

    #[test]
    fn test_thinking_parameter_injection() {
        let adapter = SiliconFlowAdapter::with_thinking_enabled(Some(6144));
        let mut params = json!({
            "model": "deepseek-ai/DeepSeek-V3.1",
            "messages": [],
            "temperature": 0.7
        });

        adapter
            .transform_request_params(&mut params, "deepseek-ai/DeepSeek-V3.1", RequestType::Chat)
            .unwrap();

        // Should have thinking parameters
        assert_eq!(params["enable_thinking"], json!(true));
        assert_eq!(params["thinking_budget"], json!(6144));
    }
}
