//! OpenAI-Compatible Provider Modules
//!
//! This module contains provider-specific implementations and configurations
//! for various OpenAI-compatible services.

use crate::error::LlmError;
use crate::traits::ProviderCapabilities;
use std::collections::HashMap;

pub mod models;

pub use models::*;

/// Trait for OpenAI-compatible providers
pub trait OpenAiCompatibleProvider: Send + Sync + 'static {
    /// Provider identifier (e.g., "deepseek", "openrouter")
    const PROVIDER_ID: &'static str;

    /// Display name for the provider
    const DISPLAY_NAME: &'static str;

    /// Provider description
    const DESCRIPTION: &'static str;

    /// Default base URL for the provider
    const DEFAULT_BASE_URL: &'static str;

    /// Default model for the provider
    const DEFAULT_MODEL: &'static str;

    /// Validate provider-specific configuration
    fn validate_config(config: &super::config::OpenAiCompatibleConfig) -> Result<(), LlmError>;

    /// Transform provider-specific parameters
    fn transform_params(params: &mut HashMap<String, serde_json::Value>) -> Result<(), LlmError>;

    /// Get supported capabilities for this provider
    fn supported_capabilities() -> ProviderCapabilities;
}

/// `DeepSeek` provider implementation
pub struct DeepSeekProvider;

impl OpenAiCompatibleProvider for DeepSeekProvider {
    const PROVIDER_ID: &'static str = "deepseek";
    const DISPLAY_NAME: &'static str = "DeepSeek";
    const DESCRIPTION: &'static str = "DeepSeek AI models with reasoning capabilities";
    const DEFAULT_BASE_URL: &'static str = "https://api.deepseek.com/v1";
    const DEFAULT_MODEL: &'static str = "deepseek-chat";

    fn validate_config(config: &super::config::OpenAiCompatibleConfig) -> Result<(), LlmError> {
        // DeepSeek-specific validation
        if config.api_key.is_empty() {
            return Err(LlmError::ConfigurationError(
                "DeepSeek API key is required".to_string(),
            ));
        }

        // Validate reasoning and coding parameters
        if let Some(reasoning) = config.provider_params.get("reasoning")
            && !reasoning.is_boolean()
        {
            return Err(LlmError::ConfigurationError(
                "reasoning parameter must be boolean".to_string(),
            ));
        }

        if let Some(coding) = config.provider_params.get("coding")
            && !coding.is_boolean()
        {
            return Err(LlmError::ConfigurationError(
                "coding parameter must be boolean".to_string(),
            ));
        }

        Ok(())
    }

    fn transform_params(params: &mut HashMap<String, serde_json::Value>) -> Result<(), LlmError> {
        // DeepSeek-specific parameter transformations

        // Handle reasoning mode - switch to reasoner model if enabled
        if let Some(reasoning) = params.get("reasoning")
            && reasoning.as_bool() == Some(true)
        {
            // This would be handled in the config transformation
            // The model switch happens in the config conversion
        }

        // Handle coding mode - switch to coder model if enabled
        if let Some(coding) = params.get("coding")
            && coding.as_bool() == Some(true)
        {
            // This would be handled in the config transformation
            // The model switch happens in the config conversion
        }

        // Handle thinking budget for reasoning models
        if let Some(budget) = params.get("thinking_budget")
            && let Some(budget_val) = budget.as_u64()
        {
            // Add to request parameters (this would be used in request building)
            params.insert(
                "max_reasoning_tokens".to_string(),
                serde_json::Value::Number(budget_val.into()),
            );
        }

        Ok(())
    }

    fn supported_capabilities() -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_custom_feature("reasoning", true)
            .with_custom_feature("coding", true)
    }
}

/// `OpenRouter` provider implementation
pub struct OpenRouterProvider;

impl OpenAiCompatibleProvider for OpenRouterProvider {
    const PROVIDER_ID: &'static str = "openrouter";
    const DISPLAY_NAME: &'static str = "OpenRouter";
    const DESCRIPTION: &'static str = "OpenRouter unified API for multiple LLM providers";
    const DEFAULT_BASE_URL: &'static str = "https://openrouter.ai/api/v1";
    const DEFAULT_MODEL: &'static str = "openai/gpt-4o";

    fn validate_config(config: &super::config::OpenAiCompatibleConfig) -> Result<(), LlmError> {
        // OpenRouter-specific validation
        if config.api_key.is_empty() {
            return Err(LlmError::ConfigurationError(
                "OpenRouter API key is required".to_string(),
            ));
        }

        // Validate site_url if provided
        if let Some(site_url) = config.provider_params.get("site_url")
            && let Some(url_str) = site_url.as_str()
            && !url_str.starts_with("http://")
            && !url_str.starts_with("https://")
        {
            return Err(LlmError::ConfigurationError(
                "site_url must be a valid HTTP/HTTPS URL".to_string(),
            ));
        }

        Ok(())
    }

    fn transform_params(params: &mut HashMap<String, serde_json::Value>) -> Result<(), LlmError> {
        // OpenRouter-specific parameter transformations

        // Transform site_url to HTTP-Referer header
        if let Some(site_url) = params.remove("site_url")
            && let Some(url_str) = site_url.as_str()
        {
            params.insert(
                "http_referer".to_string(),
                serde_json::Value::String(url_str.to_string()),
            );
        }

        // Transform app_name to X-Title header
        if let Some(app_name) = params.remove("app_name")
            && let Some(name_str) = app_name.as_str()
        {
            params.insert(
                "x_title".to_string(),
                serde_json::Value::String(name_str.to_string()),
            );
        }

        // Handle fallback models
        if let Some(fallback_models) = params.get("fallback_models")
            && let Some(models_array) = fallback_models.as_array()
        {
            let models: Vec<String> = models_array
                .iter()
                .filter_map(|v| v.as_str().map(std::string::ToString::to_string))
                .collect();
            params.insert(
                "fallback_models".to_string(),
                serde_json::Value::Array(
                    models.into_iter().map(serde_json::Value::String).collect(),
                ),
            );
        }

        Ok(())
    }

    fn supported_capabilities() -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_custom_feature("model_routing", true)
            .with_custom_feature("fallback_models", true)
    }
}

/// xAI provider implementation
pub struct XAIProvider;

impl OpenAiCompatibleProvider for XAIProvider {
    const PROVIDER_ID: &'static str = "xai";
    const DISPLAY_NAME: &'static str = "xAI";
    const DESCRIPTION: &'static str = "xAI Grok models";
    const DEFAULT_BASE_URL: &'static str = "https://api.x.ai/v1";
    const DEFAULT_MODEL: &'static str = "grok-3";

    fn validate_config(_config: &super::config::OpenAiCompatibleConfig) -> Result<(), LlmError> {
        // xAI-specific validation
        Ok(())
    }

    fn transform_params(_params: &mut HashMap<String, serde_json::Value>) -> Result<(), LlmError> {
        // xAI-specific parameter transformations
        Ok(())
    }

    fn supported_capabilities() -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_vision()
            .with_custom_feature("real_time_info", true)
    }
}

/// SiliconFlow provider implementation
#[derive(Debug, Clone)]
pub struct SiliconFlowProvider;

impl OpenAiCompatibleProvider for SiliconFlowProvider {
    const PROVIDER_ID: &'static str = "siliconflow";
    const DISPLAY_NAME: &'static str = "SiliconFlow";
    const DESCRIPTION: &'static str = "SiliconFlow AI models with reranking capabilities";
    const DEFAULT_BASE_URL: &'static str = "https://api.siliconflow.cn/v1";
    const DEFAULT_MODEL: &'static str = "deepseek-chat";

    fn validate_config(_config: &super::config::OpenAiCompatibleConfig) -> Result<(), LlmError> {
        // SiliconFlow-specific validation
        Ok(())
    }

    fn transform_params(params: &mut HashMap<String, serde_json::Value>) -> Result<(), LlmError> {
        // SiliconFlow-specific parameter transformations for image generation

        // Map OpenAI 'n' parameter to SiliconFlow 'batch_size'
        if let Some(n_value) = params.remove("n") {
            params.insert("batch_size".to_string(), n_value);
        }

        // Map OpenAI 'size' parameter to SiliconFlow 'image_size'
        if let Some(size_value) = params.remove("size") {
            params.insert("image_size".to_string(), size_value);
        }

        // Add SiliconFlow-specific default parameters for image generation
        if params.contains_key("prompt") && params.contains_key("model") {
            // Set default inference steps if not provided
            if !params.contains_key("num_inference_steps") {
                params.insert(
                    "num_inference_steps".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(20)),
                );
            }

            // Set default guidance scale if not provided
            if !params.contains_key("guidance_scale") {
                params.insert(
                    "guidance_scale".to_string(),
                    serde_json::Value::Number(serde_json::Number::from_f64(7.5).unwrap()),
                );
            }
        }

        Ok(())
    }

    fn supported_capabilities() -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_embedding()
            .with_custom_feature("rerank", true)
            .with_custom_feature("image_generation", true)
            .with_custom_feature("audio", true)
    }
}

/// Groq provider implementation
pub struct GroqProvider;

impl OpenAiCompatibleProvider for GroqProvider {
    const PROVIDER_ID: &'static str = "groq";
    const DISPLAY_NAME: &'static str = "Groq";
    const DESCRIPTION: &'static str = "Groq fast inference engine";
    const DEFAULT_BASE_URL: &'static str = "https://api.groq.com/openai/v1";
    const DEFAULT_MODEL: &'static str = "llama-3.1-70b-versatile";

    fn validate_config(_config: &super::config::OpenAiCompatibleConfig) -> Result<(), LlmError> {
        // Groq-specific validation
        Ok(())
    }

    fn transform_params(_params: &mut HashMap<String, serde_json::Value>) -> Result<(), LlmError> {
        // Groq-specific parameter transformations
        Ok(())
    }

    fn supported_capabilities() -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_custom_feature("fast_inference", true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::openai_compatible::config::OpenAiCompatibleConfig;

    #[test]
    fn test_deepseek_provider_validation() {
        // Test empty API key
        let config = OpenAiCompatibleConfig::new("deepseek".to_string(), "".to_string());
        assert!(DeepSeekProvider::validate_config(&config).is_err());

        // Test valid config
        let config = OpenAiCompatibleConfig::new("deepseek".to_string(), "test-key".to_string());
        assert!(DeepSeekProvider::validate_config(&config).is_ok());
    }

    #[test]
    fn test_deepseek_param_transformation() {
        let mut params = HashMap::new();
        params.insert("reasoning".to_string(), serde_json::Value::Bool(true));
        params.insert(
            "thinking_budget".to_string(),
            serde_json::Value::Number(1000.into()),
        );

        assert!(DeepSeekProvider::transform_params(&mut params).is_ok());
        assert!(params.contains_key("max_reasoning_tokens"));
    }

    #[test]
    fn test_openrouter_provider_validation() {
        // Test empty API key
        let config = OpenAiCompatibleConfig::new("openrouter".to_string(), "".to_string());
        assert!(OpenRouterProvider::validate_config(&config).is_err());

        // Test valid config
        let config = OpenAiCompatibleConfig::new("openrouter".to_string(), "test-key".to_string());
        assert!(OpenRouterProvider::validate_config(&config).is_ok());
    }

    #[test]
    fn test_openrouter_param_transformation() {
        let mut params = HashMap::new();
        params.insert(
            "site_url".to_string(),
            serde_json::Value::String("https://example.com".to_string()),
        );
        params.insert(
            "app_name".to_string(),
            serde_json::Value::String("Test App".to_string()),
        );

        assert!(OpenRouterProvider::transform_params(&mut params).is_ok());
        assert!(params.contains_key("http_referer"));
        assert!(params.contains_key("x_title"));
        assert!(!params.contains_key("site_url"));
        assert!(!params.contains_key("app_name"));
    }

    #[test]
    fn test_provider_capabilities() {
        let deepseek_caps = DeepSeekProvider::supported_capabilities();
        assert!(deepseek_caps.supports("chat"));
        assert!(deepseek_caps.supports("streaming"));
        assert!(deepseek_caps.supports("tools"));
        assert!(deepseek_caps.supports("reasoning"));
        assert!(deepseek_caps.supports("coding"));

        let openrouter_caps = OpenRouterProvider::supported_capabilities();
        assert!(openrouter_caps.supports("chat"));
        assert!(openrouter_caps.supports("streaming"));
        assert!(openrouter_caps.supports("vision"));
        assert!(openrouter_caps.supports("model_routing"));
        assert!(openrouter_caps.supports("fallback_models"));
    }
}
