//! Provider Module
//!
//! Contains specific implementations for each LLM provider.

#[cfg(feature = "anthropic")]
pub mod anthropic;
#[cfg(feature = "google")]
pub mod gemini;
#[cfg(feature = "groq")]
pub mod groq;
#[cfg(feature = "ollama")]
pub mod ollama;
#[cfg(feature = "openai")]
pub mod openai;
#[cfg(feature = "openai")]
pub mod openai_compatible;
#[cfg(feature = "xai")]
pub mod xai;

// Provider builder methods and convenience functions
pub mod builders;
pub mod convenience;

// Re-export main types
#[cfg(feature = "anthropic")]
pub use anthropic::AnthropicClient;
#[cfg(feature = "google")]
pub use gemini::GeminiClient;
#[cfg(feature = "groq")]
pub use groq::GroqClient;
#[cfg(feature = "ollama")]
pub use ollama::OllamaClient;
#[cfg(feature = "openai")]
pub use openai::{
    ListResponsesQuery, OpenAiClient, OpenAiResponses, ResponseMetadata, ResponseStatus,
    ResponsesApiCapability,
};
// Note: OpenAI-compatible providers now use OpenAI client directly
// Model constants are still available through openai_compatible module
#[cfg(feature = "xai")]
pub use xai::XaiClient;

use crate::traits::ProviderCapabilities;
use crate::types::ProviderType;

/// Provider Information
#[derive(Debug, Clone)]
pub struct ProviderInfo {
    /// Provider type
    pub provider_type: ProviderType,
    /// Provider name
    pub name: &'static str,
    /// Description
    pub description: &'static str,
    /// Supported capabilities
    pub capabilities: ProviderCapabilities,
    /// Default base URL
    pub default_base_url: &'static str,
    /// Supported models
    pub supported_models: Vec<&'static str>,
}

/// Get information for all supported providers
pub fn get_supported_providers() -> Vec<ProviderInfo> {
    let mut providers = Vec::new();

    #[cfg(feature = "openai")]
    providers.push(ProviderInfo {
        provider_type: ProviderType::OpenAi,
        name: "OpenAI",
        description: "OpenAI GPT models including GPT-4, GPT-3.5, and specialized models",
        capabilities: ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_audio()
            .with_embedding()
            .with_custom_feature("structured_output", true)
            .with_custom_feature("batch_processing", true),
        default_base_url: "https://api.openai.com/v1",
        supported_models: {
            use crate::constants::openai;
            let mut models = Vec::new();
            models.extend_from_slice(openai::gpt_4o::ALL);
            models.extend_from_slice(openai::gpt_4_1::ALL);
            models.extend_from_slice(openai::gpt_4_5::ALL);
            models.extend_from_slice(openai::gpt_4_turbo::ALL);
            models.extend_from_slice(openai::gpt_4::ALL);
            models.extend_from_slice(openai::o1::ALL);
            models.extend_from_slice(openai::o3::ALL);
            models.extend_from_slice(openai::o4::ALL);
            models.extend_from_slice(openai::gpt_5::ALL);
            models.extend_from_slice(openai::gpt_3_5::ALL);
            models.extend_from_slice(openai::audio::ALL);
            models.extend_from_slice(openai::images::ALL);
            models.extend_from_slice(openai::embeddings::ALL);
            models.extend_from_slice(openai::moderation::ALL);
            models
        },
    });

    #[cfg(feature = "anthropic")]
    providers.push(ProviderInfo {
        provider_type: ProviderType::Anthropic,
        name: "Anthropic",
        description: "Anthropic Claude models with advanced reasoning capabilities",
        capabilities: ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_custom_feature("prompt_caching", true)
            .with_custom_feature("thinking_mode", true),
        default_base_url: "https://api.anthropic.com",
        supported_models: {
            use crate::constants::anthropic;
            let mut models = Vec::new();
            models.extend_from_slice(anthropic::claude_opus_4_1::ALL);
            models.extend_from_slice(anthropic::claude_opus_4::ALL);
            models.extend_from_slice(anthropic::claude_sonnet_4::ALL);
            models.extend_from_slice(anthropic::claude_sonnet_3_7::ALL);
            models.extend_from_slice(anthropic::claude_sonnet_3_5::ALL);
            models.extend_from_slice(anthropic::claude_haiku_3_5::ALL);
            models.extend_from_slice(anthropic::claude_haiku_3::ALL);
            models.extend_from_slice(anthropic::claude_opus_3::ALL);
            models.extend_from_slice(anthropic::claude_sonnet_3::ALL);
            models
        },
    });

    #[cfg(feature = "google")]
    providers.push(ProviderInfo {
        provider_type: ProviderType::Gemini,
        name: "Google Gemini",
        description: "Google Gemini models with multimodal capabilities and code execution",
        capabilities: ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_custom_feature("code_execution", true)
            .with_custom_feature("thinking_mode", true)
            .with_custom_feature("safety_settings", true)
            .with_custom_feature("cached_content", true)
            .with_custom_feature("json_schema", true),
        default_base_url: "https://generativelanguage.googleapis.com/v1beta",
        supported_models: {
            use crate::constants::gemini;
            let mut models = Vec::new();
            models.extend_from_slice(gemini::gemini_2_5_pro::ALL);
            models.extend_from_slice(gemini::gemini_2_5_flash::ALL);
            models.extend_from_slice(gemini::gemini_2_5_flash_lite::ALL);
            models.extend_from_slice(gemini::gemini_2_0_flash::ALL);
            models.extend_from_slice(gemini::gemini_2_0_flash_lite::ALL);
            models.extend_from_slice(gemini::gemini_1_5_flash::ALL);
            models.extend_from_slice(gemini::gemini_1_5_flash_8b::ALL);
            models.extend_from_slice(gemini::gemini_1_5_pro::ALL);
            models
        },
    });

    #[cfg(feature = "ollama")]
    providers.push(ProviderInfo {
        provider_type: ProviderType::Ollama,
        name: "Ollama",
        description: "Local Ollama models with full control and privacy",
        capabilities: ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_embedding()
            .with_custom_feature("completion", true)
            .with_custom_feature("model_management", true)
            .with_custom_feature("local_models", true),
        default_base_url: "http://localhost:11434",
        supported_models: vec![
            "llama3.2:latest",
            "llama3.2:3b",
            "llama3.2:1b",
            "llama3.1:latest",
            "llama3.1:8b",
            "llama3.1:70b",
            "mistral:latest",
            "mistral:7b",
            "codellama:latest",
            "codellama:7b",
            "codellama:13b",
            "codellama:34b",
            "phi3:latest",
            "phi3:mini",
            "phi3:medium",
            "gemma:latest",
            "gemma:2b",
            "gemma:7b",
            "qwen2:latest",
            "qwen2:0.5b",
            "qwen2:1.5b",
            "qwen2:7b",
            "qwen2:72b",
            "deepseek-coder:latest",
            "deepseek-coder:6.7b",
            "deepseek-coder:33b",
            "nomic-embed-text:latest",
            "all-minilm:latest",
        ],
    });

    #[cfg(feature = "xai")]
    providers.push(ProviderInfo {
        provider_type: ProviderType::XAI,
        name: "xAI",
        description: "xAI Grok models with advanced reasoning capabilities",
        capabilities: ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_custom_feature("reasoning", true)
            .with_custom_feature("deferred_completion", true)
            .with_custom_feature("structured_outputs", true),
        default_base_url: "https://api.x.ai/v1",
        supported_models: crate::providers::xai::models::all_models(),
    });

    #[cfg(feature = "groq")]
    providers.push(ProviderInfo {
        provider_type: ProviderType::Groq,
        name: "Groq",
        description: "Groq models with ultra-fast inference",
        capabilities: ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_custom_feature("fast_inference", true),
        default_base_url: "https://api.groq.com/openai/v1",
        supported_models: crate::providers::groq::models::all_models(),
    });

    providers
}

/// Get provider information by provider type
pub fn get_provider_info(provider_type: &ProviderType) -> Option<ProviderInfo> {
    get_supported_providers()
        .into_iter()
        .find(|info| &info.provider_type == provider_type)
}

/// Check if a model is supported by the provider
pub fn is_model_supported(provider_type: &ProviderType, model: &str) -> bool {
    if let Some(info) = get_provider_info(provider_type) {
        info.supported_models.contains(&model)
    } else {
        false
    }
}

/// Get the default model for a provider
pub const fn get_default_model(provider_type: &ProviderType) -> Option<&'static str> {
    #[cfg(any(feature = "openai", feature = "anthropic", feature = "google"))]
    use crate::types::models::model_constants as models;

    match provider_type {
        #[cfg(feature = "openai")]
        ProviderType::OpenAi => Some(models::openai::GPT_4O), // Popular and capable
        #[cfg(feature = "anthropic")]
        ProviderType::Anthropic => Some(models::anthropic::CLAUDE_SONNET_3_5), // Good balance
        #[cfg(feature = "google")]
        ProviderType::Gemini => Some(models::gemini::GEMINI_2_5_FLASH), // Fast and capable
        #[cfg(feature = "ollama")]
        ProviderType::Ollama => Some("llama3.2:latest"),
        #[cfg(feature = "xai")]
        ProviderType::XAI => Some("grok-3-latest"),
        #[cfg(feature = "groq")]
        ProviderType::Groq => Some("llama-3.3-70b-versatile"),
        ProviderType::Custom(_) => None,

        // For disabled features, return None
        #[cfg(not(feature = "openai"))]
        ProviderType::OpenAi => None,
        #[cfg(not(feature = "anthropic"))]
        ProviderType::Anthropic => None,
        #[cfg(not(feature = "google"))]
        ProviderType::Gemini => None,
        #[cfg(not(feature = "ollama"))]
        ProviderType::Ollama => None,
        #[cfg(not(feature = "xai"))]
        ProviderType::XAI => None,
        #[cfg(not(feature = "groq"))]
        ProviderType::Groq => None,
    }
}

/// Provider Factory
pub struct ProviderFactory;

impl ProviderFactory {
    /// Validate provider configuration
    pub fn validate_config(
        provider_type: &ProviderType,
        api_key: &str,
        model: &str,
    ) -> Result<(), crate::error::LlmError> {
        // Check API key
        if api_key.is_empty() {
            return Err(crate::error::LlmError::MissingApiKey(format!(
                "API key is required for {provider_type}"
            )));
        }

        // Check model support
        if !is_model_supported(provider_type, model) {
            return Err(crate::error::LlmError::ModelNotSupported(format!(
                "Model '{model}' is not supported by {provider_type}"
            )));
        }

        Ok(())
    }

    /// Get the recommended configuration for a provider
    pub fn get_recommended_config(provider_type: &ProviderType) -> crate::types::CommonParams {
        match provider_type {
            ProviderType::OpenAi => crate::types::CommonParams {
                model: get_default_model(provider_type)
                    .unwrap_or("gpt-4o")
                    .to_string(),
                temperature: Some(0.7),
                max_tokens: Some(4096),
                top_p: Some(1.0),
                stop_sequences: None,
                seed: None,
            },
            ProviderType::Anthropic => crate::types::CommonParams {
                model: get_default_model(provider_type)
                    .unwrap_or("claude-3-5-sonnet-20241022")
                    .to_string(),
                temperature: Some(0.7),
                max_tokens: Some(4096),
                top_p: Some(1.0),
                stop_sequences: None,
                seed: None,
            },
            ProviderType::Gemini => crate::types::CommonParams {
                model: get_default_model(provider_type)
                    .unwrap_or("gemini-1.5-flash")
                    .to_string(),
                temperature: Some(0.7),
                max_tokens: Some(8192),
                top_p: Some(0.95),
                stop_sequences: None,
                seed: None,
            },
            ProviderType::Ollama => crate::types::CommonParams {
                model: get_default_model(provider_type)
                    .unwrap_or("llama3.2:latest")
                    .to_string(),
                temperature: Some(0.7),
                max_tokens: Some(4096),
                top_p: Some(0.9),
                stop_sequences: None,
                seed: None,
            },
            ProviderType::XAI => crate::types::CommonParams {
                model: get_default_model(provider_type)
                    .unwrap_or("grok-3-latest")
                    .to_string(),
                temperature: Some(0.7),
                max_tokens: Some(4096),
                top_p: Some(1.0),
                stop_sequences: None,
                seed: None,
            },
            ProviderType::Groq => crate::types::CommonParams {
                model: get_default_model(provider_type)
                    .unwrap_or("llama-3.3-70b-versatile")
                    .to_string(),
                temperature: Some(0.7),
                max_tokens: Some(8192),
                top_p: Some(1.0),
                stop_sequences: None,
                seed: None,
            },
            ProviderType::Custom(_) => crate::types::CommonParams::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_supported_providers() {
        let providers = get_supported_providers();
        assert!(!providers.is_empty());

        let openai_provider = providers
            .iter()
            .find(|p| p.provider_type == ProviderType::OpenAi);
        assert!(openai_provider.is_some());

        let anthropic_provider = providers
            .iter()
            .find(|p| p.provider_type == ProviderType::Anthropic);
        assert!(anthropic_provider.is_some());
    }

    #[test]
    fn test_model_support() {
        assert!(is_model_supported(&ProviderType::OpenAi, "gpt-4"));
        assert!(is_model_supported(
            &ProviderType::Anthropic,
            "claude-3-5-sonnet-20241022"
        ));
        assert!(!is_model_supported(&ProviderType::OpenAi, "claude-3-opus"));
    }

    #[test]
    fn test_default_models() {
        use crate::models;

        assert_eq!(
            get_default_model(&ProviderType::OpenAi),
            Some(models::openai::GPT_4O)
        );
        assert_eq!(
            get_default_model(&ProviderType::Anthropic),
            Some(models::anthropic::CLAUDE_SONNET_3_5)
        );
        assert_eq!(
            get_default_model(&ProviderType::Gemini),
            Some(models::gemini::GEMINI_2_5_FLASH)
        );
    }

    #[test]
    fn test_config_validation() {
        use crate::models::openai;

        let result =
            ProviderFactory::validate_config(&ProviderType::OpenAi, "test-key", openai::GPT_4);
        assert!(result.is_ok());

        let result = ProviderFactory::validate_config(&ProviderType::OpenAi, "", openai::GPT_4);
        assert!(result.is_err());

        let result =
            ProviderFactory::validate_config(&ProviderType::OpenAi, "test-key", "invalid-model");
        assert!(result.is_err());
    }
}
