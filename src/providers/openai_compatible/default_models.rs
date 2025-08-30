//! Default Models Configuration for OpenAI-Compatible Providers
//!
//! This module centralizes default model configurations for all OpenAI-compatible providers.
//! It provides a single source of truth for default models, making it easy to update
//! and maintain provider-specific defaults.

use std::collections::HashMap;

/// Default model configurations for OpenAI-compatible providers
#[derive(Debug, Clone)]
pub struct DefaultModelConfig {
    /// Default chat model
    pub chat_model: &'static str,
    /// Default embedding model (if supported)
    pub embedding_model: Option<&'static str>,
    /// Default image generation model (if supported)
    pub image_model: Option<&'static str>,
    /// Default rerank model (if supported)
    pub rerank_model: Option<&'static str>,
}

impl DefaultModelConfig {
    /// Create a new default model configuration
    pub const fn new(chat_model: &'static str) -> Self {
        Self {
            chat_model,
            embedding_model: None,
            image_model: None,
            rerank_model: None,
        }
    }

    /// Set embedding model
    pub const fn with_embedding(mut self, model: &'static str) -> Self {
        self.embedding_model = Some(model);
        self
    }

    /// Set image generation model
    pub const fn with_image(mut self, model: &'static str) -> Self {
        self.image_model = Some(model);
        self
    }

    /// Set rerank model
    pub const fn with_rerank(mut self, model: &'static str) -> Self {
        self.rerank_model = Some(model);
        self
    }
}

/// Registry of default models for all OpenAI-compatible providers
pub struct DefaultModelRegistry {
    configs: HashMap<&'static str, DefaultModelConfig>,
}

impl DefaultModelRegistry {
    /// Create a new registry with all provider defaults
    pub fn new() -> Self {
        let mut configs = HashMap::new();

        // SiliconFlow defaults (using constants from models.rs)
        configs.insert(
            "siliconflow",
            DefaultModelConfig::new(super::providers::models::siliconflow::DEEPSEEK_V3_1)
                .with_embedding(super::providers::models::siliconflow::BGE_LARGE_ZH_V1_5)
                .with_image(super::providers::models::siliconflow::STABLE_DIFFUSION_3_5_LARGE)
                .with_rerank(super::providers::models::siliconflow::BGE_RERANKER_V2_M3),
        );

        // DeepSeek defaults (using constants from models.rs)
        configs.insert(
            "deepseek",
            DefaultModelConfig::new(super::providers::models::deepseek::CHAT)
                .with_embedding("deepseek-embedding"), // DeepSeek doesn't have public embedding API yet
        );

        // OpenRouter defaults (using constants from models.rs)
        configs.insert(
            "openrouter",
            DefaultModelConfig::new(super::providers::models::openrouter::openai::GPT_4O)
                .with_embedding("text-embedding-3-small"), // OpenRouter supports OpenAI embedding models
        );

        // Groq defaults (when used as OpenAI-compatible)
        configs.insert("groq", DefaultModelConfig::new("llama-3.3-70b-versatile"));

        // xAI defaults (when used as OpenAI-compatible)
        configs.insert("xai", DefaultModelConfig::new("grok-2-1212"));

        Self { configs }
    }

    /// Get default model configuration for a provider
    pub fn get_config(&self, provider_id: &str) -> Option<&DefaultModelConfig> {
        self.configs.get(provider_id)
    }

    /// Get default chat model for a provider
    pub fn get_default_chat_model(&self, provider_id: &str) -> Option<&'static str> {
        self.get_config(provider_id).map(|config| config.chat_model)
    }

    /// Get default embedding model for a provider
    pub fn get_default_embedding_model(&self, provider_id: &str) -> Option<&'static str> {
        self.get_config(provider_id)
            .and_then(|config| config.embedding_model)
    }

    /// Get default image generation model for a provider
    pub fn get_default_image_model(&self, provider_id: &str) -> Option<&'static str> {
        self.get_config(provider_id)
            .and_then(|config| config.image_model)
    }

    /// Get default rerank model for a provider
    pub fn get_default_rerank_model(&self, provider_id: &str) -> Option<&'static str> {
        self.get_config(provider_id)
            .and_then(|config| config.rerank_model)
    }

    /// Get all supported providers
    pub fn get_supported_providers(&self) -> Vec<&'static str> {
        self.configs.keys().copied().collect()
    }
}

impl Default for DefaultModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global registry instance
static DEFAULT_MODEL_REGISTRY: std::sync::LazyLock<DefaultModelRegistry> =
    std::sync::LazyLock::new(DefaultModelRegistry::new);

/// Get the global default model registry
pub fn get_registry() -> &'static DefaultModelRegistry {
    &DEFAULT_MODEL_REGISTRY
}

/// Get default chat model for a provider
pub fn get_default_chat_model(provider_id: &str) -> Option<&'static str> {
    get_registry().get_default_chat_model(provider_id)
}

/// Get default embedding model for a provider
pub fn get_default_embedding_model(provider_id: &str) -> Option<&'static str> {
    get_registry().get_default_embedding_model(provider_id)
}

/// Get default image generation model for a provider
pub fn get_default_image_model(provider_id: &str) -> Option<&'static str> {
    get_registry().get_default_image_model(provider_id)
}

/// Get default rerank model for a provider
pub fn get_default_rerank_model(provider_id: &str) -> Option<&'static str> {
    get_registry().get_default_rerank_model(provider_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_model_registry() {
        let registry = DefaultModelRegistry::new();

        // Test SiliconFlow defaults
        assert_eq!(
            registry.get_default_chat_model("siliconflow"),
            Some("deepseek-ai/DeepSeek-V3.1")
        );
        assert_eq!(
            registry.get_default_embedding_model("siliconflow"),
            Some("BAAI/bge-large-zh-v1.5")
        );
        assert_eq!(
            registry.get_default_image_model("siliconflow"),
            Some("stabilityai/stable-diffusion-3-5-large")
        );

        // Test DeepSeek defaults
        assert_eq!(
            registry.get_default_chat_model("deepseek"),
            Some("deepseek-chat")
        );

        // Test OpenRouter defaults
        assert_eq!(
            registry.get_default_chat_model("openrouter"),
            Some("openai/gpt-4o")
        );

        // Test unsupported provider
        assert_eq!(registry.get_default_chat_model("unknown"), None);
    }

    #[test]
    fn test_global_registry() {
        // Test global registry functions
        assert_eq!(
            get_default_chat_model("siliconflow"),
            Some("deepseek-ai/DeepSeek-V3.1")
        );
        assert_eq!(get_default_chat_model("deepseek"), Some("deepseek-chat"));
        assert_eq!(get_default_chat_model("openrouter"), Some("openai/gpt-4o"));
        assert_eq!(get_default_chat_model("unknown"), None);
    }

    #[test]
    fn test_supported_providers() {
        let registry = DefaultModelRegistry::new();
        let providers = registry.get_supported_providers();

        assert!(providers.contains(&"siliconflow"));
        assert!(providers.contains(&"deepseek"));
        assert!(providers.contains(&"openrouter"));
        assert!(providers.contains(&"groq"));
        assert!(providers.contains(&"xai"));
    }

    #[test]
    fn test_model_config_builder() {
        let config = DefaultModelConfig::new("test-chat")
            .with_embedding("test-embedding")
            .with_image("test-image")
            .with_rerank("test-rerank");

        assert_eq!(config.chat_model, "test-chat");
        assert_eq!(config.embedding_model, Some("test-embedding"));
        assert_eq!(config.image_model, Some("test-image"));
        assert_eq!(config.rerank_model, Some("test-rerank"));
    }
}
