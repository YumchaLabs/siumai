//! Default Models Configuration for OpenAI-Compatible Providers
//!
//! Compatibility facade over config-owned compat default tables.
//!
//! The canonical builtin source-of-truth lives in `config.rs`:
//! - provider `default_model` for the primary builder/config shortcut default
//! - explicit family defaults for embedding/image/rerank (and audio families)
//!
//! This module remains as the historical read API for callers that still expect
//! `default_models::*` helpers.

use std::collections::HashMap;

/// Default model configurations for OpenAI-compatible providers
#[derive(Debug, Clone)]
pub struct DefaultModelConfig {
    /// Optional chat model override for providers whose canonical default is not owned by config.rs
    pub chat_model: Option<&'static str>,
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
            chat_model: Some(chat_model),
            embedding_model: None,
            image_model: None,
            rerank_model: None,
        }
    }

    /// Create a config with non-chat defaults only
    pub const fn extras() -> Self {
        Self {
            chat_model: None,
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

        for (provider_id, defaults) in super::config::get_builtin_provider_family_defaults_map() {
            let mut config = if let Some(chat_model_override) = defaults.chat_model_override {
                DefaultModelConfig::new(chat_model_override)
            } else {
                DefaultModelConfig::extras()
            };

            if let Some(model) = defaults.embedding_model {
                config = config.with_embedding(model);
            }
            if let Some(model) = defaults.image_model {
                config = config.with_image(model);
            }
            if let Some(model) = defaults.rerank_model {
                config = config.with_rerank(model);
            }

            configs.insert(*provider_id, config);
        }

        Self { configs }
    }

    /// Get default model configuration for a provider
    pub fn get_config(&self, provider_id: &str) -> Option<&DefaultModelConfig> {
        self.configs.get(provider_id)
    }

    /// Get default chat model for a provider
    pub fn get_default_chat_model(&self, provider_id: &str) -> Option<&'static str> {
        self.get_config(provider_id)
            .and_then(|config| config.chat_model)
            .or_else(|| {
                super::config::get_provider_config_ref(provider_id)
                    .and_then(|config| config.default_model.as_deref())
            })
    }

    /// Get default embedding model for a provider
    pub fn get_default_embedding_model(&self, provider_id: &str) -> Option<&'static str> {
        self.get_config(provider_id)
            .and_then(|config| config.embedding_model)
            .or_else(|| super::config::get_default_embedding_model(provider_id))
    }

    /// Get default image generation model for a provider
    pub fn get_default_image_model(&self, provider_id: &str) -> Option<&'static str> {
        self.get_config(provider_id)
            .and_then(|config| config.image_model)
            .or_else(|| super::config::get_default_image_model(provider_id))
    }

    /// Get default rerank model for a provider
    pub fn get_default_rerank_model(&self, provider_id: &str) -> Option<&'static str> {
        self.get_config(provider_id)
            .and_then(|config| config.rerank_model)
            .or_else(|| super::config::get_default_rerank_model(provider_id))
    }

    /// Get all supported providers
    pub fn get_supported_providers(&self) -> Vec<&'static str> {
        let mut providers: Vec<&'static str> = self.configs.keys().copied().collect();

        for (provider_id, config) in super::config::get_builtin_provider_map() {
            if config.default_model.is_some() {
                let provider_id = provider_id.as_str();
                if !providers.contains(&provider_id) {
                    providers.push(provider_id);
                }
            }
        }

        providers.sort_unstable();
        providers
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
            Some("deepseek-ai/DeepSeek-V3")
        );
        assert_eq!(
            registry.get_default_embedding_model("siliconflow"),
            Some("BAAI/bge-large-zh-v1.5")
        );
        assert_eq!(
            registry.get_default_image_model("siliconflow"),
            Some("stabilityai/stable-diffusion-3.5-large")
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
        assert_eq!(
            registry.get_default_embedding_model("openrouter"),
            Some("text-embedding-3-small")
        );
        assert_eq!(
            registry.get_default_chat_model("mistral"),
            Some("mistral-large-latest")
        );
        assert_eq!(
            registry.get_default_chat_model("together"),
            Some("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
        );
        assert_eq!(
            registry.get_default_chat_model("fireworks"),
            Some("accounts/fireworks/models/llama-v3p1-8b-instruct")
        );
        assert_eq!(
            registry.get_default_embedding_model("fireworks"),
            Some("nomic-ai/nomic-embed-text-v1.5")
        );
        assert_eq!(
            registry.get_default_rerank_model("jina"),
            Some("jina-reranker-m0")
        );
        assert_eq!(
            registry.get_default_rerank_model("voyageai"),
            Some("rerank-2")
        );

        // Test unsupported provider
        assert_eq!(registry.get_default_chat_model("unknown"), None);
    }

    #[test]
    fn test_global_registry() {
        // Test global registry functions
        assert_eq!(
            get_default_chat_model("siliconflow"),
            Some("deepseek-ai/DeepSeek-V3")
        );
        assert_eq!(get_default_chat_model("deepseek"), Some("deepseek-chat"));
        assert_eq!(get_default_chat_model("openrouter"), Some("openai/gpt-4o"));
        assert_eq!(
            get_default_embedding_model("openrouter"),
            Some("text-embedding-3-small")
        );
        assert_eq!(
            get_default_embedding_model("siliconflow"),
            Some("BAAI/bge-large-zh-v1.5")
        );
        assert_eq!(
            get_default_image_model("together"),
            Some("black-forest-labs/FLUX.1-schnell")
        );
        assert_eq!(get_default_rerank_model("jina"), Some("jina-reranker-m0"));
        assert_eq!(
            get_default_chat_model("mistral"),
            Some("mistral-large-latest")
        );
        assert_eq!(
            get_default_chat_model("together"),
            Some("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
        );
        assert_eq!(
            get_default_chat_model("fireworks"),
            Some("accounts/fireworks/models/llama-v3p1-8b-instruct")
        );
        assert_eq!(get_default_chat_model("unknown"), None);
    }

    #[test]
    fn test_supported_providers() {
        let registry = DefaultModelRegistry::new();
        let providers = registry.get_supported_providers();

        assert!(providers.contains(&"siliconflow"));
        assert!(providers.contains(&"deepseek"));
        assert!(providers.contains(&"openrouter"));
        assert!(providers.contains(&"mistral"));
        assert!(providers.contains(&"cohere"));
        assert!(providers.contains(&"together"));
        assert!(providers.contains(&"fireworks"));
        assert!(providers.contains(&"groq"));
        assert!(providers.contains(&"xai"));
    }

    #[test]
    fn test_model_config_builder() {
        let config = DefaultModelConfig::new("test-chat")
            .with_embedding("test-embedding")
            .with_image("test-image")
            .with_rerank("test-rerank");

        assert_eq!(config.chat_model, Some("test-chat"));
        assert_eq!(config.embedding_model, Some("test-embedding"));
        assert_eq!(config.image_model, Some("test-image"));
        assert_eq!(config.rerank_model, Some("test-rerank"));
    }

    #[test]
    fn test_model_config_extras_builder() {
        let config = DefaultModelConfig::extras().with_embedding("test-embedding");

        assert_eq!(config.chat_model, None);
        assert_eq!(config.embedding_model, Some("test-embedding"));
    }
}
