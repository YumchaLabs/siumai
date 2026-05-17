//! OpenAI-Compatible Providers Configuration
//!
//! This module contains centralized configuration for all OpenAI-compatible providers.
//! Inspired by Cherry Studio's provider configuration system.

use crate::providers::openai_compatible::ProviderAdapter;
use crate::standards::openai::compat::provider_registry::{
    ConfigurableAdapter, ProviderConfig, ProviderFieldMappings,
};

mod builtin_providers;
mod family_defaults;

pub(crate) use builtin_providers::get_builtin_provider_map;
pub use builtin_providers::get_builtin_providers;
pub(crate) use family_defaults::{
    get_builtin_provider_family_defaults_map, get_default_embedding_model, get_default_image_model,
    get_default_rerank_model, get_default_speech_model, get_default_transcription_model,
};

#[cfg(test)]
pub(crate) use family_defaults::get_provider_family_defaults_ref;

#[cfg(test)]
use crate::providers::openai_compatible::groq as groq_models;

/// Build a generic OpenAI-compatible provider configuration.
///
/// This mirrors AI SDK's `createOpenAICompatible({ name, baseURL, ... })` path: callers provide a
/// provider name and endpoint, and Siumai uses the plain OpenAI-compatible adapter without applying
/// any built-in provider-specific transforms.
pub fn generic_provider_config(provider_id: &str, name: &str, base_url: &str) -> ProviderConfig {
    ProviderConfig {
        id: provider_id.to_string(),
        name: name.to_string(),
        base_url: base_url.to_string(),
        field_mappings: ProviderFieldMappings::default(),
        capabilities: vec![
            "tools".to_string(),
            "vision".to_string(),
            "embedding".to_string(),
            "image_generation".to_string(),
        ],
        default_model: None,
        supports_reasoning: false,
        api_key_env: None,
        api_key_env_aliases: Vec::new(),
    }
}

/// Get provider configuration by ID
pub fn get_provider_config(provider_id: &str) -> Option<ProviderConfig> {
    get_provider_config_ref(provider_id).cloned()
}

pub(crate) fn get_provider_config_ref(provider_id: &str) -> Option<&'static ProviderConfig> {
    get_builtin_provider_map().get(provider_id)
}

pub(crate) fn is_hidden_compat_alias(provider_id: &str) -> bool {
    matches!(provider_id, "together" | "moonshot")
}

/// List all available provider IDs
pub fn list_provider_ids() -> Vec<String> {
    get_builtin_provider_map()
        .keys()
        .filter(|provider_id| !is_hidden_compat_alias(provider_id))
        .cloned()
        .collect()
}

/// Check if a provider supports a specific capability
pub fn provider_supports_capability(provider_id: &str, capability: &str) -> bool {
    if let Some(config) = get_provider_config(provider_id) {
        ConfigurableAdapter::new(config)
            .capabilities()
            .supports(capability)
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_facade_keeps_family_defaults_split() {
        let source = include_str!("config.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap_or_default();
        let family_defaults = include_str!("config/family_defaults.rs");

        assert!(
            source.contains("mod family_defaults;"),
            "OpenAI-compatible config facade should keep family defaults in a dedicated module"
        );
        assert!(
            !source.contains("struct ProviderFamilyDefaults"),
            "OpenAI-compatible config facade should not own provider family defaults data"
        );
        assert!(
            family_defaults.contains("struct ProviderFamilyDefaults"),
            "provider family defaults should live in config/family_defaults.rs"
        );
    }

    #[test]
    fn config_facade_keeps_builtin_provider_registry_split() {
        let source = include_str!("config.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap_or_default();
        let builtin_providers = include_str!("config/builtin_providers.rs");

        assert!(
            source.contains("mod builtin_providers;"),
            "OpenAI-compatible config facade should keep built-in provider registry data in a dedicated module"
        );
        for forbidden in ["fn build_builtin_providers", "static BUILTIN_PROVIDERS"] {
            assert!(
                !source.contains(forbidden),
                "OpenAI-compatible config facade should not own `{forbidden}`"
            );
            assert!(
                builtin_providers.contains(forbidden),
                "built-in provider registry data should own `{forbidden}`"
            );
        }
    }

    #[test]
    fn test_builtin_providers() {
        let providers = get_builtin_providers();

        // Should have all major providers
        assert!(providers.contains_key("deepseek"));
        assert!(providers.contains_key("openrouter"));
        assert!(providers.contains_key("togetherai"));
        assert!(providers.contains_key("deepinfra"));
        assert!(providers.contains_key("vertex-maas"));
        assert!(providers.contains_key("fireworks"));
        assert!(providers.contains_key("moonshotai"));
        assert!(providers.contains_key("alibaba"));

        // DeepSeek should support reasoning
        let deepseek = providers.get("deepseek").unwrap();
        assert!(deepseek.supports_reasoning);
        assert!(deepseek.capabilities.contains(&"reasoning".to_string()));

        // All providers should have valid URLs
        for (id, config) in &providers {
            assert!(
                !config.base_url.is_empty(),
                "Provider {} has empty base_url",
                id
            );
            assert!(
                config.base_url.starts_with("https://"),
                "Provider {} has invalid URL",
                id
            );
        }
    }

    #[test]
    fn test_provider_capabilities() {
        assert!(provider_supports_capability("deepseek", "reasoning"));
        assert!(provider_supports_capability("openrouter", "tools"));
        assert!(provider_supports_capability("openrouter", "embedding"));
        assert!(provider_supports_capability("perplexity", "tools"));
        assert!(provider_supports_capability("siliconflow", "rerank"));
        assert!(provider_supports_capability("siliconflow", "embedding"));
        assert!(provider_supports_capability(
            "siliconflow",
            "image_generation"
        ));
        assert!(provider_supports_capability("siliconflow", "speech"));
        assert!(provider_supports_capability("siliconflow", "transcription"));
        assert!(provider_supports_capability("siliconflow", "audio"));
        assert!(provider_supports_capability("togetherai", "embedding"));
        assert!(provider_supports_capability("togetherai", "completion"));
        assert!(provider_supports_capability(
            "togetherai",
            "image_generation"
        ));
        assert!(provider_supports_capability("togetherai", "speech"));
        assert!(provider_supports_capability("togetherai", "transcription"));
        assert!(provider_supports_capability("togetherai", "audio"));
        assert!(provider_supports_capability("deepinfra", "embedding"));
        assert!(provider_supports_capability("deepinfra", "completion"));
        assert!(!provider_supports_capability(
            "deepinfra",
            "image_generation"
        ));
        assert!(provider_supports_capability("moonshotai", "reasoning"));
        assert!(!provider_supports_capability("moonshotai", "completion"));
        assert!(provider_supports_capability("alibaba", "reasoning"));
        assert!(provider_supports_capability("qwen", "reasoning"));
        assert!(provider_supports_capability("vertex-maas", "embedding"));
        assert!(!provider_supports_capability(
            "vertex-maas",
            "image_generation"
        ));
        assert!(provider_supports_capability("mistral", "embedding"));
        assert!(!provider_supports_capability("mistral", "completion"));
        assert!(!provider_supports_capability("perplexity", "completion"));
        assert!(provider_supports_capability("fireworks", "completion"));
        assert!(!provider_supports_capability("groq", "vision"));
        assert!(provider_supports_capability("groq", "transcription"));
        assert!(provider_supports_capability("groq", "audio"));
        assert!(!provider_supports_capability("openrouter", "speech"));
        assert!(!provider_supports_capability("openrouter", "transcription"));
        assert!(!provider_supports_capability("openrouter", "audio"));
        assert!(!provider_supports_capability("xai", "speech"));
        assert!(!provider_supports_capability("xai", "transcription"));
        assert!(!provider_supports_capability("xai", "audio"));
        assert!(!provider_supports_capability("fireworks", "speech"));
        assert!(provider_supports_capability("fireworks", "transcription"));
        assert!(provider_supports_capability("fireworks", "audio"));
    }

    #[test]
    fn test_provider_config_retrieval() {
        let config = get_provider_config("deepseek").unwrap();
        assert_eq!(config.id, "deepseek");
        assert_eq!(config.name, "DeepSeek");
        assert_eq!(config.base_url, "https://api.deepseek.com");
        assert_eq!(config.api_key_env.as_deref(), Some("DEEPSEEK_API_KEY"));

        let config = get_provider_config("deepinfra").unwrap();
        assert_eq!(config.id, "deepinfra");
        assert_eq!(config.name, "DeepInfra");
        assert_eq!(config.base_url, "https://api.deepinfra.com/v1/openai");
        assert_eq!(config.api_key_env.as_deref(), Some("DEEPINFRA_API_KEY"));

        let config = get_provider_config("togetherai").unwrap();
        assert_eq!(config.id, "togetherai");
        assert_eq!(config.api_key_env.as_deref(), Some("TOGETHER_API_KEY"));
        assert_eq!(
            config.api_key_env_aliases,
            vec!["TOGETHER_AI_API_KEY".to_string()]
        );

        let config = get_provider_config("moonshotai").unwrap();
        assert_eq!(config.id, "moonshotai");
        assert_eq!(config.base_url, "https://api.moonshot.ai/v1");
        assert_eq!(config.api_key_env.as_deref(), Some("MOONSHOT_API_KEY"));

        let config = get_provider_config("mistral").unwrap();
        assert_eq!(config.id, "mistral");
        assert_eq!(config.api_key_env.as_deref(), Some("MISTRAL_API_KEY"));

        let config = get_provider_config("groq").unwrap();
        assert_eq!(config.id, "groq");
        assert_eq!(config.base_url, "https://api.groq.com/openai/v1");
        assert_eq!(config.api_key_env.as_deref(), Some("GROQ_API_KEY"));

        let config = get_provider_config("xai").unwrap();
        assert_eq!(config.id, "xai");
        assert_eq!(config.api_key_env.as_deref(), Some("XAI_API_KEY"));

        let config = get_provider_config("perplexity").unwrap();
        assert_eq!(config.id, "perplexity");
        assert_eq!(config.api_key_env.as_deref(), Some("PERPLEXITY_API_KEY"));

        let config = get_provider_config("fireworks").unwrap();
        assert_eq!(config.id, "fireworks");
        assert_eq!(config.api_key_env.as_deref(), Some("FIREWORKS_API_KEY"));

        let config = get_provider_config("qwen").unwrap();
        assert_eq!(config.id, "qwen");
        assert_eq!(config.api_key_env.as_deref(), Some("ALIBABA_API_KEY"));
        assert_eq!(
            config.api_key_env_aliases,
            vec!["DASHSCOPE_API_KEY".to_string()]
        );

        let config = get_provider_config("alibaba").unwrap();
        assert_eq!(config.id, "alibaba");
        assert_eq!(
            config.base_url,
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        );
        assert_eq!(config.api_key_env.as_deref(), Some("ALIBABA_API_KEY"));
        assert_eq!(
            config.api_key_env_aliases,
            vec!["DASHSCOPE_API_KEY".to_string(), "QWEN_API_KEY".to_string()]
        );

        assert!(get_provider_config("nonexistent").is_none());
    }

    #[test]
    fn test_moonshotai_canonical_id_hides_legacy_alias() {
        let provider_ids = list_provider_ids();
        assert!(
            provider_ids
                .iter()
                .any(|provider_id| provider_id == "moonshotai")
        );
        assert!(
            !provider_ids
                .iter()
                .any(|provider_id| provider_id == "moonshot")
        );

        let canonical = get_provider_config("moonshotai").expect("moonshotai provider config");
        let alias = get_provider_config("moonshot").expect("moonshot alias provider config");

        assert_eq!(canonical.id, "moonshotai");
        assert_eq!(alias.id, "moonshotai");
        assert_eq!(alias.base_url, canonical.base_url);
        assert_eq!(alias.default_model, canonical.default_model);
        assert!(provider_supports_capability("moonshotai", "reasoning"));
        assert!(!provider_supports_capability("moonshotai", "completion"));
    }

    #[test]
    fn test_completion_capable_hybrid_presets_keep_explicit_completion_metadata() {
        for provider_id in ["together", "togetherai", "deepinfra", "fireworks"] {
            let config = get_provider_config(provider_id)
                .unwrap_or_else(|| panic!("missing provider config for {provider_id}"));
            assert!(
                config.capabilities.iter().any(|cap| cap == "completion"),
                "expected {provider_id} compat preset metadata to advertise completion explicitly"
            );
        }
    }

    #[test]
    fn test_provider_family_defaults() {
        let openrouter =
            get_provider_family_defaults_ref("openrouter").expect("openrouter defaults");
        assert_eq!(openrouter.chat_model_override, Some("openai/gpt-4o"));
        assert_eq!(openrouter.embedding_model, Some("text-embedding-3-small"));

        let siliconflow =
            get_provider_family_defaults_ref("siliconflow").expect("siliconflow defaults");
        assert_eq!(siliconflow.embedding_model, Some("BAAI/bge-large-zh-v1.5"));
        assert_eq!(
            siliconflow.image_model,
            Some("stabilityai/stable-diffusion-3.5-large")
        );
        assert_eq!(siliconflow.rerank_model, Some("BAAI/bge-reranker-v2-m3"));
        assert_eq!(
            siliconflow.speech_model,
            Some("FunAudioLLM/CosyVoice2-0.5B")
        );
        assert_eq!(
            siliconflow.transcription_model,
            Some("FunAudioLLM/SenseVoiceSmall")
        );

        let togetherai =
            get_provider_family_defaults_ref("togetherai").expect("togetherai defaults");
        assert_eq!(
            togetherai.embedding_model,
            Some("togethercomputer/m2-bert-80M-8k-retrieval")
        );
        assert_eq!(
            togetherai.image_model,
            Some("black-forest-labs/FLUX.1-schnell")
        );
        assert_eq!(togetherai.speech_model, Some("cartesia/sonic-2"));
        assert_eq!(
            togetherai.transcription_model,
            Some("openai/whisper-large-v3")
        );

        let groq = get_provider_family_defaults_ref("groq").expect("groq defaults");
        assert_eq!(groq.transcription_model, Some(groq_models::TRANSCRIPTION));

        let deepinfra = get_provider_family_defaults_ref("deepinfra").expect("deepinfra defaults");
        assert_eq!(deepinfra.embedding_model, Some("BAAI/bge-base-en-v1.5"));
        assert_eq!(
            deepinfra.image_model,
            Some("black-forest-labs/FLUX-1-schnell")
        );
        assert_eq!(deepinfra.speech_model, None);
        assert_eq!(deepinfra.transcription_model, None);

        let fireworks = get_provider_family_defaults_ref("fireworks").expect("fireworks defaults");
        assert_eq!(
            fireworks.embedding_model,
            Some("nomic-ai/nomic-embed-text-v1.5")
        );
        assert_eq!(
            fireworks.image_model,
            Some("accounts/fireworks/models/flux-1-dev-fp8")
        );
        assert_eq!(fireworks.transcription_model, Some("whisper-v3"));
        assert_eq!(fireworks.speech_model, None);

        let mistral = get_provider_family_defaults_ref("mistral").expect("mistral defaults");
        assert_eq!(mistral.embedding_model, Some("mistral-embed"));
        assert_eq!(mistral.image_model, None);

        assert_eq!(
            get_default_embedding_model("infini"),
            Some("text-embedding-3-small")
        );
        assert_eq!(get_default_rerank_model("voyageai"), Some("rerank-2"));
        assert_eq!(get_default_rerank_model("jina"), Some("jina-reranker-m0"));
        assert_eq!(get_default_speech_model("openrouter"), None);
        assert_eq!(get_default_transcription_model("openrouter"), None);
    }
}
