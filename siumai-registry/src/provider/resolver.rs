use crate::provider::ids;

/// Normalize provider id aliases into a canonical id used across the registry.
pub fn normalize_provider_id(raw: &str) -> String {
    match raw.trim() {
        "google" => ids::GEMINI.to_string(),
        ids::GOOGLE_VERTEX_ALIAS => ids::VERTEX.to_string(),
        ids::GOOGLE_VERTEX_MAAS_ALIAS | ids::GOOGLE_VERTEX_MAAS_DOTTED_ALIAS | "vertexMaas" => {
            ids::VERTEX_MAAS.to_string()
        }
        "google-vertex-anthropic" => ids::ANTHROPIC_VERTEX.to_string(),
        other => other.to_string(),
    }
}

/// Normalize a model id for provider-owned compatibility aliases.
///
/// This intentionally lives in the registry layer because it depends on concrete provider ids and
/// vendor model namespaces. `siumai-core` must treat model ids as opaque strings.
pub fn normalize_model_id(provider_id: &str, model: &str) -> String {
    #[cfg(any(
        feature = "openai",
        feature = "google-vertex",
        feature = "togetherai",
        feature = "deepinfra",
        feature = "deepseek",
        feature = "xai",
        feature = "groq",
    ))]
    {
        siumai_provider_openai_compatible::providers::openai_compatible::normalize_model_id(
            provider_id,
            model,
        )
    }

    #[cfg(not(any(
        feature = "openai",
        feature = "google-vertex",
        feature = "togetherai",
        feature = "deepinfra",
        feature = "deepseek",
        feature = "xai",
        feature = "groq",
    )))]
    {
        let _ = provider_id;
        model.trim().to_string()
    }
}

/// Whether a canonical provider id should be treated as an OpenAI-compatible provider.
///
/// This is used for behaviors like model alias normalization that are specific to OpenAI-compatible
/// providers (not native OpenAI, Azure OpenAI, or non-OpenAI provider families).
#[cfg(any(test, feature = "openai", feature = "deepseek", feature = "deepinfra"))]
pub fn is_openai_compatible_provider_id(provider_id: &str) -> bool {
    match provider_id {
        // Native / built-in families (not OpenAI-compatible)
        ids::OPENAI | ids::OPENAI_CHAT | ids::OPENAI_RESPONSES => false,
        ids::AZURE | ids::AZURE_CHAT => false,
        ids::ANTHROPIC | ids::ANTHROPIC_VERTEX => false,
        ids::GEMINI | ids::VERTEX => false,
        ids::OLLAMA | ids::FIREWORKS | ids::XAI | ids::GROQ | ids::MINIMAXI => false,
        // Anything else is treated as OpenAI-compatible (custom providers).
        _ => true,
    }
}

// Keep inference intentionally conservative:
// - only infer when the model prefix is strongly associated with a provider
// - do not infer OpenAI / OpenAI-compatible providers because prefixes overlap heavily
#[cfg(any(
    test,
    feature = "openai",
    feature = "azure",
    feature = "anthropic",
    feature = "google",
    feature = "google-vertex",
    feature = "cohere",
    feature = "togetherai",
    feature = "bedrock",
    feature = "deepinfra",
    feature = "ollama",
    feature = "deepseek",
    feature = "xai",
    feature = "groq",
    feature = "minimaxi"
))]
pub fn infer_provider_id_from_model(model: &str) -> Option<String> {
    let model = model.trim();
    if model.is_empty() {
        return None;
    }

    #[cfg(feature = "anthropic")]
    {
        if model.starts_with("claude") {
            return Some(ids::ANTHROPIC.to_string());
        }
    }

    #[cfg(feature = "google")]
    {
        if model.starts_with("gemini")
            || model.contains("/models/gemini")
            || model.contains("models/gemini")
            || model.contains("publishers/google/models/gemini")
        {
            return Some(ids::GEMINI.to_string());
        }
    }

    #[cfg(feature = "google-vertex")]
    {
        if model.starts_with("imagen")
            || model.contains("/models/imagen")
            || model.contains("models/imagen")
            || model.contains("publishers/google/models/imagen")
        {
            return Some(ids::VERTEX.to_string());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_provider_id_aliases() {
        assert_eq!(normalize_provider_id("google"), "gemini");
        assert_eq!(normalize_provider_id("google-vertex"), "vertex");
        assert_eq!(normalize_provider_id("google-vertex-maas"), "vertex-maas");
        assert_eq!(normalize_provider_id("vertex.maas"), "vertex-maas");
        assert_eq!(normalize_provider_id("vertexMaas"), "vertex-maas");
        assert_eq!(
            normalize_provider_id("google-vertex-anthropic"),
            "anthropic-vertex"
        );
    }

    #[test]
    fn openai_compatible_provider_id_predicate() {
        assert!(!is_openai_compatible_provider_id("openai"));
        assert!(!is_openai_compatible_provider_id("openai-chat"));
        assert!(!is_openai_compatible_provider_id("openai-responses"));
        assert!(!is_openai_compatible_provider_id("azure"));
        assert!(!is_openai_compatible_provider_id("azure-chat"));

        assert!(is_openai_compatible_provider_id("deepseek"));
        assert!(is_openai_compatible_provider_id("openrouter"));
    }

    #[test]
    fn normalize_model_id_applies_deepseek_aliases() {
        assert_eq!(
            normalize_model_id("deepseek", "deepseek-v3"),
            "deepseek-chat"
        );
        assert_eq!(
            normalize_model_id("deepseek", "deepseek-r1"),
            "deepseek-reasoner"
        );
        assert_eq!(normalize_model_id("deepseek", "chat"), "deepseek-chat");
        assert_eq!(
            normalize_model_id("deepseek", "reasoner"),
            "deepseek-reasoner"
        );
        assert_eq!(
            normalize_model_id("deepseek", "deepseek-chat"),
            "deepseek-chat"
        );
    }

    #[test]
    fn normalize_model_id_applies_openrouter_vendor_prefixes() {
        assert_eq!(
            normalize_model_id("openrouter", "gpt-4o-mini"),
            "openai/gpt-4o-mini"
        );
        assert_eq!(
            normalize_model_id("openrouter", "claude-3-5-sonnet"),
            "anthropic/claude-3.5-sonnet"
        );
        assert_eq!(
            normalize_model_id("openrouter", "gemini-2.5-pro"),
            "google/gemini-2.5-pro"
        );
        assert_eq!(
            normalize_model_id("openrouter", "llama-3.1-70b-instruct"),
            "meta-llama/llama-3.1-70b-instruct"
        );
        assert_eq!(
            normalize_model_id("openrouter", "llama-3.1-sonar-small-128k-online"),
            "perplexity/llama-3.1-sonar-small-128k-online"
        );
        assert_eq!(
            normalize_model_id("openrouter", "command-r-plus"),
            "cohere/command-r-plus"
        );
        assert_eq!(
            normalize_model_id("openrouter", "openai/gpt-4o-mini"),
            "openai/gpt-4o-mini"
        );
    }

    #[test]
    fn normalize_model_id_applies_openai_compatible_vendor_aliases() {
        assert_eq!(
            normalize_model_id("siliconflow", "deepseek-v3.1"),
            "deepseek-ai/DeepSeek-V3.1"
        );
        assert_eq!(
            normalize_model_id("siliconflow", "qwen-2.5-72b-instruct"),
            "Qwen/Qwen2.5-72B-Instruct"
        );
        assert_eq!(
            normalize_model_id("siliconflow", "kimi-k2-instruct"),
            "moonshotai/Kimi-K2-Instruct"
        );
        assert_eq!(
            normalize_model_id("together", "llama-3.1-8b-instruct"),
            "meta-llama/llama-3.1-8b-instruct"
        );
        assert_eq!(
            normalize_model_id("fireworks", "llama-v3p1-8b-instruct"),
            "accounts/fireworks/models/llama-v3p1-8b-instruct"
        );
    }

    #[test]
    fn resolve_provider_applies_normalization() {
        let id = normalize_provider_id("google");
        assert_eq!(id, "gemini".to_string());
    }
}
