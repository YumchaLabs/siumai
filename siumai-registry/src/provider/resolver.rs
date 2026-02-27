/// Normalize provider id aliases into a canonical id used across the registry.
pub fn normalize_provider_id(raw: &str) -> String {
    match raw.trim() {
        "google" => "gemini".to_string(),
        "google-vertex" => "vertex".to_string(),
        "google-vertex-anthropic" => "anthropic-vertex".to_string(),
        other => other.to_string(),
    }
}

/// Whether a canonical provider id should be treated as an OpenAI-compatible provider.
///
/// This is used for behaviors like model alias normalization that are specific to OpenAI-compatible
/// providers (not native OpenAI, Azure OpenAI, or non-OpenAI provider families).
#[cfg(any(test, feature = "openai"))]
pub fn is_openai_compatible_provider_id(provider_id: &str) -> bool {
    match provider_id {
        // Native / built-in families (not OpenAI-compatible)
        "openai" | "openai-chat" | "openai-responses" => false,
        "azure" | "azure-chat" => false,
        "anthropic" | "anthropic-vertex" => false,
        "gemini" | "vertex" => false,
        "ollama" | "xai" | "groq" | "minimaxi" => false,
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
    feature = "ollama",
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
            return Some("anthropic".to_string());
        }
    }

    #[cfg(feature = "google")]
    {
        if model.starts_with("gemini")
            || model.contains("/models/gemini")
            || model.contains("models/gemini")
            || model.contains("publishers/google/models/gemini")
        {
            return Some("gemini".to_string());
        }
    }

    #[cfg(feature = "google-vertex")]
    {
        if model.starts_with("imagen")
            || model.contains("/models/imagen")
            || model.contains("models/imagen")
            || model.contains("publishers/google/models/imagen")
        {
            return Some("vertex".to_string());
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
    fn resolve_provider_applies_normalization() {
        let id = normalize_provider_id("google");
        assert_eq!(id, "gemini".to_string());
    }
}
