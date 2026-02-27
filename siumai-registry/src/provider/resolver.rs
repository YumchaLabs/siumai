use crate::types::ProviderType;

/// Normalize provider id aliases into a canonical id used across the registry.
pub fn normalize_provider_id(raw: &str) -> String {
    match raw.trim() {
        "google" => "gemini".to_string(),
        "google-vertex" => "vertex".to_string(),
        "google-vertex-anthropic" => "anthropic-vertex".to_string(),
        other => other.to_string(),
    }
}

/// Resolve a canonical provider id and its corresponding ProviderType.
pub fn resolve_provider(raw_provider_id: &str) -> (String, ProviderType) {
    let provider_id = normalize_provider_id(raw_provider_id);
    let provider_type = provider_type_for_provider_id(&provider_id);
    (provider_id, provider_type)
}

/// Map a canonical provider id to its ProviderType.
///
/// Note: some ids are routing variants (e.g. `openai-chat`) and still map to the same ProviderType.
pub fn provider_type_for_provider_id(provider_id: &str) -> ProviderType {
    match provider_id {
        "openai" | "openai-chat" | "openai-responses" => ProviderType::OpenAi,
        "azure" | "azure-chat" => ProviderType::Custom("azure".to_string()),
        "anthropic" => ProviderType::Anthropic,
        "anthropic-vertex" => ProviderType::Custom("anthropic-vertex".to_string()),
        "gemini" => ProviderType::Gemini,
        "vertex" => ProviderType::Custom("vertex".to_string()),
        other => ProviderType::from_name(other),
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
pub fn infer_provider_type_from_model(model: &str) -> Option<ProviderType> {
    let model = model.trim();
    if model.is_empty() {
        return None;
    }

    #[cfg(feature = "anthropic")]
    {
        if model.starts_with("claude") {
            return Some(ProviderType::Anthropic);
        }
    }

    #[cfg(feature = "google")]
    {
        if model.starts_with("gemini")
            || model.contains("/models/gemini")
            || model.contains("models/gemini")
            || model.contains("publishers/google/models/gemini")
        {
            return Some(ProviderType::Gemini);
        }
    }

    #[cfg(feature = "google-vertex")]
    {
        if model.starts_with("imagen")
            || model.contains("/models/imagen")
            || model.contains("models/imagen")
            || model.contains("publishers/google/models/imagen")
        {
            return Some(ProviderType::Custom("vertex".to_string()));
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
    fn provider_type_for_variants() {
        assert_eq!(
            provider_type_for_provider_id("openai"),
            ProviderType::OpenAi
        );
        assert_eq!(
            provider_type_for_provider_id("openai-chat"),
            ProviderType::OpenAi
        );
        assert_eq!(
            provider_type_for_provider_id("openai-responses"),
            ProviderType::OpenAi
        );

        assert_eq!(
            provider_type_for_provider_id("azure"),
            ProviderType::Custom("azure".to_string())
        );
        assert_eq!(
            provider_type_for_provider_id("azure-chat"),
            ProviderType::Custom("azure".to_string())
        );

        assert_eq!(
            provider_type_for_provider_id("anthropic-vertex"),
            ProviderType::Custom("anthropic-vertex".to_string())
        );
        assert_eq!(
            provider_type_for_provider_id("vertex"),
            ProviderType::Custom("vertex".to_string())
        );
    }

    #[test]
    fn resolve_provider_applies_normalization() {
        let (id, ty) = resolve_provider("google");
        assert_eq!(id, "gemini");
        assert_eq!(ty, ProviderType::Gemini);
    }
}
