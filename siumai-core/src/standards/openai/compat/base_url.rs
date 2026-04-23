//! Shared base-URL normalization helpers for OpenAI-compatible providers.

/// Normalize a provider-specific text-family base URL.
///
/// Most OpenAI-compatible providers treat the configured `base_url` as the final API prefix.
/// DeepInfra is different: its public provider surface accepts a root API base and then routes
/// text-family requests through `{root}/openai/...`.
pub fn normalize_text_base_url(provider_id: &str, base_url: &str) -> String {
    if provider_id == "deepinfra" {
        deepinfra_text_base_url(base_url)
    } else {
        base_url.to_string()
    }
}

/// Normalize DeepInfra's root API base URL from either the public root, `/openai`, or
/// `/inference` prefix.
pub fn deepinfra_root_base_url(base_url: &str) -> String {
    if base_url.trim().is_empty() {
        return base_url.to_string();
    }

    let mut out = base_url.trim_end_matches('/').to_string();
    for suffix in ["/openai", "/inference"] {
        if let Some(stripped) = out.strip_suffix(suffix) {
            out = stripped.to_string();
        }
    }
    out
}

/// Normalize DeepInfra's text-family API prefix.
pub fn deepinfra_text_base_url(base_url: &str) -> String {
    let root_base_url = deepinfra_root_base_url(base_url);
    if root_base_url.trim().is_empty() {
        return root_base_url;
    }

    format!("{}/openai", root_base_url.trim_end_matches('/'))
}

#[cfg(test)]
mod tests {
    use super::{deepinfra_root_base_url, deepinfra_text_base_url, normalize_text_base_url};

    #[test]
    fn deepinfra_root_base_url_accepts_root_and_family_prefixes() {
        assert_eq!(
            deepinfra_root_base_url("https://api.deepinfra.com/v1"),
            "https://api.deepinfra.com/v1"
        );
        assert_eq!(
            deepinfra_root_base_url("https://api.deepinfra.com/v1/openai"),
            "https://api.deepinfra.com/v1"
        );
        assert_eq!(
            deepinfra_root_base_url("https://api.deepinfra.com/v1/inference"),
            "https://api.deepinfra.com/v1"
        );
        assert_eq!(
            deepinfra_root_base_url("https://api.deepinfra.com/v1/openai/"),
            "https://api.deepinfra.com/v1"
        );
    }

    #[test]
    fn deepinfra_text_base_url_always_points_at_openai_family_root() {
        assert_eq!(
            deepinfra_text_base_url("https://api.deepinfra.com/v1"),
            "https://api.deepinfra.com/v1/openai"
        );
        assert_eq!(
            deepinfra_text_base_url("https://api.deepinfra.com/v1/openai"),
            "https://api.deepinfra.com/v1/openai"
        );
        assert_eq!(
            deepinfra_text_base_url("https://api.deepinfra.com/v1/inference"),
            "https://api.deepinfra.com/v1/openai"
        );
    }

    #[test]
    fn normalize_text_base_url_only_special_cases_deepinfra() {
        assert_eq!(
            normalize_text_base_url("deepinfra", "https://api.deepinfra.com/v1"),
            "https://api.deepinfra.com/v1/openai"
        );
        assert_eq!(
            normalize_text_base_url("deepseek", "https://api.deepseek.com"),
            "https://api.deepseek.com"
        );
    }
}
