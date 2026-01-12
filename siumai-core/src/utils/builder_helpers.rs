//! Builder Helper Functions
//!
//! Shared utility functions for both `SiumaiBuilder` and provider-specific builders
//! to reduce code duplication and ensure consistent behavior.

use crate::error::LlmError;

/// Get API key with environment variable fallback
///
/// Priority: explicit parameter > environment variable `{PROVIDER_ID}_API_KEY`
///
/// # Arguments
/// * `api_key` - Explicitly provided API key (if any)
/// * `provider_id` - Provider identifier (e.g., "moonshot", "deepseek")
///
/// # Returns
/// * `Ok(String)` - The API key
/// * `Err(LlmError)` - If no API key is found
///
/// # Example
/// ```rust,ignore
/// let api_key = get_api_key_with_env(None, "moonshot")?;
/// // Reads from MOONSHOT_API_KEY environment variable
/// ```
pub fn get_api_key_with_env(
    api_key: Option<String>,
    provider_id: &str,
) -> Result<String, LlmError> {
    get_api_key_with_envs(api_key, provider_id, None, &[])
}

/// Get API key with configurable environment variable fallbacks.
///
/// Priority:
/// - explicit parameter
/// - `api_key_env` (if provided)
/// - `api_key_env_aliases` (in order)
/// - default environment variable `{PROVIDER_ID}_API_KEY`
///
/// This is primarily used by "OpenAI-compatible" vendor presets where a provider id
/// may not map cleanly to a POSIX env var name (e.g. leading digits) or when a
/// provider has a widely-adopted env var name that differs from our default.
pub fn get_api_key_with_envs(
    api_key: Option<String>,
    provider_id: &str,
    api_key_env: Option<&str>,
    api_key_env_aliases: &[String],
) -> Result<String, LlmError> {
    if let Some(key) = api_key {
        return Ok(key);
    }

    let default_env = format!("{}_API_KEY", provider_id.to_uppercase());
    let mut candidates: Vec<String> = Vec::with_capacity(2 + api_key_env_aliases.len());

    if let Some(name) = api_key_env
        && !name.trim().is_empty()
    {
        candidates.push(name.to_string());
    }
    for name in api_key_env_aliases {
        if !name.trim().is_empty() {
            candidates.push(name.to_string());
        }
    }
    candidates.push(default_env);

    // Dedupe while keeping stable order.
    let mut seen = std::collections::HashSet::<String>::new();
    candidates.retain(|k| seen.insert(k.clone()));

    for env_key in &candidates {
        if let Ok(v) = std::env::var(env_key)
            && !v.is_empty()
        {
            return Ok(v);
        }
    }

    Err(LlmError::ConfigurationError(format!(
        "API key is required for {} (missing {} or explicit .api_key())",
        provider_id,
        candidates.join(", ")
    )))
}

/// Get effective model with default fallback
///
/// Priority: explicit model > default model from registry
///
/// # Arguments
/// * `model` - Explicitly provided model (if any)
/// * `provider_id` - Provider identifier
///
/// # Returns
/// The effective model string (may be empty if no default exists)
///
/// # Example
/// ```rust,ignore
/// let model = get_effective_model("", "moonshot");
/// // Returns "kimi-k2-0905-preview" (default for moonshot)
/// ```
pub fn get_effective_model(model: &str, _provider_id: &str) -> String {
    if !model.is_empty() {
        model.to_string()
    } else {
        String::new()
    }
}

/// Normalize model ID (handle provider-specific aliases)
///
/// Some providers use special model ID formats:
/// - OpenRouter: `openai/gpt-4o` → `openai/gpt-4o`
/// - DeepSeek: `chat` → `deepseek-chat`
///
/// # Arguments
/// * `provider_id` - Provider identifier
/// * `model` - Model ID to normalize
///
/// # Returns
/// Normalized model ID
///
/// # Example
/// ```rust,ignore
/// let normalized = normalize_model_id("deepseek", "chat");
/// // Returns "deepseek-chat"
/// ```
pub fn normalize_model_id(provider_id: &str, model: &str) -> String {
    crate::utils::model_alias::normalize_model_id(provider_id, model)
}

/// Resolve CommonParams for a given model id using optional context overrides.
///
/// This helper is shared between registry factories and other construction
/// paths to avoid duplicating the "if empty, fall back to model_id" logic.
pub fn resolve_common_params(
    ctx_params: Option<crate::types::CommonParams>,
    model_id: &str,
) -> crate::types::CommonParams {
    let mut common = ctx_params.unwrap_or_else(|| crate::types::CommonParams {
        model: model_id.to_string(),
        ..Default::default()
    });
    if common.model.is_empty() {
        common.model = model_id.to_string();
    }
    common
}

/// Resolve base URL with simple override semantics.
///
/// If a custom base URL is provided, it is used as‑is (apart from removing a
/// trailing `/`). The `default_url` is only used when no custom URL is given.
///
/// This matches the behavior of providers in the Vercel AI SDK, where:
/// - `baseURL` is treated as the full API prefix (including any path),
/// - the library does not try to infer or append additional path segments
///   like `/v1` or `/v1beta` when a custom base is supplied.
///
/// # Arguments
/// * `custom_url` - Custom base URL (if any)
/// * `default_url` - Provider's default base URL
///
/// # Returns
/// Resolved base URL
///
/// # Example
/// ```rust,ignore
/// // Custom URL without path
/// let url = resolve_base_url(
///     Some("https://my-server.com".to_string()),
///     "https://api.moonshot.cn/v1"
/// );
/// // Returns "https://my-server.com"
///
/// // Custom URL with path
/// let url = resolve_base_url(
///     Some("https://my-server.com/api/v2".to_string()),
///     "https://api.moonshot.cn/v1"
/// );
/// // Returns "https://my-server.com/api/v2"
/// ```
pub fn resolve_base_url(custom_url: Option<String>, default_url: &str) -> String {
    let url = custom_url.unwrap_or_else(|| default_url.to_string());
    url.trim_end_matches('/').to_string()
}

// Note: ProviderAdapter-based helpers are provider-owned and should live outside
// `siumai-core` to avoid protocol coupling.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_api_key_with_env() {
        // Test with explicit API key
        let result = get_api_key_with_env(Some("test-key".to_string()), "moonshot");
        assert_eq!(result.unwrap(), "test-key");

        // Test with missing API key (should fail)
        let result = get_api_key_with_env(None, "nonexistent_provider_xyz");
        assert!(result.is_err());
    }

    #[test]
    fn test_get_api_key_with_envs_uses_primary_env_when_available() {
        let expected = std::env::var("PATH").expect("PATH should be set in test environment");
        let result = get_api_key_with_envs(
            None,
            "custom_provider",
            Some("PATH"),
            &["SIUMAI_TEST_MISSING_ALIAS".to_string()],
        )
        .expect("api key");
        assert_eq!(result, expected);
    }

    #[test]
    fn test_get_api_key_with_envs_falls_back_to_aliases() {
        let expected = std::env::var("PATH").expect("PATH should be set in test environment");
        let result = get_api_key_with_envs(
            None,
            "custom_provider",
            Some("SIUMAI_TEST_MISSING_PRIMARY"),
            &["PATH".to_string()],
        )
        .expect("api key");
        assert_eq!(result, expected);
    }

    #[test]
    fn test_get_effective_model() {
        // Test with explicit model
        let model = get_effective_model("custom-model", "moonshot");
        assert_eq!(model, "custom-model");

        // Core crate does not provide provider-specific defaults.
        let model = get_effective_model("", "moonshot");
        assert!(model.is_empty());
    }

    #[test]
    fn test_normalize_model_id() {
        // Test DeepSeek alias
        let normalized = normalize_model_id("deepseek", "chat");
        assert_eq!(normalized, "deepseek-chat");

        // Test non-aliased model
        let normalized = normalize_model_id("moonshot", "kimi-k2-0905-preview");
        assert_eq!(normalized, "kimi-k2-0905-preview");
    }

    #[test]
    fn test_resolve_base_url() {
        // Test custom URL without path
        let url = resolve_base_url(
            Some("https://my-server.com".to_string()),
            "https://api.moonshot.cn/v1",
        );
        assert_eq!(url, "https://my-server.com");

        // Test custom URL with path
        let url = resolve_base_url(
            Some("https://my-server.com/api/v2".to_string()),
            "https://api.moonshot.cn/v1",
        );
        assert_eq!(url, "https://my-server.com/api/v2");

        // Test no custom URL
        let url = resolve_base_url(None, "https://api.moonshot.cn/v1");
        assert_eq!(url, "https://api.moonshot.cn/v1");

        // Test custom URL with trailing slash
        let url = resolve_base_url(
            Some("https://my-server.com/".to_string()),
            "https://api.moonshot.cn/v1",
        );
        assert_eq!(url, "https://my-server.com");
    }
}
