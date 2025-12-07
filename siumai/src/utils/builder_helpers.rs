//! Builder Helper Functions
//!
//! Shared utility functions for both `SiumaiBuilder` and provider-specific builders
//! to reduce code duplication and ensure consistent behavior.

use crate::error::LlmError;
#[cfg(feature = "openai")]
use std::sync::Arc;

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
    let env_key = format!("{}_API_KEY", provider_id.to_uppercase());
    api_key
        .or_else(|| std::env::var(&env_key).ok())
        .ok_or_else(|| {
            LlmError::ConfigurationError(format!(
                "API key is required for {} (missing {} environment variable or explicit .api_key())",
                provider_id, env_key
            ))
        })
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
        #[cfg(feature = "openai")]
        {
            crate::providers::openai_compatible::default_models::get_default_chat_model(
                _provider_id,
            )
            .unwrap_or("")
            .to_string()
        }
        #[cfg(not(feature = "openai"))]
        {
            String::new()
        }
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

/// Resolve base URL using provider adapter
///
/// Variant of `resolve_base_url` that works with `ProviderAdapter`.
///
/// # Arguments
/// * `custom_url` - Custom base URL (if any)
/// * `adapter` - Provider adapter
///
/// # Returns
/// Resolved base URL
#[cfg(feature = "openai")]
pub fn resolve_base_url_with_adapter(
    custom_url: Option<String>,
    adapter: &Arc<dyn crate::providers::openai_compatible::ProviderAdapter>,
) -> String {
    let url = custom_url.unwrap_or_else(|| adapter.base_url().to_string());
    url.trim_end_matches('/').to_string()
}

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
    fn test_get_effective_model() {
        // Test with explicit model
        let model = get_effective_model("custom-model", "moonshot");
        assert_eq!(model, "custom-model");

        // Test with empty model (should use default)
        #[cfg(feature = "openai")]
        {
            let model = get_effective_model("", "moonshot");
            assert!(!model.is_empty()); // Should have a default
        }
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
