//! MiniMaxi Utility Functions
//!
//! Helper functions for MiniMaxi provider implementation.

use crate::core::ProviderContext;
use crate::types::HttpConfig;

pub(super) fn build_context(
    api_key: &str,
    base_url: &str,
    http_config: &HttpConfig,
) -> ProviderContext {
    ProviderContext::new(
        "minimaxi",
        base_url.to_string(),
        Some(api_key.to_string()),
        http_config.headers.clone(),
    )
}

/// Resolve the API root base URL for MiniMaxi endpoints that live under `/v1/...`.
///
/// MiniMaxi uses:
/// - Anthropic-compatible chat base: `https://api.minimaxi.com/anthropic`
/// - OpenAI-compatible base: `https://api.minimaxi.com/v1`
/// - File management base (per OpenAPI): `https://api.minimaxi.com` + `/v1/files/...`
///
/// This helper normalizes a user-provided base URL (or the chat base) into the API root.
pub(super) fn resolve_api_root_base_url(base_url: &str) -> String {
    let mut s = base_url.trim().trim_end_matches('/').to_string();

    // Strip a trailing `/v1` if present.
    if s.ends_with("/v1") {
        s.truncate(s.len().saturating_sub(3));
        s = s.trim_end_matches('/').to_string();
    }

    // Strip a trailing `/anthropic` if present.
    if s.ends_with("/anthropic") {
        s.truncate(s.len().saturating_sub("/anthropic".len()));
        s = s.trim_end_matches('/').to_string();
    }

    // Handle the combined form: `/anthropic/v1`.
    if s.ends_with("/v1") {
        s.truncate(s.len().saturating_sub(3));
        s = s.trim_end_matches('/').to_string();
    }

    s
}
