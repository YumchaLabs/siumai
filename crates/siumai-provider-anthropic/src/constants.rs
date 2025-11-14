//! Anthropic provider constants.
//!
//! These mirror the defaults used in the aggregator crate so that
//! provider-specific configuration can live close to the provider
//! implementation.

/// Default Anthropic API base endpoint.
///
/// This matches the default used by the aggregator when no base URL
/// is configured explicitly.
pub const ANTHROPIC_V1_ENDPOINT: &str = "https://api.anthropic.com";

/// Messages API relative path segment.
///
/// The full URL is typically constructed as
/// `format!("{}/v1/messages", base_url.trim_end_matches('/'))`.
pub fn messages_path() -> &'static str {
    "/v1/messages"
}
