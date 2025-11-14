//! MiniMaxi Utility Functions
//!
//! Helper functions for MiniMaxi provider implementation.

use std::collections::HashMap;

/// Create OpenAI-style authentication headers for MiniMaxi
///
/// MiniMaxi's audio, image, video, and music APIs use OpenAI-compatible
/// authentication with Bearer token, while chat API uses Anthropic-style
/// x-api-key authentication.
///
/// # Arguments
///
/// * `api_key` - The MiniMaxi API key
///
/// # Returns
///
/// A HashMap containing the Authorization header with Bearer token
pub(super) fn create_openai_auth_headers(api_key: &str) -> HashMap<String, String> {
    // When the external MiniMaxi provider crate is enabled, delegate header
    // construction to it so that provider-specific behavior lives close to
    // the provider implementation.
    #[cfg(feature = "provider-minimaxi-external")]
    {
        return siumai_provider_minimaxi::build_openai_auth_headers(api_key);
    }

    // Fallback: keep the existing in-crate behavior.
    #[cfg(not(feature = "provider-minimaxi-external"))]
    {
        let mut headers = HashMap::new();
        headers.insert("Authorization".to_string(), format!("Bearer {}", api_key));
        headers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_openai_auth_headers() {
        let headers = create_openai_auth_headers("test-api-key");

        assert_eq!(headers.len(), 1);
        assert_eq!(
            headers.get("Authorization"),
            Some(&"Bearer test-api-key".to_string())
        );
    }
}
