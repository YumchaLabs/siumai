//! Anthropic-specific response metadata

use serde::{Deserialize, Serialize};

/// Anthropic-specific metadata from chat responses
///
/// This includes information about prompt caching and thinking tokens.
///
/// # Example
///
/// ```rust,ignore
/// use siumai::types::provider_metadata::AnthropicMetadata;
///
/// if let Some(meta) = response.anthropic_metadata() {
///     if let Some(cache_tokens) = meta.cache_read_input_tokens {
///         println!("Cache hit! Saved {} tokens", cache_tokens);
///     }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnthropicMetadata {
    /// Number of input tokens used to create the cache entry
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,

    /// Number of input tokens read from the cache
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,

    /// Number of tokens used for thinking/reasoning
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_tokens: Option<u32>,

    /// Raw thinking content (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
}

impl super::FromMetadata for AnthropicMetadata {
    fn from_metadata(
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Option<Self> {
        serde_json::from_value(serde_json::to_value(metadata).ok()?).ok()
    }
}
