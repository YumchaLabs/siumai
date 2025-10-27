//! OpenAI-specific response metadata

use serde::{Deserialize, Serialize};

/// OpenAI-specific metadata from chat responses
///
/// This includes information about reasoning tokens, service tier, and other
/// OpenAI-specific response details.
///
/// # Example
///
/// ```rust,ignore
/// use siumai::types::provider_metadata::OpenAiMetadata;
///
/// if let Some(meta) = response.openai_metadata() {
///     if let Some(reasoning_tokens) = meta.reasoning_tokens {
///         println!("Reasoning tokens used: {}", reasoning_tokens);
///     }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenAiMetadata {
    /// Number of tokens used for reasoning (o1/o3 models)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,

    /// System fingerprint for this response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,

    /// Service tier used for this request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,

    /// Revised prompt (for image generation)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
}

impl super::FromMetadata for OpenAiMetadata {
    fn from_metadata(
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Option<Self> {
        serde_json::from_value(serde_json::to_value(metadata).ok()?).ok()
    }
}
