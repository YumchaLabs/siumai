//! OpenAI-specific response metadata

use serde::{Deserialize, Serialize};

/// A normalized "source" entry (Vercel-aligned), typically produced from web search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct OpenAiSource {
    /// Source identifier (stable within a response).
    pub id: String,

    /// Source type (currently only "url" for web search results).
    pub source_type: String,

    /// Source URL.
    pub url: String,

    /// Optional title.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    /// Tool call id that produced this source (when applicable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    /// Media type for document sources (e.g. "text/plain").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,

    /// Filename for document sources.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,

    /// Provider-specific metadata for the source (e.g. fileId/containerId/index).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<serde_json::Value>,

    /// Provider-native snippet/summary if present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
}

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

    /// Sources extracted from provider-hosted tool results (Vercel-aligned).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sources: Option<Vec<OpenAiSource>>,
}

impl super::FromMetadata for OpenAiMetadata {
    fn from_metadata(
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Option<Self> {
        serde_json::from_value(serde_json::to_value(metadata).ok()?).ok()
    }
}
