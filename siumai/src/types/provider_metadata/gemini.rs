//! Gemini-specific response metadata

use serde::{Deserialize, Serialize};

/// Gemini-specific metadata from chat responses
///
/// This includes information about grounding, search results, and other
/// Gemini-specific response details.
///
/// # Example
///
/// ```rust,ignore
/// use siumai::types::provider_metadata::GeminiMetadata;
///
/// if let Some(meta) = response.gemini_metadata() {
///     if let Some(grounding_metadata) = &meta.grounding_metadata {
///         println!("Grounding support: {:?}", grounding_metadata.grounding_support);
///     }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GeminiMetadata {
    /// Grounding metadata (for search-grounded responses)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grounding_metadata: Option<GroundingMetadata>,

    /// Safety ratings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_ratings: Option<Vec<SafetyRating>>,
}

/// Grounding metadata for Gemini responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundingMetadata {
    /// Grounding support information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grounding_support: Option<Vec<GroundingSupport>>,

    /// Search entry point (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_entry_point: Option<SearchEntryPoint>,
}

/// Grounding support information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundingSupport {
    /// Segment of the response that is grounded
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segment: Option<Segment>,

    /// Grounding chunk indices
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grounding_chunk_indices: Option<Vec<u32>>,

    /// Confidence scores
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence_scores: Option<Vec<f32>>,
}

/// Segment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    /// Start index
    pub start_index: u32,
    /// End index
    pub end_index: u32,
    /// Text content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// Search entry point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchEntryPoint {
    /// Rendered content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rendered_content: Option<String>,
}

/// Safety rating
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyRating {
    /// Category
    pub category: String,
    /// Probability
    pub probability: String,
    /// Blocked
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocked: Option<bool>,
}

impl super::FromMetadata for GeminiMetadata {
    fn from_metadata(
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Option<Self> {
        serde_json::from_value(serde_json::to_value(metadata).ok()?).ok()
    }
}
