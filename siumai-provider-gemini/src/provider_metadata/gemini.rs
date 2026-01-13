//! Gemini-specific response metadata.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A normalized "source" entry (Vercel-aligned), extracted from grounding chunks.
pub use crate::standards::gemini::GeminiSource;

/// Gemini-specific metadata from chat responses.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GeminiMetadata {
    /// Grounding metadata (for search-grounded responses)
    #[serde(skip_serializing_if = "Option::is_none", rename = "groundingMetadata")]
    pub grounding_metadata: Option<GroundingMetadata>,

    /// Sources extracted from provider-hosted grounding chunks (Vercel-aligned).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sources: Option<Vec<GeminiSource>>,

    /// URL context metadata (for url_context tool responses)
    #[serde(skip_serializing_if = "Option::is_none", rename = "urlContextMetadata")]
    pub url_context_metadata: Option<UrlContextMetadata>,

    /// Safety ratings
    #[serde(skip_serializing_if = "Option::is_none", rename = "safetyRatings")]
    pub safety_ratings: Option<Vec<SafetyRating>>,

    /// Prompt feedback (content filter information for the prompt)
    #[serde(skip_serializing_if = "Option::is_none", rename = "promptFeedback")]
    pub prompt_feedback: Option<serde_json::Value>,

    /// Output only. Average log probability across all tokens in the candidate.
    #[serde(skip_serializing_if = "Option::is_none", rename = "avgLogprobs")]
    pub avg_logprobs: Option<f64>,

    /// Output only. Logprobs result payload (only present when `responseLogprobs == true`).
    #[serde(skip_serializing_if = "Option::is_none", rename = "logprobsResult")]
    pub logprobs_result: Option<GeminiLogprobsResult>,
}

/// Logprobs result payload for a candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiLogprobsResult {
    /// Length equals total number of decoding steps.
    #[serde(default, rename = "topCandidates")]
    pub top_candidates: Vec<GeminiTopCandidates>,
    /// Length equals total number of decoding steps.
    #[serde(default, rename = "chosenCandidates")]
    pub chosen_candidates: Vec<GeminiLogprobsCandidate>,
    /// Sum of log probabilities of all chosen tokens.
    #[serde(skip_serializing_if = "Option::is_none", rename = "logProbabilitySum")]
    pub log_probability_sum: Option<f64>,

    /// Preserve unknown fields for forward compatibility.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Top logprob candidates for a single decoding step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiTopCandidates {
    /// Sorted by log probability descending.
    #[serde(default)]
    pub candidates: Vec<GeminiLogprobsCandidate>,

    /// Preserve unknown fields for forward compatibility.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Logprobs token candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiLogprobsCandidate {
    /// Token string value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    /// Token ID value.
    #[serde(skip_serializing_if = "Option::is_none", rename = "tokenId")]
    pub token_id: Option<i32>,
    /// Token log probability.
    #[serde(skip_serializing_if = "Option::is_none", rename = "logProbability")]
    pub log_probability: Option<f64>,

    /// Preserve unknown fields for forward compatibility.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Grounding metadata for Gemini responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundingMetadata {
    /// List of supporting references retrieved from the specified grounding source
    #[serde(skip_serializing_if = "Option::is_none", rename = "groundingChunks")]
    pub grounding_chunks: Option<Vec<GroundingChunk>>,

    /// List of grounding support information
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "groundingSupports",
        alias = "groundingSupport"
    )]
    pub grounding_supports: Option<Vec<GroundingSupport>>,

    /// Web search queries for follow-up web searches
    #[serde(skip_serializing_if = "Option::is_none", rename = "webSearchQueries")]
    pub web_search_queries: Option<Vec<String>>,

    /// Search entry point (if applicable)
    #[serde(skip_serializing_if = "Option::is_none", rename = "searchEntryPoint")]
    pub search_entry_point: Option<SearchEntryPoint>,

    /// Preserve unknown fields for forward compatibility.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// A grounding chunk (web, retrieved context, maps).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GroundingChunk {
    Web {
        web: WebGroundingChunk,
    },
    RetrievedContext {
        #[serde(rename = "retrievedContext")]
        retrieved_context: RetrievedContextChunk,
    },
    Maps {
        maps: MapsGroundingChunk,
    },
}

/// Grounding chunk from the web
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebGroundingChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

/// Grounding chunk from retrieved context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedContextChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "fileSearchStore")]
    pub file_search_store: Option<String>,
}

/// Grounding chunk from Google Maps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapsGroundingChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "placeId")]
    pub place_id: Option<String>,
}

/// Grounding support information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundingSupport {
    /// Segment of the response that is grounded
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segment: Option<Segment>,

    /// Grounding chunk indices
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "groundingChunkIndices"
    )]
    pub grounding_chunk_indices: Option<Vec<u32>>,

    /// Confidence scores
    #[serde(skip_serializing_if = "Option::is_none", rename = "confidenceScores")]
    pub confidence_scores: Option<Vec<f32>>,
}

/// Segment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    /// Start index
    #[serde(rename = "startIndex")]
    pub start_index: u32,
    /// End index
    #[serde(rename = "endIndex")]
    pub end_index: u32,
    /// Text content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// Search entry point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchEntryPoint {
    /// Rendered content
    #[serde(skip_serializing_if = "Option::is_none", rename = "renderedContent")]
    pub rendered_content: Option<String>,
}

/// URL context metadata for Gemini responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UrlContextMetadata {
    /// List of URL contexts
    #[serde(skip_serializing_if = "Option::is_none", rename = "urlMetadata")]
    pub url_metadata: Option<Vec<UrlMetadata>>,
}

/// Single URL retrieval context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UrlMetadata {
    /// URL retrieved by the tool
    #[serde(skip_serializing_if = "Option::is_none", rename = "retrievedUrl")]
    pub retrieved_url: Option<String>,
    /// URL retrieval status
    #[serde(skip_serializing_if = "Option::is_none", rename = "urlRetrievalStatus")]
    pub url_retrieval_status: Option<UrlRetrievalStatus>,
}

/// URL retrieval status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UrlRetrievalStatus {
    /// Default value. This value is unused.
    #[serde(rename = "URL_RETRIEVAL_STATUS_UNSPECIFIED")]
    Unspecified,
    /// URL was successfully retrieved.
    #[serde(rename = "URL_RETRIEVAL_STATUS_SUCCESS")]
    Success,
    /// URL could not be retrieved due to an error.
    #[serde(rename = "URL_RETRIEVAL_STATUS_ERROR")]
    Error,
    /// URL could not be retrieved because the content is paywalled.
    #[serde(rename = "URL_RETRIEVAL_STATUS_PAYWALL")]
    Paywall,
    /// URL could not be retrieved because the content is unsafe.
    #[serde(rename = "URL_RETRIEVAL_STATUS_UNSAFE")]
    Unsafe,
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

impl crate::types::provider_metadata::FromMetadata for GeminiMetadata {
    fn from_metadata(
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Option<Self> {
        serde_json::from_value(serde_json::to_value(metadata).ok()?).ok()
    }
}

/// Typed helper for Gemini metadata extraction from `ChatResponse`.
pub trait GeminiChatResponseExt {
    fn gemini_metadata(&self) -> Option<GeminiMetadata>;
}

impl GeminiChatResponseExt for crate::types::ChatResponse {
    fn gemini_metadata(&self) -> Option<GeminiMetadata> {
        use crate::types::provider_metadata::FromMetadata;
        let meta = self.provider_metadata.as_ref()?;
        let inner = meta
            .get("google")
            .or_else(|| meta.get("vertex"))
            .or_else(|| meta.get("gemini"))?;
        GeminiMetadata::from_metadata(inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemini_metadata_parses_logprobs_fields() {
        let mut resp = crate::types::ChatResponse::new(crate::types::MessageContent::Text(
            "hello".to_string(),
        ));

        let mut inner = HashMap::new();
        inner.insert("avgLogprobs".to_string(), serde_json::json!(-0.1));
        inner.insert(
            "logprobsResult".to_string(),
            serde_json::json!({
                "chosenCandidates": [
                    { "token": "h", "tokenId": 1, "logProbability": -0.1 }
                ]
            }),
        );

        let mut outer = HashMap::new();
        outer.insert("google".to_string(), inner);
        resp.provider_metadata = Some(outer);

        let meta = resp.gemini_metadata().expect("gemini metadata");
        assert_eq!(meta.avg_logprobs, Some(-0.1));
        assert!(meta.logprobs_result.is_some());
        let chosen = meta.logprobs_result.unwrap().chosen_candidates;
        assert_eq!(chosen.len(), 1);
        assert_eq!(chosen[0].token.as_deref(), Some("h"));
    }
}
