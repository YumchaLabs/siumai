//! Gemini-specific response metadata.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A normalized "source" entry (Vercel-aligned), extracted from grounding chunks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct GeminiSource {
    /// Source identifier (stable within a response).
    pub id: String,

    /// Source type ("url" or "document").
    pub source_type: String,

    /// Source URL (only for `source_type = "url"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    /// Optional title.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    /// Media type for document sources (e.g. "application/pdf").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,

    /// Filename for document sources.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
}

/// Gemini-specific metadata from chat responses.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GeminiMetadata {
    /// Grounding metadata (for search-grounded responses)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grounding_metadata: Option<GroundingMetadata>,

    /// Sources extracted from provider-hosted grounding chunks (Vercel-aligned).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sources: Option<Vec<GeminiSource>>,

    /// URL context metadata (for url_context tool responses)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url_context_metadata: Option<UrlContextMetadata>,

    /// Safety ratings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_ratings: Option<Vec<SafetyRating>>,
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
        let meta = self.provider_metadata.as_ref()?.get("gemini")?;
        GeminiMetadata::from_metadata(meta)
    }
}

