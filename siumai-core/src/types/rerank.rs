//! Rerank API Types
//!
//! This module defines types for document reranking functionality,
//! primarily used by providers like SiliconFlow that offer reranking services.

use serde::{Deserialize, Serialize};

use super::ProviderOptionsMap;

/// Request for reranking documents based on a query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankRequest {
    /// Model to use for reranking
    pub model: String,

    /// The search query to rank documents against
    pub query: String,

    /// List of documents to rerank
    pub documents: Vec<String>,

    /// Optional instruction for the reranker (supported by some models)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instruction: Option<String>,

    /// Number of most relevant documents to return
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_n: Option<u32>,

    /// Whether to return document text in response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_documents: Option<bool>,

    /// Maximum number of chunks per document (provider-specific)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_chunks_per_doc: Option<u32>,

    /// Number of token overlaps between chunks (provider-specific)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub overlap_tokens: Option<u32>,

    /// Open provider options map (Vercel-aligned).
    #[serde(
        default,
        rename = "providerOptions",
        skip_serializing_if = "ProviderOptionsMap::is_empty"
    )]
    pub provider_options_map: ProviderOptionsMap,
}

/// Response from reranking operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResponse {
    /// Unique identifier for the rerank request
    pub id: String,

    /// Ranked results
    pub results: Vec<RerankResult>,

    /// Token usage information
    pub tokens: RerankTokenUsage,
}

/// Individual rerank result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResult {
    /// Original document content (if return_documents is true)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<RerankDocument>,

    /// Index of the document in the original input array
    pub index: u32,

    /// Relevance score (higher means more relevant)
    pub relevance_score: f64,
}

/// Document content in rerank result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankDocument {
    /// The document text
    pub text: String,
}

/// Token usage information for rerank operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankTokenUsage {
    /// Number of input tokens processed
    pub input_tokens: u32,

    /// Number of output tokens generated
    pub output_tokens: u32,
}

impl RerankRequest {
    /// Create a new rerank request with required fields
    pub fn new(model: String, query: String, documents: Vec<String>) -> Self {
        Self {
            model,
            query,
            documents,
            instruction: None,
            top_n: None,
            return_documents: None,
            max_chunks_per_doc: None,
            overlap_tokens: None,
            provider_options_map: ProviderOptionsMap::default(),
        }
    }

    /// Replace the full provider options map (open JSON map).
    pub fn with_provider_options_map(mut self, map: ProviderOptionsMap) -> Self {
        self.provider_options_map = map;
        self
    }

    /// Set provider options for a provider id (open JSON map).
    pub fn with_provider_option(
        mut self,
        provider_id: impl AsRef<str>,
        options: serde_json::Value,
    ) -> Self {
        self.provider_options_map.insert(provider_id, options);
        self
    }

    /// Set the instruction for the reranker
    pub fn with_instruction(mut self, instruction: String) -> Self {
        self.instruction = Some(instruction);
        self
    }

    /// Set the number of top results to return
    pub fn with_top_n(mut self, top_n: u32) -> Self {
        self.top_n = Some(top_n);
        self
    }

    /// Set whether to return document text
    pub fn with_return_documents(mut self, return_documents: bool) -> Self {
        self.return_documents = Some(return_documents);
        self
    }

    /// Set maximum chunks per document
    pub fn with_max_chunks_per_doc(mut self, max_chunks: u32) -> Self {
        self.max_chunks_per_doc = Some(max_chunks);
        self
    }

    /// Set overlap tokens between chunks
    pub fn with_overlap_tokens(mut self, overlap: u32) -> Self {
        self.overlap_tokens = Some(overlap);
        self
    }
}

impl RerankResponse {
    /// Get the most relevant document index
    pub fn top_result_index(&self) -> Option<u32> {
        self.results.first().map(|r| r.index)
    }

    /// Get all document indices sorted by relevance
    pub fn sorted_indices(&self) -> Vec<u32> {
        self.results.iter().map(|r| r.index).collect()
    }

    /// Get relevance scores for all results
    pub fn relevance_scores(&self) -> Vec<f64> {
        self.results.iter().map(|r| r.relevance_score).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rerank_request_creation() {
        let request = RerankRequest::new(
            "BAAI/bge-reranker-v2-m3".to_string(),
            "Apple".to_string(),
            vec!["apple".to_string(), "banana".to_string()],
        );

        assert_eq!(request.model, "BAAI/bge-reranker-v2-m3");
        assert_eq!(request.query, "Apple");
        assert_eq!(request.documents.len(), 2);
        assert!(request.instruction.is_none());
    }

    #[test]
    fn test_rerank_request_builder() {
        let request = RerankRequest::new(
            "test-model".to_string(),
            "test query".to_string(),
            vec!["doc1".to_string()],
        )
        .with_instruction("Please rerank".to_string())
        .with_top_n(5)
        .with_return_documents(true);

        assert_eq!(request.instruction, Some("Please rerank".to_string()));
        assert_eq!(request.top_n, Some(5));
        assert_eq!(request.return_documents, Some(true));
    }

    #[test]
    fn test_rerank_response_methods() {
        let response = RerankResponse {
            id: "test-id".to_string(),
            results: vec![
                RerankResult {
                    document: None,
                    index: 2,
                    relevance_score: 0.9,
                },
                RerankResult {
                    document: None,
                    index: 0,
                    relevance_score: 0.7,
                },
            ],
            tokens: RerankTokenUsage {
                input_tokens: 100,
                output_tokens: 10,
            },
        };

        assert_eq!(response.top_result_index(), Some(2));
        assert_eq!(response.sorted_indices(), vec![2, 0]);
        assert_eq!(response.relevance_scores(), vec![0.9, 0.7]);
    }
}
