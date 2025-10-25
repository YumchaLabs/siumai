//! OpenAI Rerank API Standard
//!
//! This module implements the OpenAI-style Rerank API format.
//! Currently supported by providers like SiliconFlow.
//!
//! ## Supported Providers
//!
//! - SiliconFlow
//! - (More providers may adopt this standard in the future)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use siumai::standards::openai::rerank::OpenAiRerankStandard;
//!
//! // Standard OpenAI-style rerank implementation
//! let standard = OpenAiRerankStandard::new();
//!
//! // With provider-specific adapter
//! let standard = OpenAiRerankStandard::with_adapter(
//!     Arc::new(MyCustomAdapter)
//! );
//! ```

use crate::core::RerankTransformers;
use crate::error::LlmError;
use crate::execution::transformers::rerank_request::RerankRequestTransformer;
use crate::execution::transformers::rerank_response::RerankResponseTransformer;
use crate::types::{RerankRequest, RerankResponse};
use std::sync::Arc;

/// OpenAI Rerank API Standard
///
/// Represents the OpenAI-style Rerank API format.
/// Can be used by any provider that implements OpenAI-compatible rerank API.
#[derive(Clone)]
pub struct OpenAiRerankStandard {
    /// Optional adapter for provider-specific differences
    adapter: Option<Arc<dyn OpenAiRerankAdapter>>,
}

impl OpenAiRerankStandard {
    /// Create a new standard OpenAI Rerank implementation
    pub fn new() -> Self {
        Self { adapter: None }
    }

    /// Create with a provider-specific adapter
    pub fn with_adapter(adapter: Arc<dyn OpenAiRerankAdapter>) -> Self {
        Self {
            adapter: Some(adapter),
        }
    }

    /// Create transformers for rerank requests
    pub fn create_transformers(&self, provider_id: &str) -> RerankTransformers {
        let request_tx = Arc::new(OpenAiRerankRequestTransformer {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        });

        let response_tx = Arc::new(OpenAiRerankResponseTransformer {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        });

        RerankTransformers {
            request: request_tx,
            response: response_tx,
        }
    }
}

impl Default for OpenAiRerankStandard {
    fn default() -> Self {
        Self::new()
    }
}

/// Adapter trait for provider-specific differences in OpenAI Rerank API
///
/// Implement this trait to handle provider-specific variations of the OpenAI Rerank API.
pub trait OpenAiRerankAdapter: Send + Sync {
    /// Transform request JSON before sending
    ///
    /// This is called after the standard OpenAI request transformation.
    /// Use this to add provider-specific fields or modify existing ones.
    fn transform_request(
        &self,
        _req: &RerankRequest,
        _body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    /// Transform response JSON after receiving
    ///
    /// This is called before the standard OpenAI response transformation.
    /// Use this to normalize provider-specific response fields.
    fn transform_response(&self, _resp: &mut serde_json::Value) -> Result<(), LlmError> {
        Ok(())
    }

    /// Get provider-specific endpoint path
    ///
    /// Default is "/rerank" (OpenAI-style)
    fn rerank_endpoint(&self) -> &str {
        "/rerank"
    }

    /// Get provider-specific headers
    ///
    /// Default is standard OpenAI headers (Authorization: Bearer <token>)
    fn build_headers(
        &self,
        _api_key: &str,
        _base_headers: &mut reqwest::header::HeaderMap,
    ) -> Result<(), LlmError> {
        Ok(())
    }
}

/// OpenAI Rerank Request Transformer
struct OpenAiRerankRequestTransformer {
    #[allow(dead_code)] // kept for future provider-specific branching/logging
    provider_id: String,
    adapter: Option<Arc<dyn OpenAiRerankAdapter>>,
}

impl RerankRequestTransformer for OpenAiRerankRequestTransformer {
    fn transform(&self, req: &RerankRequest) -> Result<serde_json::Value, LlmError> {
        // Build standard OpenAI-style rerank request
        let mut body = serde_json::json!({
            "model": req.model,
            "query": req.query,
            "documents": req.documents,
        });

        // Add optional parameters
        if let Some(top_n) = req.top_n {
            body["top_n"] = serde_json::json!(top_n);
        }

        if let Some(return_documents) = req.return_documents {
            body["return_documents"] = serde_json::json!(return_documents);
        }

        // Apply adapter transformations
        if let Some(adapter) = &self.adapter {
            adapter.transform_request(req, &mut body)?;
        }

        Ok(body)
    }
}

/// OpenAI Rerank Response Transformer
struct OpenAiRerankResponseTransformer {
    #[allow(dead_code)] // kept for future provider-specific branching/logging
    provider_id: String,
    adapter: Option<Arc<dyn OpenAiRerankAdapter>>,
}

impl RerankResponseTransformer for OpenAiRerankResponseTransformer {
    fn transform(&self, mut resp: serde_json::Value) -> Result<RerankResponse, LlmError> {
        // Apply adapter transformations
        if let Some(adapter) = &self.adapter {
            adapter.transform_response(&mut resp)?;
        }

        // Parse standard OpenAI-style rerank response
        let results = resp["results"]
            .as_array()
            .ok_or_else(|| LlmError::ParseError("Missing 'results' field".into()))?
            .iter()
            .map(|item| {
                let index = item["index"]
                    .as_u64()
                    .ok_or_else(|| LlmError::ParseError("Missing 'index' field".into()))?
                    as u32;

                let relevance_score = item["relevance_score"].as_f64().ok_or_else(|| {
                    LlmError::ParseError("Missing 'relevance_score' field".into())
                })?;

                let document = if let Some(doc) = item.get("document") {
                    if doc.is_string() {
                        Some(crate::types::RerankDocument {
                            text: doc.as_str().unwrap().to_string(),
                        })
                    } else if let Some(text) = doc.get("text").and_then(|t| t.as_str()) {
                        Some(crate::types::RerankDocument {
                            text: text.to_string(),
                        })
                    } else {
                        None
                    }
                } else {
                    None
                };

                Ok(crate::types::RerankResult {
                    index,
                    relevance_score,
                    document,
                })
            })
            .collect::<Result<Vec<_>, LlmError>>()?;

        // Parse usage information
        let tokens = if let Some(usage) = resp.get("usage") {
            crate::types::RerankTokenUsage {
                input_tokens: usage["input_tokens"].as_u64().unwrap_or(0) as u32,
                output_tokens: usage["output_tokens"].as_u64().unwrap_or(0) as u32,
            }
        } else {
            crate::types::RerankTokenUsage {
                input_tokens: 0,
                output_tokens: 0,
            }
        };

        Ok(RerankResponse {
            id: resp["id"].as_str().unwrap_or("").to_string(),
            results,
            tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_rerank_standard_new() {
        let standard = OpenAiRerankStandard::new();
        assert!(standard.adapter.is_none());
    }

    #[test]
    fn test_openai_rerank_standard_with_adapter() {
        struct TestAdapter;
        impl OpenAiRerankAdapter for TestAdapter {}

        let standard = OpenAiRerankStandard::with_adapter(Arc::new(TestAdapter));
        assert!(standard.adapter.is_some());
    }

    #[test]
    fn test_rerank_request_transformer() {
        let standard = OpenAiRerankStandard::new();
        let transformers = standard.create_transformers("test");

        let req = RerankRequest {
            model: "test-model".to_string(),
            query: "test query".to_string(),
            documents: vec!["doc1".to_string(), "doc2".to_string()],
            instruction: None,
            top_n: Some(5),
            return_documents: Some(true),
            max_chunks_per_doc: None,
            overlap_tokens: None,
        };

        let result = transformers.request.transform(&req);
        assert!(result.is_ok());

        let body = result.unwrap();
        assert_eq!(body["model"], "test-model");
        assert_eq!(body["query"], "test query");
        assert_eq!(body["top_n"], 5);
        assert_eq!(body["return_documents"], true);
    }

    #[test]
    fn test_rerank_response_transformer() {
        let standard = OpenAiRerankStandard::new();
        let transformers = standard.create_transformers("test");

        let resp = serde_json::json!({
            "id": "rerank-123",
            "results": [
                {
                    "index": 0,
                    "relevance_score": 0.95,
                    "document": "doc1"
                },
                {
                    "index": 1,
                    "relevance_score": 0.85,
                    "document": "doc2"
                }
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 10
            }
        });

        let result = transformers.response.transform(resp);
        assert!(result.is_ok());

        let rerank_resp = result.unwrap();
        assert_eq!(rerank_resp.id, "rerank-123");
        assert_eq!(rerank_resp.results.len(), 2);
        assert_eq!(rerank_resp.results[0].index, 0);
        assert_eq!(rerank_resp.results[0].relevance_score, 0.95);
        assert_eq!(rerank_resp.results[1].index, 1);
        assert_eq!(rerank_resp.results[1].relevance_score, 0.85);
        assert_eq!(rerank_resp.tokens.input_tokens, 100);
        assert_eq!(rerank_resp.tokens.output_tokens, 10);
    }
}
