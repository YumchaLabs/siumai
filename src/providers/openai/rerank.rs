//! OpenAI Rerank Capability Implementation
//!
//! This module provides rerank functionality for OpenAI-compatible providers
//! that support document reranking, such as SiliconFlow.

use async_trait::async_trait;
use reqwest::Client;
use secrecy::ExposeSecret;
use serde_json::json;

use crate::error::LlmError;
use crate::providers::openai::config::OpenAiConfig;
use crate::traits::RerankCapability;
use crate::types::{RerankRequest, RerankResponse};

/// OpenAI-compatible rerank capability implementation
#[derive(Debug, Clone)]
pub struct OpenAiRerank {
    /// API key for authentication
    pub api_key: secrecy::SecretString,
    /// Base URL for the API
    pub base_url: String,
    /// HTTP client for making requests
    pub http_client: Client,
    /// Optional organization ID
    pub organization: Option<String>,
    /// Optional project ID
    pub project: Option<String>,
}

impl OpenAiRerank {
    /// Create a new OpenAI rerank capability instance
    pub fn new(
        api_key: secrecy::SecretString,
        base_url: String,
        http_client: Client,
        organization: Option<String>,
        project: Option<String>,
    ) -> Self {
        Self {
            api_key,
            base_url,
            http_client,
            organization,
            project,
        }
    }

    /// Create from OpenAI configuration
    pub fn from_config(config: &OpenAiConfig, http_client: Client) -> Self {
        Self::new(
            config.api_key.clone(),
            config.base_url.clone(),
            http_client,
            config.organization.clone(),
            config.project.clone(),
        )
    }

    /// Build the rerank request URL
    fn build_url(&self) -> String {
        format!("{}/rerank", self.base_url.trim_end_matches('/'))
    }

    /// Build request headers
    fn build_headers(&self) -> Result<reqwest::header::HeaderMap, LlmError> {
        let mut headers = reqwest::header::HeaderMap::new();

        // Authorization header
        let auth_value = format!("Bearer {}", self.api_key.expose_secret());
        headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_str(&auth_value).map_err(|e| {
                LlmError::provider_error("siliconflow", format!("Invalid API key: {e}"))
            })?,
        );

        // Content-Type header
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/json"),
        );

        // Organization header (if provided)
        if let Some(org) = &self.organization {
            headers.insert(
                "OpenAI-Organization",
                reqwest::header::HeaderValue::from_str(org).map_err(|e| {
                    LlmError::provider_error("siliconflow", format!("Invalid organization: {e}"))
                })?,
            );
        }

        // Project header (if provided)
        if let Some(project) = &self.project {
            headers.insert(
                "OpenAI-Project",
                reqwest::header::HeaderValue::from_str(project).map_err(|e| {
                    LlmError::provider_error("siliconflow", format!("Invalid project: {e}"))
                })?,
            );
        }

        Ok(headers)
    }

    /// Convert rerank request to JSON payload
    fn build_payload(&self, request: &RerankRequest) -> serde_json::Value {
        let mut payload = json!({
            "model": request.model,
            "query": request.query,
            "documents": request.documents,
        });

        // Add optional fields
        if let Some(instruction) = &request.instruction {
            payload["instruction"] = json!(instruction);
        }
        if let Some(top_n) = request.top_n {
            payload["top_n"] = json!(top_n);
        }
        if let Some(return_documents) = request.return_documents {
            payload["return_documents"] = json!(return_documents);
        }
        if let Some(max_chunks) = request.max_chunks_per_doc {
            payload["max_chunks_per_doc"] = json!(max_chunks);
        }
        if let Some(overlap) = request.overlap_tokens {
            payload["overlap_tokens"] = json!(overlap);
        }

        payload
    }
}

#[async_trait]
impl RerankCapability for OpenAiRerank {
    /// Rerank documents based on their relevance to a query
    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, LlmError> {
        let url = self.build_url();
        let headers = self.build_headers()?;
        let payload = self.build_payload(&request);

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&payload)
            .send()
            .await
            .map_err(|e| {
                LlmError::provider_error(
                    "siliconflow",
                    format!("Failed to send rerank request: {e}"),
                )
            })?;

        let status = response.status();
        let response_text = response.text().await.map_err(|e| {
            LlmError::provider_error("siliconflow", format!("Failed to read response: {e}"))
        })?;

        if !status.is_success() {
            return Err(LlmError::api_error(
                status.as_u16(),
                format!(
                    "Rerank request failed with status {}: {}",
                    status, response_text
                ),
            ));
        }

        let rerank_response: RerankResponse =
            serde_json::from_str(&response_text).map_err(|e| {
                LlmError::provider_error(
                    "siliconflow",
                    format!("Failed to parse rerank response: {e}"),
                )
            })?;

        Ok(rerank_response)
    }

    /// Get the maximum number of documents that can be reranked
    fn max_documents(&self) -> Option<u32> {
        // SiliconFlow and most providers have reasonable limits
        Some(1000)
    }

    /// Get supported rerank models for this provider
    fn supported_models(&self) -> Vec<String> {
        // Return common rerank models - this could be made configurable
        vec![
            "BAAI/bge-reranker-v2-m3".to_string(),
            "Pro/BAAI/bge-reranker-v2-m3".to_string(),
            "Qwen/Qwen3-Reranker-8B".to_string(),
            "Qwen/Qwen3-Reranker-4B".to_string(),
            "Qwen/Qwen3-Reranker-0.6B".to_string(),
            "netease-youdao/bce-reranker-base_v1".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use secrecy::SecretString;

    #[test]
    fn test_build_url() {
        let rerank = OpenAiRerank::new(
            SecretString::from("test-key"),
            "https://api.siliconflow.cn/v1".to_string(),
            Client::new(),
            None,
            None,
        );

        assert_eq!(rerank.build_url(), "https://api.siliconflow.cn/v1/rerank");
    }

    #[test]
    fn test_build_url_with_trailing_slash() {
        let rerank = OpenAiRerank::new(
            SecretString::from("test-key"),
            "https://api.siliconflow.cn/v1/".to_string(),
            Client::new(),
            None,
            None,
        );

        assert_eq!(rerank.build_url(), "https://api.siliconflow.cn/v1/rerank");
    }

    #[test]
    fn test_build_payload() {
        let rerank = OpenAiRerank::new(
            SecretString::from("test-key"),
            "https://api.test.com/v1".to_string(),
            Client::new(),
            None,
            None,
        );

        let request = RerankRequest::new(
            "test-model".to_string(),
            "test query".to_string(),
            vec!["doc1".to_string(), "doc2".to_string()],
        )
        .with_top_n(5)
        .with_return_documents(true);

        let payload = rerank.build_payload(&request);

        assert_eq!(payload["model"], "test-model");
        assert_eq!(payload["query"], "test query");
        assert_eq!(payload["top_n"], 5);
        assert_eq!(payload["return_documents"], true);
    }

    #[test]
    fn test_supported_models() {
        let rerank = OpenAiRerank::new(
            SecretString::from("test-key"),
            "https://api.test.com/v1".to_string(),
            Client::new(),
            None,
            None,
        );

        let models = rerank.supported_models();
        assert!(!models.is_empty());
        assert!(models.contains(&"BAAI/bge-reranker-v2-m3".to_string()));
    }

    #[test]
    fn test_max_documents() {
        let rerank = OpenAiRerank::new(
            SecretString::from("test-key"),
            "https://api.test.com/v1".to_string(),
            Client::new(),
            None,
            None,
        );

        assert_eq!(rerank.max_documents(), Some(1000));
    }
}
