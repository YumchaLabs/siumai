//! OpenAI Rerank Capability Implementation
//!
//! This module provides rerank functionality for OpenAI-compatible providers
//! that support document reranking, such as SiliconFlow.

use async_trait::async_trait;
use reqwest::Client;
use secrecy::ExposeSecret;
use std::sync::Arc;

use crate::error::LlmError;
use crate::execution::http::transport::HttpTransport;
use crate::providers::openai::config::OpenAiConfig;
use crate::traits::RerankCapability;
use crate::types::{HttpConfig, RerankRequest, RerankResponse};

/// OpenAI-compatible rerank capability implementation
#[derive(Clone)]
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
    /// HTTP configuration for custom headers/proxy/user-agent
    pub http_config: HttpConfig,
    /// Optional custom HTTP transport (Vercel-style "custom fetch" parity).
    pub http_transport: Option<Arc<dyn HttpTransport>>,
}

impl std::fmt::Debug for OpenAiRerank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("OpenAiRerank");
        ds.field("base_url", &self.base_url)
            .field("http_config", &self.http_config);

        if self.organization.is_some() {
            ds.field("has_organization", &true);
        }
        if self.project.is_some() {
            ds.field("has_project", &true);
        }
        if self.http_transport.is_some() {
            ds.field("has_http_transport", &true);
        }

        ds.finish()
    }
}

impl OpenAiRerank {
    /// Create a new OpenAI rerank capability instance
    pub fn new(
        api_key: secrecy::SecretString,
        base_url: String,
        http_client: Client,
        organization: Option<String>,
        project: Option<String>,
        http_config: HttpConfig,
        http_transport: Option<Arc<dyn HttpTransport>>,
    ) -> Self {
        Self {
            api_key,
            base_url,
            http_client,
            organization,
            project,
            http_config,
            http_transport,
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
            config.http_config.clone(),
            config.http_transport.clone(),
        )
    }
}

#[async_trait]
impl RerankCapability for OpenAiRerank {
    /// Rerank documents based on their relevance to a query
    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, LlmError> {
        use crate::execution::executors::rerank::{RerankExecutor, RerankExecutorBuilder};

        // OpenAI's public API does not expose a rerank endpoint. This helper is intended
        // for OpenAI-compatible providers (e.g., SiliconFlow) that implement `/rerank`.
        if self.base_url.to_lowercase().contains("api.openai.com") {
            return Err(LlmError::UnsupportedOperation(
                "Rerank is not supported by OpenAI. Use OpenAI-compatible providers (e.g., siliconflow) instead."
                    .to_string(),
            ));
        }

        let ctx = crate::core::ProviderContext::new(
            "openai",
            self.base_url.clone(),
            Some(self.api_key.expose_secret().to_string()),
            self.http_config.headers.clone(),
        )
        .with_org_project(self.organization.clone(), self.project.clone());

        let spec = std::sync::Arc::new(crate::providers::openai::spec::OpenAiSpecWithRerank::new());
        let mut builder = RerankExecutorBuilder::new("openai", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx);

        if let Some(transport) = self.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

        let exec = builder.build_for_request(&request);

        RerankExecutor::execute(&*exec, request).await
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
    fn test_rerank_url_via_spec() {
        use crate::core::ProviderSpec;

        let rerank = OpenAiRerank::new(
            SecretString::from("test-key"),
            "https://api.siliconflow.cn/v1".to_string(),
            Client::new(),
            None,
            None,
            crate::types::HttpConfig::default(),
            None,
        );
        let req = RerankRequest::new(
            "BAAI/bge-reranker-v2-m3".to_string(),
            "q".to_string(),
            vec!["doc".to_string()],
        );
        let ctx = crate::core::ProviderContext::new(
            "openai",
            rerank.base_url.clone(),
            Some(rerank.api_key.expose_secret().to_string()),
            rerank.http_config.headers.clone(),
        )
        .with_org_project(rerank.organization.clone(), rerank.project.clone());
        let spec = crate::providers::openai::spec::OpenAiSpec::new();

        assert_eq!(
            spec.rerank_url(&req, &ctx),
            "https://api.siliconflow.cn/v1/rerank"
        );
    }

    #[test]
    fn test_rerank_url_via_spec_with_trailing_slash() {
        use crate::core::ProviderSpec;

        let rerank = OpenAiRerank::new(
            SecretString::from("test-key"),
            "https://api.siliconflow.cn/v1/".to_string(),
            Client::new(),
            None,
            None,
            crate::types::HttpConfig::default(),
            None,
        );
        let req = RerankRequest::new(
            "BAAI/bge-reranker-v2-m3".to_string(),
            "q".to_string(),
            vec!["doc".to_string()],
        );
        let ctx = crate::core::ProviderContext::new(
            "openai",
            rerank.base_url.clone(),
            Some(rerank.api_key.expose_secret().to_string()),
            rerank.http_config.headers.clone(),
        )
        .with_org_project(rerank.organization.clone(), rerank.project.clone());
        let spec = crate::providers::openai::spec::OpenAiSpec::new();

        assert_eq!(
            spec.rerank_url(&req, &ctx),
            "https://api.siliconflow.cn/v1/rerank"
        );
    }

    #[test]
    fn test_supported_models() {
        let rerank = OpenAiRerank::new(
            SecretString::from("test-key"),
            "https://api.test.com/v1".to_string(),
            Client::new(),
            None,
            None,
            crate::types::HttpConfig::default(),
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
            crate::types::HttpConfig::default(),
            None,
        );

        assert_eq!(rerank.max_documents(), Some(1000));
    }
}
