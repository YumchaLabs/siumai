//! Ollama Embeddings Implementation
//!
//! This module provides the Ollama implementation of embedding capabilities,
//! supporting all features including model options, truncation control, and keep-alive management.

use async_trait::async_trait;
use std::sync::Arc;

use crate::core::ProviderContext;
use crate::error::LlmError;
use crate::execution::executors::embedding::{EmbeddingExecutor, EmbeddingExecutorBuilder};
use crate::retry_api::RetryOptions;
use crate::traits::{EmbeddingCapability, EmbeddingExtensions};
use crate::types::{EmbeddingModelInfo, EmbeddingRequest, EmbeddingResponse, HttpConfig};

use super::config::OllamaParams;

/// Ollama embeddings capability implementation.
///
/// This struct provides a comprehensive implementation of Ollama's embedding capabilities,
/// including support for model options, truncation control, and keep-alive management.
///
/// # Supported Models
/// - nomic-embed-text (8192 dimensions)
/// - all-minilm (384 dimensions)
/// - mxbai-embed-large (1024 dimensions)
/// - snowflake-arctic-embed (1024 dimensions)
///
/// # API Reference
/// <https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings>
#[derive(Clone)]
pub struct OllamaEmbeddings {
    /// Base URL for Ollama API
    base_url: String,
    /// Default model to use
    default_model: String,
    /// HTTP client
    http_client: reqwest::Client,
    /// HTTP configuration
    http_config: HttpConfig,
    /// Ollama-specific parameters
    ollama_params: OllamaParams,
    /// Unified retry options (optional)
    retry_options: Option<RetryOptions>,
    /// Optional custom HTTP transport (Vercel-style "custom fetch" parity).
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
}

impl std::fmt::Debug for OllamaEmbeddings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OllamaEmbeddings")
            .field("base_url", &self.base_url)
            .field("default_model", &self.default_model)
            .field(
                "stream_disable_compression",
                &self.http_config.stream_disable_compression,
            )
            .field("has_retry", &self.retry_options.is_some())
            .field("has_http_transport", &self.http_transport.is_some())
            .finish()
    }
}

impl OllamaEmbeddings {
    /// Create a new Ollama embeddings instance
    pub fn new(
        base_url: String,
        default_model: String,
        http_client: reqwest::Client,
        http_config: HttpConfig,
        ollama_params: OllamaParams,
        http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    ) -> Self {
        Self {
            base_url,
            default_model,
            http_client,
            http_config,
            ollama_params,
            retry_options: None,
            http_transport,
        }
    }

    /// Set unified retry options for embedding requests.
    pub fn with_retry_options(mut self, retry: Option<RetryOptions>) -> Self {
        self.retry_options = retry;
        self
    }

    /// Get the default embedding model
    fn default_model(&self) -> &str {
        &self.default_model
    }

    fn build_embedding_executor(
        &self,
        request: &EmbeddingRequest,
    ) -> Arc<crate::execution::executors::embedding::HttpEmbeddingExecutor> {
        let ctx = ProviderContext::new(
            "ollama",
            self.base_url.clone(),
            None,
            self.http_config.headers.clone(),
        );
        let spec = Arc::new(super::spec::OllamaSpecWithConfig::new(
            self.ollama_params.clone(),
            self.default_model.clone(),
        ));

        let mut builder = EmbeddingExecutorBuilder::new("ollama", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx);

        if let Some(transport) = self.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        builder.build_for_request(request)
    }

    /// Get model information for Ollama embedding models
    fn get_model_info(&self, model_id: &str) -> EmbeddingModelInfo {
        match model_id {
            "nomic-embed-text" | "nomic-embed-text:latest" => EmbeddingModelInfo::new(
                model_id.to_string(),
                "Nomic Embed Text".to_string(),
                8192,
                8192,
            ),

            "all-minilm" | "all-minilm:latest" => {
                EmbeddingModelInfo::new(model_id.to_string(), "All MiniLM".to_string(), 384, 512)
            }

            "mxbai-embed-large" | "mxbai-embed-large:latest" => EmbeddingModelInfo::new(
                model_id.to_string(),
                "MxBai Embed Large".to_string(),
                1024,
                512,
            ),

            "snowflake-arctic-embed" | "snowflake-arctic-embed:latest" => EmbeddingModelInfo::new(
                model_id.to_string(),
                "Snowflake Arctic Embed".to_string(),
                1024,
                512,
            ),

            _ => EmbeddingModelInfo::new(
                model_id.to_string(),
                model_id.to_string(),
                1024, // Default dimension
                512,  // Default max tokens
            ),
        }
    }
}

#[async_trait]
impl EmbeddingCapability for OllamaEmbeddings {
    async fn embed(&self, input: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        let request = EmbeddingRequest::new(input);
        let exec = self.build_embedding_executor(&request);
        exec.execute(request).await
    }

    fn embedding_dimension(&self) -> usize {
        let model = self.default_model();
        self.get_model_info(model).dimension
    }

    fn max_tokens_per_embedding(&self) -> usize {
        let model = self.default_model();
        self.get_model_info(model).max_input_tokens
    }

    fn supported_embedding_models(&self) -> Vec<String> {
        vec![
            "nomic-embed-text".to_string(),
            "all-minilm".to_string(),
            "mxbai-embed-large".to_string(),
            "snowflake-arctic-embed".to_string(),
        ]
    }
}

#[async_trait]
impl EmbeddingExtensions for OllamaEmbeddings {
    async fn embed_with_config(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        let exec = self.build_embedding_executor(&request);
        exec.execute(request).await
    }

    async fn list_embedding_models(&self) -> Result<Vec<EmbeddingModelInfo>, LlmError> {
        let models = self.supported_embedding_models();
        let model_infos = models
            .into_iter()
            .map(|id| self.get_model_info(&id))
            .collect();
        Ok(model_infos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_options::OllamaOptions;

    #[test]
    fn test_embedding_dimensions() {
        let config = OllamaParams::default();
        let http_config = HttpConfig::default();
        let client = reqwest::Client::new();
        let embeddings = OllamaEmbeddings::new(
            "http://localhost:11434".to_string(),
            "nomic-embed-text".to_string(),
            client,
            http_config,
            config,
            None,
        );

        assert_eq!(embeddings.embedding_dimension(), 8192);
        assert_eq!(embeddings.max_tokens_per_embedding(), 8192);
    }

    #[test]
    fn test_supported_models() {
        let config = OllamaParams::default();
        let http_config = HttpConfig::default();
        let client = reqwest::Client::new();
        let embeddings = OllamaEmbeddings::new(
            "http://localhost:11434".to_string(),
            "nomic-embed-text".to_string(),
            client,
            http_config,
            config,
            None,
        );

        let models = embeddings.supported_embedding_models();
        assert!(models.contains(&"nomic-embed-text".to_string()));
        assert!(models.contains(&"all-minilm".to_string()));
        assert!(models.contains(&"mxbai-embed-large".to_string()));
        assert!(models.contains(&"snowflake-arctic-embed".to_string()));
    }

    #[test]
    fn test_model_info() {
        let config = OllamaParams::default();
        let http_config = HttpConfig::default();
        let client = reqwest::Client::new();
        let embeddings = OllamaEmbeddings::new(
            "http://localhost:11434".to_string(),
            "nomic-embed-text".to_string(),
            client,
            http_config,
            config,
            None,
        );

        let info = embeddings.get_model_info("nomic-embed-text");
        assert_eq!(info.id, "nomic-embed-text");
        assert_eq!(info.dimension, 8192);
        assert_eq!(info.max_input_tokens, 8192);
    }

    #[test]
    fn test_build_request() {
        let params = OllamaParams::default();
        let opts = OllamaOptions::new().with_param("truncate", serde_json::json!(false));
        let req = EmbeddingRequest::new(vec!["test text".to_string()]).with_provider_option(
            "ollama",
            serde_json::to_value(&opts).unwrap_or(serde_json::Value::Null),
        );

        let request = crate::providers::ollama::utils::build_embedding_request(
            &req,
            "nomic-embed-text",
            &params,
        )
        .unwrap();

        assert_eq!(request.model, "nomic-embed-text");
        assert_eq!(request.truncate, Some(false));

        if let serde_json::Value::String(text) = &request.input {
            assert_eq!(text, "test text");
        } else {
            panic!("Expected single string input");
        }
    }

    #[test]
    fn test_build_request_multiple_inputs() {
        let params = OllamaParams::default();
        let req = EmbeddingRequest::new(vec!["text1".to_string(), "text2".to_string()]);
        let request =
            crate::providers::ollama::utils::build_embedding_request(&req, "all-minilm", &params)
                .unwrap();

        assert_eq!(request.model, "all-minilm");

        if let serde_json::Value::Array(texts) = &request.input {
            assert_eq!(texts.len(), 2);
            assert_eq!(texts[0], serde_json::Value::String("text1".to_string()));
            assert_eq!(texts[1], serde_json::Value::String("text2".to_string()));
        } else {
            panic!("Expected array input");
        }
    }
}
