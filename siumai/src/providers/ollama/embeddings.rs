//! Ollama Embeddings Implementation
//!
//! This module provides the Ollama implementation of embedding capabilities,
//! supporting all features including model options, truncation control, and keep-alive management.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::LlmError;
use crate::traits::{
    EmbeddingCapability, EmbeddingExtensions, OllamaEmbeddingCapability as OllamaEmbeddingTrait,
};
use crate::types::{
    EmbeddingModelInfo, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage, HttpConfig,
};

use super::config::OllamaParams;
use super::utils::{build_headers, build_model_options, validate_model_name};

/// Ollama embeddings API request structure
#[derive(Debug, Clone, Serialize)]
struct OllamaEmbedRequest {
    /// Model name
    model: String,
    /// Input text or list of texts
    input: serde_json::Value,
    /// Truncate input to fit context length
    #[serde(skip_serializing_if = "Option::is_none")]
    truncate: Option<bool>,
    /// Additional model options
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<HashMap<String, serde_json::Value>>,
    /// Keep model loaded duration
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<String>,
}

/// Ollama embeddings API response structure
#[derive(Debug, Clone, Deserialize)]
struct OllamaEmbedResponse {
    /// Model used
    model: String,
    /// Generated embeddings
    embeddings: Vec<Vec<f64>>,
    /// Total duration in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    total_duration: Option<u64>,
    /// Load duration in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    load_duration: Option<u64>,
    /// Prompt evaluation count
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_eval_count: Option<u32>,
}

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
#[derive(Debug, Clone)]
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
}

impl OllamaEmbeddings {
    /// Create a new Ollama embeddings instance
    pub fn new(
        base_url: String,
        default_model: String,
        http_client: reqwest::Client,
        http_config: HttpConfig,
        ollama_params: OllamaParams,
    ) -> Self {
        Self {
            base_url,
            default_model,
            http_client,
            http_config,
            ollama_params,
        }
    }

    /// Get the default embedding model
    fn default_model(&self) -> &str {
        &self.default_model
    }

    /// Build the request body for Ollama API
    fn build_request(
        &self,
        input: &[String],
        model: Option<&str>,
        truncate: Option<bool>,
        options: Option<&HashMap<String, serde_json::Value>>,
        keep_alive: Option<&str>,
    ) -> Result<OllamaEmbedRequest, LlmError> {
        let model = model.unwrap_or(self.default_model()).to_string();
        validate_model_name(&model)?;

        // Convert input to appropriate format
        let input_value = if input.len() == 1 {
            serde_json::Value::String(input[0].clone())
        } else {
            serde_json::Value::Array(
                input
                    .iter()
                    .map(|t| serde_json::Value::String(t.clone()))
                    .collect(),
            )
        };

        // Build model options
        let model_options = build_model_options(
            None, // temperature (not applicable for embeddings)
            None, // max_tokens (not applicable for embeddings)
            None, // top_p (not applicable for embeddings)
            None, // frequency_penalty (not applicable for embeddings)
            None, // presence_penalty (not applicable for embeddings)
            options.or(self.ollama_params.options.as_ref()),
        );

        Ok(OllamaEmbedRequest {
            model,
            input: input_value,
            truncate: truncate.or(Some(true)), // Default to true for safety
            options: if model_options.is_empty() {
                None
            } else {
                Some(model_options)
            },
            keep_alive: keep_alive
                .map(|s| s.to_string())
                .or_else(|| self.ollama_params.keep_alive.clone()),
        })
    }

    /// Make HTTP request to Ollama API
    async fn make_request(
        &self,
        request: OllamaEmbedRequest,
    ) -> Result<OllamaEmbedResponse, LlmError> {
        let headers = build_headers(&self.http_config.headers)?;
        let url = crate::utils::url::join_url(&self.base_url, "api/embed");

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::api_error(
                status_code,
                format!("Ollama API error: {status_code} - {error_text}"),
            ));
        }

        let ollama_response: OllamaEmbedResponse = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(format!("Failed to parse Ollama response: {e}")))?;

        Ok(ollama_response)
    }

    /// Convert Ollama response to our standard format
    fn convert_response(&self, ollama_response: OllamaEmbedResponse) -> EmbeddingResponse {
        // Convert f64 to f32
        let embeddings: Vec<Vec<f32>> = ollama_response
            .embeddings
            .into_iter()
            .map(|embedding| embedding.into_iter().map(|x| x as f32).collect())
            .collect();

        let usage = ollama_response
            .prompt_eval_count
            .map(|count| EmbeddingUsage::new(count, count));

        let mut response = EmbeddingResponse::new(embeddings, ollama_response.model);
        if let Some(usage) = usage {
            response = response.with_usage(usage);
        }

        // Add timing metadata
        if let Some(total_duration) = ollama_response.total_duration {
            response = response.with_metadata(
                "total_duration_ns".to_string(),
                serde_json::Value::Number(serde_json::Number::from(total_duration)),
            );
        }
        if let Some(load_duration) = ollama_response.load_duration {
            response = response.with_metadata(
                "load_duration_ns".to_string(),
                serde_json::Value::Number(serde_json::Number::from(load_duration)),
            );
        }

        response
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
        if input.is_empty() {
            return Err(LlmError::InvalidInput("Input cannot be empty".to_string()));
        }

        let request = self.build_request(&input, None, None, None, None)?;
        let response = self.make_request(request).await?;
        Ok(self.convert_response(response))
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
        if request.input.is_empty() {
            return Err(LlmError::InvalidInput("Input cannot be empty".to_string()));
        }

        // Extract Ollama-specific parameters
        let truncate = request
            .provider_params
            .get("truncate")
            .and_then(|v| v.as_bool());

        let keep_alive = request
            .provider_params
            .get("keep_alive")
            .and_then(|v| v.as_str());

        let options = request
            .provider_params
            .get("options")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect::<HashMap<String, serde_json::Value>>()
            });

        let ollama_request = self.build_request(
            &request.input,
            request.model.as_deref(),
            truncate,
            options.as_ref(),
            keep_alive,
        )?;

        let response = self.make_request(ollama_request).await?;
        Ok(self.convert_response(response))
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

#[async_trait]
impl OllamaEmbeddingTrait for OllamaEmbeddings {
    async fn embed_with_model_options(
        &self,
        input: Vec<String>,
        model: String,
        options: HashMap<String, serde_json::Value>,
    ) -> Result<EmbeddingResponse, LlmError> {
        if input.is_empty() {
            return Err(LlmError::InvalidInput("Input cannot be empty".to_string()));
        }

        let request = self.build_request(&input, Some(&model), None, Some(&options), None)?;
        let response = self.make_request(request).await?;
        Ok(self.convert_response(response))
    }

    async fn embed_with_truncation(
        &self,
        input: Vec<String>,
        truncate: bool,
    ) -> Result<EmbeddingResponse, LlmError> {
        if input.is_empty() {
            return Err(LlmError::InvalidInput("Input cannot be empty".to_string()));
        }

        let request = self.build_request(&input, None, Some(truncate), None, None)?;
        let response = self.make_request(request).await?;
        Ok(self.convert_response(response))
    }

    async fn embed_with_keep_alive(
        &self,
        input: Vec<String>,
        keep_alive: String,
    ) -> Result<EmbeddingResponse, LlmError> {
        if input.is_empty() {
            return Err(LlmError::InvalidInput("Input cannot be empty".to_string()));
        }

        let request = self.build_request(&input, None, None, None, Some(&keep_alive))?;
        let response = self.make_request(request).await?;
        Ok(self.convert_response(response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        );

        let info = embeddings.get_model_info("nomic-embed-text");
        assert_eq!(info.id, "nomic-embed-text");
        assert_eq!(info.dimension, 8192);
        assert_eq!(info.max_input_tokens, 8192);
    }

    #[test]
    fn test_build_request() {
        let config = OllamaParams::default();
        let http_config = HttpConfig::default();
        let client = reqwest::Client::new();
        let embeddings = OllamaEmbeddings::new(
            "http://localhost:11434".to_string(),
            "nomic-embed-text".to_string(),
            client,
            http_config,
            config,
        );

        let input = vec!["test text".to_string()];
        let request = embeddings
            .build_request(&input, None, Some(false), None, None)
            .unwrap();

        assert_eq!(request.model, "nomic-embed-text");
        assert_eq!(request.truncate, Some(false));

        // Test single input format
        if let serde_json::Value::String(text) = &request.input {
            assert_eq!(text, "test text");
        } else {
            panic!("Expected single string input");
        }
    }

    #[test]
    fn test_build_request_multiple_inputs() {
        let config = OllamaParams::default();
        let http_config = HttpConfig::default();
        let client = reqwest::Client::new();
        let embeddings = OllamaEmbeddings::new(
            "http://localhost:11434".to_string(),
            "all-minilm".to_string(),
            client,
            http_config,
            config,
        );

        let input = vec!["text1".to_string(), "text2".to_string()];
        let request = embeddings
            .build_request(&input, None, None, None, None)
            .unwrap();

        assert_eq!(request.model, "all-minilm");

        // Test multiple input format
        if let serde_json::Value::Array(texts) = &request.input {
            assert_eq!(texts.len(), 2);
            assert_eq!(texts[0], serde_json::Value::String("text1".to_string()));
            assert_eq!(texts[1], serde_json::Value::String("text2".to_string()));
        } else {
            panic!("Expected array input");
        }
    }
}
