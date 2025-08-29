//! OpenAI Compatible Client
//!
//! This module provides a client implementation for OpenAI-compatible providers.

use super::{openai_config::OpenAiCompatibleConfig, types::RequestType};
use crate::error::LlmError;
use crate::stream::ChatStream;
use crate::traits::{ChatCapability, EmbeddingCapability, RerankCapability};
use crate::types::*;
use async_trait::async_trait;

/// OpenAI compatible client
///
/// This is a separate client implementation that uses the adapter system
/// to handle provider-specific differences without modifying the core OpenAI client.
pub struct OpenAiCompatibleClient {
    config: OpenAiCompatibleConfig,
    http_client: reqwest::Client,
}

impl OpenAiCompatibleClient {
    /// Create a new OpenAI compatible client
    pub async fn new(config: OpenAiCompatibleConfig) -> Result<Self, LlmError> {
        // Validate configuration
        config.validate()?;

        // Validate model with adapter
        if !config.default_model.is_empty() {
            config.adapter.validate_model(&config.default_model)?;
        }

        Ok(Self {
            config,
            http_client: reqwest::Client::new(),
        })
    }

    /// Get the provider ID
    pub fn provider_id(&self) -> &str {
        &self.config.provider_id
    }

    /// Get the current model
    pub fn model(&self) -> &str {
        &self.config.default_model
    }

    /// Build request parameters for chat
    async fn build_chat_request(
        &self,
        messages: Vec<ChatMessage>,
    ) -> Result<serde_json::Value, LlmError> {
        let mut params = serde_json::json!({
            "model": self.config.default_model,
            "messages": messages,
            "stream": false,
        });

        // Apply adapter transformations
        self.config.adapter.transform_request_params(
            &mut params,
            &self.config.default_model,
            RequestType::Chat,
        )?;

        Ok(params)
    }

    /// Send HTTP request
    async fn send_request(
        &self,
        params: serde_json::Value,
        endpoint: &str,
    ) -> Result<reqwest::Response, LlmError> {
        let mut headers = reqwest::header::HeaderMap::new();

        // Add authorization header
        headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_str(&format!("Bearer {}", self.config.api_key))
                .map_err(|e| LlmError::ConfigurationError(format!("Invalid API key: {}", e)))?,
        );

        // Add content type
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/json"),
        );

        // Add custom headers from config
        for (key, value) in &self.config.custom_headers {
            headers.insert(key, value.clone());
        }

        // Add adapter custom headers
        let adapter_headers = self.config.adapter.custom_headers();
        for (key, value) in adapter_headers.iter() {
            headers.insert(key, value.clone());
        }

        let url = format!(
            "{}/{}",
            self.config.base_url.trim_end_matches('/'),
            endpoint
        );

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&params)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::api_error(
                status.as_u16(),
                format!("HTTP {}: {}", status, error_text),
            ));
        }

        Ok(response)
    }
}

#[async_trait]
impl ChatCapability for OpenAiCompatibleClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        let mut params = self.build_chat_request(messages).await?;

        // Add tools if provided
        if let Some(tools) = tools {
            params["tools"] = serde_json::to_value(tools)
                .map_err(|e| LlmError::ParseError(format!("Failed to serialize tools: {}", e)))?;
        }

        let response = self.send_request(params, "chat/completions").await?;

        let response_text = response
            .text()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        let chat_response: ChatResponse = serde_json::from_str(&response_text)
            .map_err(|e| LlmError::ParseError(format!("Failed to parse response: {}", e)))?;

        Ok(chat_response)
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        let _ = (messages, tools);

        // For now, return an error indicating streaming is not yet implemented
        // This will be implemented in the streaming module
        Err(LlmError::UnsupportedOperation(
            "Streaming support will be implemented in the streaming module".to_string(),
        ))
    }
}

#[async_trait]
impl EmbeddingCapability for OpenAiCompatibleClient {
    async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        let mut params = serde_json::json!({
            "model": self.config.default_model,
            "input": texts,
        });

        // Apply adapter transformations
        self.config.adapter.transform_request_params(
            &mut params,
            &self.config.default_model,
            RequestType::Embedding,
        )?;

        let response = self.send_request(params, "embeddings").await?;

        let response_text = response
            .text()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        // For now, create a simple response - proper parsing will be implemented later
        let embedding_response = EmbeddingResponse::new(
            vec![vec![0.0; 1536]], // Placeholder embedding
            self.config.default_model.clone(),
        );

        // TODO: Parse actual response when EmbeddingResponse implements Deserialize
        let _ = response_text; // Suppress unused variable warning

        Ok(embedding_response)
    }

    fn embedding_dimension(&self) -> usize {
        // Default dimension, could be made configurable per model
        1536
    }
}

#[async_trait]
impl RerankCapability for OpenAiCompatibleClient {
    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, LlmError> {
        let mut params = serde_json::json!({
            "model": request.model,
            "query": request.query,
            "documents": request.documents,
            "top_n": request.top_n,
        });

        // Apply adapter transformations
        self.config.adapter.transform_request_params(
            &mut params,
            &self.config.default_model,
            RequestType::Rerank,
        )?;

        let response = self.send_request(params, "rerank").await?;

        let response_text = response
            .text()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        let rerank_response: RerankResponse = serde_json::from_str(&response_text)
            .map_err(|e| LlmError::ParseError(format!("Failed to parse response: {}", e)))?;

        Ok(rerank_response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::openai_compatible::providers::siliconflow::SiliconFlowAdapter;

    #[tokio::test]
    async fn test_client_creation() {
        let config = OpenAiCompatibleConfig::new(
            "test",
            "test-key",
            "https://api.test.com/v1",
            Box::new(SiliconFlowAdapter),
        )
        .with_model("test-model");

        let client = OpenAiCompatibleClient::new(config).await.unwrap();
        assert_eq!(client.provider_id(), "test");
        assert_eq!(client.model(), "test-model");
    }

    #[tokio::test]
    async fn test_client_validation() {
        // Invalid config should fail
        let config = OpenAiCompatibleConfig::new(
            "",
            "test-key",
            "https://api.test.com/v1",
            Box::new(SiliconFlowAdapter),
        );

        assert!(OpenAiCompatibleClient::new(config).await.is_err());
    }
}
