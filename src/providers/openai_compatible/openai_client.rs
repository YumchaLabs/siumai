//! OpenAI Compatible Client
//!
//! This module provides a client implementation for OpenAI-compatible providers.

use super::{openai_config::OpenAiCompatibleConfig, types::RequestType};
use crate::client::LlmClient;
use crate::error::LlmError;
use crate::providers::openai::utils::convert_messages;
use crate::stream::ChatStream;
use crate::traits::{
    ChatCapability, EmbeddingCapability, ImageGenerationCapability, ModelListingCapability,
    RerankCapability,
};
use crate::types::*;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// OpenAI Compatible Chat Response with provider-specific fields
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiCompatibleChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAiCompatibleChoice>,
    pub usage: Option<OpenAiCompatibleUsage>,
}

/// OpenAI Compatible Choice with provider-specific fields
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiCompatibleChoice {
    pub index: u32,
    pub message: OpenAiCompatibleMessage,
    pub finish_reason: Option<String>,
}

/// OpenAI Compatible Message with provider-specific fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiCompatibleMessage {
    pub role: String,
    pub content: Option<serde_json::Value>,
    pub tool_calls: Option<Vec<OpenAiCompatibleToolCall>>,
    pub tool_call_id: Option<String>,

    // Provider-specific thinking/reasoning fields
    pub thinking: Option<String>,          // Standard thinking field
    pub reasoning_content: Option<String>, // DeepSeek reasoning field
    pub reasoning: Option<String>,         // Alternative reasoning field
}

/// OpenAI Compatible Tool Call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiCompatibleToolCall {
    pub id: String,
    pub r#type: String,
    pub function: Option<OpenAiCompatibleFunction>,
}

/// OpenAI Compatible Function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiCompatibleFunction {
    pub name: String,
    pub arguments: String,
}

/// OpenAI Compatible Usage
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiCompatibleUsage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

/// OpenAI compatible client
///
/// This is a separate client implementation that uses the adapter system
/// to handle provider-specific differences without modifying the core OpenAI client.
#[derive(Clone)]
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
        if !config.model.is_empty() {
            config.adapter.validate_model(&config.model)?;
        }

        // Create HTTP client with configuration
        let http_client = Self::build_http_client(&config)?;

        Ok(Self {
            config,
            http_client,
        })
    }

    /// Create a new OpenAI compatible client with custom HTTP client
    pub async fn with_http_client(
        config: OpenAiCompatibleConfig,
        http_client: reqwest::Client,
    ) -> Result<Self, LlmError> {
        // Validate configuration
        config.validate()?;

        // Validate model with adapter
        if !config.model.is_empty() {
            config.adapter.validate_model(&config.model)?;
        }

        Ok(Self {
            config,
            http_client,
        })
    }

    /// Build headers for requests
    fn build_headers(
        &self,
        _request_type: RequestType,
    ) -> Result<reqwest::header::HeaderMap, LlmError> {
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

        // Add headers from HTTP config
        for (key, value) in &self.config.http_config.headers {
            let header_name =
                reqwest::header::HeaderName::from_bytes(key.as_bytes()).map_err(|e| {
                    LlmError::ConfigurationError(format!("Invalid header name '{}': {}", key, e))
                })?;
            let header_value = reqwest::header::HeaderValue::from_str(value).map_err(|e| {
                LlmError::ConfigurationError(format!("Invalid header value '{}': {}", value, e))
            })?;
            headers.insert(header_name, header_value);
        }

        // Add custom headers from config
        for (key, value) in &self.config.custom_headers {
            headers.insert(key, value.clone());
        }

        // Add adapter custom headers
        let adapter_headers = self.config.adapter.custom_headers();
        for (key, value) in adapter_headers.iter() {
            headers.insert(key, value.clone());
        }

        Ok(headers)
    }

    /// Build HTTP client with configuration
    fn build_http_client(config: &OpenAiCompatibleConfig) -> Result<reqwest::Client, LlmError> {
        let mut builder = reqwest::Client::builder();

        // Apply timeout settings
        if let Some(timeout) = config.http_config.timeout {
            builder = builder.timeout(timeout);
        }

        if let Some(connect_timeout) = config.http_config.connect_timeout {
            builder = builder.connect_timeout(connect_timeout);
        }

        // Apply proxy settings
        if let Some(proxy_url) = &config.http_config.proxy {
            let proxy = reqwest::Proxy::all(proxy_url)
                .map_err(|e| LlmError::ConfigurationError(format!("Invalid proxy URL: {}", e)))?;
            builder = builder.proxy(proxy);
        }

        // Apply user agent
        if let Some(user_agent) = &config.http_config.user_agent {
            builder = builder.user_agent(user_agent);
        }

        // Build the client
        builder
            .build()
            .map_err(|e| LlmError::HttpError(format!("Failed to create HTTP client: {}", e)))
    }

    /// Get the provider ID
    pub fn provider_id(&self) -> &str {
        &self.config.provider_id
    }

    /// Get the current model
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// Build request parameters for chat
    async fn build_chat_request(
        &self,
        messages: Vec<ChatMessage>,
    ) -> Result<serde_json::Value, LlmError> {
        // Convert ChatMessage to OpenAI format using the existing utility
        let openai_messages = convert_messages(&messages)?;

        let mut params = serde_json::json!({
            "model": self.config.model,
            "messages": openai_messages,
            "stream": false,
        });

        // Apply adapter transformations
        self.config.adapter.transform_request_params(
            &mut params,
            &self.config.model,
            RequestType::Chat,
        )?;

        Ok(params)
    }

    /// Convert OpenAI Compatible response to ChatResponse (based on OpenAI's parse_chat_response)
    fn parse_chat_response(
        &self,
        response: OpenAiCompatibleChatResponse,
    ) -> Result<ChatResponse, LlmError> {
        // Extract thinking content from multiple possible fields first (before consuming choices)
        let mut thinking_content: Option<String> = None;

        // Use dynamic field accessor for thinking content extraction
        let field_mappings = self.config.adapter.get_field_mappings(&self.config.model);
        let field_accessor = self.config.adapter.get_field_accessor();

        // Convert response to JSON for dynamic field access
        if let Ok(response_json) = serde_json::to_value(&response) {
            thinking_content =
                field_accessor.extract_thinking_content(&response_json, &field_mappings);
        }

        // Now extract the choice (consuming the vector)
        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::ApiError {
                code: 500,
                message: "No choices in response".to_string(),
                details: None,
            })?;

        let content = if let Some(content) = choice.message.content {
            match content {
                serde_json::Value::String(text) => {
                    // If no thinking content found in dedicated fields, check for thinking tags
                    if thinking_content.is_none() {
                        let thinking_regex = regex::Regex::new(r"<think>(.*?)</think>").unwrap();
                        if let Some(captures) = thinking_regex.captures(&text) {
                            thinking_content = Some(captures.get(1).unwrap().as_str().to_string());
                            // Remove thinking tags from main content
                            let cleaned_text =
                                thinking_regex.replace_all(&text, "").trim().to_string();
                            MessageContent::Text(cleaned_text)
                        } else {
                            MessageContent::Text(text)
                        }
                    } else {
                        // If thinking content was found in dedicated fields, use content as-is
                        MessageContent::Text(text)
                    }
                }
                serde_json::Value::Array(parts) => {
                    let mut content_parts = Vec::new();
                    for part in parts {
                        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                            content_parts.push(ContentPart::Text {
                                text: text.to_string(),
                            });
                        }
                    }
                    MessageContent::MultiModal(content_parts)
                }
                _ => MessageContent::Text(String::new()),
            }
        } else {
            MessageContent::Text(String::new())
        };

        // Convert tool calls
        let tool_calls = choice.message.tool_calls.map(|calls| {
            calls
                .into_iter()
                .map(|call| ToolCall {
                    id: call.id,
                    r#type: call.r#type,
                    function: call.function.map(|f| FunctionCall {
                        name: f.name,
                        arguments: f.arguments,
                    }),
                })
                .collect()
        });

        // Convert usage
        let usage = response.usage.map(|u| Usage {
            prompt_tokens: u.prompt_tokens.unwrap_or(0),
            completion_tokens: u.completion_tokens.unwrap_or(0),
            total_tokens: u.total_tokens.unwrap_or(0),
            cached_tokens: None,
            reasoning_tokens: None,
        });

        // Convert finish reason
        let finish_reason = choice.finish_reason.map(|reason| match reason.as_str() {
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            "tool_calls" => FinishReason::ToolCalls,
            "content_filter" => FinishReason::ContentFilter,
            _ => FinishReason::Other(reason),
        });

        Ok(ChatResponse {
            id: Some(response.id),
            content,
            model: Some(response.model),
            usage,
            finish_reason,
            tool_calls,
            thinking: thinking_content,
            metadata: HashMap::new(),
        })
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

        // Add headers from HTTP config
        for (key, value) in &self.config.http_config.headers {
            let header_name =
                reqwest::header::HeaderName::from_bytes(key.as_bytes()).map_err(|e| {
                    LlmError::ConfigurationError(format!("Invalid header name '{}': {}", key, e))
                })?;
            let header_value = reqwest::header::HeaderValue::from_str(value).map_err(|e| {
                LlmError::ConfigurationError(format!("Invalid header value '{}': {}", value, e))
            })?;
            headers.insert(header_name, header_value);
        }

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

        // Parse as OpenAI Compatible response first
        let openai_response: OpenAiCompatibleChatResponse = serde_json::from_str(&response_text)
            .map_err(|e| {
                LlmError::ParseError(format!("Failed to parse OpenAI Compatible response: {}", e))
            })?;

        // Convert to our ChatResponse format using the same logic as OpenAI
        let chat_response = self.parse_chat_response(openai_response)?;

        Ok(chat_response)
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        // Create a ChatRequest from the parameters
        let request = crate::types::ChatRequest {
            messages,
            tools,
            common_params: self.config.common_params.clone(),
            provider_params: None,
            http_config: Some(self.config.http_config.clone()),
            web_search: None,
            stream: true,
        };

        // Create streaming client
        let streaming_client = super::streaming::OpenAiCompatibleStreaming::new(
            self.config.clone(),
            self.config.adapter.clone(),
            self.http_client.clone(),
        );

        streaming_client.create_chat_stream(request).await
    }
}

#[async_trait]
impl EmbeddingCapability for OpenAiCompatibleClient {
    async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        let mut params = serde_json::json!({
            "model": self.config.model,
            "input": texts,
        });

        // Apply adapter transformations
        self.config.adapter.transform_request_params(
            &mut params,
            &self.config.model,
            RequestType::Embedding,
        )?;

        let response = self.send_request(params, "embeddings").await?;

        let response_text = response
            .text()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        // Parse the embedding response
        let embedding_response =
            parse_embedding_response(&response_text, &self.config.provider_id)?;

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
            &self.config.model,
            RequestType::Rerank,
        )?;

        let response = self.send_request(params, "rerank").await?;

        let response_text = response
            .text()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        let rerank_response = parse_rerank_response(&response_text, &self.config.provider_id)?;

        Ok(rerank_response)
    }
}

impl OpenAiCompatibleClient {
    /// List available models from the provider
    async fn list_models_internal(&self) -> Result<Vec<ModelInfo>, LlmError> {
        let url = format!("{}/models", self.config.base_url.trim_end_matches('/'));

        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_str(&format!("Bearer {}", self.config.api_key))
                .map_err(|e| LlmError::InvalidInput(format!("Invalid API key: {}", e)))?,
        );
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/json"),
        );

        // Add headers from HTTP config
        for (key, value) in &self.config.http_config.headers {
            let header_name =
                reqwest::header::HeaderName::from_bytes(key.as_bytes()).map_err(|e| {
                    LlmError::ConfigurationError(format!("Invalid header name '{}': {}", key, e))
                })?;
            let header_value = reqwest::header::HeaderValue::from_str(value).map_err(|e| {
                LlmError::ConfigurationError(format!("Invalid header value '{}': {}", value, e))
            })?;
            headers.insert(header_name, header_value);
        }

        // Add custom headers from config
        for (key, value) in &self.config.custom_headers {
            headers.insert(key, value.clone());
        }

        let response = self.http_client.get(&url).headers(headers).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!(
                    "{} Models API error: {}",
                    self.config.provider_id, error_text
                ),
                details: serde_json::from_str(&error_text).ok(),
            });
        }

        let models_response: serde_json::Value = response.json().await?;

        // Parse OpenAI-compatible models response
        let models = models_response
            .get("data")
            .and_then(|data| data.as_array())
            .ok_or_else(|| LlmError::ParseError("Invalid models response format".to_string()))?;

        let mut model_infos = Vec::new();
        for model in models {
            if let Some(model_id) = model.get("id").and_then(|id| id.as_str()) {
                let model_info = ModelInfo {
                    id: model_id.to_string(),
                    name: Some(model_id.to_string()),
                    description: model
                        .get("description")
                        .and_then(|d| d.as_str())
                        .map(|s| s.to_string()),
                    owned_by: model
                        .get("owned_by")
                        .and_then(|o| o.as_str())
                        .unwrap_or(&self.config.provider_id)
                        .to_string(),
                    created: model.get("created").and_then(|c| c.as_u64()),
                    capabilities: self.determine_model_capabilities(model_id),
                    context_window: None, // Not typically provided by OpenAI-compatible APIs
                    max_output_tokens: None, // Not typically provided by OpenAI-compatible APIs
                    input_cost_per_token: None, // Not typically provided by OpenAI-compatible APIs
                    output_cost_per_token: None, // Not typically provided by OpenAI-compatible APIs
                };
                model_infos.push(model_info);
            }
        }

        Ok(model_infos)
    }

    /// Determine model capabilities based on model ID
    fn determine_model_capabilities(&self, model_id: &str) -> Vec<String> {
        let mut capabilities = vec!["chat".to_string()];

        // Add capabilities based on model name patterns
        if model_id.contains("embed") || model_id.contains("embedding") {
            capabilities.push("embedding".to_string());
        }

        if model_id.contains("rerank") || model_id.contains("bge-reranker") {
            capabilities.push("rerank".to_string());
        }

        if model_id.contains("flux")
            || model_id.contains("stable-diffusion")
            || model_id.contains("kolors")
        {
            capabilities.push("image_generation".to_string());
        }

        // Add thinking capability for supported models
        if self
            .config
            .adapter
            .get_model_config(model_id)
            .supports_thinking
        {
            capabilities.push("thinking".to_string());
        }

        capabilities
    }

    /// Get detailed information about a specific model
    async fn get_model_internal(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        // First try to get from list
        let models = self.list_models_internal().await?;

        if let Some(model) = models.into_iter().find(|m| m.id == model_id) {
            return Ok(model);
        }

        // If not found in list, create a basic model info
        Ok(ModelInfo {
            id: model_id.clone(),
            name: Some(model_id.clone()),
            description: Some(format!("{} model: {}", self.config.provider_id, model_id)),
            owned_by: self.config.provider_id.clone(),
            created: None,
            capabilities: self.determine_model_capabilities(&model_id),
            context_window: None,
            max_output_tokens: None,
            input_cost_per_token: None,
            output_cost_per_token: None,
        })
    }
}

#[async_trait]
impl ModelListingCapability for OpenAiCompatibleClient {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.list_models_internal().await
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        self.get_model_internal(model_id).await
    }
}

#[async_trait]
impl ImageGenerationCapability for OpenAiCompatibleClient {
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        // Check if provider supports image generation
        if !self.config.adapter.supports_image_generation() {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not support image generation",
                self.config.provider_id
            )));
        }

        let url = format!("{}/images/generations", self.config.base_url);

        // Build headers
        let headers = self.build_headers(RequestType::ImageGeneration)?;

        // Transform request parameters if needed
        let mut request = request;
        self.config.adapter.transform_image_request(&mut request)?;

        // Make request
        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                LlmError::HttpError(format!("Failed to send image generation request: {}", e))
            })?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::api_error(
                status_code,
                format!("Image generation failed: {}", error_text),
            ));
        }

        let image_response: ImageGenerationResponse = response.json().await.map_err(|e| {
            LlmError::ParseError(format!("Failed to parse image generation response: {}", e))
        })?;

        Ok(image_response)
    }

    fn get_supported_sizes(&self) -> Vec<String> {
        self.config.adapter.get_supported_image_sizes()
    }

    fn get_supported_formats(&self) -> Vec<String> {
        self.config.adapter.get_supported_image_formats()
    }

    fn supports_image_editing(&self) -> bool {
        self.config.adapter.supports_image_editing()
    }

    fn supports_image_variations(&self) -> bool {
        self.config.adapter.supports_image_variations()
    }
}

impl LlmClient for OpenAiCompatibleClient {
    fn provider_name(&self) -> &'static str {
        self.config.adapter.provider_id()
    }

    fn supported_models(&self) -> Vec<String> {
        // Return a basic list - could be enhanced with adapter-specific models
        vec![self.config.model.clone()]
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        let adapter_caps = self.config.adapter.capabilities();

        // Convert adapter capabilities to library capabilities
        let mut caps = crate::traits::ProviderCapabilities::new();

        if adapter_caps.chat {
            caps = caps.with_chat();
        }
        if adapter_caps.streaming {
            caps = caps.with_streaming();
        }
        if adapter_caps.embedding {
            caps = caps.with_embedding();
        }
        if adapter_caps.tools {
            caps = caps.with_tools();
        }
        if adapter_caps.vision {
            caps = caps.with_vision();
        }

        caps
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new((*self).clone())
    }

    fn as_embedding_capability(&self) -> Option<&dyn EmbeddingCapability> {
        if self.config.adapter.capabilities().embedding {
            Some(self)
        } else {
            None
        }
    }

    fn as_image_generation_capability(&self) -> Option<&dyn ImageGenerationCapability> {
        if self.config.adapter.supports_image_generation() {
            Some(self)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::openai_compatible::registry::ConfigurableAdapter;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_client_creation() {
        let provider_config = crate::providers::openai_compatible::registry::ProviderConfig {
            id: "test".to_string(),
            name: "Test Provider".to_string(),
            base_url: "https://api.test.com/v1".to_string(),
            field_mappings:
                crate::providers::openai_compatible::registry::ProviderFieldMappings::default(),
            capabilities: vec!["chat".to_string()],
            default_model: Some("test-model".to_string()),
            supports_reasoning: false,
        };

        let config = OpenAiCompatibleConfig::new(
            "test",
            "test-key",
            "https://api.test.com/v1",
            Arc::new(ConfigurableAdapter::new(provider_config)),
        )
        .with_model("test-model");

        let client = OpenAiCompatibleClient::new(config).await.unwrap();
        assert_eq!(client.provider_id(), "test");
        assert_eq!(client.model(), "test-model");
    }

    #[tokio::test]
    async fn test_client_validation() {
        let provider_config = crate::providers::openai_compatible::registry::ProviderConfig {
            id: "test".to_string(),
            name: "Test Provider".to_string(),
            base_url: "https://api.test.com/v1".to_string(),
            field_mappings:
                crate::providers::openai_compatible::registry::ProviderFieldMappings::default(),
            capabilities: vec!["chat".to_string()],
            default_model: Some("test-model".to_string()),
            supports_reasoning: false,
        };

        // Invalid config should fail
        let config = OpenAiCompatibleConfig::new(
            "",
            "test-key",
            "https://api.test.com/v1",
            Arc::new(ConfigurableAdapter::new(provider_config)),
        );

        assert!(OpenAiCompatibleClient::new(config).await.is_err());
    }
}

/// SiliconFlow-specific rerank response structure
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SiliconFlowRerankResponse {
    pub id: String,
    pub results: Vec<RerankResult>,
    pub meta: SiliconFlowMeta,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SiliconFlowMeta {
    pub tokens: RerankTokenUsage,
}

/// Parse rerank response based on provider
fn parse_rerank_response(
    response_text: &str,
    provider_id: &str,
) -> Result<RerankResponse, LlmError> {
    // First try to parse as standard format
    if let Ok(standard_response) = serde_json::from_str::<RerankResponse>(response_text) {
        return Ok(standard_response);
    }

    // If that fails and it's SiliconFlow, try SiliconFlow format
    if provider_id == "siliconflow" {
        let sf_response: SiliconFlowRerankResponse =
            serde_json::from_str(response_text).map_err(|e| {
                LlmError::ParseError(format!("Failed to parse SiliconFlow rerank response: {e}"))
            })?;

        // Convert to standard format
        Ok(RerankResponse {
            id: sf_response.id,
            results: sf_response.results,
            tokens: sf_response.meta.tokens,
        })
    } else {
        // For other providers, return the original parsing error
        Err(LlmError::ParseError(format!(
            "Failed to parse rerank response for provider {}: {}",
            provider_id,
            serde_json::from_str::<RerankResponse>(response_text).unwrap_err()
        )))
    }
}

/// OpenAI-compatible embedding response structure
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct OpenAiCompatibleEmbeddingResponse {
    pub object: String,
    pub data: Vec<OpenAiCompatibleEmbeddingData>,
    pub model: String,
    pub usage: Option<EmbeddingUsage>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct OpenAiCompatibleEmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

/// Parse embedding response based on provider
fn parse_embedding_response(
    response_text: &str,
    _provider_id: &str,
) -> Result<EmbeddingResponse, LlmError> {
    // Try to parse as OpenAI-compatible format
    let openai_response: OpenAiCompatibleEmbeddingResponse = serde_json::from_str(response_text)
        .map_err(|e| LlmError::ParseError(format!("Failed to parse embedding response: {e}")))?;

    // Convert to standard format
    let embeddings = openai_response
        .data
        .into_iter()
        .map(|data| data.embedding)
        .collect();

    let mut response = EmbeddingResponse::new(embeddings, openai_response.model);
    response.usage = openai_response.usage;

    Ok(response)
}
