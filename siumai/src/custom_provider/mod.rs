//! Custom Provider Framework
//!
//! This module provides a framework for implementing custom AI providers,
//! allowing users to easily integrate new AI services into the library.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::streaming::ChatStream;
use crate::traits::*;
use crate::types::*;

// Documentation-only guide; exclude from normal builds to reduce compile overhead
#[cfg(doc)]
pub mod guide;

/// Custom provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomProviderConfig {
    /// Provider name
    pub name: String,
    /// Base URL for the API
    pub base_url: String,
    /// API key or authentication token
    pub api_key: String,
    /// Default model to use (optional)
    pub model: Option<String>,
    /// Custom headers
    pub headers: HashMap<String, String>,
    /// Request timeout in seconds
    pub timeout: Option<u64>,
    /// Custom parameters
    pub custom_params: HashMap<String, serde_json::Value>,
}

impl CustomProviderConfig {
    /// Create a new custom provider configuration
    pub fn new<S: Into<String>>(name: S, base_url: S, api_key: S) -> Self {
        Self {
            name: name.into(),
            base_url: base_url.into(),
            api_key: api_key.into(),
            model: None,
            headers: HashMap::new(),
            timeout: Some(30),
            custom_params: HashMap::new(),
        }
    }

    /// Set the default model
    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Add a custom header
    pub fn with_header<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// Set timeout
    pub const fn with_timeout(mut self, timeout_seconds: u64) -> Self {
        self.timeout = Some(timeout_seconds);
        self
    }

    /// Add a custom parameter
    pub fn with_param<K: Into<String>, V: Serialize>(mut self, key: K, value: V) -> Self {
        if let Ok(json_value) = serde_json::to_value(value) {
            self.custom_params.insert(key.into(), json_value);
        }
        self
    }
}

/// Custom provider trait
///
/// Implement this trait to create a custom AI provider
#[async_trait]
pub trait CustomProvider: Send + Sync {
    /// Get provider name
    fn name(&self) -> &str;

    /// Get supported models
    fn supported_models(&self) -> Vec<String>;

    /// Get provider capabilities
    fn capabilities(&self) -> ProviderCapabilities;

    /// Send a chat request
    async fn chat(&self, request: CustomChatRequest) -> Result<CustomChatResponse, LlmError>;

    /// Send a streaming chat request
    async fn chat_stream(&self, request: CustomChatRequest) -> Result<ChatStream, LlmError>;

    /// Validate configuration
    fn validate_config(&self, config: &CustomProviderConfig) -> Result<(), LlmError> {
        if config.name.is_empty() {
            return Err(LlmError::InvalidParameter(
                "Provider name cannot be empty".to_string(),
            ));
        }
        if config.base_url.is_empty() {
            return Err(LlmError::InvalidParameter(
                "Base URL cannot be empty".to_string(),
            ));
        }
        if config.api_key.is_empty() {
            return Err(LlmError::InvalidParameter(
                "API key cannot be empty".to_string(),
            ));
        }
        Ok(())
    }

    /// Transform request before sending
    fn transform_request(&self, _request: &mut CustomChatRequest) -> Result<(), LlmError> {
        // Default implementation does nothing
        Ok(())
    }

    /// Transform response after receiving
    fn transform_response(&self, _response: &mut CustomChatResponse) -> Result<(), LlmError> {
        // Default implementation does nothing
        Ok(())
    }
}

/// Custom chat request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomChatRequest {
    /// Messages
    pub messages: Vec<ChatMessage>,
    /// Model name
    pub model: String,
    /// Tools (if supported)
    pub tools: Option<Vec<Tool>>,
    /// Stream flag
    pub stream: bool,
    /// Custom parameters
    pub params: HashMap<String, serde_json::Value>,
}

impl CustomChatRequest {
    /// Create a new custom chat request
    pub fn new(messages: Vec<ChatMessage>, model: String) -> Self {
        Self {
            messages,
            model,
            tools: None,
            stream: false,
            params: HashMap::new(),
        }
    }

    /// Add tools
    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Enable streaming
    pub const fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Add a parameter
    pub fn with_param<K: Into<String>, V: Serialize>(mut self, key: K, value: V) -> Self {
        if let Ok(json_value) = serde_json::to_value(value) {
            self.params.insert(key.into(), json_value);
        }
        self
    }
}

/// Custom chat response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomChatResponse {
    /// Response content
    pub content: String,
    /// Tool calls (if any)
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Usage information
    pub usage: Option<Usage>,
    /// Finish reason
    pub finish_reason: Option<String>,
    /// Custom metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl CustomChatResponse {
    /// Create a new custom chat response
    pub fn new(content: String) -> Self {
        Self {
            content,
            tool_calls: None,
            usage: None,
            finish_reason: None,
            metadata: HashMap::new(),
        }
    }

    /// Add tool calls
    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = Some(tool_calls);
        self
    }

    /// Add usage information
    pub const fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Set finish reason
    pub fn with_finish_reason<S: Into<String>>(mut self, reason: S) -> Self {
        self.finish_reason = Some(reason.into());
        self
    }

    /// Add metadata
    pub fn with_metadata<K: Into<String>, V: Serialize>(mut self, key: K, value: V) -> Self {
        if let Ok(json_value) = serde_json::to_value(value) {
            self.metadata.insert(key.into(), json_value);
        }
        self
    }

    /// Convert to standard `ChatResponse`
    pub fn to_chat_response(&self, _provider_name: &str) -> ChatResponse {
        ChatResponse {
            id: None,
            content: MessageContent::Text(self.content.clone()),
            model: None,
            usage: self.usage.clone(),
            finish_reason: self.finish_reason.as_ref().map(|r| match r.as_str() {
                "stop" => FinishReason::Stop,
                "length" => FinishReason::Length,
                "tool_calls" => FinishReason::ToolCalls,
                "content_filter" => FinishReason::ContentFilter,
                _ => FinishReason::Other(r.clone()),
            }),
            tool_calls: self.tool_calls.clone(),
            thinking: None,
            metadata: self.metadata.clone(),
        }
    }
}

/// Custom provider client wrapper
pub struct CustomProviderClient {
    /// Custom provider implementation
    provider: Box<dyn CustomProvider>,
    /// Configuration
    config: CustomProviderConfig,
    /// HTTP client
    http_client: reqwest::Client,
}

impl CustomProviderClient {
    /// Create a new custom provider client
    pub fn new(
        provider: Box<dyn CustomProvider>,
        config: CustomProviderConfig,
    ) -> Result<Self, LlmError> {
        // Validate configuration
        provider.validate_config(&config)?;

        // Build default HTTP client with custom configuration
        let mut client_builder = reqwest::Client::builder();
        if let Some(timeout) = config.timeout {
            client_builder = client_builder.timeout(std::time::Duration::from_secs(timeout));
        }
        let http_client = client_builder.build().map_err(|e| {
            LlmError::ConfigurationError(format!("Failed to create HTTP client: {e}"))
        })?;

        Self::with_http_client(provider, config, http_client)
    }

    /// Create a new custom provider client with a custom HTTP client
    pub fn with_http_client(
        provider: Box<dyn CustomProvider>,
        config: CustomProviderConfig,
        http_client: reqwest::Client,
    ) -> Result<Self, LlmError> {
        // Validate configuration
        provider.validate_config(&config)?;

        Ok(Self {
            provider,
            config,
            http_client,
        })
    }

    /// Get the underlying provider
    pub fn provider(&self) -> &dyn CustomProvider {
        self.provider.as_ref()
    }

    /// Get the configuration
    pub const fn config(&self) -> &CustomProviderConfig {
        &self.config
    }

    /// Get the HTTP client
    pub const fn http_client(&self) -> &reqwest::Client {
        &self.http_client
    }
}

#[async_trait]
impl ChatCapability for CustomProviderClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        // Use configured model, or first supported model as fallback
        let model = self
            .config
            .model
            .clone()
            .or_else(|| self.provider.supported_models().first().cloned())
            .unwrap_or_else(|| "default".to_string());

        let mut request = CustomChatRequest::new(messages, model);

        if let Some(tools) = tools {
            request = request.with_tools(tools);
        }

        // Add custom parameters from config
        for (key, value) in &self.config.custom_params {
            request.params.insert(key.clone(), value.clone());
        }

        let mut response = self.provider.chat(request).await?;
        self.provider.transform_response(&mut response)?;

        Ok(response.to_chat_response(&self.config.name))
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        // Use configured model, or first supported model as fallback
        let model = self
            .config
            .model
            .clone()
            .or_else(|| self.provider.supported_models().first().cloned())
            .unwrap_or_else(|| "default".to_string());

        let mut request = CustomChatRequest::new(messages, model).with_stream(true);

        if let Some(tools) = tools {
            request = request.with_tools(tools);
        }

        // Add custom parameters from config
        for (key, value) in &self.config.custom_params {
            request.params.insert(key.clone(), value.clone());
        }

        self.provider.chat_stream(request).await
    }
}

impl LlmClient for CustomProviderClient {
    fn provider_name(&self) -> &'static str {
        // Note: This returns a static str, but the actual name is dynamic
        // In a real implementation, you might want to use a different approach
        "custom"
    }

    fn supported_models(&self) -> Vec<String> {
        self.provider.supported_models()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.provider.capabilities()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        // Custom providers need to implement their own cloning logic
        // This is a placeholder implementation
        panic!("Custom provider cloning not implemented")
    }
}

/// Helper trait for building custom providers
pub trait CustomProviderBuilder {
    /// Build the custom provider
    fn build(self) -> Result<Box<dyn CustomProvider>, LlmError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_custom_provider_config_creation() {
        let config = CustomProviderConfig::new("test-provider", "https://api.test.com", "test-key");

        assert_eq!(config.name, "test-provider");
        assert_eq!(config.base_url, "https://api.test.com");
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.model, None);
        assert!(config.headers.is_empty());
        assert_eq!(config.timeout, Some(30));
        assert!(config.custom_params.is_empty());
    }

    #[test]
    fn test_custom_provider_config_with_model() {
        let config = CustomProviderConfig::new("test-provider", "https://api.test.com", "test-key")
            .with_model("test-model-v1")
            .with_header("Authorization", "Bearer token")
            .with_timeout(60)
            .with_param("temperature", 0.7);

        assert_eq!(config.model, Some("test-model-v1".to_string()));
        assert_eq!(
            config.headers.get("Authorization"),
            Some(&"Bearer token".to_string())
        );
        assert_eq!(config.timeout, Some(60));
        assert!(config.custom_params.contains_key("temperature"));
    }
}
