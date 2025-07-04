//! `Groq` Client Implementation
//!
//! Main client implementation that aggregates all Groq capabilities.

use async_trait::async_trait;

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::stream::ChatStream;
use crate::traits::{ChatCapability, ProviderCapabilities};
use crate::types::*;

use super::chat::GroqChatCapability;
use super::config::GroqConfig;

/// `Groq` client that implements all capabilities
#[derive(Clone)]
pub struct GroqClient {
    /// Configuration
    config: GroqConfig,
    /// HTTP client
    http_client: reqwest::Client,
    /// Chat capability
    chat_capability: GroqChatCapability,
}

impl GroqClient {
    /// Create a new `Groq` client
    pub fn new(config: GroqConfig, http_client: reqwest::Client) -> Self {
        let chat_capability = GroqChatCapability::new(
            config.api_key.clone(),
            config.base_url.clone(),
            http_client.clone(),
            config.http_config.clone(),
        );

        Self {
            config,
            http_client,
            chat_capability,
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &GroqConfig {
        &self.config
    }

    /// Get the HTTP client
    pub fn http_client(&self) -> &reqwest::Client {
        &self.http_client
    }

    /// Get chat capability
    pub fn chat_capability(&self) -> &GroqChatCapability {
        &self.chat_capability
    }
}

#[async_trait]
impl LlmClient for GroqClient {
    fn provider_name(&self) -> &'static str {
        "groq"
    }

    fn supported_models(&self) -> Vec<String> {
        GroqConfig::supported_models()
            .iter()
            .map(|&s| s.to_string())
            .collect()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[async_trait]
impl ChatCapability for GroqClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        self.chat_capability.chat_with_tools(messages, tools).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        self.chat_capability.chat_stream(messages, tools).await
    }
}
