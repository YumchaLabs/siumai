//! `Groq` Client Implementation
//!
//! Main client implementation that aggregates all Groq capabilities.

use async_trait::async_trait;

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::stream::ChatStream;
use crate::traits::{ChatCapability, ModelListingCapability, ProviderCapabilities};
use crate::types::*;

use super::api::GroqModels;
use super::chat::GroqChatCapability;
use super::config::GroqConfig;

/// `Groq` client that implements all capabilities
pub struct GroqClient {
    /// Configuration
    config: GroqConfig,
    /// HTTP client
    http_client: reqwest::Client,
    /// Chat capability
    chat_capability: GroqChatCapability,
    /// Models capability
    models_capability: GroqModels,
    /// Tracing configuration
    tracing_config: Option<crate::tracing::TracingConfig>,
    /// Tracing guard to keep tracing system active
    _tracing_guard: Option<Option<tracing_appender::non_blocking::WorkerGuard>>,
}

impl Clone for GroqClient {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            http_client: self.http_client.clone(),
            chat_capability: self.chat_capability.clone(),
            models_capability: self.models_capability.clone(),
            tracing_config: self.tracing_config.clone(),
            _tracing_guard: None, // Don't clone the tracing guard
        }
    }
}

impl std::fmt::Debug for GroqClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroqClient")
            .field("provider_name", &"groq")
            .field("model", &self.config.common_params.model)
            .field("base_url", &self.config.base_url)
            .field("temperature", &self.config.common_params.temperature)
            .field("max_tokens", &self.config.common_params.max_tokens)
            .field("top_p", &self.config.common_params.top_p)
            .field("seed", &self.config.common_params.seed)
            .field("has_tracing", &self.tracing_config.is_some())
            .finish()
    }
}

impl GroqClient {
    /// Create a new `Groq` client
    pub fn new(config: GroqConfig, http_client: reqwest::Client) -> Self {
        let chat_capability = GroqChatCapability::new(
            config.api_key.clone(),
            config.base_url.clone(),
            http_client.clone(),
            config.http_config.clone(),
            config.common_params.clone(),
        );

        let models_capability = GroqModels::new(
            config.api_key.clone(),
            config.base_url.clone(),
            http_client.clone(),
            config.http_config.clone(),
        );

        Self {
            config,
            http_client,
            chat_capability,
            models_capability,
            tracing_config: None,
            _tracing_guard: None,
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

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
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

impl GroqClient {
    /// Set the tracing guard to keep tracing system active
    pub(crate) fn set_tracing_guard(
        &mut self,
        guard: Option<Option<tracing_appender::non_blocking::WorkerGuard>>,
    ) {
        self._tracing_guard = guard;
    }

    /// Set the tracing configuration
    pub(crate) fn set_tracing_config(&mut self, config: Option<crate::tracing::TracingConfig>) {
        self.tracing_config = config;
    }
}

#[async_trait]
impl ModelListingCapability for GroqClient {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.models_capability.list_models().await
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        self.models_capability.get_model(model_id).await
    }
}
