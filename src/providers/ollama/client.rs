//! Ollama Client Implementation
//!
//! Main client that aggregates all Ollama capabilities.

use async_trait::async_trait;

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::retry_api::RetryOptions;
use crate::stream::ChatStream;
use crate::traits::{
    ChatCapability, EmbeddingCapability, LlmProvider, ModelListingCapability, ProviderCapabilities,
};
use crate::types::*;

use super::chat::OllamaChatCapability;
use super::completion::OllamaCompletionCapability;
use super::config::{OllamaConfig, OllamaParams};
use super::embeddings::OllamaEmbeddings;
use super::get_default_models;
use super::models::OllamaModelsCapability;
use super::streaming::OllamaStreaming;

/// Ollama Client
pub struct OllamaClient {
    /// Chat capability implementation
    chat_capability: OllamaChatCapability,
    /// Completion capability implementation
    completion_capability: OllamaCompletionCapability,
    /// Embedding capability implementation
    embedding_capability: OllamaEmbeddings,
    /// Models capability implementation
    models_capability: OllamaModelsCapability,
    /// Streaming capability implementation
    streaming_capability: OllamaStreaming,
    /// Common parameters
    common_params: CommonParams,
    /// Ollama-specific parameters
    ollama_params: OllamaParams,
    /// HTTP client for making requests
    http_client: reqwest::Client,
    /// Base URL for Ollama API
    base_url: String,
    /// Tracing configuration
    tracing_config: Option<crate::tracing::TracingConfig>,
    /// Tracing guard to keep tracing system active (retained but not read)
    #[allow(dead_code)]
    _tracing_guard: Option<tracing_appender::non_blocking::WorkerGuard>,
    /// Unified retry options for chat
    retry_options: Option<RetryOptions>,
}

impl Clone for OllamaClient {
    fn clone(&self) -> Self {
        Self {
            chat_capability: self.chat_capability.clone(),
            completion_capability: self.completion_capability.clone(),
            embedding_capability: self.embedding_capability.clone(),
            models_capability: self.models_capability.clone(),
            streaming_capability: self.streaming_capability.clone(),
            common_params: self.common_params.clone(),
            ollama_params: self.ollama_params.clone(),
            http_client: self.http_client.clone(),
            base_url: self.base_url.clone(),
            tracing_config: self.tracing_config.clone(),
            _tracing_guard: None, // Don't clone the tracing guard
            retry_options: self.retry_options.clone(),
        }
    }
}

impl std::fmt::Debug for OllamaClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OllamaClient")
            .field("provider_name", &"ollama")
            .field("model", &self.common_params.model)
            .field("base_url", &self.base_url)
            .field("temperature", &self.common_params.temperature)
            .field("max_tokens", &self.common_params.max_tokens)
            .field("top_p", &self.common_params.top_p)
            .field("seed", &self.common_params.seed)
            .field("keep_alive", &self.ollama_params.keep_alive)
            .field("num_ctx", &self.ollama_params.num_ctx)
            .field("num_gpu", &self.ollama_params.num_gpu)
            .field("has_tracing", &self.tracing_config.is_some())
            .finish()
    }
}

impl OllamaClient {
    /// Creates a new Ollama client with configuration and HTTP client
    pub fn new(config: OllamaConfig, http_client: reqwest::Client) -> Self {
        let chat_capability = OllamaChatCapability::new(
            config.base_url.clone(),
            http_client.clone(),
            config.http_config.clone(),
            config.ollama_params.clone(),
        );

        let completion_capability = OllamaCompletionCapability::new(
            config.base_url.clone(),
            http_client.clone(),
            config.http_config.clone(),
            config.ollama_params.clone(),
        );

        let embedding_capability = OllamaEmbeddings::new(
            config.base_url.clone(),
            if config.common_params.model.is_empty() {
                "nomic-embed-text".to_string()
            } else {
                config.common_params.model.clone()
            },
            http_client.clone(),
            config.http_config.clone(),
            config.ollama_params.clone(),
        );

        let models_capability = OllamaModelsCapability::new(
            config.base_url.clone(),
            http_client.clone(),
            config.http_config.clone(),
        );

        let streaming_capability = OllamaStreaming::new(http_client.clone());

        Self {
            chat_capability,
            completion_capability,
            embedding_capability,
            models_capability,
            streaming_capability,
            common_params: config.common_params,
            ollama_params: config.ollama_params,
            http_client,
            base_url: config.base_url,
            tracing_config: None,
            _tracing_guard: None,
            retry_options: None,
        }
    }

    /// Creates a new Ollama client with configuration
    pub fn new_with_config(config: OllamaConfig) -> Self {
        let http_client = reqwest::Client::new();
        Self::new(config, http_client)
    }

    /// Get the base URL
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Set the tracing guard to keep tracing system active
    pub(crate) fn set_tracing_guard(
        &mut self,
        guard: Option<tracing_appender::non_blocking::WorkerGuard>,
    ) {
        self._tracing_guard = guard;
    }

    /// Set the tracing configuration
    pub(crate) fn set_tracing_config(&mut self, config: Option<crate::tracing::TracingConfig>) {
        self.tracing_config = config;
    }

    /// Set unified retry options
    pub fn set_retry_options(&mut self, options: Option<RetryOptions>) {
        self.retry_options = options;
    }

    /// Get common parameters
    pub const fn common_params(&self) -> &CommonParams {
        &self.common_params
    }

    /// Get Ollama-specific parameters
    pub const fn ollama_params(&self) -> &OllamaParams {
        &self.ollama_params
    }

    /// Get chat capability (for testing and debugging)
    pub const fn chat_capability(&self) -> &super::chat::OllamaChatCapability {
        &self.chat_capability
    }

    /// Update common parameters
    pub fn with_common_params(mut self, params: CommonParams) -> Self {
        self.common_params = params;
        self
    }

    /// Update Ollama-specific parameters
    pub fn with_ollama_params(mut self, params: OllamaParams) -> Self {
        self.ollama_params = params;
        self
    }

    /// Set model
    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.common_params.model = model.into();
        self
    }

    /// Set temperature
    pub const fn with_temperature(mut self, temperature: f32) -> Self {
        self.common_params.temperature = Some(temperature);
        self
    }

    /// Set max tokens
    pub const fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.common_params.max_tokens = Some(max_tokens);
        self
    }

    /// Set keep alive duration
    pub fn with_keep_alive<S: Into<String>>(mut self, duration: S) -> Self {
        self.ollama_params.keep_alive = Some(duration.into());
        self
    }

    /// Enable raw mode
    pub const fn with_raw(mut self, raw: bool) -> Self {
        self.ollama_params.raw = Some(raw);
        self
    }

    /// Set output format
    pub fn with_format<S: Into<String>>(mut self, format: S) -> Self {
        self.ollama_params.format = Some(format.into());
        self
    }

    /// Add model option
    pub fn with_option<K: Into<String>>(mut self, key: K, value: serde_json::Value) -> Self {
        let mut options = self.ollama_params.options.unwrap_or_default();
        options.insert(key.into(), value);
        self.ollama_params.options = Some(options);
        self
    }

    /// Enable reasoning mode for reasoning models
    pub fn with_reasoning(mut self, enabled: bool) -> Self {
        self.ollama_params.think = Some(enabled);
        self
    }

    /// Enable reasoning mode (convenience method)
    pub fn with_reasoning_enabled(mut self) -> Self {
        self.ollama_params.think = Some(true);
        self
    }

    /// Disable reasoning mode (convenience method)
    pub fn with_reasoning_disabled(mut self) -> Self {
        self.ollama_params.think = Some(false);
        self
    }

    /// Enable thinking mode for thinking models (deprecated alias)
    #[deprecated(
        since = "0.7.1",
        note = "Use `with_reasoning()` instead for consistency"
    )]
    pub fn with_thinking(self, enabled: bool) -> Self {
        self.with_reasoning(enabled)
    }

    /// Enable thinking mode (deprecated alias)
    #[deprecated(
        since = "0.7.1",
        note = "Use `with_reasoning_enabled()` instead for consistency"
    )]
    pub fn with_think(self) -> Self {
        self.with_reasoning_enabled()
    }

    /// Disable thinking mode (deprecated alias)
    #[deprecated(
        since = "0.7.1",
        note = "Use `with_reasoning_disabled()` instead for consistency"
    )]
    pub fn with_no_think(self) -> Self {
        self.with_reasoning_disabled()
    }

    /// Generate text completion (using /api/generate endpoint)
    pub async fn generate(&self, prompt: String) -> Result<String, LlmError> {
        self.completion_capability.generate(prompt).await
    }

    /// Generate text completion with streaming
    pub async fn generate_stream(&self, prompt: String) -> Result<ChatStream, LlmError> {
        self.completion_capability.generate_stream(prompt).await
    }

    /// Check if Ollama server is running
    pub async fn health_check(&self) -> Result<bool, LlmError> {
        let url = format!("{}/api/version", self.base_url);

        match self.http_client.get(&url).send().await {
            Ok(response) => Ok(response.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    /// Get Ollama version
    pub async fn version(&self) -> Result<String, LlmError> {
        let url = format!("{}/api/version", self.base_url);

        let response = self.http_client.get(&url).send().await?;

        if !response.status().is_success() {
            return Err(LlmError::HttpError(format!(
                "Failed to get version: {}",
                response.status()
            )));
        }

        let version_response: super::types::OllamaVersionResponse = response.json().await?;
        Ok(version_response.version)
    }
}

#[async_trait]
impl ChatCapability for OllamaClient {
    /// Chat with tools implementation
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        let call = || {
            let req = ChatRequest {
                messages: messages.clone(),
                tools: tools.clone(),
                common_params: self.common_params.clone(),
                provider_params: None,
                http_config: None,
                web_search: None,
                stream: false,
                telemetry: None,
            };
            async move { self.chat_capability.chat(req).await }
        };

        if let Some(opts) = &self.retry_options {
            crate::retry_api::retry_with(call, opts.clone()).await
        } else {
            call().await
        }
    }

    /// Streaming chat with tools
    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        // Create a ChatRequest with proper common_params
        let request = ChatRequest {
            messages,
            tools,
            common_params: self.common_params.clone(),
            provider_params: None,
            http_config: None,
            web_search: None,
            stream: true,
            telemetry: None,
        };

        let headers = crate::providers::ollama::utils::build_headers(
            &self.chat_capability.http_config.headers,
        )?;
        let body = self.chat_capability.build_chat_request_body(&request)?;
        let url = format!("{}/api/chat", self.base_url);

        // Use the dedicated streaming capability
        self.streaming_capability
            .clone()
            .create_chat_stream(url, headers, body)
            .await
    }
}

#[async_trait]
impl EmbeddingCapability for OllamaClient {
    async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        if let Some(opts) = &self.retry_options {
            let cap = self.embedding_capability.clone();
            crate::retry_api::retry_with(
                || {
                    let texts = texts.clone();
                    let cap = cap.clone();
                    async move { cap.embed(texts).await }
                },
                opts.clone(),
            )
            .await
        } else {
            self.embedding_capability.embed(texts).await
        }
    }

    fn embedding_dimension(&self) -> usize {
        self.embedding_capability.embedding_dimension()
    }
}

#[async_trait]
impl ModelListingCapability for OllamaClient {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.models_capability.list_models().await
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        self.models_capability.get_model(model_id).await
    }
}

impl LlmClient for OllamaClient {
    fn provider_name(&self) -> &'static str {
        "ollama"
    }

    fn supported_models(&self) -> Vec<String> {
        get_default_models()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_embedding()
            .with_custom_feature("completion", true)
            .with_custom_feature("model_management", true)
            .with_custom_feature("local_models", true)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_embedding_capability(&self) -> Option<&dyn EmbeddingCapability> {
        Some(self)
    }
}

impl LlmProvider for OllamaClient {
    fn provider_name(&self) -> &'static str {
        "ollama"
    }

    fn supported_models(&self) -> Vec<String> {
        get_default_models()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_embedding()
            .with_custom_feature("completion", true)
            .with_custom_feature("model_management", true)
            .with_custom_feature("local_models", true)
    }

    fn http_client(&self) -> &reqwest::Client {
        &self.http_client
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let config = OllamaConfig::default();
        let client = OllamaClient::new_with_config(config);

        assert_eq!(LlmProvider::provider_name(&client), "ollama");
        assert_eq!(client.base_url(), "http://localhost:11434");
    }

    #[test]
    fn test_client_builder_pattern() {
        let config = OllamaConfig::default();
        let client = OllamaClient::new_with_config(config)
            .with_model("llama3.2")
            .with_temperature(0.7)
            .with_max_tokens(1000)
            .with_keep_alive("10m")
            .with_raw(true)
            .with_format("json")
            .with_option(
                "top_p",
                serde_json::Value::Number(serde_json::Number::from_f64(0.9).unwrap()),
            );

        assert_eq!(client.common_params().model, "llama3.2".to_string());
        assert_eq!(client.common_params().temperature, Some(0.7));
        assert_eq!(client.common_params().max_tokens, Some(1000));
        assert_eq!(client.ollama_params().keep_alive, Some("10m".to_string()));
        assert_eq!(client.ollama_params().raw, Some(true));
        assert_eq!(client.ollama_params().format, Some("json".to_string()));
    }
}
