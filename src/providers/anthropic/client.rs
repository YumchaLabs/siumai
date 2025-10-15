//! Anthropic Client Implementation
//!
//! Main client structure that aggregates all Anthropic capabilities.

use async_trait::async_trait;

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::executors::chat::{ChatExecutor, HttpChatExecutor};
// use crate::transformers::{request::RequestTransformer, response::ResponseTransformer};
use crate::params::AnthropicParams;
use crate::retry_api::RetryOptions;
use crate::stream::ChatStream;
use crate::traits::*;
use crate::types::*;
use std::sync::Arc;

use super::models::AnthropicModels;
use super::types::AnthropicSpecificParams;
use super::utils::get_default_models;

/// Anthropic Client
pub struct AnthropicClient {
    /// API key and endpoint configuration
    api_key: String,
    base_url: String,
    http_client: reqwest::Client,
    http_config: HttpConfig,
    /// Models capability implementation
    models_capability: AnthropicModels,
    /// Common parameters
    common_params: CommonParams,
    /// Anthropic-specific parameters
    anthropic_params: AnthropicParams,
    /// Anthropic-specific configuration
    specific_params: AnthropicSpecificParams,
    /// Tracing configuration
    tracing_config: Option<crate::tracing::TracingConfig>,
    /// Tracing guard to keep tracing system active (retained but not read)
    #[allow(dead_code)]
    _tracing_guard: Option<tracing_appender::non_blocking::WorkerGuard>,
    /// Unified retry options for chat
    retry_options: Option<RetryOptions>,
}

impl Clone for AnthropicClient {
    fn clone(&self) -> Self {
        Self {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            http_client: self.http_client.clone(),
            http_config: self.http_config.clone(),
            models_capability: self.models_capability.clone(),
            common_params: self.common_params.clone(),
            anthropic_params: self.anthropic_params.clone(),
            specific_params: self.specific_params.clone(),
            tracing_config: self.tracing_config.clone(),
            _tracing_guard: None, // Don't clone the tracing guard
            retry_options: self.retry_options.clone(),
        }
    }
}

impl std::fmt::Debug for AnthropicClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicClient")
            .field("provider_name", &"anthropic")
            .field("model", &self.common_params.model)
            .field("base_url", &self.base_url)
            .field("temperature", &self.common_params.temperature)
            .field("max_tokens", &self.common_params.max_tokens)
            .field("top_p", &self.common_params.top_p)
            .field("seed", &self.common_params.seed)
            .field(
                "beta_features_count",
                &self.specific_params.beta_features.len(),
            )
            .field(
                "thinking_enabled",
                &self
                    .specific_params
                    .thinking_config
                    .as_ref()
                    .map(|c| c.is_enabled())
                    .unwrap_or(false),
            )
            .field(
                "cache_control_enabled",
                &self.specific_params.cache_control.is_some(),
            )
            .field("has_tracing", &self.tracing_config.is_some())
            .finish()
    }
}

impl AnthropicClient {
    /// Creates a new Anthropic client
    pub fn new(
        api_key: String,
        base_url: String,
        http_client: reqwest::Client,
        common_params: CommonParams,
        anthropic_params: AnthropicParams,
        http_config: HttpConfig,
    ) -> Self {
        let specific_params = AnthropicSpecificParams::default();

        let models_capability = AnthropicModels::new(
            api_key.clone(),
            base_url.clone(),
            http_client.clone(),
            http_config.clone(),
        );

        Self {
            api_key,
            base_url,
            http_client,
            http_config,
            models_capability,
            common_params,
            anthropic_params,
            specific_params,
            tracing_config: None,
            _tracing_guard: None,
            retry_options: None,
        }
    }

    /// Get Anthropic-specific parameters
    pub const fn specific_params(&self) -> &AnthropicSpecificParams {
        &self.specific_params
    }

    /// Get common parameters (for testing and debugging)
    pub const fn common_params(&self) -> &CommonParams {
        &self.common_params
    }

    // Chat capability getter removed after executors migration
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

    /// Update Anthropic-specific parameters
    pub fn with_specific_params(mut self, params: AnthropicSpecificParams) -> Self {
        self.specific_params = params;
        self
    }

    /// Enable beta features
    pub fn with_beta_features(mut self, features: Vec<String>) -> Self {
        self.specific_params.beta_features = features;
        self
    }

    /// Enable prompt caching
    pub fn with_cache_control(mut self, cache_control: super::cache::CacheControl) -> Self {
        self.specific_params.cache_control = Some(cache_control);
        self
    }

    /// Enable thinking mode with specified budget tokens
    pub fn with_thinking_mode(mut self, budget_tokens: Option<u32>) -> Self {
        let config = budget_tokens.map(super::thinking::ThinkingConfig::enabled);
        self.specific_params.thinking_config = config;
        self
    }

    /// Enable thinking mode with default budget (10k tokens)
    pub fn with_thinking_enabled(mut self) -> Self {
        self.specific_params.thinking_config =
            Some(super::thinking::ThinkingConfig::enabled(10000));
        self
    }

    /// Set custom metadata
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.specific_params.metadata = Some(metadata);
        self
    }

    /// Add a beta feature
    pub fn add_beta_feature(mut self, feature: String) -> Self {
        self.specific_params.beta_features.push(feature);
        self
    }

    /// Enable prompt caching with ephemeral type
    pub fn with_ephemeral_cache(self) -> Self {
        self.with_cache_control(super::cache::CacheControl::ephemeral())
    }
}

impl AnthropicClient {
    async fn chat_with_tools_inner(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        // Unified ChatRequest
        let request = ChatRequest {
            messages,
            tools,
            common_params: self.common_params.clone(),
            provider_params: None,
            http_config: None,
            web_search: None,
            stream: false,
        };

        // Executors wiring
        let http = self.http_client.clone();
        let base = self.base_url.clone();
        let api_key = self.api_key.clone();
        let custom_headers = self.http_config.headers.clone();
        let req_tx = super::transformers::AnthropicRequestTransformer::new(Some(
            self.specific_params.clone(),
        ));
        let resp_tx = super::transformers::AnthropicResponseTransformer;
        let headers_builder = move || {
            let mut headers = super::utils::build_headers(&api_key, &custom_headers)?;
            crate::utils::http_headers::inject_tracing_headers(&mut headers);
            Ok(headers)
        };
        let exec = Arc::new(HttpChatExecutor {
            provider_id: "anthropic".to_string(),
            http_client: http,
            request_transformer: Arc::new(req_tx),
            response_transformer: Arc::new(resp_tx),
            stream_transformer: None,
            build_url: Box::new(move |_stream| format!("{}/v1/messages", base)),
            build_headers: Box::new(headers_builder),
            before_send: None,
        });
        exec.execute(request).await
    }
}

#[async_trait]
impl ChatCapability for AnthropicClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        if let Some(opts) = &self.retry_options {
            crate::retry_api::retry_with(
                || {
                    let m = messages.clone();
                    let t = tools.clone();
                    async move { self.chat_with_tools_inner(m, t).await }
                },
                opts.clone(),
            )
            .await
        } else {
            self.chat_with_tools_inner(messages, tools).await
        }
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        let request = ChatRequest {
            messages,
            tools,
            common_params: self.common_params.clone(),
            provider_params: None,
            http_config: None,
            web_search: None,
            stream: true,
        };

        let http = self.http_client.clone();
        let base = self.base_url.clone();
        let api_key = self.api_key.clone();
        let custom_headers = self.http_config.headers.clone();
        let req_tx = super::transformers::AnthropicRequestTransformer::new(Some(
            self.specific_params.clone(),
        ));
        let resp_tx = super::transformers::AnthropicResponseTransformer;
        let stream_converter =
            super::streaming::AnthropicEventConverter::new(self.anthropic_params.clone());
        let stream_tx = super::transformers::AnthropicStreamChunkTransformer {
            provider_id: "anthropic".to_string(),
            inner: stream_converter,
        };
        let headers_builder = move || {
            let mut headers = super::utils::build_headers(&api_key, &custom_headers)?;
            crate::utils::http_headers::inject_tracing_headers(&mut headers);
            Ok(headers)
        };
        let exec = Arc::new(HttpChatExecutor {
            provider_id: "anthropic".to_string(),
            http_client: http,
            request_transformer: Arc::new(req_tx),
            response_transformer: Arc::new(resp_tx),
            stream_transformer: Some(Arc::new(stream_tx)),
            build_url: Box::new(move |_stream| format!("{}/v1/messages", base)),
            build_headers: Box::new(headers_builder),
            before_send: None,
        });
        exec.execute_stream(request).await
    }
}

#[async_trait]
impl ModelListingCapability for AnthropicClient {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.models_capability.list_models().await
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        self.models_capability.get_model(model_id).await
    }
}

impl LlmClient for AnthropicClient {
    fn provider_name(&self) -> &'static str {
        "anthropic"
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
            .with_custom_feature("prompt_caching", true)
            .with_custom_feature("thinking_mode", true)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_client_creation() {
        let client = AnthropicClient::new(
            "test-key".to_string(),
            "https://api.anthropic.com".to_string(),
            reqwest::Client::new(),
            CommonParams::default(),
            AnthropicParams::default(),
            HttpConfig::default(),
        );

        assert_eq!(client.provider_name(), "anthropic");
        assert!(!client.supported_models().is_empty());
    }

    #[test]
    fn test_anthropic_client_with_specific_params() {
        let client = AnthropicClient::new(
            "test-key".to_string(),
            "https://api.anthropic.com".to_string(),
            reqwest::Client::new(),
            CommonParams::default(),
            AnthropicParams::default(),
            HttpConfig::default(),
        )
        .with_beta_features(vec!["feature1".to_string(), "feature2".to_string()])
        .with_thinking_enabled()
        .with_ephemeral_cache();

        assert_eq!(client.specific_params().beta_features.len(), 2);
        assert!(client.specific_params().thinking_config.is_some());
        assert!(
            client
                .specific_params()
                .thinking_config
                .as_ref()
                .unwrap()
                .is_enabled()
        );
        assert!(client.specific_params().cache_control.is_some());
    }

    #[test]
    fn test_anthropic_client_beta_features() {
        let client = AnthropicClient::new(
            "test-key".to_string(),
            "https://api.anthropic.com".to_string(),
            reqwest::Client::new(),
            CommonParams::default(),
            AnthropicParams::default(),
            HttpConfig::default(),
        )
        .add_beta_feature("computer-use-2024-10-22".to_string())
        .add_beta_feature("prompt-caching-2024-07-31".to_string());

        assert_eq!(client.specific_params().beta_features.len(), 2);
        assert!(
            client
                .specific_params()
                .beta_features
                .contains(&"computer-use-2024-10-22".to_string())
        );
        assert!(
            client
                .specific_params()
                .beta_features
                .contains(&"prompt-caching-2024-07-31".to_string())
        );
    }
}
