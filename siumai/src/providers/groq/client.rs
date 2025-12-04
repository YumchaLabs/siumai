//! `Groq` Client Implementation
//!
//! Main client implementation that aggregates all Groq capabilities.

use async_trait::async_trait;

use crate::client::LlmClient;
use crate::error::LlmError;

use crate::streaming::ChatStream;
use crate::traits::{ChatCapability, ModelListingCapability, ProviderCapabilities};
use crate::types::*;

use super::api::GroqModels;
use super::chat::GroqChatCapability;
use super::config::GroqConfig;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::retry_api::RetryOptions;
use std::sync::Arc;

// Split capability implementations into submodules (no public API changes)
mod audio;

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
    tracing_config: Option<crate::observability::tracing::TracingConfig>,
    /// Tracing guard to keep tracing system active
    /// NOTE: Tracing subscriber functionality has been moved to siumai-extras
    #[allow(dead_code)]
    _tracing_guard: Option<()>,
    /// Unified retry options for chat
    retry_options: Option<RetryOptions>,
    /// Optional HTTP interceptors applied to all chat requests
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
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
            retry_options: self.retry_options.clone(),
            http_interceptors: self.http_interceptors.clone(),
        }
    }
}

impl std::fmt::Debug for GroqClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroqClient")
            .field("provider_id", &"groq")
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
            retry_options: None,
            http_interceptors: Vec::new(),
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

    /// Install model-level middlewares for chat requests.
    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<std::sync::Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.chat_capability = self.chat_capability.clone().with_middlewares(middlewares);
        self
    }
}

#[async_trait]
impl LlmClient for GroqClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("groq")
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
            .with_custom_feature("audio", true)
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
        if let Some(opts) = &self.retry_options {
            let mut opts = opts.clone();
            if opts.provider.is_none() {
                opts.provider = Some(crate::types::ProviderType::Groq);
            }
            crate::retry_api::retry_with(
                || {
                    let m = messages.clone();
                    let t = tools.clone();
                    async move { self.chat_capability.chat_with_tools(m, t).await }
                },
                opts,
            )
            .await
        } else {
            self.chat_capability.chat_with_tools(messages, tools).await
        }
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        self.chat_capability.chat_stream(messages, tools).await
    }

    async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.chat_request_via_spec(request).await
    }

    async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        self.chat_stream_request_via_spec(request).await
    }
}

// Audio capability implementation moved to client/audio.rs (no behavior change)

impl GroqClient {
    /// Create provider context for this client
    fn build_context(&self) -> crate::core::ProviderContext {
        use secrecy::ExposeSecret;
        crate::core::ProviderContext::new(
            "groq",
            self.config.base_url.clone(),
            Some(self.config.api_key.expose_secret().to_string()),
            self.config.http_config.headers.clone(),
        )
    }

    /// Create chat executor using the builder pattern
    fn build_chat_executor(
        &self,
        request: &ChatRequest,
    ) -> Arc<crate::execution::executors::chat::HttpChatExecutor> {
        use crate::core::ProviderSpec;
        use crate::execution::executors::chat::ChatExecutorBuilder;

        let ctx = self.build_context();
        let spec = Arc::new(crate::providers::groq::spec::GroqSpec);
        let bundle = spec.choose_chat_transformers(request, &ctx);
        let before_send_hook = spec.chat_before_send(request, &ctx);

        let mut builder = ChatExecutorBuilder::new("groq", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_transformer_bundle(bundle)
            .with_stream_disable_compression(self.config.http_config.stream_disable_compression)
            .with_interceptors(self.http_interceptors.clone())
            .with_middlewares(self.chat_capability.middlewares.clone());

        if let Some(hook) = before_send_hook {
            builder = builder.with_before_send(hook);
        }

        builder.build()
    }

    /// Execute chat request via spec (unified implementation)
    async fn chat_request_via_spec(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        use crate::execution::executors::chat::ChatExecutor;

        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute(&*exec, request).await
    }

    /// Execute streaming chat request via spec (unified implementation)
    async fn chat_stream_request_via_spec(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStream, LlmError> {
        use crate::execution::executors::chat::ChatExecutor;

        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute_stream(&*exec, request).await
    }

    /// Set the tracing configuration
    pub(crate) fn set_tracing_config(
        &mut self,
        config: Option<crate::observability::tracing::TracingConfig>,
    ) {
        self.tracing_config = config;
    }

    /// Set unified retry options
    pub fn set_retry_options(&mut self, options: Option<RetryOptions>) {
        self.retry_options = options;
    }

    /// Install HTTP interceptors for all chat requests.
    pub fn with_http_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors.clone();
        let mut cap = self.chat_capability.clone();
        cap.interceptors = interceptors;
        self.chat_capability = cap;
        self
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
