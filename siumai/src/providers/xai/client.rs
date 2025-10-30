//! `xAI` Client Implementation
//!
//! Main client for the `xAI` provider that aggregates all capabilities.

use async_trait::async_trait;

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::execution::middleware::language_model::LanguageModelMiddleware;

use crate::execution::http::interceptor::HttpInterceptor;
use crate::retry_api::RetryOptions;
use crate::streaming::ChatStream;
use crate::traits::{ChatCapability, ModelListingCapability, ProviderCapabilities};
use crate::types::*;
use std::sync::Arc;

use super::api::XaiModels;
use super::chat::XaiChatCapability;
use super::config::XaiConfig;

/// `xAI` Client
///
/// Main client that provides access to all `xAI` capabilities.
/// This client implements the `LlmClient` trait for unified access
/// and also provides `xAI`-specific functionality.
pub struct XaiClient {
    /// Chat capability
    pub chat_capability: XaiChatCapability,
    /// Models capability
    pub models_capability: XaiModels,
    /// Common parameters
    pub common_params: CommonParams,
    /// HTTP client
    pub http_client: reqwest::Client,
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

impl Clone for XaiClient {
    fn clone(&self) -> Self {
        Self {
            chat_capability: self.chat_capability.clone(),
            models_capability: self.models_capability.clone(),
            common_params: self.common_params.clone(),
            http_client: self.http_client.clone(),
            tracing_config: self.tracing_config.clone(),
            _tracing_guard: None, // Don't clone the tracing guard
            retry_options: self.retry_options.clone(),
            http_interceptors: self.http_interceptors.clone(),
        }
    }
}

impl XaiClient {
    /// Create a new `xAI` client
    pub async fn new(config: XaiConfig) -> Result<Self, LlmError> {
        // Validate configuration
        config.validate()?;

        // Use unified HTTP client builder (now includes proxy, user_agent, headers, etc.)
        let http_client =
            crate::execution::http::client::build_http_client_from_config(&config.http_config)?;

        Self::with_http_client(config, http_client).await
    }

    /// Create a new `xAI` client with a custom HTTP client
    pub async fn with_http_client(
        config: XaiConfig,
        http_client: reqwest::Client,
    ) -> Result<Self, LlmError> {
        // Validate configuration
        config.validate()?;

        // Create chat capability
        let chat_capability = XaiChatCapability::new(
            config.api_key.clone(),
            config.base_url.clone(),
            http_client.clone(),
            config.http_config.clone(),
            config.common_params.clone(),
        );

        // Create models capability
        let models_capability = XaiModels::new(
            config.api_key.clone(),
            config.base_url.clone(),
            http_client.clone(),
            config.http_config.clone(),
        );

        Ok(Self {
            chat_capability,
            models_capability,
            common_params: config.common_params,
            http_client,
            tracing_config: None,
            _tracing_guard: None,
            retry_options: None,
            http_interceptors: Vec::new(),
        })
    }

    /// Get the current configuration
    pub fn config(&self) -> XaiConfig {
        XaiConfig {
            api_key: self.chat_capability.api_key.clone(),
            base_url: self.chat_capability.base_url.clone(),
            common_params: self.common_params.clone(),
            http_config: self.chat_capability.http_config.clone(),
            web_search_config: WebSearchConfig::default(),
        }
    }

    /// Install model-level middlewares for chat requests.
    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<std::sync::Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.chat_capability = self.chat_capability.clone().with_middlewares(middlewares);
        self
    }

    /// Update common parameters
    pub fn with_common_params(mut self, params: CommonParams) -> Self {
        self.common_params = params;
        self
    }

    /// Update model
    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.common_params.model = model.into();
        self
    }

    /// Update temperature
    pub const fn with_temperature(mut self, temperature: f32) -> Self {
        self.common_params.temperature = Some(temperature);
        self
    }

    /// Update max tokens
    pub const fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.common_params.max_tokens = Some(max_tokens);
        self
    }
}

#[async_trait]
impl LlmClient for XaiClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("xai")
    }

    fn supported_models(&self) -> Vec<String> {
        crate::providers::xai::models::all_models()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_custom_feature("reasoning", true)
            .with_custom_feature("deferred_completion", true)
            .with_custom_feature("structured_outputs", true)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }
}

impl XaiClient {
    async fn chat_with_tools_inner(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        // Delegate to capability's trait method (no provider params)
        self.chat_capability.chat_with_tools(messages, tools).await
    }

    /// Create provider context for this client
    fn build_context(&self) -> crate::core::ProviderContext {
        use secrecy::ExposeSecret;
        crate::core::ProviderContext::new(
            "xai",
            self.chat_capability.base_url.clone(),
            Some(self.chat_capability.api_key.expose_secret().to_string()),
            self.chat_capability.http_config.headers.clone(),
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
        let spec = Arc::new(crate::providers::xai::spec::XaiSpec);
        let bundle = spec.choose_chat_transformers(request, &ctx);
        let before_send_hook = spec.chat_before_send(request, &ctx);

        let mut builder = ChatExecutorBuilder::new("xai", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_transformer_bundle(bundle)
            .with_stream_disable_compression(
                self.chat_capability.http_config.stream_disable_compression,
            )
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
}

#[async_trait]
impl ChatCapability for XaiClient {
    /// Chat with tools implementation
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

    /// Chat stream implementation
    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        // Now that XaiChatCapability has the correct common_params, we can use the trait method directly
        self.chat_capability.chat_stream(messages, tools).await
    }

    async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.chat_request_via_spec(request).await
    }

    async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        self.chat_stream_request_via_spec(request).await
    }
}

/// `xAI`-specific methods
impl XaiClient {
    /// Chat with reasoning effort (for thinking models)
    pub async fn chat_with_reasoning(
        &self,
        messages: Vec<ChatMessage>,
        _reasoning_effort: &str,
    ) -> Result<ChatResponse, LlmError> {
        let request = ChatRequest::builder()
            .messages(messages)
            .common_params(self.common_params.clone())
            .build();

        self.chat_capability.chat_request(request).await
    }

    /// Create a deferred completion
    pub async fn create_deferred_completion(
        &self,
        messages: Vec<ChatMessage>,
    ) -> Result<String, LlmError> {
        let request = ChatRequest::builder()
            .messages(messages)
            .common_params(self.common_params.clone())
            .build();

        // This would return a request_id instead of a full response
        // Implementation would need to handle the deferred response format
        let _response = self.chat_capability.chat_request(request).await?;

        // For now, return a placeholder - this would need proper implementation
        // to handle xAI's deferred completion API response format
        Err(LlmError::UnsupportedOperation(
            "Deferred completion not implemented yet".to_string(),
        ))
    }

    /// Get a deferred completion result
    pub async fn get_deferred_completion(
        &self,
        request_id: &str,
    ) -> Result<ChatResponse, LlmError> {
        use secrecy::ExposeSecret;
        use std::sync::Arc;

        // Build URL
        let url = format!(
            "{}/chat/deferred-completion/{}",
            self.chat_capability.base_url, request_id
        );

        // Use unified common HTTP helper so interceptors/retry/tracing apply consistently
        let ctx = crate::core::ProviderContext::new(
            "xai",
            self.chat_capability.base_url.clone(),
            Some(self.chat_capability.api_key.expose_secret().to_string()),
            self.chat_capability.http_config.headers.clone(),
        );
        let spec = Arc::new(crate::providers::xai::spec::XaiSpec);
        let config = crate::execution::executors::common::HttpExecutionConfig {
            provider_id: "xai".to_string(),
            http_client: self.http_client.clone(),
            provider_spec: spec,
            provider_context: ctx,
            interceptors: self.http_interceptors.clone(),
            retry_options: self.retry_options.clone(),
        };

        let result =
            crate::execution::executors::common::execute_get_request(&config, &url, None).await?;

        match result.status {
            200 => {
                // Not yet implemented: map to UnsupportedOperation to match previous behavior
                Err(LlmError::UnsupportedOperation(
                    "Get deferred completion not implemented yet".to_string(),
                ))
            }
            202 => Err(LlmError::ApiError {
                code: 202,
                message: "Deferred completion not ready yet".to_string(),
                details: Some(result.json),
            }),
            // Non-2xx are already converted to Err by execute_get_request
            other => Err(LlmError::ApiError {
                code: other,
                message: "Unexpected deferred completion status".to_string(),
                details: Some(result.json),
            }),
        }
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
impl ModelListingCapability for XaiClient {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.models_capability.list_models().await
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        self.models_capability.get_model(model_id).await
    }
}
