//! `xAI` Client Implementation
//!
//! Main client for the `xAI` provider that aggregates all capabilities.

use async_trait::async_trait;
use std::time::Duration;

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::middleware::language_model::LanguageModelMiddleware;
use crate::retry_api::RetryOptions;
use crate::stream::ChatStream;
use crate::traits::{ChatCapability, ModelListingCapability, ProviderCapabilities};
use crate::types::*;
use crate::utils::http_interceptor::HttpInterceptor;
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
    tracing_config: Option<crate::tracing::TracingConfig>,
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
        config
            .validate()
            .map_err(|e| LlmError::InvalidInput(format!("Invalid xAI configuration: {e}")))?;

        // Create HTTP client with timeout
        let http_client = reqwest::Client::builder()
            .timeout(
                config
                    .http_config
                    .timeout
                    .unwrap_or(Duration::from_secs(30)),
            )
            .build()
            .map_err(|e| LlmError::HttpError(format!("Failed to create HTTP client: {e}")))?;

        Self::with_http_client(config, http_client).await
    }

    /// Create a new `xAI` client with a custom HTTP client
    pub async fn with_http_client(
        config: XaiConfig,
        http_client: reqwest::Client,
    ) -> Result<Self, LlmError> {
        // Validate configuration
        config
            .validate()
            .map_err(|e| LlmError::InvalidInput(format!("Invalid xAI configuration: {e}")))?;

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
    fn provider_name(&self) -> &'static str {
        "xai"
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
        // Route via transformers + executor path to preserve provider_params
        use crate::executors::chat::{ChatExecutor, HttpChatExecutor};
        let http = self.http_client.clone();
        let base = self.chat_capability.base_url.clone();
        let api_key = self.chat_capability.api_key.clone();
        let custom_headers = self.chat_capability.http_config.headers.clone();
        let req_tx = super::transformers::XaiRequestTransformer;
        let resp_tx = super::transformers::XaiResponseTransformer;
        let api_key_clone = api_key.clone();
        let custom_headers_clone = custom_headers.clone();
        let headers_builder = move || {
            let api_key = api_key_clone.clone();
            let custom_headers = custom_headers_clone.clone();
            Box::pin(async move { super::utils::build_headers(&api_key, &custom_headers) })
                as std::pin::Pin<
                    Box<
                        dyn std::future::Future<
                                Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>,
                            > + Send,
                    >,
                >
        };
        let exec = HttpChatExecutor {
            provider_id: "xai".to_string(),
            http_client: http,
            request_transformer: std::sync::Arc::new(req_tx),
            response_transformer: std::sync::Arc::new(resp_tx),
            stream_transformer: None,
            json_stream_converter: None,
            stream_disable_compression: self.chat_capability.http_config.stream_disable_compression,
            interceptors: self.http_interceptors.clone(),
            middlewares: self.chat_capability.middlewares.clone(),
            build_url: Box::new(move |_stream, _req| format!("{}/chat/completions", base)),
            build_headers: std::sync::Arc::new(headers_builder),
            before_send: None,
        };
        exec.execute(request).await
    }

    async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        use crate::executors::chat::{ChatExecutor, HttpChatExecutor};
        let http = self.http_client.clone();
        let base = self.chat_capability.base_url.clone();
        let api_key = self.chat_capability.api_key.clone();
        let custom_headers = self.chat_capability.http_config.headers.clone();
        let req_tx = super::transformers::XaiRequestTransformer;
        let resp_tx = super::transformers::XaiResponseTransformer;
        let inner = super::streaming::XaiEventConverter::new();
        let stream_tx = super::transformers::XaiStreamChunkTransformer {
            provider_id: "xai".to_string(),
            inner,
        };
        let api_key_clone = api_key.clone();
        let custom_headers_clone = custom_headers.clone();
        let headers_builder = move || {
            let api_key = api_key_clone.clone();
            let custom_headers = custom_headers_clone.clone();
            Box::pin(async move { super::utils::build_headers(&api_key, &custom_headers) })
                as std::pin::Pin<
                    Box<
                        dyn std::future::Future<
                                Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>,
                            > + Send,
                    >,
                >
        };
        let exec = HttpChatExecutor {
            provider_id: "xai".to_string(),
            http_client: http,
            request_transformer: std::sync::Arc::new(req_tx),
            response_transformer: std::sync::Arc::new(resp_tx),
            stream_transformer: Some(std::sync::Arc::new(stream_tx)),
            json_stream_converter: None,
            stream_disable_compression: self.chat_capability.http_config.stream_disable_compression,
            interceptors: self.http_interceptors.clone(),
            middlewares: self.chat_capability.middlewares.clone(),
            build_url: Box::new(move |_stream, _req| format!("{}/chat/completions", base)),
            build_headers: std::sync::Arc::new(headers_builder),
            before_send: None,
        };
        exec.execute_stream(request).await
    }
}

/// `xAI`-specific methods
impl XaiClient {
    /// Chat with reasoning effort (for thinking models)
    pub async fn chat_with_reasoning(
        &self,
        messages: Vec<ChatMessage>,
        reasoning_effort: &str,
    ) -> Result<ChatResponse, LlmError> {
        let mut provider_params = std::collections::HashMap::new();
        provider_params.insert(
            "reasoning_effort".to_string(),
            serde_json::Value::String(reasoning_effort.to_string()),
        );

        let request = ChatRequest {
            messages,
            tools: None,
            common_params: self.common_params.clone(),
            ..Default::default()
        };

        self.chat_capability.chat_request(request).await
    }

    /// Create a deferred completion
    pub async fn create_deferred_completion(
        &self,
        messages: Vec<ChatMessage>,
    ) -> Result<String, LlmError> {
        let request = ChatRequest {
            messages,
            tools: None,
            common_params: self.common_params.clone(),
            ..Default::default()
        };

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
        let url = format!(
            "{}/chat/deferred-completion/{}",
            self.chat_capability.base_url, request_id
        );
        let headers = super::utils::build_headers(
            &self.chat_capability.api_key,
            &self.chat_capability.http_config.headers,
        )?;

        let response = self.http_client.get(&url).headers(headers).send().await?;

        match response.status().as_u16() {
            200 => {
                let _xai_response: super::types::XaiChatResponse = response.json().await?;
                // We need to make parse_chat_response public or create a wrapper
                Err(LlmError::UnsupportedOperation(
                    "Get deferred completion not implemented yet".to_string(),
                ))
            }
            202 => Err(LlmError::ApiError {
                code: 202,
                message: "Deferred completion not ready yet".to_string(),
                details: None,
            }),
            _ => {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_default();
                Err(LlmError::ApiError {
                    code: status.as_u16(),
                    message: format!("xAI API error: {error_text}"),
                    details: serde_json::from_str(&error_text).ok(),
                })
            }
        }
    }

    /// Set the tracing guard to keep tracing system active
    /// NOTE: Tracing subscriber functionality has been moved to siumai-extras
    pub(crate) fn set_tracing_guard(&mut self, guard: Option<()>) {
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
