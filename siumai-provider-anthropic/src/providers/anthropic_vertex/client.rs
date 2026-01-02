//! Anthropic on Vertex AI client
//!
//! This client reuses Anthropic transformers but targets Vertex AI publisher endpoints,
//! authenticating via `Authorization: Bearer <token>` headers.

use async_trait::async_trait;
use backoff::ExponentialBackoffBuilder;
use reqwest::Client as HttpClient;
use std::sync::Arc;
use std::time::Duration;

use crate::error::LlmError;
use crate::execution::executors::chat::HttpChatExecutor;
use crate::execution::executors::common::{HttpExecutionConfig, execute_get_request};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::retry_api::RetryOptions;
use crate::streaming::ChatStream;
use crate::traits::{ChatCapability, ModelListingCapability};
use crate::types::{ChatMessage, ChatRequest, ChatResponse, ModelInfo};

/// Minimal config for Vertex Anthropic client (delegate to SiumaiBuilder for common params)
#[derive(Debug, Clone)]
pub struct VertexAnthropicConfig {
    pub base_url: String,
    pub model: String,
    pub http_config: crate::types::HttpConfig,
}

#[derive(Clone)]
pub struct VertexAnthropicClient {
    http_client: HttpClient,
    config: VertexAnthropicConfig,
    /// Optional HTTP interceptors applied to chat requests
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional model-level middlewares
    model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    /// Optional retry options for chat calls
    retry_options: Option<RetryOptions>,
}

impl VertexAnthropicClient {
    pub fn new(config: VertexAnthropicConfig, http_client: HttpClient) -> Self {
        Self {
            http_client,
            config,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
            retry_options: None,
        }
    }

    fn parse_model_id(name: &str) -> String {
        // Vertex model resource typically ends with "/models/{id}"; extract the trailing id.
        match name.rsplit_once("/models/") {
            Some((_, id)) => id.to_string(),
            None => name.to_string(),
        }
    }

    /// Create provider context for this client
    fn build_context(&self) -> crate::core::ProviderContext {
        crate::core::ProviderContext::new(
            "anthropic-vertex",
            self.config.base_url.clone(),
            None,
            self.config.http_config.headers.clone(),
        )
    }

    /// Create chat executor using the builder pattern
    fn build_chat_executor(&self, request: &ChatRequest) -> Arc<HttpChatExecutor> {
        use crate::core::ProviderSpec;
        use crate::execution::executors::chat::ChatExecutorBuilder;

        let ctx = self.build_context();
        let spec = Arc::new(super::spec::VertexAnthropicSpec::new(
            self.config.base_url.clone(),
            self.config.model.clone(),
            self.config.http_config.headers.clone(),
        ));
        let bundle = spec.choose_chat_transformers(request, &ctx);
        let before_send_hook = spec.chat_before_send(request, &ctx);

        let mut builder = ChatExecutorBuilder::new("anthropic-vertex", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_transformer_bundle(bundle)
            .with_stream_disable_compression(self.config.http_config.stream_disable_compression)
            .with_interceptors(self.http_interceptors.clone())
            .with_middlewares(self.model_middlewares.clone());

        if let Some(hook) = before_send_hook {
            builder = builder.with_before_send(hook);
        }

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        builder.build()
    }

    fn build_models_http_config(&self) -> HttpExecutionConfig {
        let ctx = self.build_context();
        let spec = Arc::new(super::spec::VertexAnthropicSpec::new(
            self.config.base_url.clone(),
            self.config.model.clone(),
            self.config.http_config.headers.clone(),
        ));
        crate::execution::wiring::HttpExecutionWiring::new(
            "anthropic-vertex",
            self.http_client.clone(),
            ctx,
        )
        .with_interceptors(self.http_interceptors.clone())
        .with_retry_options(self.retry_options.clone())
        .config(spec)
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
impl ChatCapability for VertexAnthropicClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        // Build request once and wrap the unified execution path with configurable retry.
        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(crate::types::CommonParams {
                model: self.config.model.clone(),
                ..Default::default()
            })
            .http_config(self.config.http_config.clone());
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        let req = builder.build();

        self.chat_request_via_spec(req).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<ChatStream, LlmError> {
        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(crate::types::CommonParams {
                model: self.config.model.clone(),
                ..Default::default()
            })
            .http_config(self.config.http_config.clone())
            .stream(true);
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        let req = builder.build();

        self.chat_stream_request_via_spec(req).await
    }
}

fn anthropic_backoff_executor() -> crate::retry_api::BackoffRetryExecutor {
    let backoff = ExponentialBackoffBuilder::new()
        .with_initial_interval(Duration::from_millis(1000))
        .with_max_interval(Duration::from_secs(60))
        .with_multiplier(1.5)
        .with_max_elapsed_time(Some(Duration::from_secs(300)))
        .build();
    crate::retry_api::BackoffRetryExecutor::with_backoff(backoff)
}

impl VertexAnthropicClient {
    /// Install HTTP interceptors for all chat requests.
    pub fn with_http_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors;
        self
    }

    /// Install model-level middlewares for chat requests.
    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.model_middlewares = middlewares;
        self
    }

    /// Set unified retry options for chat calls.
    pub fn set_retry_options(&mut self, options: Option<RetryOptions>) {
        self.retry_options = options.map(|mut opts| {
            if matches!(opts.backend, crate::retry_api::RetryBackend::Backoff)
                && opts.backoff_executor.is_none()
            {
                opts.backoff_executor = Some(anthropic_backoff_executor());
            }
            opts
        });
    }
}

#[async_trait]
impl ModelListingCapability for VertexAnthropicClient {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        let config = self.build_models_http_config();
        let url = config.provider_spec.models_url(&config.provider_context);
        let res = execute_get_request(&config, &url, None).await?;
        let json = res.json;
        let mut out = Vec::new();
        if let Some(arr) = json.get("models").and_then(|v| v.as_array()) {
            for m in arr {
                let raw_name = m.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let id = if raw_name.is_empty() {
                    self.config.model.clone()
                } else {
                    Self::parse_model_id(raw_name)
                };
                let display = m
                    .get("displayName")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                out.push(ModelInfo {
                    id,
                    name: display,
                    description: None,
                    owned_by: "Anthropic@Vertex".to_string(),
                    created: None,
                    capabilities: vec![
                        "chat".to_string(),
                        "streaming".to_string(),
                        "tools".to_string(),
                    ],
                    context_window: None,
                    max_output_tokens: None,
                    input_cost_per_token: None,
                    output_cost_per_token: None,
                });
            }
        }
        if out.is_empty() {
            // Fallback to configured model when list is empty
            out.push(ModelInfo {
                id: self.config.model.clone(),
                name: None,
                description: None,
                owned_by: "Anthropic@Vertex".to_string(),
                created: None,
                capabilities: vec![
                    "chat".to_string(),
                    "streaming".to_string(),
                    "tools".to_string(),
                ],
                context_window: None,
                max_output_tokens: None,
                input_cost_per_token: None,
                output_cost_per_token: None,
            });
        }
        Ok(out)
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        // Attempt to fetch a specific model; if endpoint is unavailable, fallback to minimal info
        let url =
            crate::utils::url::join_url(&self.config.base_url, &format!("models/{}", model_id));
        let config = self.build_models_http_config();
        let res = match execute_get_request(&config, &url, None).await {
            Ok(res) => res,
            Err(_) => {
                // Fallback minimal info
                return Ok(ModelInfo {
                    id: model_id,
                    name: None,
                    description: None,
                    owned_by: "Anthropic@Vertex".to_string(),
                    created: None,
                    capabilities: vec![
                        "chat".to_string(),
                        "streaming".to_string(),
                        "tools".to_string(),
                    ],
                    context_window: None,
                    max_output_tokens: None,
                    input_cost_per_token: None,
                    output_cost_per_token: None,
                });
            }
        };

        let json = res.json;
        let raw_name = json.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let id = if raw_name.is_empty() {
            model_id
        } else {
            Self::parse_model_id(raw_name)
        };
        let display = json
            .get("displayName")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        Ok(ModelInfo {
            id,
            name: display,
            description: None,
            owned_by: "Anthropic@Vertex".to_string(),
            created: None,
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            context_window: None,
            max_output_tokens: None,
            input_cost_per_token: None,
            output_cost_per_token: None,
        })
    }
}

impl crate::client::LlmClient for VertexAnthropicClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("anthropic")
    }

    fn supported_models(&self) -> Vec<String> {
        vec![self.config.model.clone()]
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn crate::client::LlmClient> {
        Box::new(self.clone())
    }

    fn as_model_listing_capability(&self) -> Option<&dyn crate::traits::ModelListingCapability> {
        Some(self)
    }
}
