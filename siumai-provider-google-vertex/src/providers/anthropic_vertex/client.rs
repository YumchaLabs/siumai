//! Anthropic on Vertex AI client
//!
//! This client reuses Anthropic transformers but targets Vertex AI publisher endpoints,
//! authenticating via `Authorization: Bearer <token>` headers.

use async_trait::async_trait;
use backoff::ExponentialBackoffBuilder;
use reqwest::Client as HttpClient;
use std::sync::Arc;
use std::time::Duration;

use crate::auth::TokenProvider;
use crate::error::LlmError;
use crate::execution::executors::chat::HttpChatExecutor;
use crate::execution::executors::common::{HttpExecutionConfig, execute_get_request};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::provider_options::anthropic_vertex::{
    VertexAnthropicOptions, VertexAnthropicStructuredOutputMode, VertexAnthropicThinkingMode,
};
use crate::retry_api::RetryOptions;
use crate::streaming::ChatStream;
use crate::traits::{ChatCapability, ModelListingCapability};
use crate::types::{ChatMessage, ChatRequest, ChatResponse, ModelInfo};

/// Minimal config for Vertex Anthropic client (delegate to SiumaiBuilder for common params)
#[derive(Clone)]
pub struct VertexAnthropicConfig {
    pub base_url: String,
    pub model: String,
    pub http_config: crate::types::HttpConfig,
    /// Default provider-owned request options merged before request-local overrides.
    pub default_provider_options_map: crate::types::ProviderOptionsMap,
    /// Optional custom HTTP transport (Vercel-style "custom fetch" parity).
    pub http_transport:
        Option<std::sync::Arc<dyn crate::execution::http::transport::HttpTransport>>,
    /// Optional Bearer token provider (e.g. ADC).
    pub token_provider: Option<Arc<dyn TokenProvider>>,
    /// Optional HTTP interceptors applied to all requests built from this config.
    pub http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional model-level middlewares applied before provider mapping (chat only).
    pub model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
}

impl std::fmt::Debug for VertexAnthropicConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("VertexAnthropicConfig");
        ds.field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("http_config", &self.http_config)
            .field(
                "default_provider_options_map",
                &self.default_provider_options_map,
            );

        if self.token_provider.is_some() {
            ds.field("has_token_provider", &true);
        }

        ds.finish()
    }
}

impl VertexAnthropicConfig {
    pub fn new<B: Into<String>, M: Into<String>>(base_url: B, model: M) -> Self {
        Self {
            base_url: base_url.into(),
            model: model.into(),
            http_config: crate::types::HttpConfig::default(),
            default_provider_options_map: crate::types::ProviderOptionsMap::default(),
            http_transport: None,
            token_provider: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
        }
    }

    pub fn with_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_http_config(mut self, http_config: crate::types::HttpConfig) -> Self {
        self.http_config = http_config;
        self
    }

    pub fn with_http_transport(
        mut self,
        transport: std::sync::Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.http_transport = Some(transport);
        self
    }

    /// Merge provider default options into this config.
    pub fn with_provider_options_map(
        mut self,
        provider_options_map: crate::types::ProviderOptionsMap,
    ) -> Self {
        self.default_provider_options_map
            .merge_overrides(provider_options_map);
        self
    }

    /// Merge Anthropic-on-Vertex default chat options into this config.
    pub fn with_anthropic_vertex_options(mut self, options: VertexAnthropicOptions) -> Self {
        let value =
            serde_json::to_value(options).expect("Anthropic-on-Vertex options should serialize");
        match (
            self.default_provider_options_map.get("anthropic").cloned(),
            value,
        ) {
            (Some(serde_json::Value::Object(mut base)), serde_json::Value::Object(extra)) => {
                for (key, value) in extra {
                    base.insert(key, value);
                }
                self.default_provider_options_map
                    .insert("anthropic", serde_json::Value::Object(base));
            }
            (_, value) => {
                self.default_provider_options_map.insert("anthropic", value);
            }
        }
        self
    }

    /// Configure Anthropic-on-Vertex thinking mode defaults.
    pub fn with_thinking_mode(self, config: VertexAnthropicThinkingMode) -> Self {
        self.with_anthropic_vertex_options(VertexAnthropicOptions::new().with_thinking_mode(config))
    }

    /// Configure Anthropic-on-Vertex structured-output routing defaults.
    pub fn with_structured_output_mode(self, mode: VertexAnthropicStructuredOutputMode) -> Self {
        self.with_anthropic_vertex_options(
            VertexAnthropicOptions::new().with_structured_output_mode(mode),
        )
    }

    /// Configure Anthropic-on-Vertex parallel tool-use defaults.
    pub fn with_disable_parallel_tool_use(self, disabled: bool) -> Self {
        self.with_anthropic_vertex_options(
            VertexAnthropicOptions::new().with_disable_parallel_tool_use(disabled),
        )
    }

    /// Configure Anthropic-on-Vertex reasoning replay defaults.
    pub fn with_send_reasoning(self, send_reasoning: bool) -> Self {
        self.with_anthropic_vertex_options(
            VertexAnthropicOptions::new().with_send_reasoning(send_reasoning),
        )
    }

    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http_config.timeout = Some(timeout);
        self
    }

    pub fn with_connect_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http_config.connect_timeout = Some(timeout);
        self
    }

    pub fn with_http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.http_config.stream_disable_compression = disable;
        self
    }

    pub fn with_token_provider(mut self, token_provider: Arc<dyn TokenProvider>) -> Self {
        self.token_provider = Some(token_provider);
        self
    }

    pub fn with_http_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors;
        self
    }

    pub fn with_http_interceptor(mut self, interceptor: Arc<dyn HttpInterceptor>) -> Self {
        self.http_interceptors.push(interceptor);
        self
    }

    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.model_middlewares = middlewares;
        self
    }

    pub fn with_bearer_token<S: Into<String>>(mut self, token: S) -> Self {
        let token = token.into();
        let trimmed = token.trim();
        if !trimmed.is_empty() {
            self.http_config
                .headers
                .insert("Authorization".to_string(), format!("Bearer {trimmed}"));
        }
        self
    }

    pub fn with_authorization<S: Into<String>>(mut self, value: S) -> Self {
        let value = value.into();
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            self.http_config
                .headers
                .insert("Authorization".to_string(), trimmed.to_string());
        }
        self
    }

    pub fn validate(&self) -> Result<(), LlmError> {
        if self.base_url.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Vertex Anthropic requires a non-empty base_url".to_string(),
            ));
        }
        if self.model.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Vertex Anthropic requires a non-empty model id".to_string(),
            ));
        }
        Ok(())
    }
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
    /// Construct a `VertexAnthropicClient` from a config-first `VertexAnthropicConfig`.
    pub fn from_config(config: VertexAnthropicConfig) -> Result<Self, LlmError> {
        let http_client =
            crate::execution::http::client::build_http_client_from_config(&config.http_config)?;
        Self::with_http_client(config, http_client)
    }

    /// Construct a `VertexAnthropicClient` from a config with a caller-supplied HTTP client.
    pub fn with_http_client(
        config: VertexAnthropicConfig,
        http_client: HttpClient,
    ) -> Result<Self, LlmError> {
        config.validate()?;
        let http_interceptors = config.http_interceptors.clone();
        let model_middlewares = config.model_middlewares.clone();
        Ok(Self::new(config, http_client)
            .with_http_interceptors(http_interceptors)
            .with_model_middlewares(model_middlewares))
    }

    pub fn new(config: VertexAnthropicConfig, http_client: HttpClient) -> Self {
        Self {
            http_client,
            config,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
            retry_options: None,
        }
    }

    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    #[cfg(test)]
    pub(crate) fn _debug_has_token_provider(&self) -> bool {
        self.config.token_provider.is_some()
    }

    fn parse_model_id(name: &str) -> String {
        // Vertex model resource typically ends with "/models/{id}"; extract the trailing id.
        match name.rsplit_once("/models/") {
            Some((_, id)) => id.to_string(),
            None => name.to_string(),
        }
    }

    /// Create provider context for this client.
    async fn build_context(&self) -> crate::core::ProviderContext {
        super::context::build_context(&self.config).await
    }

    /// Create chat executor using the builder pattern
    async fn build_chat_executor(&self, request: &ChatRequest) -> Arc<HttpChatExecutor> {
        use crate::core::ProviderSpec;
        use crate::execution::executors::chat::ChatExecutorBuilder;

        let ctx = self.build_context().await;
        let model = if request.common_params.model.trim().is_empty() {
            self.config.model.clone()
        } else {
            request.common_params.model.clone()
        };
        let spec = Arc::new(super::spec::VertexAnthropicSpec::new(
            self.config.base_url.clone(),
            model,
            self.config.http_config.headers.clone(),
        ));
        let bundle = spec.choose_chat_transformers(request, &ctx);

        let mut builder = ChatExecutorBuilder::new("anthropic-vertex", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_transformer_bundle(bundle)
            .with_stream_disable_compression(self.config.http_config.stream_disable_compression)
            .with_interceptors(self.http_interceptors.clone())
            .with_middlewares(self.model_middlewares.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        builder.build()
    }

    fn prepare_chat_request(
        &self,
        mut request: ChatRequest,
        stream: bool,
    ) -> Result<ChatRequest, LlmError> {
        if request.common_params.model.trim().is_empty() {
            request.common_params.model = self.config.model.clone();
        }
        if request.common_params.model.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Vertex Anthropic request requires a non-empty model id".to_string(),
            ));
        }
        if request.http_config.is_none() {
            request.http_config = Some(self.config.http_config.clone());
        }
        if !self.config.default_provider_options_map.is_empty() {
            let mut merged = self.config.default_provider_options_map.clone();
            merged.merge_overrides(std::mem::take(&mut request.provider_options_map));
            request.provider_options_map = merged;
        }
        request.stream = stream;
        Ok(request)
    }

    async fn build_models_http_config(&self) -> HttpExecutionConfig {
        let ctx = self.build_context().await;
        let spec = Arc::new(super::spec::VertexAnthropicSpec::new(
            self.config.base_url.clone(),
            self.config.model.clone(),
            self.config.http_config.headers.clone(),
        ));
        let mut wiring = crate::execution::wiring::HttpExecutionWiring::new(
            "anthropic-vertex",
            self.http_client.clone(),
            ctx,
        )
        .with_interceptors(self.http_interceptors.clone())
        .with_retry_options(self.retry_options.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            wiring = wiring.with_transport(transport);
        }

        wiring.config(spec)
    }

    /// Execute chat request via spec (unified implementation)
    async fn chat_request_via_spec(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        use crate::execution::executors::chat::ChatExecutor;

        let exec = self.build_chat_executor(&request).await;
        ChatExecutor::execute(&*exec, request).await
    }

    /// Execute streaming chat request via spec (unified implementation)
    async fn chat_stream_request_via_spec(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStream, LlmError> {
        use crate::execution::executors::chat::ChatExecutor;

        let exec = self.build_chat_executor(&request).await;
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
        let req = self.prepare_chat_request(builder.build(), false)?;

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
        let req = self.prepare_chat_request(builder.build(), true)?;

        self.chat_stream_request_via_spec(req).await
    }

    async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        let request = self.prepare_chat_request(request, false)?;
        self.chat_request_via_spec(request).await
    }

    async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        let request = self.prepare_chat_request(request, true)?;
        self.chat_stream_request_via_spec(request).await
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
        let config = self.build_models_http_config().await;
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
        let config = self.build_models_http_config().await;
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
        std::borrow::Cow::Borrowed("anthropic-vertex")
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

    fn as_chat_capability(&self) -> Option<&dyn crate::traits::ChatCapability> {
        Some(self)
    }

    fn as_model_listing_capability(&self) -> Option<&dyn crate::traits::ModelListingCapability> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::transport::{
        HttpTransport, HttpTransportRequest, HttpTransportResponse, HttpTransportStreamBody,
        HttpTransportStreamResponse,
    };
    use crate::providers::anthropic_vertex::{
        AnthropicChatResponseExt, VertexAnthropicChatRequestExt, VertexAnthropicOptions,
        VertexAnthropicThinkingMode,
    };
    use async_trait::async_trait;
    use futures_util::StreamExt;
    use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
    use std::{
        sync::{Arc, Mutex},
        time::Duration,
    };

    #[derive(Clone, Default)]
    struct CaptureTransport {
        last: Arc<Mutex<Option<HttpTransportRequest>>>,
        last_stream: Arc<Mutex<Option<HttpTransportRequest>>>,
    }

    impl CaptureTransport {
        fn take(&self) -> Option<HttpTransportRequest> {
            self.last.lock().expect("lock").take()
        }

        fn take_stream(&self) -> Option<HttpTransportRequest> {
            self.last_stream.lock().expect("lock").take()
        }
    }

    #[derive(Clone)]
    struct FixtureStreamTransport {
        last_stream: Arc<Mutex<Option<HttpTransportRequest>>>,
        body: Arc<Vec<u8>>,
    }

    impl FixtureStreamTransport {
        fn new(body: Vec<u8>) -> Self {
            Self {
                last_stream: Arc::new(Mutex::new(None)),
                body: Arc::new(body),
            }
        }

        fn take_stream(&self) -> Option<HttpTransportRequest> {
            self.last_stream.lock().expect("lock").take()
        }
    }

    #[async_trait]
    impl HttpTransport for FixtureStreamTransport {
        async fn execute_json(
            &self,
            _request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportResponse {
                status: 501,
                headers,
                body: br#"{"error":{"type":"test_error","message":"json unsupported in test"}}"#
                    .to_vec(),
            })
        }

        async fn execute_stream(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportStreamResponse, LlmError> {
            *self.last_stream.lock().expect("lock") = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(
                CONTENT_TYPE,
                HeaderValue::from_static("text/event-stream; charset=utf-8"),
            );

            Ok(HttpTransportStreamResponse {
                status: 200,
                headers,
                body: HttpTransportStreamBody::from_bytes((*self.body).clone()),
            })
        }
    }

    #[derive(Clone)]
    struct FixtureJsonTransport {
        last: Arc<Mutex<Option<HttpTransportRequest>>>,
        body: Arc<Vec<u8>>,
    }

    impl FixtureJsonTransport {
        fn new(body: Vec<u8>) -> Self {
            Self {
                last: Arc::new(Mutex::new(None)),
                body: Arc::new(body),
            }
        }

        fn take(&self) -> Option<HttpTransportRequest> {
            self.last.lock().expect("lock").take()
        }
    }

    #[async_trait]
    impl HttpTransport for FixtureJsonTransport {
        async fn execute_json(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last.lock().expect("lock") = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportResponse {
                status: 200,
                headers,
                body: (*self.body).clone(),
            })
        }

        async fn execute_stream(
            &self,
            _request: HttpTransportRequest,
        ) -> Result<HttpTransportStreamResponse, LlmError> {
            let mut headers = HeaderMap::new();
            headers.insert(
                CONTENT_TYPE,
                HeaderValue::from_static("text/event-stream; charset=utf-8"),
            );

            Ok(HttpTransportStreamResponse {
                status: 501,
                headers,
                body: HttpTransportStreamBody::from_bytes(
                    br#"data: {"error":{"type":"test_error","message":"stream unsupported in test"}}"#
                        .to_vec(),
                ),
            })
        }
    }

    fn vertex_anthropic_json_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "value": { "type": "string" }
            },
            "required": ["value"],
            "additionalProperties": false
        })
    }

    fn make_vertex_anthropic_structured_output_request(model: &str) -> ChatRequest {
        ChatRequest::builder()
            .model(model)
            .messages(vec![ChatMessage::user("hi").build()])
            .response_format(crate::types::ResponseFormat::json_schema(
                vertex_anthropic_json_schema(),
            ))
            .build()
    }

    fn vertex_anthropic_structured_output_success_stream_body(model: &str) -> Vec<u8> {
        let mut body = String::new();
        body.push_str(&format!(
            r#"data: {{"type":"message_start","message":{{"id":"msg_test","model":"{model}","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":15,"output_tokens":0}}}}}}"#
        ));
        body.push_str("\n\n");
        body.push_str(
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        );
        body.push_str("\n\n");

        let part1 =
            serde_json::to_string("{\"value\":\"te").expect("serialize anthropic-vertex text part");
        body.push_str(&format!(
            r#"data: {{"type":"content_block_delta","index":0,"delta":{{"type":"text_delta","text":{part1}}}}}"#
        ));
        body.push_str("\n\n");

        let part2 = serde_json::to_string("st\"}").expect("serialize anthropic-vertex text part");
        body.push_str(&format!(
            r#"data: {{"type":"content_block_delta","index":0,"delta":{{"type":"text_delta","text":{part2}}}}}"#
        ));
        body.push_str("\n\n");

        body.push_str(r#"data: {"type":"content_block_stop","index":0}"#);
        body.push_str("\n\n");
        body.push_str(
            r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":15,"output_tokens":4}}"#,
        );
        body.push_str("\n\n");
        body.into_bytes()
    }

    fn vertex_anthropic_structured_output_interrupted_stream_body(model: &str) -> Vec<u8> {
        let mut body = String::new();
        body.push_str(&format!(
            r#"data: {{"type":"message_start","message":{{"id":"msg_test","model":"{model}","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":15,"output_tokens":0}}}}}}"#
        ));
        body.push_str("\n\n");
        body.push_str(
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        );
        body.push_str("\n\n");

        let partial =
            serde_json::to_string("{\"value\":").expect("serialize anthropic-vertex partial");
        body.push_str(&format!(
            r#"data: {{"type":"content_block_delta","index":0,"delta":{{"type":"text_delta","text":{partial}}}}}"#
        ));
        body.push_str("\n\n");
        body.into_bytes()
    }

    fn vertex_anthropic_structured_output_success_response_body(model: &str) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "{\"value\":\"test\"}"
                }
            ],
            "model": model,
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 15,
                "output_tokens": 4
            }
        }))
        .expect("serialize anthropic-vertex success response")
    }

    fn vertex_anthropic_structured_output_invalid_response_body(model: &str) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "sorry, not valid json"
                }
            ],
            "model": model,
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 15,
                "output_tokens": 4
            }
        }))
        .expect("serialize anthropic-vertex invalid response")
    }

    fn make_vertex_anthropic_reasoning_request(model: &str) -> ChatRequest {
        ChatRequest::builder()
            .model(model)
            .messages(vec![ChatMessage::user("hi").build()])
            .build()
            .with_anthropic_vertex_options(
                VertexAnthropicOptions::new()
                    .with_thinking_mode(VertexAnthropicThinkingMode::enabled(Some(2048))),
            )
    }

    fn vertex_anthropic_reasoning_response_body(model: &str) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "id": "msg_reasoning",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Count the letters carefully. The word strawberry contains three r characters.",
                    "signature": "sig-1"
                },
                {
                    "type": "redacted_thinking",
                    "data": "redacted-blob"
                },
                {
                    "type": "text",
                    "text": "There are three letter r's in strawberry."
                }
            ],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 18,
                "output_tokens": 24
            }
        }))
        .expect("serialize anthropic-vertex reasoning response")
    }

    fn vertex_anthropic_reasoning_stream_body(model: &str) -> Vec<u8> {
        let mut body = String::new();
        body.push_str(&format!(
            r#"data: {{"type":"message_start","message":{{"id":"msg_reasoning","model":"{model}","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":18,"output_tokens":1}}}}}}"#
        ));
        body.push_str("\n\n");
        body.push_str(
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
        );
        body.push_str("\n\n");
        body.push_str(
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Count the letters carefully. "}}"#,
        );
        body.push_str("\n\n");
        body.push_str(
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"The word strawberry contains three r characters."}}"#,
        );
        body.push_str("\n\n");
        body.push_str(
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-1"}}"#,
        );
        body.push_str("\n\n");
        body.push_str(r#"data: {"type":"content_block_stop","index":0}"#);
        body.push_str("\n\n");
        body.push_str(
            r#"data: {"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"redacted-blob"}}"#,
        );
        body.push_str("\n\n");
        body.push_str(r#"data: {"type":"content_block_stop","index":1}"#);
        body.push_str("\n\n");
        body.push_str(
            r#"data: {"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}"#,
        );
        body.push_str("\n\n");
        body.push_str(
            r#"data: {"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"There are three letter r's in strawberry."}}"#,
        );
        body.push_str("\n\n");
        body.push_str(r#"data: {"type":"content_block_stop","index":2}"#);
        body.push_str("\n\n");
        body.push_str(
            r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":18,"output_tokens":24}}"#,
        );
        body.push_str("\n\n");
        body.push_str(r#"data: {"type":"message_stop"}"#);
        body.push_str("\n\n");
        body.into_bytes()
    }

    fn header_value(req: &HttpTransportRequest, name: &str) -> Option<String> {
        req.headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(ToString::to_string)
    }

    fn assert_vertex_anthropic_structured_output_stream_request(
        req: &HttpTransportRequest,
        base_url: &str,
        model: &str,
    ) {
        assert_eq!(
            req.url,
            format!("{base_url}/models/{model}:streamRawPredict")
        );
        assert_eq!(req.body["stream"], serde_json::json!(true));
        assert_eq!(
            header_value(req, "accept"),
            Some("text/event-stream".to_string())
        );
        assert_eq!(
            header_value(req, "authorization"),
            Some("Bearer test-token".to_string())
        );
        assert_eq!(
            req.body["anthropic_version"],
            serde_json::json!("vertex-2023-10-16")
        );
        assert_eq!(
            req.body["output_format"],
            serde_json::json!({
                "type": "json_schema",
                "schema": vertex_anthropic_json_schema()
            })
        );
        assert!(req.body.get("model").is_none());
        assert!(req.body.get("tools").is_none());
    }

    fn assert_vertex_anthropic_structured_output_request(
        req: &HttpTransportRequest,
        base_url: &str,
        model: &str,
    ) {
        assert_eq!(req.url, format!("{base_url}/models/{model}:rawPredict"));
        assert_eq!(
            header_value(req, "authorization"),
            Some("Bearer test-token".to_string())
        );
        assert_eq!(
            req.body["anthropic_version"],
            serde_json::json!("vertex-2023-10-16")
        );
        assert_eq!(
            req.body["output_format"],
            serde_json::json!({
                "type": "json_schema",
                "schema": vertex_anthropic_json_schema()
            })
        );
        assert!(req.body.get("stream").is_none());
        assert!(req.body.get("model").is_none());
        assert!(req.body.get("tools").is_none());
    }

    fn assert_vertex_anthropic_reasoning_request(
        req: &HttpTransportRequest,
        base_url: &str,
        model: &str,
        stream: bool,
    ) {
        let suffix = if stream {
            ":streamRawPredict"
        } else {
            ":rawPredict"
        };
        assert_eq!(req.url, format!("{base_url}/models/{model}{suffix}"));
        assert_eq!(
            header_value(req, "authorization"),
            Some("Bearer test-token".to_string())
        );
        if stream {
            assert_eq!(
                header_value(req, "accept"),
                Some("text/event-stream".to_string())
            );
            assert_eq!(req.body["stream"], serde_json::json!(true));
        } else {
            assert!(req.body.get("stream").is_none());
        }
        assert_eq!(
            req.body["anthropic_version"],
            serde_json::json!("vertex-2023-10-16")
        );
        assert_eq!(
            req.body["thinking"],
            serde_json::json!({
                "type": "enabled",
                "budget_tokens": 2048
            })
        );
        assert!(req.body.get("model").is_none());
    }

    async fn collect_stream_events(mut stream: ChatStream) -> Vec<crate::types::ChatStreamEvent> {
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            match item {
                Ok(event) => events.push(event),
                Err(err) => panic!("collect anthropic-vertex client stream event failed: {err:?}"),
            }
        }
        events
    }

    #[async_trait]
    impl HttpTransport for CaptureTransport {
        async fn execute_json(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last.lock().expect("lock") = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportResponse {
                status: 401,
                headers,
                body: br#"{"error":{"type":"auth_error","message":"unauthorized"}}"#.to_vec(),
            })
        }

        async fn execute_stream(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportStreamResponse, LlmError> {
            *self.last_stream.lock().expect("lock") = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportStreamResponse {
                status: 401,
                headers,
                body: HttpTransportStreamBody::from_bytes(
                    br#"{"error":{"type":"auth_error","message":"unauthorized"}}"#.to_vec(),
                ),
            })
        }
    }

    #[test]
    fn vertex_anthropic_llmclient_exposes_chat_capability() {
        let cfg = VertexAnthropicConfig {
            base_url: "https://example.invalid".to_string(),
            model: "claude-3-5-sonnet-20241022".to_string(),
            http_config: crate::types::HttpConfig::default(),
            default_provider_options_map: crate::types::ProviderOptionsMap::default(),
            http_transport: None,
            token_provider: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
        };
        let client = VertexAnthropicClient::new(cfg, reqwest::Client::new());
        let llm: &dyn crate::client::LlmClient = &client;
        assert_eq!(
            llm.provider_id(),
            std::borrow::Cow::Borrowed("anthropic-vertex")
        );
        assert!(llm.as_chat_capability().is_some());
        assert!(llm.capabilities().chat);
    }

    #[test]
    fn vertex_anthropic_config_http_convenience_helpers() {
        let config =
            VertexAnthropicConfig::new("https://example.invalid", "claude-3-5-sonnet-20241022")
                .with_timeout(Duration::from_secs(14))
                .with_connect_timeout(Duration::from_secs(4))
                .with_http_stream_disable_compression(true)
                .with_http_interceptor(Arc::new(
                    crate::execution::http::interceptor::LoggingInterceptor,
                ));

        assert_eq!(config.http_config.timeout, Some(Duration::from_secs(14)));
        assert_eq!(
            config.http_config.connect_timeout,
            Some(Duration::from_secs(4))
        );
        assert!(config.http_config.stream_disable_compression);
        assert_eq!(config.http_interceptors.len(), 1);
    }

    #[test]
    fn prepare_chat_request_for_stream_sets_stream_and_fills_defaults() {
        let cfg = VertexAnthropicConfig::new(
            "https://example.invalid/v1/projects/p/locations/us/publishers/anthropic",
            "claude-3-5-sonnet-20241022",
        )
        .with_http_config(crate::types::HttpConfig::default());
        let client = VertexAnthropicClient::from_config(cfg).expect("from_config ok");

        let request = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .build();

        let prepared = client
            .prepare_chat_request(request, true)
            .expect("prepare stream request");

        assert!(prepared.stream);
        assert_eq!(prepared.common_params.model, "claude-3-5-sonnet-20241022");
        assert!(prepared.http_config.is_some());
    }

    #[test]
    fn prepare_chat_request_merges_config_default_provider_options_before_request_overrides() {
        let cfg =
            VertexAnthropicConfig::new("https://example.invalid", "claude-3-5-sonnet-20241022")
                .with_thinking_mode(VertexAnthropicThinkingMode::enabled(Some(2048)))
                .with_structured_output_mode(VertexAnthropicStructuredOutputMode::JsonTool)
                .with_send_reasoning(false);
        let client = VertexAnthropicClient::from_config(cfg).expect("from_config ok");

        let request = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .build()
            .with_anthropic_vertex_options(
                VertexAnthropicOptions::new()
                    .with_structured_output_mode(VertexAnthropicStructuredOutputMode::OutputFormat)
                    .with_disable_parallel_tool_use(true),
            );

        let prepared = client
            .prepare_chat_request(request, false)
            .expect("prepare request");

        let value = prepared
            .provider_options_map
            .get("anthropic")
            .expect("anthropic options present");
        assert_eq!(value["thinking_mode"]["enabled"], serde_json::json!(true));
        assert_eq!(
            value["thinking_mode"]["thinking_budget"],
            serde_json::json!(2048)
        );
        assert_eq!(
            value["structured_output_mode"],
            serde_json::json!("outputFormat")
        );
        assert_eq!(value["send_reasoning"], serde_json::json!(false));
        assert_eq!(value["disable_parallel_tool_use"], serde_json::json!(true));
    }

    #[test]
    fn prepare_chat_request_for_non_stream_clears_stream_and_preserves_explicit_model() {
        let cfg = VertexAnthropicConfig::new(
            "https://example.invalid/v1/projects/p/locations/us/publishers/anthropic",
            "claude-3-5-sonnet-20241022",
        )
        .with_http_config(crate::types::HttpConfig::default());
        let client = VertexAnthropicClient::from_config(cfg).expect("from_config ok");

        let request = ChatRequest::builder()
            .model("claude-3-7-sonnet-20250219")
            .messages(vec![ChatMessage::user("hi").build()])
            .stream(true)
            .build();

        let prepared = client
            .prepare_chat_request(request, false)
            .expect("prepare non-stream request");

        assert!(!prepared.stream);
        assert_eq!(prepared.common_params.model, "claude-3-7-sonnet-20250219");
        assert!(prepared.http_config.is_some());
    }

    #[tokio::test]
    async fn chat_request_uses_explicit_request_model_in_raw_predict_url() {
        let transport = CaptureTransport::default();
        let cfg = VertexAnthropicConfig::new(
            "https://example.invalid/v1/projects/p/locations/us/publishers/anthropic",
            "claude-3-5-sonnet-20241022",
        )
        .with_authorization("Bearer test-token")
        .with_http_transport(Arc::new(transport.clone()));
        let client = VertexAnthropicClient::from_config(cfg).expect("from_config ok");

        let request = ChatRequest::builder()
            .model("claude-3-7-sonnet-20250219")
            .messages(vec![ChatMessage::user("hi").build()])
            .build();

        let err = client
            .chat_request(request)
            .await
            .expect_err("transport should stop request");
        assert!(matches!(err, LlmError::ApiError { code: 401, .. }));

        let captured = transport.take().expect("captured request");
        assert!(
            captured
                .url
                .contains("/models/claude-3-7-sonnet-20250219:rawPredict"),
            "unexpected url: {}",
            captured.url
        );
        assert!(transport.take_stream().is_none());
    }

    #[tokio::test]
    async fn chat_stream_request_uses_explicit_request_model_in_stream_raw_predict_url() {
        let transport = CaptureTransport::default();
        let cfg = VertexAnthropicConfig::new(
            "https://example.invalid/v1/projects/p/locations/us/publishers/anthropic",
            "claude-3-5-sonnet-20241022",
        )
        .with_authorization("Bearer test-token")
        .with_http_transport(Arc::new(transport.clone()));
        let client = VertexAnthropicClient::from_config(cfg).expect("from_config ok");

        let request = ChatRequest::builder()
            .model("claude-3-7-sonnet-20250219")
            .messages(vec![ChatMessage::user("hi").build()])
            .build();

        let err = match client.chat_stream_request(request).await {
            Ok(_) => panic!("transport should stop stream request"),
            Err(err) => err,
        };
        assert!(matches!(err, LlmError::ApiError { code: 401, .. }));

        let captured = transport.take_stream().expect("captured stream request");
        assert!(
            captured
                .url
                .contains("/models/claude-3-7-sonnet-20250219:streamRawPredict"),
            "unexpected url: {}",
            captured.url
        );
        assert!(transport.take().is_none());
    }

    #[tokio::test]
    async fn anthropic_vertex_client_structured_output_stream_end_extracts_json_and_preserves_stream_end()
     {
        let model = "claude-sonnet-4-5-latest";
        let base_url = "https://example.invalid/v1/projects/p/locations/us/publishers/anthropic";
        let transport = FixtureStreamTransport::new(
            vertex_anthropic_structured_output_success_stream_body(model),
        );
        let client = VertexAnthropicClient::from_config(
            VertexAnthropicConfig::new(base_url, model)
                .with_bearer_token("test-token")
                .with_http_transport(Arc::new(transport.clone())),
        )
        .expect("from_config ok");

        let events = collect_stream_events(
            client
                .chat_stream_request(make_vertex_anthropic_structured_output_request(model))
                .await
                .expect("anthropic-vertex stream ok"),
        )
        .await;

        let content = events
            .iter()
            .filter_map(|event| match event {
                crate::types::ChatStreamEvent::ContentDelta { delta, .. } => Some(delta.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(content, "{\"value\":\"test\"}");

        let end = events
            .iter()
            .find_map(|event| match event {
                crate::types::ChatStreamEvent::StreamEnd { response } => Some(response),
                _ => None,
            })
            .expect("expected stream end");
        assert_eq!(end.finish_reason, Some(crate::types::FinishReason::Stop));
        assert_eq!(end.id.as_deref(), Some("msg_test"));
        assert_eq!(end.model.as_deref(), Some(model));
        assert_eq!(end.text().unwrap_or_default(), "{\"value\":\"test\"}");

        let value = siumai_core::structured_output::extract_json_value_from_stream(Box::pin(
            futures::stream::iter(events.into_iter().map(Ok::<_, LlmError>)),
        ))
        .await
        .expect("structured output value");
        assert_eq!(value["value"], "test");

        let req = transport.take_stream().expect("captured stream request");
        assert_vertex_anthropic_structured_output_stream_request(&req, base_url, model);
    }

    #[tokio::test]
    async fn anthropic_vertex_client_structured_output_response_extracts_json_and_preserves_response()
     {
        let model = "claude-sonnet-4-5-latest";
        let base_url = "https://example.invalid/v1/projects/p/locations/us/publishers/anthropic";
        let transport = FixtureJsonTransport::new(
            vertex_anthropic_structured_output_success_response_body(model),
        );
        let client = VertexAnthropicClient::from_config(
            VertexAnthropicConfig::new(base_url, model)
                .with_bearer_token("test-token")
                .with_http_transport(Arc::new(transport.clone())),
        )
        .expect("from_config ok");

        let response = client
            .chat_request(make_vertex_anthropic_structured_output_request(model))
            .await
            .expect("anthropic-vertex response ok");

        assert_eq!(
            response.finish_reason,
            Some(crate::types::FinishReason::Stop)
        );
        assert_eq!(response.id.as_deref(), Some("msg_test"));
        assert_eq!(response.model.as_deref(), Some(model));
        assert_eq!(response.text().unwrap_or_default(), "{\"value\":\"test\"}");

        let value = siumai_core::structured_output::extract_json_value_from_response(&response)
            .expect("structured output value");
        assert_eq!(value["value"], "test");

        let req = transport.take().expect("captured request");
        assert_vertex_anthropic_structured_output_request(&req, base_url, model);
    }

    #[tokio::test]
    async fn anthropic_vertex_client_structured_output_response_returns_invalid_json_error() {
        let model = "claude-sonnet-4-5-latest";
        let base_url = "https://example.invalid/v1/projects/p/locations/us/publishers/anthropic";
        let transport = FixtureJsonTransport::new(
            vertex_anthropic_structured_output_invalid_response_body(model),
        );
        let client = VertexAnthropicClient::from_config(
            VertexAnthropicConfig::new(base_url, model)
                .with_bearer_token("test-token")
                .with_http_transport(Arc::new(transport.clone())),
        )
        .expect("from_config ok");

        let response = client
            .chat_request(make_vertex_anthropic_structured_output_request(model))
            .await
            .expect("anthropic-vertex response ok");

        let err = siumai_core::structured_output::extract_json_value_from_response(&response)
            .expect_err("invalid response should fail");

        match err {
            LlmError::ParseError(message) => {
                assert!(message.contains("no valid JSON candidate found"))
            }
            other => panic!("expected ParseError, got {other:?}"),
        }

        let req = transport.take().expect("captured request");
        assert_vertex_anthropic_structured_output_request(&req, base_url, model);
    }

    #[tokio::test]
    async fn anthropic_vertex_client_reasoning_response_preserves_metadata_and_request_shape() {
        let model = "claude-sonnet-4-5-latest";
        let base_url = "https://example.invalid/v1/projects/p/locations/us/publishers/anthropic";
        let transport = FixtureJsonTransport::new(vertex_anthropic_reasoning_response_body(model));
        let client = VertexAnthropicClient::from_config(
            VertexAnthropicConfig::new(base_url, model)
                .with_bearer_token("test-token")
                .with_http_transport(Arc::new(transport.clone())),
        )
        .expect("from_config ok");

        let response = client
            .chat_request(make_vertex_anthropic_reasoning_request(model))
            .await
            .expect("anthropic-vertex response ok");

        let expected_reasoning =
            "Count the letters carefully. The word strawberry contains three r characters."
                .to_string();
        assert_eq!(
            response.content_text(),
            Some("There are three letter r's in strawberry.")
        );
        assert_eq!(response.reasoning(), vec![expected_reasoning]);
        assert_eq!(
            response.finish_reason,
            Some(crate::types::FinishReason::Stop)
        );

        let anthropic_meta = response
            .anthropic_metadata()
            .expect("expected typed anthropic metadata");
        assert_eq!(anthropic_meta.thinking_signature.as_deref(), Some("sig-1"));
        assert_eq!(
            anthropic_meta.redacted_thinking_data.as_deref(),
            Some("redacted-blob")
        );

        let req = transport.take().expect("captured request");
        assert_vertex_anthropic_reasoning_request(&req, base_url, model, false);
    }

    #[tokio::test]
    async fn anthropic_vertex_client_reasoning_stream_preserves_metadata_and_request_shape() {
        let model = "claude-sonnet-4-5-latest";
        let base_url = "https://example.invalid/v1/projects/p/locations/us/publishers/anthropic";
        let transport = FixtureStreamTransport::new(vertex_anthropic_reasoning_stream_body(model));
        let client = VertexAnthropicClient::from_config(
            VertexAnthropicConfig::new(base_url, model)
                .with_bearer_token("test-token")
                .with_http_transport(Arc::new(transport.clone())),
        )
        .expect("from_config ok");

        let mut stream = client
            .chat_stream_request(make_vertex_anthropic_reasoning_request(model))
            .await
            .expect("anthropic-vertex stream ok");

        let mut reasoning = String::new();
        let mut end = None;
        while let Some(item) = stream.next().await {
            match item.expect("stream event ok") {
                crate::types::ChatStreamEvent::ThinkingDelta { delta } => {
                    reasoning.push_str(&delta)
                }
                crate::types::ChatStreamEvent::StreamEnd { response } => {
                    end = Some(response);
                    break;
                }
                _ => {}
            }
        }

        let response = end.expect("expected stream end");
        assert_eq!(
            reasoning,
            "Count the letters carefully. The word strawberry contains three r characters."
        );
        assert_eq!(
            response.content_text(),
            Some("There are three letter r's in strawberry.")
        );
        assert_eq!(
            response.finish_reason,
            Some(crate::types::FinishReason::Stop)
        );

        let anthropic_meta = response
            .anthropic_metadata()
            .expect("expected typed anthropic metadata");
        assert_eq!(anthropic_meta.thinking_signature.as_deref(), Some("sig-1"));
        assert_eq!(
            anthropic_meta.redacted_thinking_data.as_deref(),
            Some("redacted-blob")
        );

        let req = transport.take_stream().expect("captured stream request");
        assert_vertex_anthropic_reasoning_request(&req, base_url, model, true);
    }

    #[tokio::test]
    async fn anthropic_vertex_client_structured_output_stream_returns_incomplete_json_error() {
        let model = "claude-sonnet-4-5-latest";
        let base_url = "https://example.invalid/v1/projects/p/locations/us/publishers/anthropic";
        let transport = FixtureStreamTransport::new(
            vertex_anthropic_structured_output_interrupted_stream_body(model),
        );
        let client = VertexAnthropicClient::from_config(
            VertexAnthropicConfig::new(base_url, model)
                .with_bearer_token("test-token")
                .with_http_transport(Arc::new(transport.clone())),
        )
        .expect("from_config ok");

        let err = siumai_core::structured_output::extract_json_value_from_stream(
            client
                .chat_stream_request(make_vertex_anthropic_structured_output_request(model))
                .await
                .expect("anthropic-vertex stream ok"),
        )
        .await
        .expect_err("interrupted stream should fail");

        match err {
            LlmError::ParseError(message) => {
                assert!(message.contains("stream ended before a complete JSON value was produced"))
            }
            other => panic!("expected ParseError, got {other:?}"),
        }

        let req = transport.take_stream().expect("captured stream request");
        assert_vertex_anthropic_structured_output_stream_request(&req, base_url, model);
    }
}
