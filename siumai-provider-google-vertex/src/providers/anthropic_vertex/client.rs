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
            .field("http_config", &self.http_config);

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

    pub fn with_token_provider(mut self, token_provider: Arc<dyn TokenProvider>) -> Self {
        self.token_provider = Some(token_provider);
        self
    }

    pub fn with_http_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors;
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
    use async_trait::async_trait;
    use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
    use std::sync::{Arc, Mutex};

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
}
