//! Google Vertex AI client (minimal).
//!
//! This client currently focuses on Vertex Imagen via the `:predict` endpoint.

use async_trait::async_trait;
use reqwest::Client as HttpClient;
use std::borrow::Cow;
use std::sync::Arc;

use crate::auth::TokenProvider;
use crate::client::LlmClient;
use crate::core::ProviderSpec;
use crate::error::LlmError;
use crate::execution::executors::image::{ImageExecutor, ImageExecutorBuilder};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::retry_api::RetryOptions;
use crate::streaming::ChatStream;
use crate::traits::{
    ChatCapability, EmbeddingCapability, EmbeddingExtensions, ImageExtras,
    ImageGenerationCapability, ProviderCapabilities,
};
use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, EmbeddingRequest, EmbeddingResponse, ImageEditRequest,
    ImageGenerationRequest, ImageGenerationResponse, Tool,
};

/// Minimal config for Google Vertex client (delegate auth to HttpConfig headers / token providers).
#[derive(Clone)]
pub struct GoogleVertexConfig {
    /// Vertex base URL, typically:
    /// `https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google`
    pub base_url: String,
    /// Default model id (e.g., `imagen-3.0-generate-002`).
    pub model: String,
    /// Optional API key (express mode). When set and no `Authorization` header is present,
    /// it will be passed as the `key` query parameter.
    pub api_key: Option<String>,
    /// Per-request HTTP config (headers, timeouts, etc.).
    pub http_config: crate::types::HttpConfig,
    /// Optional custom HTTP transport (Vercel-style "custom fetch" parity).
    pub http_transport:
        Option<std::sync::Arc<dyn crate::execution::http::transport::HttpTransport>>,
    /// Optional Bearer token provider (e.g., ADC). When present, an `Authorization` header
    /// will be injected automatically if one is not already set.
    pub token_provider: Option<Arc<dyn TokenProvider>>,
    /// Optional HTTP interceptors applied to all requests built from this config.
    pub http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional model-level middlewares applied before provider mapping (chat only).
    pub model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
}

impl std::fmt::Debug for GoogleVertexConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("GoogleVertexConfig");
        ds.field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("http_config", &self.http_config);

        if self.api_key.is_some() {
            ds.field("has_api_key", &true);
        }

        if self.token_provider.is_some() {
            ds.field("has_token_provider", &true);
        }

        ds.finish()
    }
}

impl GoogleVertexConfig {
    /// Create a config from an explicit Vertex base URL and model id.
    pub fn new<B: Into<String>, M: Into<String>>(base_url: B, model: M) -> Self {
        Self {
            base_url: base_url.into(),
            model: model.into(),
            api_key: None,
            http_config: crate::types::HttpConfig::default(),
            http_transport: None,
            token_provider: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
        }
    }

    /// Create an express-mode config that uses the shared Google Vertex express base URL.
    pub fn express<S: Into<String>, M: Into<String>>(api_key: S, model: M) -> Self {
        Self::new(crate::auth::vertex::GOOGLE_VERTEX_EXPRESS_BASE_URL, model).with_api_key(api_key)
    }

    /// Create an enterprise-mode config from project + location.
    pub fn enterprise<P: AsRef<str>, L: AsRef<str>, M: Into<String>>(
        project: P,
        location: L,
        model: M,
    ) -> Self {
        Self::new(
            crate::auth::vertex::google_vertex_base_url(
                project.as_ref().trim(),
                location.as_ref().trim(),
            ),
            model,
        )
    }

    pub fn with_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        let api_key = api_key.into();
        let trimmed = api_key.trim().to_string();
        self.api_key = (!trimmed.is_empty()).then_some(trimmed);
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

    pub fn validate(&self) -> Result<(), LlmError> {
        if self.base_url.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Google Vertex requires a non-empty base_url".to_string(),
            ));
        }
        if self.model.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Google Vertex requires a non-empty model id".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct GoogleVertexClient {
    http_client: HttpClient,
    config: GoogleVertexConfig,
    common_params: crate::types::CommonParams,
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    retry_options: Option<RetryOptions>,
}

impl GoogleVertexClient {
    /// Construct a `GoogleVertexClient` from a config-first `GoogleVertexConfig`.
    pub fn from_config(config: GoogleVertexConfig) -> Result<Self, LlmError> {
        let http_client =
            crate::execution::http::client::build_http_client_from_config(&config.http_config)?;
        Self::with_http_client(config, http_client)
    }

    /// Construct a `GoogleVertexClient` from a `GoogleVertexConfig` with a caller-supplied HTTP client.
    pub fn with_http_client(
        config: GoogleVertexConfig,
        http_client: HttpClient,
    ) -> Result<Self, LlmError> {
        config.validate()?;

        let http_interceptors = config.http_interceptors.clone();
        let model_middlewares = config.model_middlewares.clone();

        Ok(Self::new(config, http_client)
            .with_interceptors(http_interceptors)
            .with_model_middlewares(model_middlewares))
    }

    pub fn new(config: GoogleVertexConfig, http_client: HttpClient) -> Self {
        let common_params = crate::types::CommonParams {
            model: config.model.clone(),
            ..Default::default()
        };
        Self {
            http_client,
            config,
            common_params,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
            retry_options: None,
        }
    }

    pub fn with_common_params(mut self, common_params: crate::types::CommonParams) -> Self {
        self.common_params = common_params;
        self
    }

    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    pub fn with_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
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

    pub fn with_retry_options(mut self, retry_options: RetryOptions) -> Self {
        self.retry_options = Some(retry_options);
        self
    }

    async fn build_context(&self) -> crate::core::ProviderContext {
        super::context::build_context(&self.config).await
    }

    fn prepare_chat_request(
        &self,
        mut request: ChatRequest,
        stream: bool,
    ) -> Result<ChatRequest, LlmError> {
        if request.common_params.model.trim().is_empty() {
            request.common_params.model = self.common_params.model.clone();
        }
        if request.common_params.model.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Google Vertex chat request requires a non-empty model id".to_string(),
            ));
        }
        if request.http_config.is_none() {
            request.http_config = Some(self.config.http_config.clone());
        }
        request.stream = stream;
        Ok(request)
    }

    async fn chat_request_via_spec(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        use crate::execution::executors::chat::{ChatExecutor, ChatExecutorBuilder};

        let request = self.prepare_chat_request(request, false)?;
        let ctx = self.build_context().await;
        let spec = Arc::new(
            crate::standards::vertex_generative_ai::VertexGenerativeAiStandard::new()
                .create_spec("vertex"),
        );
        let bundle = spec.choose_chat_transformers(&request, &ctx);

        let mut exec = ChatExecutorBuilder::new("vertex", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_transformer_bundle(bundle)
            .with_stream_disable_compression(self.config.http_config.stream_disable_compression)
            .with_interceptors(self.http_interceptors.clone())
            .with_middlewares(self.model_middlewares.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            exec = exec.with_transport(transport);
        }
        if let Some(retry) = self.retry_options.clone() {
            exec = exec.with_retry_options(retry);
        }

        let exec = exec.build();
        ChatExecutor::execute(&*exec, request).await
    }

    async fn chat_stream_request_via_spec(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStream, LlmError> {
        use crate::execution::executors::chat::{ChatExecutor, ChatExecutorBuilder};

        let request = self.prepare_chat_request(request, true)?;
        let ctx = self.build_context().await;
        let spec = Arc::new(
            crate::standards::vertex_generative_ai::VertexGenerativeAiStandard::new()
                .create_spec("vertex"),
        );
        let bundle = spec.choose_chat_transformers(&request, &ctx);

        let mut exec = ChatExecutorBuilder::new("vertex", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_transformer_bundle(bundle)
            .with_stream_disable_compression(self.config.http_config.stream_disable_compression)
            .with_interceptors(self.http_interceptors.clone())
            .with_middlewares(self.model_middlewares.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            exec = exec.with_transport(transport);
        }
        if let Some(retry) = self.retry_options.clone() {
            exec = exec.with_retry_options(retry);
        }

        let exec = exec.build();
        ChatExecutor::execute_stream(&*exec, request).await
    }

    fn has_auth_header(headers: &std::collections::HashMap<String, String>) -> bool {
        headers
            .keys()
            .any(|k| k.eq_ignore_ascii_case("authorization"))
    }

    async fn inject_auth_header(
        &self,
        req_http: &mut crate::types::HttpConfig,
    ) -> Result<(), LlmError> {
        if Self::has_auth_header(&req_http.headers) {
            return Ok(());
        }
        let Some(tp) = &self.config.token_provider else {
            return Ok(());
        };
        let token = tp.token().await?;
        req_http
            .headers
            .insert("authorization".to_string(), format!("Bearer {token}"));
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn _debug_has_token_provider(&self) -> bool {
        self.config.token_provider.is_some()
    }

    #[cfg(test)]
    pub(crate) fn _debug_has_api_key(&self) -> bool {
        self.config.api_key.is_some()
    }
}

#[async_trait]
impl ImageGenerationCapability for GoogleVertexClient {
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let mut request = request;
        if request.model.is_none() {
            request.model = Some(self.config.model.clone());
        }
        let mut http_config = self.config.http_config.clone();
        self.inject_auth_header(&mut http_config).await?;
        request.http_config = Some(http_config);

        let ctx = self.build_context().await;
        let spec = Arc::new(
            crate::standards::vertex_imagen::VertexImagenStandard::new().create_spec("vertex"),
        );

        let mut builder = ImageExecutorBuilder::new("vertex", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

        let exec = if let Some(retry) = self.retry_options.clone() {
            builder
                .with_retry_options(retry)
                .build_for_request(&request)
        } else {
            builder.build_for_request(&request)
        };

        ImageExecutor::execute(&*exec, request).await
    }
}

#[async_trait]
impl ImageExtras for GoogleVertexClient {
    async fn edit_image(
        &self,
        request: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let mut request = request;
        if request.model.is_none() {
            request.model = Some(self.config.model.clone());
        }
        let mut http_config = self.config.http_config.clone();
        self.inject_auth_header(&mut http_config).await?;
        request.http_config = Some(http_config);

        let ctx = self.build_context().await;
        let spec = Arc::new(
            crate::standards::vertex_imagen::VertexImagenStandard::new().create_spec("vertex"),
        );

        let mut builder = ImageExecutorBuilder::new("vertex", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

        let selector = ImageGenerationRequest {
            model: request.model.clone(),
            ..Default::default()
        };
        let exec = if let Some(retry) = self.retry_options.clone() {
            builder
                .with_retry_options(retry)
                .build_for_request(&selector)
        } else {
            builder.build_for_request(&selector)
        };

        ImageExecutor::execute_edit(&*exec, request).await
    }

    async fn create_variation(
        &self,
        _request: crate::types::ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Vertex provider does not support image variations".to_string(),
        ))
    }

    fn get_supported_sizes(&self) -> Vec<String> {
        vec![]
    }

    fn get_supported_formats(&self) -> Vec<String> {
        vec!["b64_json".to_string()]
    }

    fn supports_image_editing(&self) -> bool {
        true
    }

    fn supports_image_variations(&self) -> bool {
        false
    }
}

#[cfg(feature = "google-vertex")]
#[async_trait]
impl ChatCapability for GoogleVertexClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(self.common_params.clone())
            .http_config(self.config.http_config.clone());
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        self.chat_request_via_spec(builder.build()).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(self.common_params.clone())
            .http_config(self.config.http_config.clone())
            .stream(true);
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        self.chat_stream_request_via_spec(builder.build()).await
    }

    async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.chat_request_via_spec(request).await
    }

    async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        self.chat_stream_request_via_spec(request).await
    }
}

#[async_trait]
impl EmbeddingCapability for GoogleVertexClient {
    async fn embed(&self, input: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        use crate::execution::executors::embedding::{EmbeddingExecutor, EmbeddingExecutorBuilder};

        let req = EmbeddingRequest::new(input).with_model(self.common_params.model.clone());

        let ctx = self.build_context().await;
        let spec = Arc::new(
            crate::standards::vertex_embedding::VertexEmbeddingStandard::new()
                .create_spec("vertex"),
        );

        let mut exec = EmbeddingExecutorBuilder::new("vertex", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            exec = exec.with_transport(transport);
        }

        let exec = if let Some(retry) = self.retry_options.clone() {
            exec.with_retry_options(retry).build_for_request(&req)
        } else {
            exec.build_for_request(&req)
        };

        EmbeddingExecutor::execute(&*exec, req).await
    }

    fn embedding_dimension(&self) -> usize {
        768
    }

    fn max_tokens_per_embedding(&self) -> usize {
        2048
    }

    fn supported_embedding_models(&self) -> Vec<String> {
        vec!["text-embedding-004".to_string()]
    }
}

#[async_trait]
impl EmbeddingExtensions for GoogleVertexClient {
    async fn embed_with_config(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        use crate::execution::executors::embedding::{EmbeddingExecutor, EmbeddingExecutorBuilder};

        let mut request = request;
        if request.model.as_deref().is_none_or(str::is_empty) {
            request.model = Some(self.common_params.model.clone());
        }

        let ctx = self.build_context().await;
        let spec = Arc::new(
            crate::standards::vertex_embedding::VertexEmbeddingStandard::new()
                .create_spec("vertex"),
        );

        let mut exec = EmbeddingExecutorBuilder::new("vertex", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            exec = exec.with_transport(transport);
        }

        let exec = if let Some(retry) = self.retry_options.clone() {
            exec.with_retry_options(retry).build_for_request(&request)
        } else {
            exec.build_for_request(&request)
        };

        EmbeddingExecutor::execute(&*exec, request).await
    }
}

impl LlmClient for GoogleVertexClient {
    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed("vertex")
    }

    fn supported_models(&self) -> Vec<String> {
        vec![self.config.model.clone()]
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_embedding()
            .with_image_generation()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_chat_capability(&self) -> Option<&dyn ChatCapability> {
        Some(self)
    }

    fn as_image_generation_capability(&self) -> Option<&dyn ImageGenerationCapability> {
        Some(self)
    }

    fn as_image_extras(&self) -> Option<&dyn ImageExtras> {
        Some(self)
    }

    fn as_embedding_capability(&self) -> Option<&dyn EmbeddingCapability> {
        Some(self)
    }
}

impl crate::traits::ModelMetadata for GoogleVertexClient {
    fn provider_id(&self) -> &str {
        "vertex"
    }

    fn model_id(&self) -> &str {
        &self.common_params.model
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
    fn google_vertex_llmclient_exposes_expected_capabilities() {
        let cfg = GoogleVertexConfig::new("https://example.invalid", "imagen-3.0-generate-002");
        let client = GoogleVertexClient::from_config(cfg).expect("from_config ok");
        let llm: &dyn crate::client::LlmClient = &client;
        assert_eq!(llm.provider_id(), std::borrow::Cow::Borrowed("vertex"));
        assert!(llm.as_chat_capability().is_some());
        assert!(llm.as_image_generation_capability().is_some());
        assert!(llm.as_embedding_capability().is_some());
        assert_eq!(crate::traits::ModelMetadata::provider_id(&client), "vertex");
        assert_eq!(
            crate::traits::ModelMetadata::model_id(&client),
            "imagen-3.0-generate-002"
        );
    }

    #[test]
    fn google_vertex_config_convenience_constructors_set_expected_defaults() {
        let express = GoogleVertexConfig::express("test-key", "gemini-2.5-pro");
        assert_eq!(
            express.base_url,
            crate::auth::vertex::GOOGLE_VERTEX_EXPRESS_BASE_URL
        );
        assert_eq!(express.api_key.as_deref(), Some("test-key"));

        let enterprise = GoogleVertexConfig::enterprise("project-1", "us-central1", "gemini");
        assert_eq!(
            enterprise.base_url,
            crate::auth::vertex::google_vertex_base_url("project-1", "us-central1")
        );
        assert_eq!(enterprise.api_key, None);
    }

    #[test]
    fn prepare_chat_request_for_stream_sets_stream_and_fills_defaults() {
        let cfg = GoogleVertexConfig::new("https://example.invalid", "gemini-2.5-flash")
            .with_http_config(crate::types::HttpConfig::default());
        let client = GoogleVertexClient::from_config(cfg).expect("from_config ok");

        let request = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .build();

        let prepared = client
            .prepare_chat_request(request, true)
            .expect("prepare stream request");

        assert!(prepared.stream);
        assert_eq!(prepared.common_params.model, "gemini-2.5-flash");
        assert!(prepared.http_config.is_some());
    }

    #[test]
    fn prepare_chat_request_for_non_stream_clears_stream_and_preserves_explicit_model() {
        let cfg = GoogleVertexConfig::new("https://example.invalid", "gemini-2.5-flash")
            .with_http_config(crate::types::HttpConfig::default());
        let client = GoogleVertexClient::from_config(cfg).expect("from_config ok");

        let request = ChatRequest::builder()
            .model("gemini-1.5-pro")
            .messages(vec![ChatMessage::user("hi").build()])
            .stream(true)
            .build();

        let prepared = client
            .prepare_chat_request(request, false)
            .expect("prepare non-stream request");

        assert!(!prepared.stream);
        assert_eq!(prepared.common_params.model, "gemini-1.5-pro");
        assert!(prepared.http_config.is_some());
    }

    #[tokio::test]
    async fn google_vertex_chat_stream_request_preserves_stable_options_at_transport_boundary() {
        let transport = CaptureTransport::default();
        let client = GoogleVertexClient::from_config(
            GoogleVertexConfig::new("https://example.invalid", "gemini-2.5-flash")
                .with_api_key("test-key")
                .with_http_transport(Arc::new(transport.clone())),
        )
        .expect("from_config ok");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model("gemini-2.5-flash")
            .messages(vec![ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "lookup_weather",
                "Look up the weather",
                serde_json::json!({
                    "type": "object",
                    "properties": { "location": { "type": "string" } },
                    "required": ["location"],
                    "additionalProperties": false
                }),
            )])
            .tool_choice(crate::types::ToolChoice::None)
            .response_format(crate::types::ResponseFormat::json_schema(schema.clone()))
            .build()
            .with_provider_option(
                "vertex",
                serde_json::json!({
                    "thinkingConfig": {
                        "thinkingBudget": 2048,
                        "includeThoughts": true
                    },
                    "structuredOutputs": true
                }),
            );

        let _ = client.chat_stream_request(request).await;

        let captured = transport.take_stream().expect("captured stream request");
        assert!(
            captured
                .url
                .contains("/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key=test-key"),
            "unexpected url: {}",
            captured.url
        );
        assert_eq!(
            captured.headers.get("accept").and_then(|v| v.to_str().ok()),
            Some("text/event-stream")
        );
        assert!(transport.take().is_none());
        assert_eq!(
            captured.body["generationConfig"]["thinkingConfig"],
            serde_json::json!({
                "thinkingBudget": 2048,
                "includeThoughts": true
            })
        );
        assert_eq!(
            captured.body["generationConfig"]["responseMimeType"],
            serde_json::json!("application/json")
        );
        assert!(
            captured.body["generationConfig"]
                .get("responseSchema")
                .is_some()
        );
        assert!(
            captured.body["generationConfig"]
                .get("responseJsonSchema")
                .is_none()
        );
        assert_eq!(
            captured.body["toolConfig"],
            serde_json::json!({
                "functionCallingConfig": { "mode": "NONE" }
            })
        );
        assert_eq!(
            captured.body["tools"][0]["functionDeclarations"][0]["name"],
            serde_json::json!("lookup_weather")
        );
        assert_eq!(
            captured.body["contents"][0]["parts"][0]["text"],
            serde_json::json!("hi")
        );
    }
}
