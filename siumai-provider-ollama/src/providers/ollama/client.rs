//! Ollama Client Implementation
//!
//! Main client that aggregates all Ollama capabilities.

use async_trait::async_trait;
use backoff::ExponentialBackoffBuilder;
use std::sync::Arc;
use std::time::Duration;

use crate::client::LlmClient;
use crate::core::ProviderContext;
use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::http::transport::HttpTransport;
use crate::execution::middleware::LanguageModelMiddleware;
use crate::retry_api::RetryOptions;
use crate::streaming::ChatStream;
use crate::traits::{
    ChatCapability, EmbeddingCapability, EmbeddingExtensions, LlmProvider, ModelListingCapability,
    ProviderCapabilities,
};
use crate::types::*;

use super::chat::OllamaChatCapability;
use super::config::{OllamaConfig, OllamaParams};
use super::embeddings::OllamaEmbeddings;
use super::get_default_models;
use super::models::OllamaModelsCapability;

/// Ollama Client
pub struct OllamaClient {
    /// Chat capability implementation
    chat_capability: OllamaChatCapability,
    /// Embedding capability implementation
    embedding_capability: OllamaEmbeddings,
    /// Models capability implementation
    models_capability: OllamaModelsCapability,
    /// Common parameters
    common_params: CommonParams,
    /// Ollama-specific parameters
    ollama_params: OllamaParams,
    /// HTTP client for making requests
    http_client: reqwest::Client,
    /// Base URL for Ollama API
    base_url: String,
    /// Tracing configuration
    tracing_config: Option<crate::observability::tracing::TracingConfig>,
    /// Tracing guard to keep tracing system active (retained but not read)
    /// NOTE: Tracing subscriber functionality has been moved to siumai-extras
    #[allow(dead_code)]
    _tracing_guard: Option<()>,
    /// Unified retry options for chat
    retry_options: Option<RetryOptions>,
    /// Optional HTTP interceptors applied to all chat requests
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional model-level middlewares
    model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    /// Optional custom HTTP transport (Vercel-style "custom fetch" parity).
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
}

impl Clone for OllamaClient {
    fn clone(&self) -> Self {
        Self {
            chat_capability: self.chat_capability.clone(),
            embedding_capability: self.embedding_capability.clone(),
            models_capability: self.models_capability.clone(),
            common_params: self.common_params.clone(),
            ollama_params: self.ollama_params.clone(),
            http_client: self.http_client.clone(),
            base_url: self.base_url.clone(),
            tracing_config: self.tracing_config.clone(),
            _tracing_guard: None, // Don't clone the tracing guard
            retry_options: self.retry_options.clone(),
            http_interceptors: self.http_interceptors.clone(),
            model_middlewares: self.model_middlewares.clone(),
            http_transport: self.http_transport.clone(),
        }
    }
}

impl std::fmt::Debug for OllamaClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OllamaClient")
            .field("provider_id", &"ollama")
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
        let http_transport = config.http_transport.clone();

        let chat_capability = OllamaChatCapability::new(
            config.base_url.clone(),
            http_client.clone(),
            config.http_config.clone(),
            config.ollama_params.clone(),
            http_transport.clone(),
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
            http_transport.clone(),
        );

        let models_capability = OllamaModelsCapability::new(
            config.base_url.clone(),
            http_client.clone(),
            config.http_config.clone(),
            http_transport.clone(),
        );

        Self {
            chat_capability,
            embedding_capability,
            models_capability,
            common_params: config.common_params,
            ollama_params: config.ollama_params,
            http_client,
            base_url: config.base_url,
            tracing_config: None,
            _tracing_guard: None,
            retry_options: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
            http_transport,
        }
    }

    /// Construct an `OllamaClient` from an `OllamaConfig` (config-first construction).
    ///
    /// This is the recommended construction style for new code that does not want to
    /// depend on the unified builder surface.
    pub fn from_config(config: OllamaConfig) -> Result<Self, LlmError> {
        config.validate()?;
        let http_interceptors = config.http_interceptors.clone();
        let model_middlewares = config.model_middlewares.clone();
        let http_client =
            crate::execution::http::client::build_http_client_from_config(&config.http_config)?;
        Ok(Self::new(config, http_client)
            .with_http_interceptors(http_interceptors)
            .with_model_middlewares(model_middlewares))
    }

    /// Construct an `OllamaClient` from an `OllamaConfig` with a caller-supplied HTTP client.
    pub fn with_http_client(
        config: OllamaConfig,
        http_client: reqwest::Client,
    ) -> Result<Self, LlmError> {
        config.validate()?;
        let http_interceptors = config.http_interceptors.clone();
        let model_middlewares = config.model_middlewares.clone();
        Ok(Self::new(config, http_client)
            .with_http_interceptors(http_interceptors)
            .with_model_middlewares(model_middlewares))
    }

    /// Creates a new Ollama client with configuration
    pub fn new_with_config(config: OllamaConfig) -> Self {
        let http_interceptors = config.http_interceptors.clone();
        let model_middlewares = config.model_middlewares.clone();
        let http_client =
            crate::execution::http::client::build_http_client_from_config(&config.http_config)
                .unwrap_or_else(|_| reqwest::Client::new());
        Self::new(config, http_client)
            .with_http_interceptors(http_interceptors)
            .with_model_middlewares(model_middlewares)
    }

    /// Get the base URL
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Get the normalized provider context used by execution helpers.
    pub fn provider_context(&self) -> ProviderContext {
        ProviderContext::new(
            "ollama",
            self.base_url.clone(),
            None,
            std::collections::HashMap::new(),
        )
    }

    /// Get the underlying HTTP client.
    pub fn http_client(&self) -> reqwest::Client {
        self.http_client.clone()
    }

    /// Get unified retry options.
    pub fn retry_options(&self) -> Option<RetryOptions> {
        self.retry_options.clone()
    }

    /// Get installed HTTP interceptors.
    pub fn http_interceptors(&self) -> Vec<Arc<dyn HttpInterceptor>> {
        self.http_interceptors.clone()
    }

    /// Get the installed custom HTTP transport.
    pub fn http_transport(&self) -> Option<Arc<dyn HttpTransport>> {
        self.http_transport.clone()
    }

    /// Set the tracing configuration
    #[doc(hidden)]
    pub fn set_tracing_config(
        &mut self,
        config: Option<crate::observability::tracing::TracingConfig>,
    ) {
        self.tracing_config = config;
    }

    /// Set unified retry options
    pub fn set_retry_options(&mut self, options: Option<RetryOptions>) {
        self.retry_options = options.map(|mut opts| {
            if matches!(opts.backend, crate::retry_api::RetryBackend::Backoff)
                && opts.backoff_executor.is_none()
            {
                opts.backoff_executor = Some(ollama_backoff_executor());
            }
            opts
        });
        self.embedding_capability = self
            .embedding_capability
            .clone()
            .with_retry_options(self.retry_options.clone());
    }

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
    pub const fn with_temperature(mut self, temperature: f64) -> Self {
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

    /// Check if Ollama server is running
    pub async fn health_check(&self) -> Result<bool, LlmError> {
        use crate::execution::executors::common::execute_get_request;

        let spec = Arc::new(crate::providers::ollama::spec::OllamaSpec::new(
            self.ollama_params.clone(),
        ));
        let config = self.http_wiring().config(spec);

        let url = format!("{}/api/version", self.base_url.trim_end_matches('/'));
        match execute_get_request(&config, &url, None).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Get Ollama version
    pub async fn version(&self) -> Result<String, LlmError> {
        use crate::execution::executors::common::execute_get_request;

        let spec = Arc::new(crate::providers::ollama::spec::OllamaSpec::new(
            self.ollama_params.clone(),
        ));
        let config = self.http_wiring().config(spec);

        let url = format!("{}/api/version", self.base_url.trim_end_matches('/'));
        let res = execute_get_request(&config, &url, None).await?;
        let version_response: super::types::OllamaVersionResponse =
            serde_json::from_value(res.json).map_err(|e| {
                LlmError::ParseError(format!("Failed to parse Ollama version response: {}", e))
            })?;
        Ok(version_response.version)
    }

    /// Create provider context for this client
    fn build_context(&self) -> crate::core::ProviderContext {
        crate::core::ProviderContext::new(
            "ollama",
            self.base_url.clone(),
            None,
            self.chat_capability.http_config.headers.clone(),
        )
    }

    fn prepare_chat_request(
        &self,
        request: ChatRequest,
        stream: bool,
    ) -> Result<ChatRequest, LlmError> {
        let request = crate::utils::chat_request::normalize_chat_request(
            request,
            crate::utils::chat_request::ChatRequestDefaults::new(&self.common_params),
            stream,
        );
        if request.common_params.model.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Model is required".to_string(),
            ));
        }
        Ok(request)
    }

    fn http_wiring(&self) -> crate::execution::wiring::HttpExecutionWiring {
        let mut wiring = crate::execution::wiring::HttpExecutionWiring::new(
            "ollama",
            self.http_client.clone(),
            self.build_context(),
        )
        .with_interceptors(self.http_interceptors.clone())
        .with_retry_options(self.retry_options.clone());

        if let Some(transport) = self.http_transport.clone() {
            wiring = wiring.with_transport(transport);
        }

        wiring
    }

    /// Create chat executor using the builder pattern
    fn build_chat_executor(
        &self,
        request: &ChatRequest,
    ) -> Arc<crate::execution::executors::chat::HttpChatExecutor> {
        use crate::core::ProviderSpec;
        use crate::execution::executors::chat::ChatExecutorBuilder;

        let ctx = self.build_context();
        let spec = Arc::new(crate::providers::ollama::spec::OllamaSpec::new(
            self.ollama_params.clone(),
        ));
        let bundle = spec.choose_chat_transformers(request, &ctx);
        let before_send_hook = spec.chat_before_send(request, &ctx);

        let mut builder = ChatExecutorBuilder::new("ollama", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_transformer_bundle(bundle)
            .with_stream_disable_compression(
                self.chat_capability.http_config.stream_disable_compression,
            )
            .with_interceptors(self.http_interceptors.clone())
            .with_middlewares(self.model_middlewares.clone());

        if let Some(transport) = self.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

        if let Some(hook) = before_send_hook {
            builder = builder.with_before_send(hook);
        }

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        builder.build()
    }

    /// Execute chat request via spec (unified implementation)
    async fn chat_request_via_spec(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        use crate::execution::executors::chat::ChatExecutor;

        let request = self.prepare_chat_request(request, false)?;
        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute(&*exec, request).await
    }

    /// Execute streaming chat request via spec (unified implementation)
    async fn chat_stream_request_via_spec(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStream, LlmError> {
        use crate::execution::executors::chat::ChatExecutor;

        let request = self.prepare_chat_request(request, true)?;
        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute_stream(&*exec, request).await
    }
}

impl crate::traits::ModelMetadata for OllamaClient {
    fn provider_id(&self) -> &str {
        "ollama"
    }

    fn model_id(&self) -> &str {
        &self.common_params.model
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
        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(self.common_params.clone());
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        self.chat_request_via_spec(builder.build()).await
    }

    /// Streaming chat with tools
    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(self.common_params.clone())
            .stream(true);
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        let request = builder.build();

        self.chat_stream_request_via_spec(request).await
    }

    async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.chat_request_via_spec(request).await
    }

    async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        self.chat_stream_request_via_spec(request).await
    }
}

#[async_trait]
impl EmbeddingCapability for OllamaClient {
    async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        self.embedding_capability.embed(texts).await
    }

    fn embedding_dimension(&self) -> usize {
        self.embedding_capability.embedding_dimension()
    }
}

#[async_trait]
impl EmbeddingExtensions for OllamaClient {
    async fn embed_with_config(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        self.embedding_capability.embed_with_config(request).await
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
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("ollama")
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

    fn as_chat_capability(&self) -> Option<&dyn ChatCapability> {
        Some(self)
    }

    fn as_embedding_capability(&self) -> Option<&dyn EmbeddingCapability> {
        Some(self)
    }

    fn as_embedding_extensions(&self) -> Option<&dyn crate::traits::EmbeddingExtensions> {
        Some(self)
    }

    fn as_model_listing_capability(&self) -> Option<&dyn crate::traits::ModelListingCapability> {
        Some(self)
    }
}

impl LlmProvider for OllamaClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("ollama")
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

fn ollama_backoff_executor() -> crate::retry_api::BackoffRetryExecutor {
    let backoff = ExponentialBackoffBuilder::new()
        .with_initial_interval(Duration::from_millis(500))
        .with_max_interval(Duration::from_secs(30))
        .with_multiplier(1.5)
        .with_max_elapsed_time(Some(Duration::from_secs(180)))
        .build();
    crate::retry_api::BackoffRetryExecutor::with_backoff(backoff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::transport::{
        HttpTransport, HttpTransportRequest, HttpTransportResponse, HttpTransportStreamBody,
        HttpTransportStreamResponse,
    };
    use crate::providers::ollama::ext::OllamaChatRequestExt;
    use async_trait::async_trait;
    use futures_util::StreamExt;
    use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
    use std::sync::Mutex;

    #[derive(Clone, Default)]
    struct CaptureTransport {
        last: Arc<Mutex<Option<HttpTransportRequest>>>,
    }

    #[async_trait]
    impl HttpTransport for CaptureTransport {
        async fn execute_json(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last.lock().unwrap() = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportResponse {
                status: 200,
                headers,
                body: br#"{"model":"llama3.2","created_at":"2026-03-23T00:00:00Z","message":{"role":"assistant","content":"ok"},"done":true}"#
                    .to_vec(),
            })
        }
    }

    #[derive(Clone, Default)]
    struct RecordingTransport {
        json_calls: Arc<Mutex<Vec<HttpTransportRequest>>>,
        stream_calls: Arc<Mutex<Vec<HttpTransportRequest>>>,
    }

    impl RecordingTransport {
        fn json_calls(&self) -> Vec<HttpTransportRequest> {
            self.json_calls.lock().unwrap().clone()
        }

        fn stream_calls(&self) -> Vec<HttpTransportRequest> {
            self.stream_calls.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl HttpTransport for RecordingTransport {
        async fn execute_json(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            self.json_calls.lock().unwrap().push(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportResponse {
                status: 200,
                headers,
                body: br#"{"model":"llama3.2","created_at":"2026-03-23T00:00:00Z","message":{"role":"assistant","content":"ok"},"done":true}"#
                    .to_vec(),
            })
        }

        async fn execute_stream(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportStreamResponse, LlmError> {
            self.stream_calls.lock().unwrap().push(request);

            let mut headers = HeaderMap::new();
            headers.insert(
                CONTENT_TYPE,
                HeaderValue::from_static("application/x-ndjson"),
            );

            Ok(HttpTransportStreamResponse {
                status: 200,
                headers,
                body: HttpTransportStreamBody::from_bytes(
                    br#"{"model":"llama3.2","message":{"role":"assistant","content":"ok"},"done":false}
{"model":"llama3.2","done":true,"prompt_eval_count":1,"eval_count":1}
"#
                    .to_vec(),
                ),
            })
        }
    }

    #[test]
    fn test_client_creation() {
        let config = OllamaConfig::default();
        let client = OllamaClient::new_with_config(config);

        assert_eq!(
            LlmProvider::provider_id(&client),
            std::borrow::Cow::Borrowed("ollama")
        );
        assert_eq!(client.base_url(), "http://localhost:11434");
    }

    #[test]
    fn ollama_client_from_config_builds_http_client() {
        let config = OllamaConfig::default();
        let client = OllamaClient::from_config(config).expect("from_config ok");
        assert_eq!(client.base_url(), "http://localhost:11434");
    }

    #[test]
    fn ollama_client_with_http_client_preserves_wrapper_context() {
        let transport = CaptureTransport::default();
        let config = OllamaConfig::builder()
            .base_url("http://localhost:11434/custom/")
            .model("llama3.2")
            .http_transport(Arc::new(transport))
            .build()
            .expect("build ollama config");

        let mut client =
            OllamaClient::with_http_client(config, reqwest::Client::new()).expect("client ok");
        client.set_retry_options(Some(RetryOptions::backoff()));

        let ctx = client.provider_context();
        assert_eq!(ctx.provider_id, "ollama");
        assert_eq!(ctx.base_url, "http://localhost:11434/custom/");
        assert!(ctx.api_key.is_none());
        assert_eq!(client.base_url(), "http://localhost:11434/custom/");
        assert!(client.retry_options().is_some());
        assert!(client.http_transport().is_some());
        assert_eq!(crate::traits::ModelMetadata::provider_id(&client), "ollama");
        assert_eq!(crate::traits::ModelMetadata::model_id(&client), "llama3.2");
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

    #[test]
    fn test_chat_executor_uses_client_params_and_provider_options_override() {
        use crate::provider_options::OllamaOptions;

        let config = OllamaConfig::default();
        let client = OllamaClient::new_with_config(config)
            .with_model("llama3.2")
            .with_keep_alive("10m")
            .with_raw(true)
            .with_format("json")
            .with_reasoning_enabled()
            .with_option(
                "top_k",
                serde_json::Value::Number(serde_json::Number::from(40)),
            );

        let request = ChatRequest::builder()
            .messages(vec![ChatMessage::user("Hello").build()])
            .common_params(client.common_params().clone())
            .build()
            .with_ollama_options(
                OllamaOptions::new()
                    .with_keep_alive("1m")
                    .with_raw_mode(false)
                    .with_param("think", serde_json::json!(false))
                    .with_param("num_ctx", serde_json::json!(4096)),
            );

        let exec = client.build_chat_executor(&request);
        let json = exec
            .request_transformer
            .as_ref()
            .expect("ollama chat executor should install a request transformer")
            .transform_chat(&request)
            .unwrap();

        assert_eq!(json["keep_alive"], "1m");
        assert_eq!(json["raw"], false);
        assert_eq!(json["format"], "json");
        assert_eq!(json["think"], false);
        assert_eq!(json["options"]["num_ctx"], 4096);
        assert_eq!(json["options"]["top_k"], 40);
    }

    #[tokio::test]
    async fn chat_stream_request_forces_stream_true_and_preserves_response_format_schema() {
        let transport = RecordingTransport::default();
        let config = OllamaConfig::builder()
            .base_url("http://localhost:11434/custom/")
            .model("llama3.2")
            .http_transport(Arc::new(transport.clone()))
            .build()
            .expect("build ollama config");
        let client = OllamaClient::from_config(config).expect("client ok");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model("llama3.2")
            .messages(vec![ChatMessage::user("Hello").build()])
            .provider_option("ollama", serde_json::json!({ "format": "json" }))
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                schema.clone(),
            ))
            .build();

        let mut stream = client
            .chat_stream_request(request)
            .await
            .expect("stream request ok");
        while let Some(event) = stream.next().await {
            let event = event.expect("stream event ok");
            if matches!(event, crate::types::ChatStreamEvent::StreamEnd { .. }) {
                break;
            }
        }

        let json_calls = transport.json_calls();
        assert!(
            json_calls.is_empty(),
            "stream path should not use execute_json"
        );

        let stream_calls = transport.stream_calls();
        assert_eq!(stream_calls.len(), 1);

        let call = &stream_calls[0];
        assert_eq!(call.url, "http://localhost:11434/custom/api/chat");
        assert_eq!(call.body["stream"], serde_json::json!(true));
        assert_eq!(call.body["format"], schema);
    }

    #[tokio::test]
    async fn chat_request_required_tool_choice_fails_fast_before_transport() {
        let transport = RecordingTransport::default();
        let config = OllamaConfig::builder()
            .base_url("http://localhost:11434/custom/")
            .model("llama3.2")
            .http_transport(Arc::new(transport.clone()))
            .build()
            .expect("build ollama config");
        let client = OllamaClient::from_config(config).expect("client ok");

        let request = ChatRequest::builder()
            .model("llama3.2")
            .messages(vec![ChatMessage::user("Hello").build()])
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "city": { "type": "string" }
                    },
                    "required": ["city"]
                }),
            )])
            .tool_choice(crate::types::ToolChoice::Required)
            .build();

        let err = client
            .chat_request(request)
            .await
            .expect_err("required tool_choice should fail before transport");
        assert!(matches!(err, LlmError::UnsupportedOperation(_)));
        assert!(transport.json_calls().is_empty());
        assert!(transport.stream_calls().is_empty());
    }

    #[tokio::test]
    async fn chat_request_fills_missing_common_params_from_client_defaults() {
        let transport = RecordingTransport::default();
        let config = OllamaConfig::builder()
            .base_url("http://localhost:11434/custom/")
            .model("llama3.2")
            .temperature(0.6)
            .max_tokens(128)
            .http_transport(Arc::new(transport.clone()))
            .build()
            .expect("build ollama config");
        let client = OllamaClient::from_config(config).expect("client ok");

        let request = ChatRequest::builder()
            .messages(vec![ChatMessage::user("Hello").build()])
            .build();

        let _response = client.chat_request(request).await.expect("request ok");
        let calls = transport.json_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].body["model"], serde_json::json!("llama3.2"));
        assert_eq!(
            calls[0].body["options"]["temperature"],
            serde_json::json!(0.6)
        );
        assert_eq!(
            calls[0].body["options"]["num_predict"],
            serde_json::json!(128)
        );
    }

    #[tokio::test]
    async fn chat_stream_request_preserves_explicit_common_params_over_client_defaults() {
        let transport = RecordingTransport::default();
        let config = OllamaConfig::builder()
            .base_url("http://localhost:11434/custom/")
            .model("llama3.2")
            .temperature(0.6)
            .max_tokens(128)
            .http_transport(Arc::new(transport.clone()))
            .build()
            .expect("build ollama config");
        let client = OllamaClient::from_config(config).expect("client ok");

        let request = ChatRequest::builder()
            .model("qwen3:latest")
            .temperature(0.2)
            .messages(vec![ChatMessage::user("Hello").build()])
            .build();

        let mut stream = client
            .chat_stream_request(request)
            .await
            .expect("stream request ok");
        while let Some(event) = stream.next().await {
            let event = event.expect("stream event ok");
            if matches!(event, crate::types::ChatStreamEvent::StreamEnd { .. }) {
                break;
            }
        }

        let calls = transport.stream_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].body["model"], serde_json::json!("qwen3:latest"));
        assert_eq!(calls[0].body["stream"], serde_json::json!(true));
        assert_eq!(
            calls[0].body["options"]["temperature"],
            serde_json::json!(0.2)
        );
        assert_eq!(
            calls[0].body["options"]["num_predict"],
            serde_json::json!(128)
        );
    }
}
