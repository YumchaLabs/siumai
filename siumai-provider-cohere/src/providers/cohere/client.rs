//! `Cohere` native unified client.

use super::config::CohereConfig;
use crate::client::LlmClient;
use crate::core::{ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::executors::chat::{ChatExecutor, ChatExecutorBuilder};
use crate::execution::executors::embedding::{EmbeddingExecutor, EmbeddingExecutorBuilder};
use crate::execution::executors::rerank::{RerankExecutor, RerankExecutorBuilder};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::http::transport::HttpTransport;
use crate::retry_api::RetryOptions;
use crate::standards::cohere::CohereSpec;
use crate::traits::{
    ChatCapability, EmbeddingCapability, EmbeddingExtensions, ModelMetadata, ProviderCapabilities,
    RerankCapability,
};
use crate::types::{
    ChatRequest, ChatResponse, EmbeddingRequest, EmbeddingResponse, RerankRequest, RerankResponse,
};
use crate::utils::chat_request::{ChatRequestDefaults, normalize_chat_request};
use async_trait::async_trait;
use secrecy::ExposeSecret;
use std::borrow::Cow;
use std::sync::Arc;

const EMBEDDING_MODELS: &[&str] = &[
    "embed-english-v3.0",
    "embed-multilingual-v3.0",
    "embed-english-light-v3.0",
    "embed-multilingual-light-v3.0",
    "embed-v4.0",
];
const MAX_EMBEDDINGS_PER_CALL: usize = 96;

#[derive(Clone)]
pub struct CohereClient {
    config: CohereConfig,
    http_client: reqwest::Client,
    retry_options: Option<RetryOptions>,
}

impl std::fmt::Debug for CohereClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CohereClient")
            .field("config", &self.config)
            .field("retry_options", &self.retry_options)
            .finish()
    }
}

impl CohereClient {
    pub fn from_config(config: CohereConfig) -> Result<Self, LlmError> {
        config.validate()?;
        let http_client =
            crate::execution::http::client::build_http_client_from_config(&config.http_config)?;
        Self::with_http_client(config, http_client)
    }

    pub fn with_http_client(
        config: CohereConfig,
        http_client: reqwest::Client,
    ) -> Result<Self, LlmError> {
        config.validate()?;
        Ok(Self {
            config,
            http_client,
            retry_options: None,
        })
    }

    pub fn with_retry_options(mut self, retry_options: RetryOptions) -> Self {
        self.retry_options = Some(retry_options);
        self
    }

    pub fn with_retry(self, retry_options: RetryOptions) -> Self {
        self.with_retry_options(retry_options)
    }

    fn provider_spec(&self) -> Arc<dyn ProviderSpec> {
        Arc::new(CohereSpec::new())
    }

    fn build_context(&self) -> ProviderContext {
        ProviderContext::new(
            "cohere",
            self.config.base_url.clone(),
            Some(self.config.api_key.expose_secret().to_string()),
            self.config.http_config.headers.clone(),
        )
    }

    fn configured_model(&self) -> Option<String> {
        let model = self.config.common_params.model.trim();
        (!model.is_empty()).then_some(model.to_string())
    }

    fn build_chat_executor(&self) -> Arc<dyn ChatExecutor> {
        let mut builder = ChatExecutorBuilder::new("cohere", self.http_client.clone())
            .with_spec(self.provider_spec())
            .with_context(self.build_context())
            .with_runtime_transformer_selection()
            .with_interceptors(self.config.http_interceptors.clone());

        if let Some(retry_options) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry_options);
        }
        if let Some(transport) = self.config.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

        builder.build()
    }

    fn build_embedding_executor(
        &self,
        request: &EmbeddingRequest,
    ) -> Arc<crate::execution::executors::embedding::HttpEmbeddingExecutor> {
        let mut builder = EmbeddingExecutorBuilder::new("cohere", self.http_client.clone())
            .with_spec(self.provider_spec())
            .with_context(self.build_context())
            .with_interceptors(self.config.http_interceptors.clone());

        if let Some(retry_options) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry_options);
        }
        if let Some(transport) = self.config.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

        builder.build_for_request(request)
    }

    fn build_rerank_executor(
        &self,
        request: &RerankRequest,
    ) -> Arc<crate::execution::executors::rerank::HttpRerankExecutor> {
        let mut builder = RerankExecutorBuilder::new("cohere", self.http_client.clone())
            .with_spec(self.provider_spec())
            .with_context(self.build_context())
            .with_interceptors(self.config.http_interceptors.clone());

        if let Some(retry_options) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry_options);
        }
        if let Some(transport) = self.config.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

        builder.build_for_request(request)
    }

    pub fn provider_context(&self) -> ProviderContext {
        self.build_context()
    }

    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    pub fn http_client(&self) -> reqwest::Client {
        self.http_client.clone()
    }

    pub fn retry_options(&self) -> Option<RetryOptions> {
        self.retry_options.clone()
    }

    pub fn http_interceptors(&self) -> Vec<Arc<dyn HttpInterceptor>> {
        self.config.http_interceptors.clone()
    }

    pub fn http_transport(&self) -> Option<Arc<dyn HttpTransport>> {
        self.config.http_transport.clone()
    }

    pub fn set_retry_options(&mut self, options: Option<RetryOptions>) {
        self.retry_options = options;
    }

    #[cfg(test)]
    pub(crate) fn _debug_base_url(&self) -> &str {
        &self.config.base_url
    }
}

#[async_trait]
impl ChatCapability for CohereClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<crate::types::ChatMessage>,
        tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        let mut request = ChatRequest::new(messages);
        request.tools = tools;
        self.chat_request(request).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<crate::types::ChatMessage>,
        tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<crate::streaming::ChatStream, LlmError> {
        let mut request = ChatRequest::new(messages);
        request.tools = tools;
        self.chat_stream_request(request).await
    }

    async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        let request = normalize_chat_request(
            request,
            ChatRequestDefaults::new(&self.config.common_params)
                .with_http_config(&self.config.http_config),
            false,
        );
        if request.common_params.model.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Cohere chat request requires a non-empty model id".to_string(),
            ));
        }

        ChatExecutor::execute(&*self.build_chat_executor(), request).await
    }

    async fn chat_stream_request(
        &self,
        request: ChatRequest,
    ) -> Result<crate::streaming::ChatStream, LlmError> {
        let request = normalize_chat_request(
            request,
            ChatRequestDefaults::new(&self.config.common_params)
                .with_http_config(&self.config.http_config),
            true,
        );
        if request.common_params.model.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Cohere chat request requires a non-empty model id".to_string(),
            ));
        }

        ChatExecutor::execute_stream(&*self.build_chat_executor(), request).await
    }
}

#[async_trait]
impl EmbeddingCapability for CohereClient {
    async fn embed(&self, input: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        let mut request = EmbeddingRequest::new(input);
        if let Some(model) = self.configured_model() {
            request = request.with_model(model);
        }
        self.embed_with_config(request).await
    }

    fn as_embedding_extensions(&self) -> Option<&dyn EmbeddingExtensions> {
        Some(self)
    }

    fn embedding_dimension(&self) -> usize {
        embedding_dimension_for_model(
            self.configured_model()
                .as_deref()
                .unwrap_or("embed-english-v3.0"),
        )
    }

    fn max_tokens_per_embedding(&self) -> usize {
        8192
    }

    fn supported_embedding_models(&self) -> Vec<String> {
        EMBEDDING_MODELS
            .iter()
            .map(|model| (*model).to_string())
            .collect()
    }
}

#[async_trait]
impl EmbeddingExtensions for CohereClient {
    async fn embed_with_config(
        &self,
        mut request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        if request.model.as_deref().unwrap_or("").trim().is_empty()
            && let Some(model) = self.configured_model()
        {
            request.model = Some(model);
        }
        if request.input.len() > MAX_EMBEDDINGS_PER_CALL {
            return Err(LlmError::InvalidInput(format!(
                "Cohere embedding requests support at most {MAX_EMBEDDINGS_PER_CALL} inputs per call, got {}",
                request.input.len()
            )));
        }
        if request.model.as_deref().unwrap_or("").trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Cohere embedding request requires a non-empty model id".to_string(),
            ));
        }

        EmbeddingExecutor::execute(&*self.build_embedding_executor(&request), request).await
    }
}

#[async_trait]
impl RerankCapability for CohereClient {
    async fn rerank(&self, mut request: RerankRequest) -> Result<RerankResponse, LlmError> {
        if request.model.trim().is_empty()
            && let Some(model) = self.configured_model()
        {
            request.model = model;
        }
        if request.model.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Cohere rerank request requires a non-empty model id".to_string(),
            ));
        }

        RerankExecutor::execute(&*self.build_rerank_executor(&request), request).await
    }

    fn supported_models(&self) -> Vec<String> {
        configured_model_or_empty(&self.config)
    }
}

impl ModelMetadata for CohereClient {
    fn provider_id(&self) -> &str {
        "cohere"
    }

    fn model_id(&self) -> &str {
        &self.config.common_params.model
    }
}

impl LlmClient for CohereClient {
    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed("cohere")
    }

    fn supported_models(&self) -> Vec<String> {
        configured_model_or_empty(&self.config)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_embedding()
            .with_rerank()
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

    fn as_embedding_extensions(&self) -> Option<&dyn EmbeddingExtensions> {
        Some(self)
    }

    fn as_rerank_capability(&self) -> Option<&dyn RerankCapability> {
        Some(self)
    }
}

fn configured_model_or_empty(config: &CohereConfig) -> Vec<String> {
    let model = config.common_params.model.trim();
    if model.is_empty() {
        Vec::new()
    } else {
        vec![model.to_string()]
    }
}

fn embedding_dimension_for_model(model: &str) -> usize {
    match model {
        "embed-english-v3.0" | "embed-multilingual-v3.0" => 1024,
        "embed-english-light-v3.0" | "embed-multilingual-light-v3.0" => 384,
        "embed-v4.0" => 1536,
        _ => 1024,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::Arc;

    #[derive(Clone, Default)]
    struct NoopInterceptor;

    impl HttpInterceptor for NoopInterceptor {}

    #[derive(Clone, Default)]
    struct NoopTransport;

    #[async_trait]
    impl HttpTransport for NoopTransport {
        async fn execute_json(
            &self,
            _request: crate::execution::http::transport::HttpTransportRequest,
        ) -> Result<crate::execution::http::transport::HttpTransportResponse, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "NoopTransport does not execute requests".to_string(),
            ))
        }
    }

    #[test]
    fn cohere_client_exposes_provider_context_and_runtime_helpers() {
        let transport = Arc::new(NoopTransport);
        let interceptor = Arc::new(NoopInterceptor);
        let config = CohereConfig::new("test-key")
            .with_base_url("https://example.com/cohere")
            .with_model("command-r")
            .with_http_transport(transport.clone())
            .with_http_interceptors(vec![interceptor]);
        let mut client = CohereClient::from_config(config).expect("client");

        client.set_retry_options(Some(RetryOptions::backoff()));

        let ctx = client.provider_context();
        assert_eq!(ctx.base_url, "https://example.com/cohere");
        assert_eq!(ctx.api_key.as_deref(), Some("test-key"));
        assert_eq!(client.base_url(), "https://example.com/cohere");
        assert!(client.retry_options().is_some());
        assert!(client.http_transport().is_some());
        assert_eq!(client.http_interceptors().len(), 1);
        assert!(client.capabilities().supports("chat"));
        assert!(client.capabilities().supports("embedding"));
        assert!(client.capabilities().supports("rerank"));
        let _http_client = client.http_client();
    }

    #[test]
    fn cohere_client_allows_config_without_default_model() {
        let client = CohereClient::from_config(CohereConfig::new("test-key")).expect("client");
        assert!(crate::client::LlmClient::supported_models(&client).is_empty());
        assert_eq!(client.model_id(), "");
    }

    #[tokio::test]
    async fn cohere_embedding_rejects_more_than_96_inputs_per_call() {
        let client = CohereClient::from_config(
            CohereConfig::new("test-key").with_model("embed-english-v3.0"),
        )
        .expect("client");
        let request = EmbeddingRequest::new(
            (0..97)
                .map(|index| format!("input-{index}"))
                .collect::<Vec<_>>(),
        );

        let err = client
            .embed_with_config(request)
            .await
            .expect_err("must fail");

        assert!(
            matches!(err, LlmError::InvalidInput(ref message) if message.contains("at most 96 inputs per call")),
            "expected invalid input limit error, got {err:?}"
        );
    }
}
