//! `Bedrock` client.

use super::config::BedrockConfig;
use crate::client::LlmClient;
use crate::core::{ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::executors::chat::{ChatExecutor, ChatExecutorBuilder};
use crate::execution::executors::rerank::{RerankExecutor, RerankExecutorBuilder};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::http::transport::HttpTransport;
use crate::retry_api::RetryOptions;
use crate::standards::bedrock::chat::BedrockChatStandard;
use crate::standards::bedrock::rerank::BedrockRerankStandard;
use crate::streaming::ChatStream;
use crate::traits::{ChatCapability, ModelMetadata, ProviderCapabilities, RerankCapability};
use crate::types::{ChatMessage, ChatRequest, ChatResponse, RerankRequest, RerankResponse, Tool};
use async_trait::async_trait;
use secrecy::ExposeSecret;
use std::borrow::Cow;
use std::sync::Arc;

/// Provider-owned Amazon Bedrock client.
#[derive(Clone)]
pub struct BedrockClient {
    config: BedrockConfig,
    http_client: reqwest::Client,
    retry_options: Option<RetryOptions>,
}

impl std::fmt::Debug for BedrockClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BedrockClient")
            .field("config", &self.config)
            .field("retry_options", &self.retry_options)
            .finish()
    }
}

impl BedrockClient {
    /// Build a client from config using an HTTP client derived from `http_config`.
    pub fn from_config(config: BedrockConfig) -> Result<Self, LlmError> {
        config.validate()?;
        let http_client =
            crate::execution::http::client::build_http_client_from_config(&config.http_config)?;
        Self::with_http_client(config, http_client)
    }

    /// Build a client from config with an explicit `reqwest::Client`.
    pub fn with_http_client(
        config: BedrockConfig,
        http_client: reqwest::Client,
    ) -> Result<Self, LlmError> {
        config.validate()?;
        Ok(Self {
            config,
            http_client,
            retry_options: None,
        })
    }

    /// Set retry options.
    pub fn with_retry_options(mut self, retry_options: RetryOptions) -> Self {
        self.retry_options = Some(retry_options);
        self
    }

    /// Alias for `with_retry_options(...)`.
    pub fn with_retry(self, retry_options: RetryOptions) -> Self {
        self.with_retry_options(retry_options)
    }

    fn supported_models_vec(&self) -> Vec<String> {
        let mut models = Vec::new();
        if !self.config.common_params.model.trim().is_empty() {
            models.push(self.config.common_params.model.clone());
        }
        if let Some(model) = self
            .config
            .default_rerank_model
            .as_ref()
            .filter(|model| !model.trim().is_empty())
            && !models.iter().any(|value| value == model)
        {
            models.push(model.clone());
        }
        models
    }

    fn chat_spec(&self) -> Arc<dyn ProviderSpec> {
        Arc::new(BedrockChatStandard::new().create_spec("bedrock"))
    }

    fn rerank_spec(&self) -> Arc<dyn ProviderSpec> {
        Arc::new(BedrockRerankStandard::new().create_spec("bedrock"))
    }

    fn api_key(&self) -> Option<String> {
        self.config
            .api_key
            .as_ref()
            .map(|value| value.expose_secret().trim().to_string())
            .filter(|value| !value.is_empty())
    }

    fn build_chat_context(&self) -> ProviderContext {
        ProviderContext::new(
            "bedrock",
            self.config.runtime_base_url.clone(),
            self.api_key(),
            self.config.http_config.headers.clone(),
        )
    }

    fn build_rerank_context(&self) -> ProviderContext {
        ProviderContext::new(
            "bedrock",
            self.config.agent_runtime_base_url.clone(),
            self.api_key(),
            self.config.http_config.headers.clone(),
        )
    }

    /// Get the runtime base URL used for chat execution.
    pub fn runtime_base_url(&self) -> &str {
        &self.config.runtime_base_url
    }

    /// Get the agent runtime base URL used for rerank execution.
    pub fn agent_runtime_base_url(&self) -> &str {
        &self.config.agent_runtime_base_url
    }

    /// Get the normalized provider context used by chat execution helpers.
    pub fn chat_provider_context(&self) -> ProviderContext {
        self.build_chat_context()
    }

    /// Get the normalized provider context used by rerank execution helpers.
    pub fn rerank_provider_context(&self) -> ProviderContext {
        self.build_rerank_context()
    }

    /// Get the underlying HTTP client.
    pub fn http_client(&self) -> reqwest::Client {
        self.http_client.clone()
    }

    /// Get installed retry options.
    pub fn retry_options(&self) -> Option<RetryOptions> {
        self.retry_options.clone()
    }

    /// Get installed HTTP interceptors.
    pub fn http_interceptors(&self) -> Vec<Arc<dyn HttpInterceptor>> {
        self.config.http_interceptors.clone()
    }

    /// Get the installed custom HTTP transport.
    pub fn http_transport(&self) -> Option<Arc<dyn HttpTransport>> {
        self.config.http_transport.clone()
    }

    /// Set retry options.
    pub fn set_retry_options(&mut self, options: Option<RetryOptions>) {
        self.retry_options = options;
    }

    fn prepare_chat_request(
        &self,
        mut request: ChatRequest,
        stream: bool,
    ) -> Result<ChatRequest, LlmError> {
        request = crate::utils::chat_request::normalize_chat_request(
            request,
            crate::utils::chat_request::ChatRequestDefaults::new(&self.config.common_params),
            stream,
        );
        if request.common_params.model.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Bedrock chat request requires a non-empty model id".to_string(),
            ));
        }
        Ok(request)
    }

    fn prepare_rerank_request(
        &self,
        mut request: RerankRequest,
    ) -> Result<RerankRequest, LlmError> {
        if request.model.trim().is_empty() {
            if let Some(model) = self
                .config
                .default_rerank_model
                .as_ref()
                .filter(|model| !model.trim().is_empty())
            {
                request.model = model.clone();
            } else {
                request.model = self.config.common_params.model.clone();
            }
        }
        if request.model.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Bedrock rerank request requires a non-empty model id".to_string(),
            ));
        }

        let mut bedrock_options = request
            .provider_options_map
            .get("bedrock")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        let map = bedrock_options.as_object_mut().ok_or_else(|| {
            LlmError::InvalidParameter(
                "providerOptions.bedrock must be a JSON object for Amazon Bedrock".to_string(),
            )
        })?;
        if !map.contains_key("region") {
            map.insert(
                "region".to_string(),
                serde_json::Value::String(self.config.region.clone()),
            );
        }
        request
            .provider_options_map
            .insert("bedrock", bedrock_options);

        Ok(request)
    }

    async fn chat_request_via_spec(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        let request = self.prepare_chat_request(request, false)?;
        let ctx = self.build_chat_context();
        let spec = self.chat_spec();
        let bundle = spec.choose_chat_transformers(&request, &ctx);

        let mut builder = ChatExecutorBuilder::new("bedrock", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_transformer_bundle(bundle)
            .with_stream_disable_compression(self.config.http_config.stream_disable_compression)
            .with_interceptors(self.config.http_interceptors.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            builder = builder.with_transport(transport);
        }
        if let Some(retry_options) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry_options);
        }

        let exec = builder.build();
        ChatExecutor::execute(&*exec, request).await
    }

    async fn chat_stream_request_via_spec(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStream, LlmError> {
        let request = self.prepare_chat_request(request, true)?;
        let ctx = self.build_chat_context();
        let spec = self.chat_spec();
        let bundle = spec.choose_chat_transformers(&request, &ctx);

        let mut builder = ChatExecutorBuilder::new("bedrock", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_transformer_bundle(bundle)
            .with_stream_disable_compression(self.config.http_config.stream_disable_compression)
            .with_interceptors(self.config.http_interceptors.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            builder = builder.with_transport(transport);
        }
        if let Some(retry_options) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry_options);
        }

        let exec = builder.build();
        ChatExecutor::execute_stream(&*exec, request).await
    }

    #[cfg(test)]
    pub(crate) fn _debug_runtime_base_url(&self) -> &str {
        &self.config.runtime_base_url
    }

    #[cfg(test)]
    pub(crate) fn _debug_agent_runtime_base_url(&self) -> &str {
        &self.config.agent_runtime_base_url
    }
}

#[async_trait]
impl ChatCapability for BedrockClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(self.config.common_params.clone());
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
            .common_params(self.config.common_params.clone())
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
impl RerankCapability for BedrockClient {
    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, LlmError> {
        let request = self.prepare_rerank_request(request)?;

        let mut builder = RerankExecutorBuilder::new("bedrock", self.http_client.clone())
            .with_spec(self.rerank_spec())
            .with_context(self.build_rerank_context())
            .with_interceptors(self.config.http_interceptors.clone());

        if let Some(retry_options) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry_options);
        }
        if let Some(transport) = self.config.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

        let exec = builder.build_for_request(&request);
        RerankExecutor::execute(&*exec, request).await
    }

    fn supported_models(&self) -> Vec<String> {
        self.supported_models_vec()
    }
}

impl ModelMetadata for BedrockClient {
    fn provider_id(&self) -> &str {
        "bedrock"
    }

    fn model_id(&self) -> &str {
        if !self.config.common_params.model.trim().is_empty() {
            &self.config.common_params.model
        } else {
            self.config.default_rerank_model.as_deref().unwrap_or("")
        }
    }
}

impl LlmClient for BedrockClient {
    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed("bedrock")
    }

    fn supported_models(&self) -> Vec<String> {
        self.supported_models_vec()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
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

    fn as_rerank_capability(&self) -> Option<&dyn RerankCapability> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::bedrock::BedrockConfig;
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
    fn bedrock_client_exposes_split_runtime_context_and_helpers() {
        let transport = Arc::new(NoopTransport);
        let interceptor = Arc::new(NoopInterceptor);
        let config = BedrockConfig::new()
            .with_runtime_base_url("https://runtime.example.com")
            .with_agent_runtime_base_url("https://agent.example.com")
            .with_api_key("test-key")
            .with_model("anthropic.claude-3-5-sonnet")
            .with_http_transport(transport.clone())
            .with_http_interceptors(vec![interceptor]);
        let mut client = BedrockClient::from_config(config).expect("client");

        client.set_retry_options(Some(RetryOptions::backoff()));

        let chat_ctx = client.chat_provider_context();
        let rerank_ctx = client.rerank_provider_context();
        assert_eq!(chat_ctx.base_url, "https://runtime.example.com");
        assert_eq!(rerank_ctx.base_url, "https://agent.example.com");
        assert_eq!(chat_ctx.api_key.as_deref(), Some("test-key"));
        assert_eq!(rerank_ctx.api_key.as_deref(), Some("test-key"));
        assert_eq!(client.runtime_base_url(), "https://runtime.example.com");
        assert_eq!(client.agent_runtime_base_url(), "https://agent.example.com");
        assert!(client.retry_options().is_some());
        assert!(client.http_transport().is_some());
        assert_eq!(client.http_interceptors().len(), 1);
        let _http_client = client.http_client();
    }

    #[test]
    fn prepare_chat_request_for_stream_sets_stream_and_fills_default_model() {
        let config = BedrockConfig::new()
            .with_model("anthropic.claude-3-5-sonnet")
            .with_http_config(crate::types::HttpConfig::default());
        let client = BedrockClient::from_config(config).expect("client");

        let request = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .build();

        let prepared = client
            .prepare_chat_request(request, true)
            .expect("prepare stream request");

        assert!(prepared.stream);
        assert_eq!(prepared.common_params.model, "anthropic.claude-3-5-sonnet");
    }

    #[test]
    fn prepare_chat_request_for_non_stream_clears_stream_and_preserves_explicit_model() {
        let config = BedrockConfig::new()
            .with_model("anthropic.claude-3-5-sonnet")
            .with_http_config(crate::types::HttpConfig::default());
        let client = BedrockClient::from_config(config).expect("client");

        let request = ChatRequest::builder()
            .model("amazon.nova-pro-v1:0")
            .messages(vec![ChatMessage::user("hi").build()])
            .stream(true)
            .build();

        let prepared = client
            .prepare_chat_request(request, false)
            .expect("prepare non-stream request");

        assert!(!prepared.stream);
        assert_eq!(prepared.common_params.model, "amazon.nova-pro-v1:0");
    }

    #[test]
    fn prepare_chat_request_merges_missing_common_params_from_config() {
        let mut config = BedrockConfig::new()
            .with_model("anthropic.claude-3-5-sonnet")
            .with_http_config(crate::types::HttpConfig::default());
        config.common_params.temperature = Some(0.2);
        config.common_params.max_tokens = Some(256);
        config.common_params.top_p = Some(0.9);
        let client = BedrockClient::from_config(config).expect("client");

        let request = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .common_params(crate::types::CommonParams {
                temperature: Some(0.7),
                ..Default::default()
            })
            .build();

        let prepared = client
            .prepare_chat_request(request, true)
            .expect("prepare stream request");

        assert!(prepared.stream);
        assert_eq!(prepared.common_params.model, "anthropic.claude-3-5-sonnet");
        assert_eq!(prepared.common_params.temperature, Some(0.7));
        assert_eq!(prepared.common_params.max_tokens, Some(256));
        assert_eq!(prepared.common_params.top_p, Some(0.9));
    }
}
