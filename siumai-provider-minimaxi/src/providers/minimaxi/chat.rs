//! MiniMaxi Chat Capability Implementation
//!
//! Implements the `ChatCapability` trait for MiniMaxi.

use async_trait::async_trait;

use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::retry_api::RetryOptions;
use crate::streaming::ChatStream;
use crate::traits::ChatCapability;
use crate::types::*;
use std::sync::Arc;

/// MiniMaxi Chat Capability Implementation
#[derive(Clone)]
pub struct MinimaxiChatCapability {
    pub api_key: String,
    pub base_url: String,
    pub http_client: reqwest::Client,
    pub common_params: crate::types::CommonParams,
    pub http_config: HttpConfig,
    pub retry_options: Option<RetryOptions>,
    /// Optional HTTP interceptors for chat requests
    pub interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional model-level middlewares for chat requests
    pub middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
}

impl MinimaxiChatCapability {
    /// Create a new MiniMaxi chat capability instance
    pub fn new(
        api_key: String,
        base_url: String,
        http_client: reqwest::Client,
        common_params: crate::types::CommonParams,
        http_config: HttpConfig,
    ) -> Self {
        Self {
            api_key,
            base_url,
            http_client,
            common_params,
            http_config,
            retry_options: None,
            interceptors: Vec::new(),
            middlewares: Vec::new(),
        }
    }

    /// Set model-level middlewares for chat requests
    pub fn with_middlewares(mut self, mws: Vec<Arc<dyn LanguageModelMiddleware>>) -> Self {
        self.middlewares = mws;
        self
    }

    /// Set unified retry options for chat requests.
    pub fn with_retry_options(mut self, retry: Option<RetryOptions>) -> Self {
        self.retry_options = retry;
        self
    }

    /// Set HTTP interceptors for chat requests.
    pub fn with_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.interceptors = interceptors;
        self
    }

    fn build_context(&self) -> crate::core::ProviderContext {
        crate::core::ProviderContext::new(
            "minimaxi",
            self.base_url.clone(),
            Some(self.api_key.clone()),
            self.http_config.headers.clone(),
        )
    }

    fn build_chat_executor(
        &self,
        request: &ChatRequest,
    ) -> Arc<crate::execution::executors::chat::HttpChatExecutor> {
        use crate::core::ProviderSpec;
        use crate::execution::executors::chat::ChatExecutorBuilder;

        let ctx = self.build_context();
        let spec = Arc::new(crate::providers::minimaxi::spec::MinimaxiSpec::new());
        let bundle = spec.choose_chat_transformers(request, &ctx);
        let before_send_hook = spec.chat_before_send(request, &ctx);

        let mut builder = ChatExecutorBuilder::new("minimaxi", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_transformer_bundle(bundle)
            .with_stream_disable_compression(self.http_config.stream_disable_compression)
            .with_interceptors(self.interceptors.clone())
            .with_middlewares(self.middlewares.clone());

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        if let Some(hook) = before_send_hook {
            builder = builder.with_before_send(hook);
        }

        builder.build()
    }
}

#[async_trait]
impl ChatCapability for MinimaxiChatCapability {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        use crate::execution::executors::chat::ChatExecutor;

        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(self.common_params.clone());
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        let request = builder.build();

        let exec = self.build_chat_executor(&request);

        if let Some(opts) = &self.retry_options {
            let mut opts = opts.clone();
            if opts.provider.is_none() {
                opts.provider = Some(crate::types::ProviderType::MiniMaxi);
            }
            crate::retry_api::retry_with(
                || {
                    let rq = request.clone();
                    let ex = exec.clone();
                    async move { ChatExecutor::execute(&*ex, rq).await }
                },
                opts,
            )
            .await
        } else {
            ChatExecutor::execute(&*exec, request).await
        }
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        use crate::execution::executors::chat::ChatExecutor;

        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(self.common_params.clone())
            .stream(true);
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        let request = builder.build();

        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute_stream(&*exec, request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Default)]
    struct NoopInterceptor;

    impl crate::execution::http::interceptor::HttpInterceptor for NoopInterceptor {}

    #[test]
    fn build_chat_executor_wires_policy_from_capability() {
        let http_config = HttpConfig::builder()
            .stream_disable_compression(false)
            .build();

        let cap = MinimaxiChatCapability::new(
            "test-key".to_string(),
            crate::providers::minimaxi::config::MinimaxiConfig::DEFAULT_BASE_URL.to_string(),
            reqwest::Client::new(),
            CommonParams {
                model: crate::providers::minimaxi::config::MinimaxiConfig::DEFAULT_MODEL
                    .to_string(),
                ..Default::default()
            },
            http_config,
        )
        .with_interceptors(vec![Arc::new(NoopInterceptor)])
        .with_retry_options(Some(RetryOptions::backoff()));

        let req = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_common_params(cap.common_params.clone());

        let exec = cap.build_chat_executor(&req);
        assert_eq!(exec.policy.interceptors.len(), 1);
        assert!(exec.policy.retry_options.is_some());
        assert!(!exec.policy.stream_disable_compression);
        assert_eq!(exec.provider_context.provider_id, "minimaxi");
    }
}
