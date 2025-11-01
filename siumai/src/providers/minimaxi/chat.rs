//! MiniMaxi Chat Capability Implementation
//!
//! Implements the `ChatCapability` trait for MiniMaxi.

use async_trait::async_trait;

use crate::core::ProviderSpec;
use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
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
    ) -> Self {
        Self {
            api_key,
            base_url,
            http_client,
            common_params,
            interceptors: Vec::new(),
            middlewares: Vec::new(),
        }
    }

    /// Set model-level middlewares for chat requests
    pub fn with_middlewares(mut self, mws: Vec<Arc<dyn LanguageModelMiddleware>>) -> Self {
        self.middlewares = mws;
        self
    }
}

#[async_trait]
impl ChatCapability for MinimaxiChatCapability {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        // Create a ChatRequest from messages and tools
        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(self.common_params.clone());
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        let request = builder.build();

        use crate::execution::executors::chat::{ChatExecutor, HttpChatExecutor};

        let ctx = crate::core::ProviderContext::new(
            "minimaxi",
            self.base_url.clone(),
            Some(self.api_key.clone()),
            std::collections::HashMap::new(),
        );

        let spec = Arc::new(crate::providers::minimaxi::spec::MinimaxiSpec::new());
        let bundle = spec.choose_chat_transformers(&request, &ctx);
        let http = self.http_client.clone();

        let exec = HttpChatExecutor {
            provider_id: "minimaxi".to_string(),
            http_client: http,
            request_transformer: bundle.request,
            response_transformer: bundle.response,
            stream_transformer: None,
            json_stream_converter: None,
            policy: crate::execution::ExecutionPolicy::new()
                .with_interceptors(self.interceptors.clone()),
            middlewares: self.middlewares.clone(),
            provider_spec: spec,
            provider_context: ctx,
        };

        exec.execute(request).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        // Create a ChatRequest from messages and tools
        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(self.common_params.clone())
            .stream(true);
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        let request = builder.build();

        use crate::execution::executors::chat::{ChatExecutor, HttpChatExecutor};

        let ctx = crate::core::ProviderContext::new(
            "minimaxi",
            self.base_url.clone(),
            Some(self.api_key.clone()),
            std::collections::HashMap::new(),
        );

        let spec = Arc::new(crate::providers::minimaxi::spec::MinimaxiSpec::new());
        let bundle = spec.choose_chat_transformers(&request, &ctx);
        let http = self.http_client.clone();

        let exec = HttpChatExecutor {
            provider_id: "minimaxi".to_string(),
            http_client: http,
            request_transformer: bundle.request,
            response_transformer: bundle.response,
            stream_transformer: bundle.stream,
            json_stream_converter: bundle.json,
            policy: crate::execution::ExecutionPolicy::new()
                .with_interceptors(self.interceptors.clone()),
            middlewares: self.middlewares.clone(),
            provider_spec: spec,
            provider_context: ctx,
        };

        exec.execute_stream(request).await
    }
}
