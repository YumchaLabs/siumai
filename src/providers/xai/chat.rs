//! `xAI` Chat Capability Implementation
//!
//! Implements the `ChatCapability` trait for `xAI`.

use async_trait::async_trait;
// use std::time::Instant; // replaced by executor path

use crate::error::LlmError;
use crate::stream::ChatStream;
use crate::traits::ChatCapability;
use crate::types::*;

// use super::types::*;
use super::utils::*;
use crate::middleware::language_model::LanguageModelMiddleware;
use crate::utils::http_interceptor::HttpInterceptor;
use std::sync::Arc;

/// `xAI` Chat Capability Implementation
#[derive(Clone)]
pub struct XaiChatCapability {
    pub api_key: String,
    pub base_url: String,
    pub http_client: reqwest::Client,
    pub http_config: HttpConfig,
    pub common_params: CommonParams,
    /// Optional HTTP interceptors for chat requests
    pub interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional model-level middlewares
    pub middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
}

impl XaiChatCapability {
    /// Create a new `xAI` chat capability instance
    pub const fn new(
        api_key: String,
        base_url: String,
        http_client: reqwest::Client,
        http_config: HttpConfig,
        common_params: CommonParams,
    ) -> Self {
        Self {
            api_key,
            base_url,
            http_client,
            http_config,
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
impl ChatCapability for XaiChatCapability {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        // Create a ChatRequest from messages and tools
        let request = ChatRequest {
            messages,
            tools,
            common_params: self.common_params.clone(),
            provider_params: None,
            http_config: None,
            web_search: None,
            stream: false,
            telemetry: None,
        };
        use crate::executors::chat::{ChatExecutor, HttpChatExecutor};
        let http = self.http_client.clone();
        let base = self.base_url.clone();
        let api_key = self.api_key.clone();
        let custom_headers = self.http_config.headers.clone();
        let req_tx = super::transformers::XaiRequestTransformer;
        let resp_tx = super::transformers::XaiResponseTransformer;
        let api_key_for_headers = api_key.clone();
        let custom_headers_for_headers = custom_headers.clone();
        let headers_builder = move || {
            let api_key = api_key_for_headers.clone();
            let custom_headers = custom_headers_for_headers.clone();
            Box::pin(async move {
                build_headers(&api_key, &custom_headers)
            }) as std::pin::Pin<Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>> + Send>>
        };
        let exec = HttpChatExecutor {
            provider_id: "xai".to_string(),
            http_client: http,
            request_transformer: std::sync::Arc::new(req_tx),
            response_transformer: std::sync::Arc::new(resp_tx),
            stream_transformer: None,
            json_stream_converter: None,
            stream_disable_compression: self.http_config.stream_disable_compression,
            interceptors: self.interceptors.clone(),
            middlewares: self.middlewares.clone(),
            build_url: Box::new(move |_stream| format!("{}/chat/completions", base)),
            build_headers: std::sync::Arc::new(headers_builder),
            before_send: None,
        };
        exec.execute(request).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        // Create a ChatRequest from messages and tools
        let request = ChatRequest {
            messages,
            tools,
            common_params: self.common_params.clone(),
            provider_params: None,
            http_config: None,
            web_search: None,
            stream: true,
            telemetry: None,
        };

        use crate::executors::chat::{ChatExecutor, HttpChatExecutor};
        let http = self.http_client.clone();
        let base = self.base_url.clone();
        let api_key = self.api_key.clone();
        let custom_headers = self.http_config.headers.clone();
        let req_tx = super::transformers::XaiRequestTransformer;
        let resp_tx = super::transformers::XaiResponseTransformer;
        let inner = super::streaming::XaiEventConverter::new();
        let stream_tx = super::transformers::XaiStreamChunkTransformer {
            provider_id: "xai".to_string(),
            inner,
        };
        let api_key_for_headers = api_key.clone();
        let custom_headers_for_headers = custom_headers.clone();
        let headers_builder = move || {
            let api_key = api_key_for_headers.clone();
            let custom_headers = custom_headers_for_headers.clone();
            Box::pin(async move {
                build_headers(&api_key, &custom_headers)
            }) as std::pin::Pin<Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>> + Send>>
        };
        let exec = HttpChatExecutor {
            provider_id: "xai".to_string(),
            http_client: http,
            request_transformer: std::sync::Arc::new(req_tx),
            response_transformer: std::sync::Arc::new(resp_tx),
            stream_transformer: Some(std::sync::Arc::new(stream_tx)),
            json_stream_converter: None,
            stream_disable_compression: self.http_config.stream_disable_compression,
            interceptors: self.interceptors.clone(),
            middlewares: self.middlewares.clone(),
            build_url: Box::new(move |_stream| format!("{}/chat/completions", base)),
            build_headers: std::sync::Arc::new(headers_builder),
            before_send: None,
        };
        exec.execute_stream(request).await
    }
}

impl XaiChatCapability {
    /// Execute chat with a fully-formed ChatRequest (used for advanced provider params flows)
    pub async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        use crate::executors::chat::{ChatExecutor, HttpChatExecutor};
        let http = self.http_client.clone();
        let base = self.base_url.clone();
        let api_key = self.api_key.clone();
        let custom_headers = self.http_config.headers.clone();
        let req_tx = super::transformers::XaiRequestTransformer;
        let resp_tx = super::transformers::XaiResponseTransformer;
        let api_key_clone = api_key.clone();
        let custom_headers_clone = custom_headers.clone();
        let headers_builder = move || {
            let api_key = api_key_clone.clone();
            let custom_headers = custom_headers_clone.clone();
            Box::pin(async move {
                build_headers(&api_key, &custom_headers)
            }) as std::pin::Pin<Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>> + Send>>
        };
        let exec = HttpChatExecutor {
            provider_id: "xai".to_string(),
            http_client: http,
            request_transformer: std::sync::Arc::new(req_tx),
            response_transformer: std::sync::Arc::new(resp_tx),
            stream_transformer: None,
            json_stream_converter: None,
            stream_disable_compression: self.http_config.stream_disable_compression,
            interceptors: self.interceptors.clone(),
            middlewares: self.middlewares.clone(),
            build_url: Box::new(move |_stream| format!("{}/chat/completions", base)),
            build_headers: std::sync::Arc::new(headers_builder),
            before_send: None,
        };
        exec.execute(request).await
    }
}
