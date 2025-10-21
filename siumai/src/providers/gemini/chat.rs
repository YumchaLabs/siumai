//! Gemini Chat Capability Implementation
//!
//! This module implements the chat functionality for Google Gemini API.

use async_trait::async_trait;
use reqwest::Client as HttpClient;
// use std::time::Instant; // removed after executors migration

use crate::error::LlmError;
use crate::types::ChatRequest;
// use crate::transformers::request::RequestTransformer;
// use crate::transformers::response::ResponseTransformer;
use crate::executors::chat::{ChatExecutor, HttpChatExecutor};
use crate::middleware::language_model::LanguageModelMiddleware;
use crate::provider_core::ProviderSpec;
use crate::streaming::ChatStream;
use crate::traits::ChatCapability;
use crate::types::{ChatMessage, Tool};
use crate::utils::http_interceptor::HttpInterceptor;
use std::sync::Arc;

use super::types::{GeminiConfig, GenerateContentRequest};
use crate::ChatResponse;

/// Gemini chat capability implementation
#[derive(Clone)]
pub struct GeminiChatCapability {
    config: GeminiConfig,
    http_client: HttpClient,
    /// Optional HTTP interceptors for chat requests
    interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional model-level middlewares for chat requests
    middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
}

impl GeminiChatCapability {
    /// Create a new Gemini chat capability
    pub fn new(
        config: GeminiConfig,
        http_client: HttpClient,
        interceptors: Vec<Arc<dyn HttpInterceptor>>,
    ) -> Self {
        Self {
            config,
            http_client,
            interceptors,
            middlewares: Vec::new(),
        }
    }

    /// Set model-level middlewares
    pub fn with_middlewares(mut self, mws: Vec<Arc<dyn LanguageModelMiddleware>>) -> Self {
        self.middlewares = mws;
        self
    }

    /// Build the request body for Gemini API
    pub fn build_request_body(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<GenerateContentRequest, LlmError> {
        super::convert::build_request_body(&self.config, messages, tools)
    }

    // Removed legacy direct convert/make_request; Executors + Transformers handle mapping/HTTP
}

#[async_trait]
impl ChatCapability for GeminiChatCapability {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        let req = ChatRequest {
            messages,
            tools,
            common_params: crate::types::CommonParams {
                model: self.config.model.clone(),
                ..Default::default()
            },
            ..Default::default()
        };

        let http = self.http_client.clone();
        let spec = crate::providers::gemini::spec::GeminiSpec;
        use secrecy::ExposeSecret;
        let ctx = crate::provider_core::ProviderContext::new(
            "gemini",
            self.config.base_url.clone(),
            Some(self.config.api_key.expose_secret().to_string()),
            self.config.http_config.headers.clone(),
        );
        let req_tx = super::transformers::GeminiRequestTransformer {
            config: self.config.clone(),
        };
        let resp_tx = super::transformers::GeminiResponseTransformer {
            config: self.config.clone(),
        };
        let http0 = http.clone();
        let ctx_for_headers = ctx.clone();
        let tp = self.config.token_provider.clone();
        let headers_builder = move || {
            let mut ctx = ctx_for_headers.clone();
            let tp = tp.clone();
            Box::pin(async move {
                if let Some(tp) = tp {
                    if let Ok(tok) = tp.token().await {
                        ctx.http_extra_headers
                            .insert("Authorization".to_string(), format!("Bearer {tok}"));
                    }
                }
                spec.build_headers(&ctx)
            })
                as std::pin::Pin<
                    Box<
                        dyn std::future::Future<
                                Output = Result<reqwest::header::HeaderMap, LlmError>,
                            > + Send,
                    >,
                >
        };
        let ctx_for_url = ctx.clone();
        let build_url = move |stream: bool, r: &ChatRequest| spec.chat_url(stream, r, &ctx_for_url);
        let exec = HttpChatExecutor {
            provider_id: "gemini".to_string(),
            http_client: http0,
            request_transformer: Arc::new(req_tx),
            response_transformer: Arc::new(resp_tx),
            stream_transformer: None,
            json_stream_converter: None,
            stream_disable_compression: self.config.http_config.stream_disable_compression,
            interceptors: self.interceptors.clone(),
            middlewares: self.middlewares.clone(),
            build_url: Box::new(build_url),
            build_headers: std::sync::Arc::new(headers_builder),
            before_send: None,
        };
        exec.execute(req).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        let req = ChatRequest {
            messages,
            tools,
            common_params: crate::types::CommonParams {
                model: self.config.model.clone(),
                ..Default::default()
            },
            stream: true,
            ..Default::default()
        };

        let http = self.http_client.clone();
        let spec = crate::providers::gemini::spec::GeminiSpec;
        use secrecy::ExposeSecret;
        let ctx = crate::provider_core::ProviderContext::new(
            "gemini",
            self.config.base_url.clone(),
            Some(self.config.api_key.expose_secret().to_string()),
            self.config.http_config.headers.clone(),
        );
        let req_tx = super::transformers::GeminiRequestTransformer {
            config: self.config.clone(),
        };
        let resp_tx = super::transformers::GeminiResponseTransformer {
            config: self.config.clone(),
        };
        let converter = super::streaming::GeminiEventConverter::new(self.config.clone());
        let stream_tx = super::transformers::GeminiStreamChunkTransformer {
            provider_id: "gemini".to_string(),
            inner: converter,
        };
        let http0 = http.clone();
        let ctx_for_headers = ctx.clone();
        let tp = self.config.token_provider.clone();
        let headers_builder = move || {
            let mut ctx = ctx_for_headers.clone();
            let tp = tp.clone();
            Box::pin(async move {
                if let Some(tp) = tp {
                    if let Ok(tok) = tp.token().await {
                        ctx.http_extra_headers
                            .insert("Authorization".to_string(), format!("Bearer {tok}"));
                    }
                }
                spec.build_headers(&ctx)
            })
                as std::pin::Pin<
                    Box<
                        dyn std::future::Future<
                                Output = Result<reqwest::header::HeaderMap, LlmError>,
                            > + Send,
                    >,
                >
        };
        let ctx_for_url = ctx.clone();
        let build_url = move |stream: bool, r: &ChatRequest| spec.chat_url(stream, r, &ctx_for_url);
        let exec = HttpChatExecutor {
            provider_id: "gemini".to_string(),
            http_client: http0,
            request_transformer: Arc::new(req_tx),
            response_transformer: Arc::new(resp_tx),
            stream_transformer: Some(Arc::new(stream_tx)),
            json_stream_converter: None,
            stream_disable_compression: self.config.http_config.stream_disable_compression,
            interceptors: self.interceptors.clone(),
            middlewares: self.middlewares.clone(),
            build_url: Box::new(build_url),
            build_headers: std::sync::Arc::new(headers_builder),
            before_send: None,
        };
        exec.execute_stream(req).await
    }
}
