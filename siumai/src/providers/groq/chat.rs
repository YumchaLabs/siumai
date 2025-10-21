//! `Groq` Chat Capability Implementation
//!
//! Implements the `ChatCapability` trait for `Groq`.

use async_trait::async_trait;
use secrecy::SecretString;
// use std::time::Instant; // replaced by executor path

use crate::error::LlmError;
use crate::middleware::language_model::LanguageModelMiddleware;
use crate::provider_core::ProviderSpec;
use crate::streaming::ChatStream;
use crate::traits::ChatCapability;
use crate::types::*;
use crate::utils::http_interceptor::HttpInterceptor;
use std::sync::Arc;

// use super::types::*;
// use super::utils::*;

/// `Groq` Chat Capability Implementation
#[derive(Clone)]
pub struct GroqChatCapability {
    pub api_key: SecretString,
    pub base_url: String,
    pub http_client: reqwest::Client,
    pub http_config: HttpConfig,
    pub common_params: CommonParams,
    /// Optional HTTP interceptors for chat requests
    pub interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional model-level middlewares for chat requests
    pub middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
}

impl GroqChatCapability {
    /// Create a new `Groq` chat capability instance
    pub fn new(
        api_key: SecretString,
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
impl ChatCapability for GroqChatCapability {
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
            ..Default::default()
        };
        use crate::executors::chat::{ChatExecutor, HttpChatExecutor};
        use secrecy::ExposeSecret;
        let ctx = crate::provider_core::ProviderContext::new(
            "groq",
            self.base_url.clone(),
            Some(self.api_key.expose_secret().to_string()),
            self.http_config.headers.clone(),
        );
        let spec = Arc::new(crate::providers::groq::spec::GroqSpec);
        let bundle = spec.choose_chat_transformers(&request, &ctx);
        let http = self.http_client.clone();

        let exec = HttpChatExecutor {
            provider_id: "groq".to_string(),
            http_client: http,
            request_transformer: bundle.request,
            response_transformer: bundle.response,
            stream_transformer: None,
            json_stream_converter: None,
            stream_disable_compression: self.http_config.stream_disable_compression,
            interceptors: self.interceptors.clone(),
            middlewares: self.middlewares.clone(),
            provider_spec: spec,
            provider_context: ctx,
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
            stream: true,
            ..Default::default()
        };

        use crate::executors::chat::{ChatExecutor, HttpChatExecutor};
        use secrecy::ExposeSecret;
        let ctx = crate::provider_core::ProviderContext::new(
            "groq",
            self.base_url.clone(),
            Some(self.api_key.expose_secret().to_string()),
            self.http_config.headers.clone(),
        );
        let spec = Arc::new(crate::providers::groq::spec::GroqSpec);
        let bundle = spec.choose_chat_transformers(&request, &ctx);
        let http = self.http_client.clone();

        let exec = HttpChatExecutor {
            provider_id: "groq".to_string(),
            http_client: http,
            request_transformer: bundle.request,
            response_transformer: bundle.response,
            stream_transformer: bundle.stream,
            json_stream_converter: bundle.json,
            stream_disable_compression: self.http_config.stream_disable_compression,
            interceptors: self.interceptors.clone(),
            middlewares: self.middlewares.clone(),
            provider_spec: spec,
            provider_context: ctx,
            before_send: None,
        };
        exec.execute_stream(request).await
    }
}

// Legacy direct methods removed; Chat goes through Executors consistently.
