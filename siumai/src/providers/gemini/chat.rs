//! Gemini Chat Capability Implementation
//!
//! This module implements the chat functionality for Google Gemini API.

use async_trait::async_trait;
use reqwest::Client as HttpClient;

use crate::error::LlmError;
use crate::execution::executors::chat::{ChatExecutor, HttpChatExecutor};
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::types::ChatRequest;

use crate::execution::http::interceptor::HttpInterceptor;
use crate::streaming::ChatStream;
use crate::traits::ChatCapability;
use crate::types::{ChatMessage, Tool};
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
}

#[async_trait]
impl ChatCapability for GeminiChatCapability {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        // Build ChatRequest using unified CommonParams
        let mut builder =
            ChatRequest::builder()
                .messages(messages)
                .common_params(crate::types::CommonParams {
                    model: self.config.common_params.model.clone(),
                    ..Default::default()
                });
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        let req = builder.build();

        // Build provider context and inject token if available
        use secrecy::ExposeSecret;
        let mut ctx = crate::core::ProviderContext::new(
            "gemini",
            self.config.base_url.clone(),
            Some(self.config.api_key.expose_secret().to_string()),
            self.config.http_config.headers.clone(),
        );
        if let Some(tp) = &self.config.token_provider
            && let Ok(tok) = tp.token().await
        {
            ctx.http_extra_headers
                .insert("Authorization".to_string(), format!("Bearer {tok}"));
        }

        // Use the unified ProviderSpec + std-gemini pipeline for non-streaming chat.
        use crate::core::ProviderSpec;
        let spec = Arc::new(crate::providers::gemini::spec::GeminiSpec);
        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let before_send_hook = spec.chat_before_send(&req, &ctx);

        let http0 = self.http_client.clone();
        let policy = {
            let base = crate::execution::ExecutionPolicy::new()
                .with_stream_disable_compression(self.config.http_config.stream_disable_compression)
                .with_interceptors(self.interceptors.clone());
            if let Some(hook) = before_send_hook {
                base.with_before_send(hook)
            } else {
                base
            }
        };

        let exec = HttpChatExecutor {
            provider_id: "gemini".to_string(),
            http_client: http0,
            request_transformer: bundle.request,
            response_transformer: bundle.response,
            stream_transformer: None,
            json_stream_converter: None,
            policy,
            middlewares: self.middlewares.clone(),
            provider_spec: spec,
            provider_context: ctx,
        };
        exec.execute(req).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        // Build ChatRequest with streaming enabled
        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(crate::types::CommonParams {
                model: self.config.common_params.model.clone(),
                ..Default::default()
            })
            .stream(true);
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        let req = builder.build();

        // Build provider context and inject token if available
        use secrecy::ExposeSecret;
        let mut ctx = crate::core::ProviderContext::new(
            "gemini",
            self.config.base_url.clone(),
            Some(self.config.api_key.expose_secret().to_string()),
            self.config.http_config.headers.clone(),
        );
        if let Some(tp) = &self.config.token_provider
            && let Ok(tok) = tp.token().await
        {
            ctx.http_extra_headers
                .insert("Authorization".to_string(), format!("Bearer {tok}"));
        }

        // Use the unified ProviderSpec + std-gemini pipeline for streaming,
        // so that streaming events follow the std-gemini ChatStreamEventCore
        // model and are bridged into ChatStreamEvent via core helpers.
        use crate::core::ProviderSpec;
        let spec = Arc::new(crate::providers::gemini::spec::GeminiSpec);
        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let before_send_hook = spec.chat_before_send(&req, &ctx);

        let http0 = self.http_client.clone();
        let policy = {
            let base = crate::execution::ExecutionPolicy::new()
                .with_stream_disable_compression(self.config.http_config.stream_disable_compression)
                .with_interceptors(self.interceptors.clone());
            if let Some(hook) = before_send_hook {
                base.with_before_send(hook)
            } else {
                base
            }
        };

        let exec = HttpChatExecutor {
            provider_id: "gemini".to_string(),
            http_client: http0,
            request_transformer: bundle.request,
            response_transformer: bundle.response,
            stream_transformer: bundle.stream,
            json_stream_converter: bundle.json,
            policy,
            middlewares: self.middlewares.clone(),
            provider_spec: spec,
            provider_context: ctx,
        };
        exec.execute_stream(req).await
    }
}
