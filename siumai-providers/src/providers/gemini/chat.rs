//! Gemini Chat Capability Implementation
//!
//! This module implements the chat functionality for Google Gemini API.

use async_trait::async_trait;
use reqwest::Client as HttpClient;

use crate::error::LlmError;
use crate::execution::executors::chat::{ChatExecutor, ChatExecutorBuilder};
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

    async fn build_chat_executor(
        &self,
        request: &ChatRequest,
    ) -> Arc<crate::execution::executors::chat::HttpChatExecutor> {
        use crate::core::ProviderSpec;

        let ctx = super::context::build_context(&self.config).await;
        let spec = Arc::new(crate::providers::gemini::spec::GeminiSpecWithConfig::new(
            self.config.clone(),
        ));
        let bundle = spec.choose_chat_transformers(request, &ctx);
        let before_send_hook = spec.chat_before_send(request, &ctx);

        let mut builder = ChatExecutorBuilder::new("gemini", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_transformer_bundle(bundle)
            .with_stream_disable_compression(self.config.http_config.stream_disable_compression)
            .with_interceptors(self.interceptors.clone())
            .with_middlewares(self.middlewares.clone());

        if let Some(hook) = before_send_hook {
            builder = builder.with_before_send(hook);
        }

        builder.build()
    }
}

#[async_trait]
impl ChatCapability for GeminiChatCapability {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
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
        let exec = self.build_chat_executor(&req).await;
        ChatExecutor::execute(&*exec, req).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
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
        let exec = self.build_chat_executor(&req).await;
        ChatExecutor::execute_stream(&*exec, req).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::HttpConfig;

    #[tokio::test]
    async fn build_chat_executor_wires_before_send_for_custom_options() {
        let cfg = GeminiConfig::new("test-key")
            .with_model("gemini-1.5-flash".to_string())
            .with_http_config(HttpConfig::builder().build());
        let cap = GeminiChatCapability::new(cfg, reqwest::Client::new(), Vec::new());

        let mut custom = std::collections::HashMap::new();
        custom.insert("my_custom".to_string(), serde_json::json!("x"));
        let req = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_common_params(crate::types::CommonParams {
                model: "gemini-1.5-flash".to_string(),
                ..Default::default()
            })
            .with_provider_options(crate::types::ProviderOptions::Custom {
                provider_id: "gemini".to_string(),
                options: custom,
            });

        let exec = cap.build_chat_executor(&req).await;
        assert!(exec.policy.before_send.is_some());
    }
}
