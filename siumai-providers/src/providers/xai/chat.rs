//! `xAI` Chat Capability Implementation
//!
//! Implements the `ChatCapability` trait for `xAI`.

use async_trait::async_trait;

use crate::core::ProviderSpec;
use crate::error::LlmError;
use crate::execution::executors::chat::{ChatExecutor, ChatExecutorBuilder};
use crate::streaming::ChatStream;
use crate::traits::ChatCapability;
use crate::types::*;

use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use secrecy::SecretString;
use std::sync::Arc;

/// `xAI` Chat Capability Implementation
#[derive(Clone)]
pub struct XaiChatCapability {
    pub api_key: SecretString,
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

    fn build_context(&self) -> crate::core::ProviderContext {
        use secrecy::ExposeSecret;
        crate::core::ProviderContext::new(
            "xai",
            self.base_url.clone(),
            Some(self.api_key.expose_secret().to_string()),
            self.http_config.headers.clone(),
        )
    }

    fn build_chat_executor(
        &self,
        request: &ChatRequest,
    ) -> Arc<crate::execution::executors::chat::HttpChatExecutor> {
        let ctx = self.build_context();
        let spec = Arc::new(crate::providers::xai::spec::XaiSpec);
        let bundle = spec.choose_chat_transformers(request, &ctx);
        let before_send_hook = spec.chat_before_send(request, &ctx);

        let mut builder = ChatExecutorBuilder::new("xai", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_transformer_bundle(bundle)
            .with_stream_disable_compression(self.http_config.stream_disable_compression)
            .with_interceptors(self.interceptors.clone())
            .with_middlewares(self.middlewares.clone());

        if let Some(hook) = before_send_hook {
            builder = builder.with_before_send(hook);
        }

        builder.build()
    }
}

#[async_trait]
impl ChatCapability for XaiChatCapability {
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
        let request = builder.build();
        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute(&*exec, request).await
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
        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute_stream(&*exec, request).await
    }
}

impl XaiChatCapability {
    /// Execute chat with a fully-formed ChatRequest (used for advanced provider params flows)
    pub async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute(&*exec, request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_chat_executor_wires_before_send_for_custom_options() {
        let http_config = HttpConfig::builder().build();
        let cap = XaiChatCapability::new(
            secrecy::SecretString::from("test-key".to_string()),
            "https://api.x.ai/v1".to_string(),
            reqwest::Client::new(),
            http_config,
            CommonParams {
                model: crate::providers::xai::models::popular::LATEST.to_string(),
                ..Default::default()
            },
        );

        let mut custom = std::collections::HashMap::new();
        custom.insert("my_custom".to_string(), serde_json::json!(true));
        let req = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_common_params(cap.common_params.clone())
            .with_provider_options(crate::types::ProviderOptions::Custom {
                provider_id: "xai".to_string(),
                options: custom,
            });

        let exec = cap.build_chat_executor(&req);
        assert!(exec.policy.before_send.is_some());
    }
}
