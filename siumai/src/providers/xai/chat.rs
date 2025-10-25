//! `xAI` Chat Capability Implementation
//!
//! Implements the `ChatCapability` trait for `xAI`.

use async_trait::async_trait;

use crate::core::ProviderSpec;
use crate::error::LlmError;
use crate::streaming::ChatStream;
use crate::traits::ChatCapability;
use crate::types::*;

// use super::types::*;
use super::utils::*;
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
            ..Default::default()
        };
        use crate::execution::executors::chat::{ChatExecutor, HttpChatExecutor};
        use secrecy::ExposeSecret;
        let ctx = crate::core::ProviderContext::new(
            "xai",
            self.base_url.clone(),
            Some(self.api_key.expose_secret().to_string()),
            self.http_config.headers.clone(),
        );
        let spec = Arc::new(crate::providers::xai::spec::XaiSpec);
        let bundle = spec.choose_chat_transformers(&request, &ctx);
        let http = self.http_client.clone();

        let exec = HttpChatExecutor {
            provider_id: "xai".to_string(),
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
            retry_options: None,
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

        use crate::execution::executors::chat::{ChatExecutor, HttpChatExecutor};
        use secrecy::ExposeSecret;
        let ctx = crate::core::ProviderContext::new(
            "xai",
            self.base_url.clone(),
            Some(self.api_key.expose_secret().to_string()),
            self.http_config.headers.clone(),
        );
        let spec = Arc::new(crate::providers::xai::spec::XaiSpec);
        let bundle = spec.choose_chat_transformers(&request, &ctx);
        let http = self.http_client.clone();

        let exec = HttpChatExecutor {
            provider_id: "xai".to_string(),
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
            retry_options: None,
        };
        exec.execute_stream(request).await
    }
}

impl XaiChatCapability {
    /// Execute chat with a fully-formed ChatRequest (used for advanced provider params flows)
    pub async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        use crate::execution::executors::chat::{ChatExecutor, HttpChatExecutor};
        use secrecy::ExposeSecret;

        // Create a simple spec for this legacy method
        struct SimpleXaiSpec {
            base_url: String,
            api_key: String,
            custom_headers: std::collections::HashMap<String, String>,
        }

        impl crate::core::ProviderSpec for SimpleXaiSpec {
            fn id(&self) -> &'static str {
                "xai"
            }
            fn capabilities(&self) -> crate::traits::ProviderCapabilities {
                crate::traits::ProviderCapabilities::new()
            }
            fn build_headers(
                &self,
                _ctx: &crate::core::ProviderContext,
            ) -> Result<reqwest::header::HeaderMap, LlmError> {
                build_headers(&self.api_key, &self.custom_headers)
            }
            fn chat_url(
                &self,
                _stream: bool,
                _req: &ChatRequest,
                _ctx: &crate::core::ProviderContext,
            ) -> String {
                format!("{}/chat/completions", self.base_url)
            }
            fn choose_chat_transformers(
                &self,
                _req: &ChatRequest,
                _ctx: &crate::core::ProviderContext,
            ) -> crate::core::ChatTransformers {
                crate::core::ChatTransformers {
                    request: std::sync::Arc::new(super::transformers::XaiRequestTransformer),
                    response: std::sync::Arc::new(super::transformers::XaiResponseTransformer),
                    stream: None,
                    json: None,
                }
            }
        }

        let spec = Arc::new(SimpleXaiSpec {
            base_url: self.base_url.clone(),
            api_key: self.api_key.expose_secret().to_string(),
            custom_headers: self.http_config.headers.clone(),
        });

        let ctx = crate::core::ProviderContext::new(
            "xai",
            self.base_url.clone(),
            Some(self.api_key.expose_secret().to_string()),
            self.http_config.headers.clone(),
        );

        let bundle = spec.choose_chat_transformers(&request, &ctx);
        let http = self.http_client.clone();

        let exec = HttpChatExecutor {
            provider_id: "xai".to_string(),
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
            retry_options: None,
        };
        exec.execute(request).await
    }
}
