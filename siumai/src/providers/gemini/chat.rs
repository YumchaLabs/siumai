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
        let req = ChatRequest {
            messages,
            tools,
            common_params: crate::types::CommonParams {
                model: self.config.common_params.model.clone(),
                ..Default::default()
            },
            ..Default::default()
        };

        let http = self.http_client.clone();
        let spec = crate::providers::gemini::spec::GeminiSpec;
        use secrecy::ExposeSecret;
        let ctx = crate::core::ProviderContext::new(
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
        // For Gemini, we need to handle token_provider dynamically
        // So we create a wrapper spec that handles token injection
        struct GeminiSpecWrapper {
            spec: crate::providers::gemini::spec::GeminiSpec,
            #[allow(dead_code)]
            token_provider: Option<Arc<dyn crate::auth::TokenProvider>>,
        }

        impl crate::core::ProviderSpec for GeminiSpecWrapper {
            fn id(&self) -> &'static str {
                self.spec.id()
            }
            fn capabilities(&self) -> crate::traits::ProviderCapabilities {
                self.spec.capabilities()
            }
            fn build_headers(
                &self,
                ctx: &crate::core::ProviderContext,
            ) -> Result<reqwest::header::HeaderMap, LlmError> {
                // Note: This is now sync, so we can't await token_provider here
                // The token should be injected into ctx before calling this
                self.spec.build_headers(ctx)
            }
            fn chat_url(
                &self,
                stream: bool,
                req: &ChatRequest,
                ctx: &crate::core::ProviderContext,
            ) -> String {
                self.spec.chat_url(stream, req, ctx)
            }
            fn choose_chat_transformers(
                &self,
                req: &ChatRequest,
                ctx: &crate::core::ProviderContext,
            ) -> crate::core::ChatTransformers {
                self.spec.choose_chat_transformers(req, ctx)
            }
        }

        // Inject token into context if available
        let mut ctx_with_token = ctx.clone();
        if let Some(tp) = &self.config.token_provider
            && let Ok(tok) = tp.token().await
        {
            ctx_with_token
                .http_extra_headers
                .insert("Authorization".to_string(), format!("Bearer {tok}"));
        }

        let spec_wrapper = Arc::new(GeminiSpecWrapper {
            spec,
            token_provider: self.config.token_provider.clone(),
        });

        let http0 = http.clone();
        let exec = HttpChatExecutor {
            provider_id: "gemini".to_string(),
            http_client: http0,
            request_transformer: Arc::new(req_tx),
            response_transformer: Arc::new(resp_tx),
            stream_transformer: None,
            json_stream_converter: None,
            policy: crate::execution::ExecutionPolicy::new()
                .with_stream_disable_compression(self.config.http_config.stream_disable_compression)
                .with_interceptors(self.interceptors.clone()),
            middlewares: self.middlewares.clone(),
            provider_spec: spec_wrapper,
            provider_context: ctx_with_token,
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
                model: self.config.common_params.model.clone(),
                ..Default::default()
            },
            stream: true,
            ..Default::default()
        };

        let http = self.http_client.clone();
        let spec = crate::providers::gemini::spec::GeminiSpec;
        use secrecy::ExposeSecret;
        let ctx = crate::core::ProviderContext::new(
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

        // For Gemini, we need to handle token_provider dynamically
        struct GeminiStreamSpecWrapper {
            spec: crate::providers::gemini::spec::GeminiSpec,
            #[allow(dead_code)]
            token_provider: Option<Arc<dyn crate::auth::TokenProvider>>,
            config: crate::providers::gemini::types::GeminiConfig,
        }

        impl crate::core::ProviderSpec for GeminiStreamSpecWrapper {
            fn id(&self) -> &'static str {
                self.spec.id()
            }
            fn capabilities(&self) -> crate::traits::ProviderCapabilities {
                self.spec.capabilities()
            }
            fn build_headers(
                &self,
                ctx: &crate::core::ProviderContext,
            ) -> Result<reqwest::header::HeaderMap, LlmError> {
                self.spec.build_headers(ctx)
            }
            fn chat_url(
                &self,
                stream: bool,
                req: &ChatRequest,
                ctx: &crate::core::ProviderContext,
            ) -> String {
                self.spec.chat_url(stream, req, ctx)
            }
            fn choose_chat_transformers(
                &self,
                _req: &ChatRequest,
                _ctx: &crate::core::ProviderContext,
            ) -> crate::core::ChatTransformers {
                let converter = super::streaming::GeminiEventConverter::new(self.config.clone());
                let stream_tx = super::transformers::GeminiStreamChunkTransformer {
                    provider_id: "gemini".to_string(),
                    inner: converter,
                };
                crate::core::ChatTransformers {
                    request: Arc::new(super::transformers::GeminiRequestTransformer {
                        config: self.config.clone(),
                    }),
                    response: Arc::new(super::transformers::GeminiResponseTransformer {
                        config: self.config.clone(),
                    }),
                    stream: Some(Arc::new(stream_tx)),
                    json: None,
                }
            }
        }

        // Inject token into context if available
        let mut ctx_with_token = ctx.clone();
        if let Some(tp) = &self.config.token_provider
            && let Ok(tok) = tp.token().await
        {
            ctx_with_token
                .http_extra_headers
                .insert("Authorization".to_string(), format!("Bearer {tok}"));
        }

        let spec_wrapper = Arc::new(GeminiStreamSpecWrapper {
            spec,
            token_provider: self.config.token_provider.clone(),
            config: self.config.clone(),
        });

        let http0 = http.clone();
        let exec = HttpChatExecutor {
            provider_id: "gemini".to_string(),
            http_client: http0,
            request_transformer: Arc::new(req_tx),
            response_transformer: Arc::new(resp_tx),
            stream_transformer: Some(Arc::new(stream_tx)),
            json_stream_converter: None,
            policy: crate::execution::ExecutionPolicy::new()
                .with_stream_disable_compression(self.config.http_config.stream_disable_compression)
                .with_interceptors(self.interceptors.clone()),
            middlewares: self.middlewares.clone(),
            provider_spec: spec_wrapper,
            provider_context: ctx_with_token,
        };
        exec.execute_stream(req).await
    }
}
