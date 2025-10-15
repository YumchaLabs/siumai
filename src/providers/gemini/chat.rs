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
use crate::stream::ChatStream;
use crate::traits::ChatCapability;
use crate::types::{ChatMessage, Tool};
use std::sync::Arc;

use super::types::{GeminiConfig, GenerateContentRequest};
use crate::ChatResponse;

/// Gemini chat capability implementation
#[derive(Debug, Clone)]
pub struct GeminiChatCapability {
    config: GeminiConfig,
    http_client: HttpClient,
}

impl GeminiChatCapability {
    /// Create a new Gemini chat capability
    pub fn new(config: GeminiConfig, http_client: HttpClient) -> Self {
        Self {
            config,
            http_client,
        }
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
            provider_params: None,
            http_config: None,
            web_search: None,
            stream: false,
        };

        let http = self.http_client.clone();
        let base = self.config.base_url.clone();
        let model = self.config.model.clone();
        let api_key = self.config.api_key.clone();
        let req_tx = super::transformers::GeminiRequestTransformer {
            config: self.config.clone(),
        };
        let resp_tx = super::transformers::GeminiResponseTransformer {
            config: self.config.clone(),
        };
        let extra = self
            .config
            .http_config
            .clone()
            .and_then(|c| Some(c.headers))
            .unwrap_or_default();
        let headers_builder = move || {
            let headers = crate::utils::http_headers::ProviderHeaders::gemini(&api_key, &extra)?;
            Ok(headers)
        };
        let exec = HttpChatExecutor {
            provider_id: "gemini".to_string(),
            http_client: http,
            request_transformer: Arc::new(req_tx),
            response_transformer: Arc::new(resp_tx),
            stream_transformer: None,
            build_url: Box::new(move |_stream| {
                crate::utils::url::join_url(&base, &format!("models/{}:generateContent", model))
            }),
            build_headers: Box::new(headers_builder),
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
            provider_params: None,
            http_config: None,
            web_search: None,
            stream: true,
        };

        let http = self.http_client.clone();
        let base = self.config.base_url.clone();
        let model = self.config.model.clone();
        let api_key = self.config.api_key.clone();
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
        let extra = self
            .config
            .http_config
            .clone()
            .and_then(|c| Some(c.headers))
            .unwrap_or_default();
        let headers_builder = move || {
            let headers = crate::utils::http_headers::ProviderHeaders::gemini(&api_key, &extra)?;
            Ok(headers)
        };
        let exec = HttpChatExecutor {
            provider_id: "gemini".to_string(),
            http_client: http,
            request_transformer: Arc::new(req_tx),
            response_transformer: Arc::new(resp_tx),
            stream_transformer: Some(Arc::new(stream_tx)),
            build_url: Box::new(move |_stream| {
                crate::utils::url::join_url(
                    &base,
                    &format!("models/{}:streamGenerateContent?alt=sse", model),
                )
            }),
            build_headers: Box::new(headers_builder),
            before_send: None,
        };
        exec.execute_stream(req).await
    }
}
