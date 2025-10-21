//! Anthropic Messages API Standard
//!
//! This module implements the Anthropic Messages API format.

use crate::error::LlmError;
use crate::provider_core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::transformers::request::RequestTransformer;
use crate::transformers::response::ResponseTransformer;
use crate::transformers::stream::StreamChunkTransformer;
use crate::types::ChatRequest;
use std::sync::Arc;

/// Anthropic Chat API Standard
#[derive(Clone)]
pub struct AnthropicChatStandard {
    adapter: Option<Arc<dyn AnthropicChatAdapter>>,
}

impl AnthropicChatStandard {
    pub fn new() -> Self {
        Self { adapter: None }
    }

    pub fn with_adapter(adapter: Arc<dyn AnthropicChatAdapter>) -> Self {
        Self {
            adapter: Some(adapter),
        }
    }

    pub fn create_spec(&self, provider_id: &'static str) -> AnthropicChatSpec {
        AnthropicChatSpec {
            provider_id,
            adapter: self.adapter.clone(),
        }
    }

    pub fn create_transformers(&self, provider_id: &str) -> ChatTransformers {
        let request_tx = Arc::new(AnthropicChatRequestTransformer {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        });

        let response_tx = Arc::new(AnthropicChatResponseTransformer {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        });

        let stream_tx = Arc::new(AnthropicChatStreamTransformer {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        });

        ChatTransformers {
            request: request_tx,
            response: response_tx,
            stream: Some(stream_tx),
            json: None,
        }
    }
}

impl Default for AnthropicChatStandard {
    fn default() -> Self {
        Self::new()
    }
}

/// Adapter trait for provider-specific differences in Anthropic Messages API
pub trait AnthropicChatAdapter: Send + Sync {
    fn transform_request(
        &self,
        _req: &ChatRequest,
        _body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    fn transform_response(&self, _resp: &mut serde_json::Value) -> Result<(), LlmError> {
        Ok(())
    }

    fn transform_sse_event(&self, _event: &mut serde_json::Value) -> Result<(), LlmError> {
        Ok(())
    }

    fn messages_endpoint(&self) -> &str {
        "/v1/messages"
    }

    fn build_headers(
        &self,
        _api_key: &str,
        _base_headers: &mut reqwest::header::HeaderMap,
    ) -> Result<(), LlmError> {
        Ok(())
    }
}

pub struct AnthropicChatSpec {
    provider_id: &'static str,
    adapter: Option<Arc<dyn AnthropicChatAdapter>>,
}

impl ProviderSpec for AnthropicChatSpec {
    fn id(&self) -> &'static str {
        self.provider_id
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<reqwest::header::HeaderMap, LlmError> {
        use reqwest::header::HeaderMap;
        let mut headers = HeaderMap::new();

        // Standard Anthropic headers
        if let Some(api_key) = &ctx.api_key {
            headers.insert(
                "x-api-key",
                api_key
                    .parse()
                    .map_err(|e| LlmError::InvalidParameter(format!("Invalid API key: {}", e)))?,
            );
            headers.insert(
                "anthropic-version",
                "2023-06-01"
                    .parse()
                    .map_err(|e| LlmError::InvalidParameter(format!("Invalid version: {}", e)))?,
            );
        }

        // Add custom headers
        for (k, v) in &ctx.http_extra_headers {
            let header_name: reqwest::header::HeaderName = k.parse().map_err(|e| {
                LlmError::InvalidParameter(format!("Invalid header name '{}': {}", k, e))
            })?;
            let header_value: reqwest::header::HeaderValue = v.parse().map_err(|e| {
                LlmError::InvalidParameter(format!("Invalid header value '{}': {}", v, e))
            })?;
            headers.insert(header_name, header_value);
        }

        // Allow adapter to modify headers
        if let Some(adapter) = &self.adapter {
            adapter.build_headers(ctx.api_key.as_deref().unwrap_or(""), &mut headers)?;
        }

        Ok(headers)
    }

    fn choose_chat_transformers(
        &self,
        _req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        let request_tx = Arc::new(AnthropicChatRequestTransformer {
            provider_id: ctx.provider_id.clone(),
            adapter: self.adapter.clone(),
        });

        let response_tx = Arc::new(AnthropicChatResponseTransformer {
            provider_id: ctx.provider_id.clone(),
            adapter: self.adapter.clone(),
        });

        let stream_tx = Arc::new(AnthropicChatStreamTransformer {
            provider_id: ctx.provider_id.clone(),
            adapter: self.adapter.clone(),
        });

        ChatTransformers {
            request: request_tx,
            response: response_tx,
            stream: Some(stream_tx),
            json: None,
        }
    }

    fn chat_url(&self, _stream: bool, _req: &ChatRequest, ctx: &ProviderContext) -> String {
        let endpoint = self
            .adapter
            .as_ref()
            .map(|a| a.messages_endpoint())
            .unwrap_or("/v1/messages");
        format!("{}{}", ctx.base_url.trim_end_matches('/'), endpoint)
    }

    fn chat_before_send(
        &self,
        req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> Option<crate::executors::BeforeSendHook> {
        // Use default custom options hook
        crate::provider_core::default_custom_options_hook(self.provider_id, req)
    }
}

#[derive(Clone)]
struct AnthropicChatRequestTransformer {
    provider_id: String,
    adapter: Option<Arc<dyn AnthropicChatAdapter>>,
}

impl RequestTransformer for AnthropicChatRequestTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        let anthropic_tx =
            crate::providers::anthropic::transformers::AnthropicRequestTransformer::new(None);
        let mut body = anthropic_tx.transform_chat(req)?;

        if let Some(adapter) = &self.adapter {
            adapter.transform_request(req, &mut body)?;
        }

        Ok(body)
    }
}

#[derive(Clone)]
struct AnthropicChatResponseTransformer {
    provider_id: String,
    adapter: Option<Arc<dyn AnthropicChatAdapter>>,
}

impl ResponseTransformer for AnthropicChatResponseTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_chat_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<crate::types::ChatResponse, LlmError> {
        let mut raw = raw.clone();
        if let Some(adapter) = &self.adapter {
            adapter.transform_response(&mut raw)?;
        }

        let anthropic_tx = crate::providers::anthropic::transformers::AnthropicResponseTransformer;
        anthropic_tx.transform_chat_response(&raw)
    }
}

#[derive(Clone)]
struct AnthropicChatStreamTransformer {
    provider_id: String,
    adapter: Option<Arc<dyn AnthropicChatAdapter>>,
}

impl StreamChunkTransformer for AnthropicChatStreamTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn convert_event(
        &self,
        event: eventsource_stream::Event,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Vec<Result<crate::streaming::ChatStreamEvent, LlmError>>,
                > + Send
                + Sync
                + '_,
        >,
    > {
        use crate::streaming::SseEventConverter;

        let inner = crate::providers::anthropic::streaming::AnthropicEventConverter::new(
            crate::params::AnthropicParams::default(),
        );

        // TODO: Apply adapter transformations to SSE events

        Box::pin(async move { inner.convert_event(event).await })
    }

    fn handle_stream_end(&self) -> Option<Result<crate::streaming::ChatStreamEvent, LlmError>> {
        None
    }
}
