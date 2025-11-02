//! OpenAI Chat Completions API Standard
//!
//! This module implements the OpenAI Chat Completions API format, which has become
//! the de facto standard for chat-based LLM APIs.
//!
//! ## Supported Providers
//!
//! - OpenAI (native)
//! - DeepSeek
//! - SiliconFlow
//! - Together
//! - OpenRouter
//! - Groq
//! - xAI
//! - Many others
//!
//! ## Usage
//!
//! ```rust,ignore
//! use siumai::standards::openai::chat::OpenAiChatStandard;
//!
//! // Standard OpenAI implementation
//! let standard = OpenAiChatStandard::new();
//!
//! // With provider-specific adapter
//! let standard = OpenAiChatStandard::with_adapter(
//!     Arc::new(MyCustomAdapter)
//! );
//! ```

use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::transformers::request::RequestTransformer;
use crate::execution::transformers::response::ResponseTransformer;
use crate::execution::transformers::stream::StreamChunkTransformer;
use crate::types::ChatRequest;
use std::sync::Arc;

/// OpenAI Chat API Standard
///
/// Represents the OpenAI Chat Completions API format.
/// Can be used by any provider that implements OpenAI-compatible chat API.
#[derive(Clone)]
pub struct OpenAiChatStandard {
    /// Optional adapter for provider-specific differences
    adapter: Option<Arc<dyn OpenAiChatAdapter>>,
}

impl OpenAiChatStandard {
    /// Create a new standard OpenAI Chat implementation
    pub fn new() -> Self {
        Self { adapter: None }
    }

    /// Create with a provider-specific adapter
    pub fn with_adapter(adapter: Arc<dyn OpenAiChatAdapter>) -> Self {
        Self {
            adapter: Some(adapter),
        }
    }

    /// Create a ProviderSpec for this standard
    pub fn create_spec(&self, provider_id: &'static str) -> OpenAiChatSpec {
        OpenAiChatSpec {
            provider_id,
            adapter: self.adapter.clone(),
        }
    }

    /// Create transformers for chat requests
    pub fn create_transformers(&self, provider_id: &str) -> ChatTransformers {
        let request_tx = Arc::new(OpenAiChatRequestTransformer {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        });

        let response_tx = Arc::new(OpenAiChatResponseTransformer {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        });

        let stream_tx = Arc::new(OpenAiChatStreamTransformer {
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

impl Default for OpenAiChatStandard {
    fn default() -> Self {
        Self::new()
    }
}

/// Adapter trait for provider-specific differences in OpenAI Chat API
///
/// Implement this trait to handle provider-specific variations of the OpenAI Chat API.
/// For example, DeepSeek uses `reasoning_content` instead of standard fields for
/// reasoning models.
pub trait OpenAiChatAdapter: Send + Sync {
    /// Transform request JSON before sending
    ///
    /// This is called after the standard OpenAI request transformation.
    /// Use this to add provider-specific fields or modify existing ones.
    fn transform_request(
        &self,
        _req: &ChatRequest,
        _body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    /// Transform response JSON after receiving
    ///
    /// This is called before the standard OpenAI response transformation.
    /// Use this to normalize provider-specific response fields.
    fn transform_response(&self, _resp: &mut serde_json::Value) -> Result<(), LlmError> {
        Ok(())
    }

    /// Transform SSE event before processing
    ///
    /// This is called for each SSE event in streaming responses.
    /// Use this to normalize provider-specific event formats.
    fn transform_sse_event(&self, _event: &mut serde_json::Value) -> Result<(), LlmError> {
        Ok(())
    }

    /// Get provider-specific endpoint path
    ///
    /// Default is "/chat/completions" (standard OpenAI)
    fn chat_endpoint(&self) -> &str {
        "/chat/completions"
    }

    /// Get provider-specific headers
    ///
    /// Default is standard OpenAI headers (Authorization: Bearer <token>)
    fn build_headers(
        &self,
        _api_key: &str,
        _base_headers: &mut reqwest::header::HeaderMap,
    ) -> Result<(), LlmError> {
        Ok(())
    }
}

/// ProviderSpec implementation for OpenAI Chat Standard
pub struct OpenAiChatSpec {
    provider_id: &'static str,
    adapter: Option<Arc<dyn OpenAiChatAdapter>>,
}

impl ProviderSpec for OpenAiChatSpec {
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

        // Standard OpenAI Authorization header
        if let Some(api_key) = &ctx.api_key {
            headers.insert(
                "Authorization",
                format!("Bearer {}", api_key)
                    .parse()
                    .map_err(|e| LlmError::InvalidParameter(format!("Invalid API key: {}", e)))?,
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
        let request_tx = Arc::new(OpenAiChatRequestTransformer {
            provider_id: ctx.provider_id.clone(),
            adapter: self.adapter.clone(),
        });

        let response_tx = Arc::new(OpenAiChatResponseTransformer {
            provider_id: ctx.provider_id.clone(),
            adapter: self.adapter.clone(),
        });

        let stream_tx = Arc::new(OpenAiChatStreamTransformer {
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
            .map(|a| a.chat_endpoint())
            .unwrap_or("/chat/completions");
        format!("{}{}", ctx.base_url.trim_end_matches('/'), endpoint)
    }

    fn chat_before_send(
        &self,
        req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        // Use default custom options hook
        crate::core::default_custom_options_hook(self.provider_id, req)
    }
}

/// Request transformer for OpenAI Chat API
#[derive(Clone)]
struct OpenAiChatRequestTransformer {
    provider_id: String,
    adapter: Option<Arc<dyn OpenAiChatAdapter>>,
}

impl RequestTransformer for OpenAiChatRequestTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        // Reuse the existing OpenAI request transformer logic
        let openai_tx = crate::providers::openai::transformers::request::OpenAiRequestTransformer;
        let mut body = openai_tx.transform_chat(req)?;

        // Apply adapter transformations if present
        if let Some(adapter) = &self.adapter {
            adapter.transform_request(req, &mut body)?;
        }

        Ok(body)
    }
}

/// Response transformer for OpenAI Chat API
#[derive(Clone)]
struct OpenAiChatResponseTransformer {
    provider_id: String,
    adapter: Option<Arc<dyn OpenAiChatAdapter>>,
}

impl ResponseTransformer for OpenAiChatResponseTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_chat_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<crate::types::ChatResponse, LlmError> {
        // Apply adapter transformations if present
        let mut raw = raw.clone();
        if let Some(adapter) = &self.adapter {
            adapter.transform_response(&mut raw)?;
        }

        // Reuse the existing OpenAI response transformer logic
        let openai_tx = crate::providers::openai::transformers::response::OpenAiResponseTransformer;
        openai_tx.transform_chat_response(&raw)
    }
}

/// Stream transformer for OpenAI Chat API
#[derive(Clone)]
struct OpenAiChatStreamTransformer {
    provider_id: String,
    adapter: Option<Arc<dyn OpenAiChatAdapter>>,
}

impl StreamChunkTransformer for OpenAiChatStreamTransformer {
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
        // Reuse the existing OpenAI stream transformer logic
        use crate::streaming::SseEventConverter;

        let chat_adapter = self.adapter.clone();

        // Choose a ProviderAdapter based on provider_id; fallback to OpenAI standard
        let provider_adapter: Arc<
            dyn crate::providers::openai_compatible::adapter::ProviderAdapter,
        > = {
            // Try configurable adapter from builtin providers registry
            let builtins = crate::providers::openai_compatible::config::get_builtin_providers();
            if let Some(conf) = builtins.get(&self.provider_id) {
                Arc::new(
                    crate::providers::openai_compatible::registry::ConfigurableAdapter::new(
                        conf.clone(),
                    ),
                )
            } else {
                Arc::new(crate::providers::openai::adapter::OpenAiStandardAdapter {
                    base_url: String::new(),
                })
            }
        };
        // Create a minimal config for the converter
        let config =
            crate::providers::openai_compatible::openai_config::OpenAiCompatibleConfig::new(
                &self.provider_id,
                "",
                "",
                provider_adapter.clone(),
            );

        let inner =
            crate::providers::openai_compatible::streaming::OpenAiCompatibleEventConverter::new(
                config,
                provider_adapter,
            );

        Box::pin(async move {
            // Apply adapter transformation to SSE event if adapter is present
            let event_to_process = if let Some(adapter) = chat_adapter {
                // Parse JSON, apply adapter transformation, then re-serialize
                match crate::streaming::parse_json_with_repair::<serde_json::Value>(&event.data) {
                    Ok(mut json) => {
                        // Apply adapter transformation
                        if let Err(e) = adapter.transform_sse_event(&mut json) {
                            return vec![Err(e)];
                        }
                        // Re-serialize to create modified event
                        let modified_data = match serde_json::to_string(&json) {
                            Ok(data) => data,
                            Err(e) => {
                                return vec![Err(LlmError::ParseError(format!(
                                    "Failed to serialize modified SSE event: {e}"
                                )))];
                            }
                        };
                        eventsource_stream::Event {
                            data: modified_data,
                            ..event
                        }
                    }
                    Err(e) => {
                        return vec![Err(LlmError::ParseError(format!(
                            "Failed to parse SSE event for adapter transformation: {e}"
                        )))];
                    }
                }
            } else {
                event
            };

            inner.convert_event(event_to_process).await
        })
    }

    fn handle_stream_end(&self) -> Option<Result<crate::streaming::ChatStreamEvent, LlmError>> {
        None
    }
}
