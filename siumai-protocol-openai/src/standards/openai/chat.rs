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
//! use siumai::experimental::standards::openai::chat::OpenAiChatStandard;
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
    /// Optional OpenAI-compatible provider adapter (field mappings, compat flags, etc.)
    provider_adapter: Option<Arc<dyn crate::standards::openai::compat::adapter::ProviderAdapter>>,
}

impl OpenAiChatStandard {
    /// Create a new standard OpenAI Chat implementation
    pub fn new() -> Self {
        Self {
            adapter: None,
            provider_adapter: None,
        }
    }

    /// Create with a provider-specific adapter
    pub fn with_adapter(adapter: Arc<dyn OpenAiChatAdapter>) -> Self {
        Self {
            adapter: Some(adapter),
            provider_adapter: None,
        }
    }

    /// Create with an OpenAI-compatible provider adapter
    pub fn with_provider_adapter(
        provider_adapter: Arc<dyn crate::standards::openai::compat::adapter::ProviderAdapter>,
    ) -> Self {
        Self {
            adapter: None,
            provider_adapter: Some(provider_adapter),
        }
    }

    /// Create with both adapters
    pub fn with_adapters(
        chat_adapter: Arc<dyn OpenAiChatAdapter>,
        provider_adapter: Arc<dyn crate::standards::openai::compat::adapter::ProviderAdapter>,
    ) -> Self {
        Self {
            adapter: Some(chat_adapter),
            provider_adapter: Some(provider_adapter),
        }
    }

    /// Create a ProviderSpec for this standard
    pub fn create_spec(&self, provider_id: &'static str) -> OpenAiChatSpec {
        OpenAiChatSpec {
            provider_id,
            adapter: self.adapter.clone(),
            provider_adapter: self.provider_adapter.clone(),
        }
    }

    /// Create transformers for chat requests
    pub fn create_transformers(&self, provider_id: &str) -> ChatTransformers {
        self.create_transformers_with_model(provider_id, None)
    }

    /// Create transformers for chat requests with a known model id
    pub fn create_transformers_with_model(
        &self,
        provider_id: &str,
        model: Option<&str>,
    ) -> ChatTransformers {
        let request_tx = Arc::new(OpenAiChatRequestTransformer {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        });

        let provider_adapter: Arc<dyn crate::standards::openai::compat::adapter::ProviderAdapter> =
            self.provider_adapter.clone().unwrap_or_else(|| {
                Arc::new(
                    crate::standards::openai::compat::adapter::OpenAiStandardAdapter {
                        base_url: String::new(),
                    },
                )
            });

        let response_tx = Arc::new(OpenAiChatResponseTransformer {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
            provider_adapter: provider_adapter.clone(),
            fallback_model: model.map(|m| m.to_string()),
        });

        let inner = build_openai_compatible_stream_converter(
            provider_id,
            model,
            None,
            None,
            provider_adapter.clone(),
        );
        let stream_tx = Arc::new(OpenAiChatStreamTransformer {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
            inner,
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
    provider_adapter: Option<Arc<dyn crate::standards::openai::compat::adapter::ProviderAdapter>>,
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
        let mut headers =
            crate::standards::openai::headers::build_openai_compatible_json_headers(ctx)?;

        // Allow adapter to modify headers
        if let Some(adapter) = &self.adapter {
            adapter.build_headers(ctx.api_key.as_deref().unwrap_or(""), &mut headers)?;
        }

        Ok(headers)
    }

    fn choose_chat_transformers(
        &self,
        req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        let request_tx = Arc::new(OpenAiChatRequestTransformer {
            provider_id: ctx.provider_id.clone(),
            adapter: self.adapter.clone(),
        });

        let provider_adapter: Arc<dyn crate::standards::openai::compat::adapter::ProviderAdapter> =
            self.provider_adapter.clone().unwrap_or_else(|| {
                Arc::new(
                    crate::standards::openai::compat::adapter::OpenAiStandardAdapter {
                        base_url: ctx.base_url.clone(),
                    },
                )
            });

        let response_tx = Arc::new(OpenAiChatResponseTransformer {
            provider_id: ctx.provider_id.clone(),
            adapter: self.adapter.clone(),
            provider_adapter: provider_adapter.clone(),
            fallback_model: Some(req.common_params.model.clone()),
        });
        let inner = build_openai_compatible_stream_converter(
            &ctx.provider_id,
            Some(&req.common_params.model),
            ctx.api_key.as_deref(),
            Some(&ctx.base_url),
            provider_adapter.clone(),
        );
        let stream_tx = Arc::new(OpenAiChatStreamTransformer {
            provider_id: ctx.provider_id.clone(),
            adapter: self.adapter.clone(),
            inner,
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
        let _ = req;
        None
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
        // Vercel parity: OpenAI/Azure OpenAI support PDF/audio file parts in Chat Completions.
        // We must build the base body with the OpenAI-chat message converter, otherwise the
        // OpenAI-compatible converter errors before we can override the `messages` field.
        let mut body = if self.provider_id == "openai" || self.provider_id == "azure" {
            if req.common_params.model.is_empty() {
                return Err(LlmError::InvalidParameter(
                    "Model must be specified".to_string(),
                ));
            }

            let system_message_mode = req
                .provider_option("openai")
                .or_else(|| req.provider_option("azure"))
                .or_else(|| req.provider_option(&self.provider_id))
                .and_then(|v| v.as_object())
                .and_then(|obj| {
                    obj.get("systemMessageMode")
                        .or_else(|| obj.get("system_message_mode"))
                        .and_then(|v| v.as_str())
                })
                .unwrap_or("system");

            let mut body = serde_json::json!({ "model": req.common_params.model });

            if let Some(t) = req.common_params.temperature {
                body["temperature"] = serde_json::json!(t);
            }
            if let Some(tp) = req.common_params.top_p {
                body["top_p"] = serde_json::json!(tp);
            }
            if let Some(seed) = req.common_params.seed {
                body["seed"] = serde_json::json!(seed);
            }
            if let Some(max) = req.common_params.max_completion_tokens {
                body["max_completion_tokens"] = serde_json::json!(max);
            } else if let Some(max) = req.common_params.max_tokens {
                body["max_tokens"] = serde_json::json!(max);
            }
            if let Some(stops) = &req.common_params.stop_sequences {
                body["stop"] = serde_json::json!(stops);
            }

            let messages_input = match system_message_mode {
                "developer" => req
                    .messages
                    .iter()
                    .cloned()
                    .map(|mut m| {
                        if matches!(m.role, crate::types::MessageRole::System) {
                            m.role = crate::types::MessageRole::Developer;
                        }
                        m
                    })
                    .collect::<Vec<_>>(),
                "remove" => req
                    .messages
                    .iter()
                    .filter(|m| !matches!(m.role, crate::types::MessageRole::System))
                    .cloned()
                    .collect::<Vec<_>>(),
                _ => req.messages.clone(),
            };

            let messages =
                crate::standards::openai::utils::convert_messages_openai_chat(&messages_input)?;
            body["messages"] = serde_json::to_value(messages)?;

            if let Some(tools) = &req.tools
                && !tools.is_empty()
            {
                let openai_tools =
                    crate::standards::openai::utils::convert_tools_to_openai_format(tools)?;
                if !openai_tools.is_empty() {
                    body["tools"] = serde_json::Value::Array(openai_tools);
                    if let Some(choice) = &req.tool_choice {
                        body["tool_choice"] =
                            crate::standards::openai::utils::convert_tool_choice(choice);
                    }
                }
            }

            if req.stream {
                body["stream"] = serde_json::Value::Bool(true);
                body["stream_options"] = serde_json::json!({ "include_usage": true });
            }

            body
        } else {
            // Reuse the existing OpenAI request transformer logic for OpenAI-compatible vendors.
            let openai_tx = crate::standards::openai::transformers::OpenAiRequestTransformer;
            openai_tx.transform_chat(req)?
        };

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
    provider_adapter: Arc<dyn crate::standards::openai::compat::adapter::ProviderAdapter>,
    fallback_model: Option<String>,
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

        // Use OpenAI-compatible response transformer with the injected provider adapter.
        // This enables provider-specific thinking field extraction and other compat behaviors.
        let model = raw
            .get("model")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .or_else(|| self.fallback_model.clone())
            .unwrap_or_default();

        let cfg = crate::standards::openai::compat::openai_config::OpenAiCompatibleConfig::new(
            &self.provider_id,
            "",
            "",
            self.provider_adapter.clone(),
        )
        .with_model(&model);

        let compat = crate::standards::openai::compat::transformers::CompatResponseTransformer {
            config: cfg,
            adapter: self.provider_adapter.clone(),
        };
        compat.transform_chat_response(&raw)
    }
}

/// Stream transformer for OpenAI Chat API
#[derive(Clone)]
struct OpenAiChatStreamTransformer {
    provider_id: String,
    adapter: Option<Arc<dyn OpenAiChatAdapter>>,
    inner: crate::standards::openai::compat::streaming::OpenAiCompatibleEventConverter,
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
        use crate::streaming::SseEventConverter;

        let chat_adapter = self.adapter.clone();
        let inner = self.inner.clone();

        Box::pin(async move {
            // Apply adapter transformation to SSE event if adapter is present
            let event_to_process = if let Some(adapter) = chat_adapter {
                // Parse JSON, apply adapter transformation, then re-serialize
                match serde_json::from_str::<serde_json::Value>(&event.data) {
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
        use crate::streaming::SseEventConverter;
        self.inner.handle_stream_end()
    }
}

fn build_openai_compatible_stream_converter(
    provider_id: &str,
    model: Option<&str>,
    api_key: Option<&str>,
    base_url: Option<&str>,
    provider_adapter: Arc<dyn crate::standards::openai::compat::adapter::ProviderAdapter>,
) -> crate::standards::openai::compat::streaming::OpenAiCompatibleEventConverter {
    let mut cfg = crate::standards::openai::compat::openai_config::OpenAiCompatibleConfig::new(
        provider_id,
        api_key.unwrap_or_default(),
        base_url.unwrap_or_default(),
        provider_adapter.clone(),
    );
    if let Some(m) = model
        && !m.is_empty()
    {
        cfg = cfg.with_model(m);
    }

    crate::standards::openai::compat::streaming::OpenAiCompatibleEventConverter::new(
        cfg,
        provider_adapter,
    )
}
