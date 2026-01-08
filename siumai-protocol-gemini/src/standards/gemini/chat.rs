//! Google Gemini Chat API Standard
//!
//! Standard + Adapter wrapper for Gemini chat format using provider transformers.

use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::transformers::{
    request::RequestTransformer, response::ResponseTransformer, stream::StreamChunkTransformer,
};
use crate::standards::gemini::headers::build_gemini_headers;
use crate::types::ChatRequest;
use std::sync::Arc;

/// Gemini Chat API Standard
#[derive(Clone)]
pub struct GeminiChatStandard {
    adapter: Option<Arc<dyn GeminiChatAdapter>>,
}

impl GeminiChatStandard {
    pub fn new() -> Self {
        Self { adapter: None }
    }

    pub fn with_adapter(adapter: Arc<dyn GeminiChatAdapter>) -> Self {
        Self {
            adapter: Some(adapter),
        }
    }

    pub fn create_spec(&self, provider_id: &'static str) -> GeminiChatSpec {
        GeminiChatSpec {
            provider_id,
            adapter: self.adapter.clone(),
        }
    }

    pub fn create_transformers(&self, provider_id: &str) -> ChatTransformers {
        self.create_transformers_with_model(provider_id, None)
    }

    pub fn create_transformers_with_model(
        &self,
        provider_id: &str,
        model: Option<&str>,
    ) -> ChatTransformers {
        let request_tx = Arc::new(GeminiChatRequestTransformer {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        });
        let response_tx = Arc::new(GeminiChatResponseTransformer {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        });
        let mut cfg = crate::standards::gemini::types::GeminiConfig::default();
        if let Some(m) = model
            && !m.is_empty()
        {
            cfg.model = m.to_string();
            cfg.common_params.model = m.to_string();
        }
        if provider_id.to_ascii_lowercase().contains("vertex") {
            cfg.provider_metadata_key = Some("vertex".to_string());
        }
        let stream_tx = Arc::new(GeminiChatStreamTransformer {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
            inner: crate::standards::gemini::streaming::GeminiEventConverter::new(cfg),
        });

        ChatTransformers {
            request: request_tx,
            response: response_tx,
            stream: Some(stream_tx),
            json: None,
        }
    }
}

impl Default for GeminiChatStandard {
    fn default() -> Self {
        Self::new()
    }
}

// === Adapter-wrapped transformers ===

#[derive(Clone)]
struct GeminiChatRequestTransformer {
    provider_id: String,
    adapter: Option<Arc<dyn GeminiChatAdapter>>,
}

impl RequestTransformer for GeminiChatRequestTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        let mut cfg = crate::standards::gemini::types::GeminiConfig::default();
        if !req.common_params.model.is_empty() {
            cfg.model = req.common_params.model.clone();
            cfg.common_params.model = req.common_params.model.clone();
        }
        let provider_tx =
            crate::standards::gemini::transformers::GeminiRequestTransformer { config: cfg };
        let mut body = provider_tx.transform_chat(req)?;
        if let Some(adapter) = &self.adapter {
            adapter.transform_request(req, &mut body)?;
        }
        Ok(body)
    }
}

#[derive(Clone)]
struct GeminiChatResponseTransformer {
    provider_id: String,
    adapter: Option<Arc<dyn GeminiChatAdapter>>,
}

impl ResponseTransformer for GeminiChatResponseTransformer {
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
        let mut cfg = crate::standards::gemini::types::GeminiConfig::default();
        if self.provider_id.to_ascii_lowercase().contains("vertex") {
            cfg.provider_metadata_key = Some("vertex".to_string());
        }
        let provider_tx =
            crate::standards::gemini::transformers::GeminiResponseTransformer { config: cfg };
        provider_tx.transform_chat_response(&raw)
    }
}

#[derive(Clone)]
struct GeminiChatStreamTransformer {
    provider_id: String,
    adapter: Option<Arc<dyn GeminiChatAdapter>>,
    inner: crate::standards::gemini::streaming::GeminiEventConverter,
}

impl StreamChunkTransformer for GeminiChatStreamTransformer {
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

        let adapter = self.adapter.clone();
        let inner = self.inner.clone();
        Box::pin(async move {
            let event_to_process = if let Some(adapter) = adapter {
                match crate::streaming::parse_json_with_repair::<serde_json::Value>(&event.data) {
                    Ok(mut json) => {
                        if let Err(e) = adapter.transform_sse_event(&mut json) {
                            return vec![Err(e)];
                        }
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

/// Adapter trait for provider-specific differences in Gemini Chat API
pub trait GeminiChatAdapter: Send + Sync {
    /// Mutate request JSON after standard transformation
    fn transform_request(
        &self,
        _req: &ChatRequest,
        _body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    /// Mutate response JSON before conversion to unified type
    fn transform_response(&self, _resp: &mut serde_json::Value) -> Result<(), LlmError> {
        Ok(())
    }

    /// Mutate SSE event payload prior to conversion
    fn transform_sse_event(&self, _event: &mut serde_json::Value) -> Result<(), LlmError> {
        Ok(())
    }

    /// Override chat endpoint (non-stream)
    fn chat_endpoint(&self, model: &str) -> String {
        format!("/models/{}:generateContent", model)
    }

    /// Override stream chat endpoint (SSE)
    fn chat_stream_endpoint(&self, model: &str) -> String {
        format!("/models/{}:streamGenerateContent?alt=sse", model)
    }

    /// Allow custom header injection
    fn build_headers(
        &self,
        _api_key: &str,
        _base_headers: &mut reqwest::header::HeaderMap,
    ) -> Result<(), LlmError> {
        Ok(())
    }
}

/// ProviderSpec for Gemini Chat standard
pub struct GeminiChatSpec {
    provider_id: &'static str,
    adapter: Option<Arc<dyn GeminiChatAdapter>>,
}

impl ProviderSpec for GeminiChatSpec {
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
        let api_key = ctx.api_key.as_deref().unwrap_or("");
        let mut headers = build_gemini_headers(api_key, &ctx.http_extra_headers)?;
        if let Some(adapter) = &self.adapter {
            adapter.build_headers(api_key, &mut headers)?;
        }
        Ok(headers)
    }

    fn choose_chat_transformers(
        &self,
        req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        GeminiChatStandard {
            adapter: self.adapter.clone(),
        }
        .create_transformers_with_model(&ctx.provider_id, Some(&req.common_params.model))
    }

    fn chat_url(&self, stream: bool, req: &ChatRequest, ctx: &ProviderContext) -> String {
        let base = ctx.base_url.trim_end_matches('/');
        let model = super::normalize_gemini_model_id(&req.common_params.model);
        if let Some(adapter) = &self.adapter {
            let endpoint = if stream {
                adapter.chat_stream_endpoint(&model)
            } else {
                adapter.chat_endpoint(&model)
            };
            format!("{}{}", base, endpoint)
        } else if stream {
            format!("{}/models/{}:streamGenerateContent?alt=sse", base, model)
        } else {
            format!("{}/models/{}:generateContent", base, model)
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ProviderContext;
    use crate::types::ChatMessage;

    #[test]
    fn chat_url_accepts_vertex_resource_style_model_ids() {
        let spec = GeminiChatStandard::new().create_spec("gemini");
        let ctx = ProviderContext::new(
            "gemini",
            "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/google".to_string(),
            Some("".to_string()),
            std::collections::HashMap::new(),
        );

        let req = ChatRequest::builder()
            .message(ChatMessage::user("hi").build())
            .model("publishers/google/models/gemini-2.0-flash")
            .build();
        assert_eq!(
            spec.chat_url(false, &req, &ctx),
            "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/google/models/gemini-2.0-flash:generateContent"
        );

        let req = ChatRequest::builder()
            .message(ChatMessage::user("hi").build())
            .stream(true)
            .model("models/gemini-2.0-flash")
            .build();
        assert_eq!(
            spec.chat_url(true, &req, &ctx),
            "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/google/models/gemini-2.0-flash:streamGenerateContent?alt=sse"
        );
    }
}
