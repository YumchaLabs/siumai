//! Google Gemini Chat API Standard
//!
//! Standard + Adapter wrapper for Gemini chat format using provider transformers.

use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::transformers::{
    request::RequestTransformer, response::ResponseTransformer, stream::StreamChunkTransformer,
};
use crate::standards::gemini::headers::build_gemini_headers;
use crate::standards::gemini::types::GeminiConfig;
use crate::types::ChatRequest;
use std::sync::Arc;

/// Gemini Chat API Standard
#[derive(Clone)]
pub struct GeminiChatStandard {
    adapter: Option<Arc<dyn GeminiChatAdapter>>,
    base_config: GeminiConfig,
}

impl GeminiChatStandard {
    pub fn new() -> Self {
        Self {
            adapter: None,
            base_config: GeminiConfig::default(),
        }
    }

    pub fn with_adapter(adapter: Arc<dyn GeminiChatAdapter>) -> Self {
        Self {
            adapter: Some(adapter),
            base_config: GeminiConfig::default(),
        }
    }

    pub fn with_base_config(mut self, base_config: GeminiConfig) -> Self {
        self.base_config = base_config;
        self
    }

    pub fn create_spec(&self, provider_id: &'static str) -> GeminiChatSpec {
        GeminiChatSpec {
            provider_id,
            adapter: self.adapter.clone(),
            base_config: self.base_config.clone(),
        }
    }

    pub fn create_transformers(&self, provider_id: &str) -> ChatTransformers {
        self.create_transformers_with_model_and_stream_options(provider_id, None, false)
    }

    pub fn create_transformers_with_model(
        &self,
        provider_id: &str,
        model: Option<&str>,
    ) -> ChatTransformers {
        self.create_transformers_with_model_and_stream_options(provider_id, model, false)
    }

    pub fn create_transformers_with_model_and_stream_options(
        &self,
        provider_id: &str,
        model: Option<&str>,
        include_raw_chunks: bool,
    ) -> ChatTransformers {
        let cfg = self.config_for_transformers(provider_id, model);
        let request_tx = Arc::new(GeminiChatRequestTransformer {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
            base_config: cfg.clone(),
        });
        let response_tx = Arc::new(GeminiChatResponseTransformer {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
            config: cfg.clone(),
        });
        let stream_tx = Arc::new(GeminiChatStreamTransformer {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
            inner: crate::standards::gemini::streaming::GeminiEventConverter::new(cfg)
                .with_include_raw_chunks(include_raw_chunks),
        });

        ChatTransformers {
            request: request_tx,
            response: response_tx,
            stream: Some(stream_tx),
            json: None,
        }
    }

    fn config_for_transformers(&self, provider_id: &str, model: Option<&str>) -> GeminiConfig {
        let mut cfg = self.base_config.clone();
        if let Some(m) = model
            && !m.is_empty()
        {
            cfg.model = m.to_string();
            cfg.common_params.model = m.to_string();
        }
        if provider_id.to_ascii_lowercase().contains("vertex")
            && cfg.provider_metadata_key.is_none()
        {
            cfg.provider_metadata_key = Some("vertex".to_string());
        }
        cfg
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
    base_config: GeminiConfig,
}

impl RequestTransformer for GeminiChatRequestTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        let mut cfg = self.base_config.clone();
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
    config: GeminiConfig,
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
        let provider_tx = crate::standards::gemini::transformers::GeminiResponseTransformer {
            config: self.config.clone(),
        };
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
                match serde_json::from_str::<serde_json::Value>(&event.data) {
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

    fn is_stream_end_event(&self, event: &eventsource_stream::Event) -> bool {
        use crate::streaming::SseEventConverter;
        self.inner.is_stream_end_event(event)
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
    base_config: GeminiConfig,
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
            base_config: self.base_config.clone(),
        }
        .create_transformers_with_model_and_stream_options(
            &ctx.provider_id,
            Some(&req.common_params.model),
            req.stream_options.include_raw_chunks,
        )
    }

    fn try_chat_url(
        &self,
        stream: bool,
        req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> Result<String, LlmError> {
        let base = ctx.base_url.trim_end_matches('/');
        let model = super::normalize_gemini_model_id(&req.common_params.model);
        let url = if let Some(adapter) = &self.adapter {
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
        };
        Ok(url)
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
    use crate::streaming::TypedStreamPart;
    use crate::types::{ChatMessage, ChatRequest};

    fn source_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start_index = source.find(start).expect("section start marker");
        let end_index = source[start_index..]
            .find(end)
            .map(|offset| start_index + offset)
            .expect("section end marker");
        &source[start_index..end_index]
    }

    #[test]
    fn chat_wrapper_keeps_request_response_stream_maps_directional() {
        let source = include_str!("chat.rs");

        let request_transformer = source_section(
            source,
            "impl RequestTransformer for GeminiChatRequestTransformer",
            "#[derive(Clone)]\nstruct GeminiChatResponseTransformer",
        );
        assert!(
            !request_transformer.contains("provider_metadata"),
            "Gemini chat request transformer wrapper must not read legacy provider_metadata"
        );
        assert!(
            !request_transformer.contains("providerMetadata"),
            "Gemini chat request transformer wrapper must not read legacy providerMetadata"
        );

        let response_transformer = source_section(
            source,
            "impl ResponseTransformer for GeminiChatResponseTransformer",
            "#[derive(Clone)]\nstruct GeminiChatStreamTransformer",
        );
        assert!(
            !response_transformer.contains("provider_options"),
            "Gemini chat response transformer wrapper must not read request provider_options"
        );
        assert!(
            !response_transformer.contains("providerOptions"),
            "Gemini chat response transformer wrapper must not read request providerOptions"
        );

        let stream_transformer = source_section(
            source,
            "impl StreamChunkTransformer for GeminiChatStreamTransformer",
            "/// Adapter trait for provider-specific differences in Gemini Chat API",
        );
        assert!(
            !stream_transformer.contains("provider_options"),
            "Gemini chat stream transformer wrapper must not read request provider_options"
        );
        assert!(
            !stream_transformer.contains("providerOptions"),
            "Gemini chat stream transformer wrapper must not read request providerOptions"
        );
    }

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
            spec.try_chat_url(false, &req, &ctx).unwrap(),
            "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/google/models/gemini-2.0-flash:generateContent"
        );

        let req = ChatRequest::builder()
            .message(ChatMessage::user("hi").build())
            .stream(true)
            .model("models/gemini-2.0-flash")
            .build();
        assert_eq!(
            spec.try_chat_url(true, &req, &ctx).unwrap(),
            "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/google/models/gemini-2.0-flash:streamGenerateContent?alt=sse"
        );
    }

    #[test]
    fn vertex_request_transformer_prefers_vertex_provider_namespace() {
        let tx = GeminiChatStandard::new().create_transformers_with_model("vertex", None);
        let req = ChatRequest::builder()
            .message(
                ChatMessage::assistant_with_content(vec![
                    crate::types::ContentPart::reasoning("thinking")
                        .with_provider_option(
                            "google",
                            serde_json::json!({ "thoughtSignature": "google_sig" }),
                        )
                        .with_provider_option(
                            "vertex",
                            serde_json::json!({ "thoughtSignature": "vertex_sig" }),
                        ),
                ])
                .build(),
            )
            .model("gemini-2.5-flash")
            .build();

        let body = tx.request.transform_chat(&req).expect("transform chat");
        let part = &body["contents"][0]["parts"][0];

        assert_eq!(part["thought"], serde_json::json!(true));
        assert_eq!(part["thoughtSignature"], serde_json::json!("vertex_sig"));
    }

    #[test]
    fn base_config_generate_id_flows_into_response_transformer() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let counter = Arc::new(AtomicUsize::new(0));
        let tx = GeminiChatStandard::new()
            .with_base_config(GeminiConfig::default().with_generate_id({
                let counter = Arc::clone(&counter);
                move || format!("vertex-custom-{}", counter.fetch_add(1, Ordering::Relaxed))
            }))
            .create_transformers_with_model("vertex", Some("gemini-2.5-flash"));

        let raw = serde_json::json!({
            "candidates": [
                {
                    "content": {
                        "parts": [
                            { "functionCall": { "name": "weather", "args": { "city": "Tokyo" } } }
                        ]
                    },
                    "groundingMetadata": {
                        "groundingChunks": [
                            { "web": { "uri": "https://example.com", "title": "Example" } }
                        ]
                    },
                    "finishReason": "STOP"
                }
            ],
            "modelVersion": "gemini-2.5-flash"
        });

        let response = tx
            .response
            .transform_chat_response(&raw)
            .expect("transform response");

        let tool_call = response.tool_calls()[0].as_tool_call().expect("tool call");
        assert_eq!(tool_call.tool_call_id, "vertex-custom-0");
        assert_eq!(
            response
                .provider_metadata
                .as_ref()
                .and_then(|meta| meta.get("vertex"))
                .and_then(|meta| meta.get("sources"))
                .and_then(|sources| sources.as_array())
                .and_then(|sources| sources.first())
                .and_then(|source| source.get("id"))
                .and_then(|value| value.as_str()),
            Some("vertex-custom-1")
        );
    }

    #[tokio::test]
    async fn choose_chat_transformers_forwards_include_raw_chunks_to_stream_converter() {
        let spec = GeminiChatStandard::new().create_spec("gemini");
        let ctx = ProviderContext::new(
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta".to_string(),
            Some("test-key".to_string()),
            std::collections::HashMap::new(),
        );
        let req = ChatRequest::builder()
            .message(ChatMessage::user("hi").build())
            .stream(true)
            .include_raw_chunks(true)
            .model("gemini-2.5-flash")
            .build();

        let tx = spec.choose_chat_transformers(&req, &ctx);
        let stream_tx = tx.stream.expect("stream transformer");
        let event = eventsource_stream::Event {
            event: "".into(),
            data: serde_json::json!({
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                { "text": "Hello" }
                            ]
                        }
                    }
                ]
            })
            .to_string(),
            id: "".into(),
            retry: None,
        };

        let result = stream_tx.convert_event(event).await;
        assert!(result.iter().any(|event| {
            matches!(
                event
                    .as_ref()
                    .ok()
                    .and_then(TypedStreamPart::try_from_chat_event),
                Some(TypedStreamPart::Raw { .. })
            )
        }));
    }
}
