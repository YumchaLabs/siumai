//! Anthropic Messages API Standard
//!
//! This module implements the Anthropic Messages API format.

use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::transformers::request::RequestTransformer;
use crate::execution::transformers::response::ResponseTransformer;
use crate::execution::transformers::stream::StreamChunkTransformer;
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

        let inner = crate::standards::anthropic::streaming::AnthropicEventConverter::new(
            super::params::AnthropicParams::default(),
        );
        let stream_tx = Arc::new(AnthropicChatStreamTransformer {
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
        req: &ChatRequest,
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

        fn citations_enabled_for_part(
            message_custom: &std::collections::HashMap<String, serde_json::Value>,
            index: usize,
        ) -> bool {
            let Some(map) = message_custom
                .get("anthropic_document_citations")
                .and_then(|v| v.as_object())
            else {
                return false;
            };

            let Some(cfg) = map.get(&index.to_string()) else {
                return false;
            };
            if let Some(b) = cfg.as_bool() {
                return b;
            }
            cfg.get("enabled")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        }

        fn document_title_for_part(
            message_custom: &std::collections::HashMap<String, serde_json::Value>,
            index: usize,
        ) -> Option<String> {
            let map = message_custom
                .get("anthropic_document_metadata")
                .and_then(|v| v.as_object())?;
            let meta = map.get(&index.to_string()).and_then(|v| v.as_object())?;
            meta.get("title")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
        }

        fn extract_citation_documents(
            req: &ChatRequest,
        ) -> Vec<crate::standards::anthropic::streaming::AnthropicCitationDocument> {
            let mut out = Vec::new();

            for msg in &req.messages {
                if msg.role != crate::types::MessageRole::User {
                    continue;
                }
                let crate::types::MessageContent::MultiModal(parts) = &msg.content else {
                    continue;
                };

                for (i, part) in parts.iter().enumerate() {
                    if !citations_enabled_for_part(&msg.metadata.custom, i) {
                        continue;
                    }

                    let crate::types::ContentPart::File {
                        media_type,
                        filename,
                        ..
                    } = part
                    else {
                        continue;
                    };

                    if media_type != "application/pdf" && media_type != "text/plain" {
                        continue;
                    }

                    let title_override = document_title_for_part(&msg.metadata.custom, i);
                    out.push(
                        crate::standards::anthropic::streaming::AnthropicCitationDocument {
                            title: title_override
                                .or_else(|| filename.clone())
                                .unwrap_or_else(|| "Untitled Document".to_string()),
                            filename: filename.clone(),
                            media_type: media_type.clone(),
                        },
                    );
                }
            }

            out
        }

        let citation_documents = extract_citation_documents(req);
        let mut stream_params = super::params::AnthropicParams::default();
        if matches!(
            req.response_format,
            Some(crate::types::chat::ResponseFormat::Json { .. })
        ) {
            fn supports_output_format(model: &str) -> bool {
                model.starts_with("claude-sonnet-4-5")
                    || model.starts_with("claude-opus-4-5")
                    || model.starts_with("claude-haiku-4-5")
            }

            let tools_empty = req.tools.as_ref().map(|t| t.is_empty()).unwrap_or(true);
            stream_params = stream_params.with_structured_output_mode(
                if supports_output_format(&req.common_params.model) && tools_empty {
                    super::params::StructuredOutputMode::OutputFormat
                } else {
                    super::params::StructuredOutputMode::JsonTool
                },
            );
        }

        let inner =
            crate::standards::anthropic::streaming::AnthropicEventConverter::new(stream_params)
                .with_citation_documents(citation_documents);
        let stream_tx = Arc::new(AnthropicChatStreamTransformer {
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
            .map(|a| a.messages_endpoint())
            .unwrap_or("/v1/messages");

        // Compatibility: allow users to pass a base URL that already includes `/v1`
        // (e.g. `ANTHROPIC_BASE_URL=https://api.anthropic.com/v1`) without producing
        // double version segments like `/v1/v1/messages`.
        let base = ctx.base_url.trim_end_matches('/');
        let endpoint = if base.ends_with("/v1") && endpoint.starts_with("/v1/") {
            &endpoint["/v1".len()..]
        } else {
            endpoint
        };

        format!("{base}{endpoint}")
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
            crate::standards::anthropic::transformers::AnthropicRequestTransformer::new(None);
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

        let anthropic_tx = crate::standards::anthropic::transformers::AnthropicResponseTransformer;
        anthropic_tx.transform_chat_response(&raw)
    }
}

#[derive(Clone)]
struct AnthropicChatStreamTransformer {
    provider_id: String,
    adapter: Option<Arc<dyn AnthropicChatAdapter>>,
    inner: crate::standards::anthropic::streaming::AnthropicEventConverter,
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

        let adapter = self.adapter.clone();
        let inner = self.inner.clone();

        Box::pin(async move {
            // Apply adapter transformation to SSE event if adapter is present
            let event_to_process = if let Some(adapter) = adapter {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, ChatRequest, ContentPart};
    use eventsource_stream::Event;

    #[tokio::test]
    async fn citation_documents_use_document_metadata_title_override() {
        let spec = AnthropicChatStandard::new().create_spec("anthropic");
        let ctx = ProviderContext::new(
            "anthropic",
            "https://api.anthropic.com".to_string(),
            Some("test".to_string()),
            std::collections::HashMap::new(),
        );

        let msg = ChatMessage::user("hi")
            .with_content_parts(vec![ContentPart::file_url(
                "https://example.com/a.pdf",
                "application/pdf",
            )])
            .anthropic_document_citations_for_part(1, true)
            .anthropic_document_metadata_for_part(1, Some("My PDF".to_string()), None)
            .build();

        let req = ChatRequest::builder().message(msg).stream(true).build();
        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let stream_tx = bundle.stream.expect("stream transformer");

        let ev = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"page_location","cited_text":"hello","document_index":0,"document_title":null,"start_page_number":1,"end_page_number":1}}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };

        let out = stream_tx.convert_event(ev).await;
        assert_eq!(out.len(), 1);
        match out.first().unwrap().as_ref().unwrap() {
            crate::streaming::ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "anthropic:source");
                assert_eq!(data["title"], serde_json::json!("My PDF"));
            }
            other => panic!("Expected source Custom event, got {:?}", other),
        }
    }
}
