//! xAI Responses SSE presets.
//!
//! This module wraps the OpenAI Responses SSE converter with xAI-specific defaults
//! to keep call sites and tests free of provider-specific mode toggles.

use crate::streaming::SseEventConverter;

/// xAI-aligned Responses SSE converter.
///
/// Presets:
/// - `StreamPartsStyle::Xai` (Vercel stream parts ids and finish shape)
/// - `WebSearchStreamMode::Xai` (tool-input-* behavior for web_search/x_search)
#[derive(Clone)]
pub struct XaiResponsesEventConverter {
    inner: crate::standards::openai::responses_sse::OpenAiResponsesEventConverter,
}

impl XaiResponsesEventConverter {
    pub fn new() -> Self {
        Self {
            inner: crate::standards::openai::responses_sse::OpenAiResponsesEventConverter::new()
                .with_stream_parts_style(
                    crate::standards::openai::responses_sse::StreamPartsStyle::Xai,
                )
                .with_web_search_stream_mode(
                    crate::standards::openai::responses_sse::WebSearchStreamMode::Xai,
                ),
        }
    }

    pub fn with_request_tools(mut self, tools: &[crate::types::Tool]) -> Self {
        self.inner = self.inner.with_request_tools(tools);
        self
    }
}

impl Default for XaiResponsesEventConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl SseEventConverter for XaiResponsesEventConverter {
    fn convert_event(
        &self,
        event: eventsource_stream::Event,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Vec<Result<crate::streaming::ChatStreamEvent, crate::error::LlmError>>,
                > + Send
                + Sync
                + '_,
        >,
    > {
        self.inner.convert_event(event)
    }

    fn handle_stream_end(
        &self,
    ) -> Option<Result<crate::streaming::ChatStreamEvent, crate::error::LlmError>> {
        self.inner.handle_stream_end()
    }

    fn handle_stream_end_events(
        &self,
    ) -> Vec<Result<crate::streaming::ChatStreamEvent, crate::error::LlmError>> {
        self.inner.handle_stream_end_events()
    }

    fn finalize_on_disconnect(&self) -> bool {
        self.inner.finalize_on_disconnect()
    }
}
