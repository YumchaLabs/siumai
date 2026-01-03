//! Stream chunk transformer wrappers for OpenAI-compatible streaming (protocol layer)

use crate::error::LlmError;
use crate::execution::transformers::stream::StreamChunkTransformer;
use crate::streaming::SseEventConverter;
use std::future::Future;
use std::pin::Pin;

/// Stream chunk transformer wrapping the OpenAI-compatible converter for OpenAI
#[derive(Clone)]
pub struct OpenAiStreamChunkTransformer {
    pub provider_id: String,
    pub inner: crate::standards::openai::compat::streaming::OpenAiCompatibleEventConverter,
}

impl StreamChunkTransformer for OpenAiStreamChunkTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn convert_event(
        &self,
        event: eventsource_stream::Event,
    ) -> Pin<
        Box<
            dyn Future<Output = Vec<Result<crate::streaming::ChatStreamEvent, LlmError>>>
                + Send
                + Sync
                + '_,
        >,
    > {
        self.inner.convert_event(event)
    }

    fn handle_stream_end(&self) -> Option<Result<crate::streaming::ChatStreamEvent, LlmError>> {
        self.inner.handle_stream_end()
    }
}

/// Stream transformer for OpenAI Responses API using the standard Responses converter.
#[cfg(feature = "openai-responses")]
#[derive(Clone)]
pub struct OpenAiResponsesStreamChunkTransformer {
    pub provider_id: String,
    pub inner: crate::standards::openai::responses_sse::OpenAiResponsesEventConverter,
}

#[cfg(feature = "openai-responses")]
impl StreamChunkTransformer for OpenAiResponsesStreamChunkTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn convert_event(
        &self,
        event: eventsource_stream::Event,
    ) -> Pin<
        Box<
            dyn Future<Output = Vec<Result<crate::streaming::ChatStreamEvent, LlmError>>>
                + Send
                + Sync
                + '_,
        >,
    > {
        self.inner.convert_event(event)
    }

    fn handle_stream_end(&self) -> Option<Result<crate::streaming::ChatStreamEvent, LlmError>> {
        None
    }
}
