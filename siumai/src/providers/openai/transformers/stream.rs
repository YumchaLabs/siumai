//! Stream chunk transformer wrappers for OpenAI and OpenAI Responses APIs

use crate::error::LlmError;
use crate::streaming::SseEventConverter;
use crate::execution::transformers::stream::StreamChunkTransformer;
use std::future::Future;
use std::pin::Pin;

/// Stream chunk transformer wrapping the OpenAI-compatible converter for OpenAI
#[derive(Clone)]
pub struct OpenAiStreamChunkTransformer {
    pub provider_id: String,
    pub inner: crate::providers::openai_compatible::streaming::OpenAiCompatibleEventConverter,
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

/// Stream transformer for OpenAI Responses API using existing converter
#[derive(Clone)]
pub struct OpenAiResponsesStreamChunkTransformer {
    pub provider_id: String,
    pub inner: crate::providers::openai::responses::OpenAiResponsesEventConverter,
}

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
