use super::*;

/// Stream chunk transformer wrapping the existing GeminiEventConverter
#[derive(Clone)]
pub struct GeminiStreamChunkTransformer {
    pub provider_id: String,
    pub inner: streaming::GeminiEventConverter,
}

impl StreamChunkTransformer for GeminiStreamChunkTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn convert_event(
        &self,
        event: Event,
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
