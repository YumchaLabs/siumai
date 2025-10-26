use async_trait::async_trait;
use futures::StreamExt;
use siumai::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

#[derive(Debug)]
struct CancelTestProvider {
    start_count: AtomicU32,
}

impl CancelTestProvider {
    fn new() -> Self {
        Self { start_count: AtomicU32::new(0) }
    }
}

#[async_trait]
impl ChatCapability for CancelTestProvider {
    async fn chat_with_tools(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        Ok(ChatResponse::text_only("ok"))
    }

    async fn chat_stream(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        self.start_count.fetch_add(1, Ordering::SeqCst);
        // Build a slow stream: delta("A"), sleep, delta("B"), sleep, end
        let s = async_stream::stream! {
            yield Ok(ChatStreamEvent::ContentDelta { delta: "A".to_string(), index: None });
            tokio::time::sleep(Duration::from_millis(25)).await;
            yield Ok(ChatStreamEvent::ContentDelta { delta: "B".to_string(), index: None });
            tokio::time::sleep(Duration::from_millis(25)).await;
            let response = ChatResponse::text_only("done");
            yield Ok(ChatStreamEvent::StreamEnd { response });
        };
        Ok(Box::pin(s))
    }
}

impl LlmClient for CancelTestProvider {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> { std::borrow::Cow::Borrowed("mock-cancel") }
    fn supported_models(&self) -> Vec<String> { vec!["mock".into()] }
    fn capabilities(&self) -> ProviderCapabilities { ProviderCapabilities::new().with_chat().with_streaming() }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn clone_box(&self) -> Box<dyn LlmClient> {
        // Preserve counter value for visibility in clones
        let cloned = CancelTestProvider::new();
        cloned.start_count.store(self.start_count.load(Ordering::SeqCst), Ordering::SeqCst);
        Box::new(cloned)
    }
}

#[tokio::test]
async fn chat_stream_with_cancel_stops_further_events() {
    let provider = CancelTestProvider::new();
    let client = siumai::provider::Siumai::new(std::sync::Arc::new(provider));

    let handle = client
        .chat_stream_with_cancel(vec![ChatMessage::user("hi").build()], None)
        .await
        .expect("start stream");

    futures::pin_mut!(handle.stream);

    // Read first delta
    let first = handle.stream.next().await.expect("first event").expect("ok");
    match first {
        ChatStreamEvent::ContentDelta { ref delta, .. } => assert_eq!(delta, "A"),
        other => panic!("expected first delta, got: {other:?}"),
    }

    // Cancel twice (idempotent)
    handle.cancel.cancel();
    handle.cancel.cancel();

    // Ensure no more events (stream ends early; no StreamEnd)
    let rest: Vec<_> = handle.stream.collect().await;
    assert!(rest.is_empty(), "cancellation should stop subsequent events");
}

