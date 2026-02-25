use async_trait::async_trait;
use futures::StreamExt;
use siumai::experimental::client::LlmClient;
use siumai::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

#[derive(Debug)]
struct CancelTestProvider {
    start_count: AtomicU32,
}

impl CancelTestProvider {
    fn new() -> Self {
        Self {
            start_count: AtomicU32::new(0),
        }
    }
}

#[async_trait]
impl ChatCapability for CancelTestProvider {
    async fn chat_with_tools(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        Ok(ChatResponse::new(MessageContent::Text("ok".to_string())))
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
            let response = ChatResponse::new(MessageContent::Text("done".to_string()));
            yield Ok(ChatStreamEvent::StreamEnd { response });
        };
        Ok(Box::pin(s))
    }
}

impl LlmClient for CancelTestProvider {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("mock-cancel")
    }
    fn supported_models(&self) -> Vec<String> {
        vec!["mock".into()]
    }
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_chat().with_streaming()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn LlmClient> {
        // Preserve counter value for visibility in clones
        let cloned = CancelTestProvider::new();
        cloned
            .start_count
            .store(self.start_count.load(Ordering::SeqCst), Ordering::SeqCst);
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

    let ChatStreamHandle { mut stream, cancel } = handle;

    // Read first delta
    let first = stream.next().await.expect("first event").expect("ok");
    match first {
        ChatStreamEvent::ContentDelta { ref delta, .. } => assert_eq!(delta, "A"),
        other => panic!("expected first delta, got: {other:?}"),
    }

    // Cancel twice (idempotent)
    cancel.cancel();
    cancel.cancel();

    // Ensure no more events (stream ends early; no StreamEnd)
    let rest: Vec<_> = stream.collect().await;
    assert!(
        rest.is_empty(),
        "cancellation should stop subsequent events"
    );
}

#[derive(Debug)]
struct SlowStartProvider {
    started: Arc<AtomicU32>,
    dropped: Arc<AtomicU32>,
}

impl SlowStartProvider {
    fn new() -> Self {
        Self {
            started: Arc::new(AtomicU32::new(0)),
            dropped: Arc::new(AtomicU32::new(0)),
        }
    }
}

#[async_trait]
impl ChatCapability for SlowStartProvider {
    async fn chat_with_tools(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        Ok(ChatResponse::new(MessageContent::Text("ok".to_string())))
    }

    async fn chat_stream(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        self.started.fetch_add(1, Ordering::SeqCst);

        struct DropGuard {
            dropped: Arc<AtomicU32>,
        }

        impl Drop for DropGuard {
            fn drop(&mut self) {
                self.dropped.fetch_add(1, Ordering::SeqCst);
            }
        }

        let _guard = DropGuard {
            dropped: Arc::clone(&self.dropped),
        };

        tokio::time::sleep(Duration::from_millis(500)).await;

        let s = async_stream::stream! {
            yield Ok(ChatStreamEvent::ContentDelta { delta: "late".to_string(), index: None });
            let response = ChatResponse::new(MessageContent::Text("done".to_string()));
            yield Ok(ChatStreamEvent::StreamEnd { response });
        };

        Ok(Box::pin(s))
    }
}

impl LlmClient for SlowStartProvider {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("mock-slow-start")
    }
    fn supported_models(&self) -> Vec<String> {
        vec!["mock".into()]
    }
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_chat().with_streaming()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(Self {
            started: Arc::clone(&self.started),
            dropped: Arc::clone(&self.dropped),
        })
    }
}

#[tokio::test]
async fn chat_stream_with_cancel_can_abort_handshake() {
    let provider = SlowStartProvider::new();
    let started = Arc::clone(&provider.started);
    let dropped = Arc::clone(&provider.dropped);

    let client = siumai::provider::Siumai::new(Arc::new(provider));

    let handle = client
        .chat_stream_with_cancel(vec![ChatMessage::user("hi").build()], None)
        .await
        .expect("start stream");

    let ChatStreamHandle { mut stream, cancel } = handle;

    let waiter = tokio::spawn(async move { stream.next().await });

    // Ensure the streaming handshake started (we are inside `chat_stream()`).
    tokio::time::timeout(Duration::from_millis(200), async {
        while started.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("handshake should start");

    cancel.cancel();

    let out = tokio::time::timeout(Duration::from_millis(200), waiter)
        .await
        .expect("cancel should stop handshake quickly")
        .expect("task ok");

    assert!(out.is_none(), "stream should end without yielding events");

    // Dropping the handshake future should drop our guard promptly.
    tokio::time::timeout(Duration::from_millis(200), async {
        while dropped.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("handshake future should be dropped after cancel");
}
