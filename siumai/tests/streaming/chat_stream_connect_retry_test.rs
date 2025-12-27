use async_trait::async_trait;
use futures::StreamExt;
use siumai::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Debug)]
struct ConnectRetryProvider {
    attempts: AtomicU32,
    fail_until: u32,
}

impl ConnectRetryProvider {
    fn new(fail_until: u32) -> Self {
        Self {
            attempts: AtomicU32::new(0),
            fail_until,
        }
    }
}

#[async_trait]
impl ChatCapability for ConnectRetryProvider {
    async fn chat_with_tools(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        Ok(ChatResponse::new(siumai::types::MessageContent::Text(
            "ok".to_string(),
        )))
    }

    async fn chat_stream(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        let n = self.attempts.fetch_add(1, Ordering::SeqCst) + 1;
        if n <= self.fail_until {
            return Err(LlmError::api_error(
                500,
                format!("connect failure attempt {n}"),
            ));
        }
        let s = async_stream::stream! {
            yield Ok(ChatStreamEvent::ContentDelta { delta: "hello".to_string(), index: None });
            let response = ChatResponse::new(siumai::types::MessageContent::Text(
                "done".to_string(),
            ));
            yield Ok(ChatStreamEvent::StreamEnd { response });
        };
        Ok(Box::pin(s))
    }
}

impl LlmClient for ConnectRetryProvider {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("mock-retry-stream")
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
        let cloned = ConnectRetryProvider::new(self.fail_until);
        cloned
            .attempts
            .store(self.attempts.load(Ordering::SeqCst), Ordering::SeqCst);
        Box::new(cloned)
    }
}

#[tokio::test]
async fn chat_stream_retries_connect_then_succeeds() {
    // Fail once, then succeed on the second attempt
    let provider = ConnectRetryProvider::new(1);
    let client = siumai::provider::Siumai::new(std::sync::Arc::new(provider))
        .with_retry_options(Some(RetryOptions::policy_default().with_max_attempts(3)));

    let stream = client
        .chat_stream(vec![ChatMessage::user("hi").build()], None)
        .await
        .expect("stream should be established after retry");

    let events: Vec<_> = stream.collect().await;
    assert!(matches!(
        events.first(),
        Some(Ok(ChatStreamEvent::ContentDelta { .. }))
    ));
    assert!(matches!(
        events.last(),
        Some(Ok(ChatStreamEvent::StreamEnd { .. }))
    ));
}
