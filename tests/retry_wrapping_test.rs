use std::sync::atomic::{AtomicU32, Ordering};

use async_trait::async_trait;
use siumai::prelude::*;

#[derive(Debug)]
struct TestProvider {
    attempts: AtomicU32,
    fail_until: u32,
}

impl TestProvider {
    fn new(fail_until: u32) -> Self {
        Self {
            attempts: AtomicU32::new(0),
            fail_until,
        }
    }
}

#[async_trait]
impl ChatCapability for TestProvider {
    async fn chat_with_tools(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        let n = self.attempts.fetch_add(1, Ordering::SeqCst) + 1;
        if n <= self.fail_until {
            // Return retryable error (Server 500)
            Err(LlmError::ApiError {
                code: 500,
                message: format!("forced failure attempt {n}"),
                details: None,
            })
        } else {
            Ok(ChatResponse {
                id: Some("ok".into()),
                content: MessageContent::Text("success".into()),
                model: Some("mock-model".into()),
                usage: None,
                finish_reason: Some(FinishReason::Stop),
                tool_calls: None,
                thinking: None,
                metadata: std::collections::HashMap::new(),
            })
        }
    }

    async fn chat_stream(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "stream not supported in test".into(),
        ))
    }
}

impl LlmClient for TestProvider {
    fn provider_name(&self) -> &'static str {
        "mock"
    }

    fn supported_models(&self) -> Vec<String> {
        vec!["mock-model".into()]
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_chat()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        // Preserve current attempt count in the clone for safety
        let current = self.attempts.load(Ordering::SeqCst);
        let cloned = TestProvider::new(self.fail_until);
        cloned.attempts.store(current, Ordering::SeqCst);
        Box::new(cloned)
    }
}

#[tokio::test]
async fn test_siumai_retries_and_succeeds_on_second_attempt() {
    // First attempt fails, second succeeds
    let provider = TestProvider::new(1);
    let client = siumai::provider::Siumai::new(Box::new(provider))
        .with_retry_options(Some(RetryOptions::policy_default().with_max_attempts(3)));

    let msgs = vec![ChatMessage::user("hi").build()];
    let resp = client
        .chat_with_tools(msgs, None)
        .await
        .expect("should succeed");
    assert_eq!(resp.text().unwrap_or_default(), "success");
}

#[tokio::test]
async fn test_siumai_retry_respects_max_attempts_and_fails() {
    // Always fail, allow only 2 attempts => error
    let provider = TestProvider::new(u32::MAX);
    let client = siumai::provider::Siumai::new(Box::new(provider))
        .with_retry_options(Some(RetryOptions::policy_default().with_max_attempts(2)));

    let msgs = vec![ChatMessage::user("hi").build()];
    let err = client
        .chat_with_tools(msgs, None)
        .await
        .expect_err("should fail");

    match err {
        LlmError::ApiError { code, .. } => assert!(code >= 500),
        other => panic!("unexpected error: {other:?}"),
    }
}
