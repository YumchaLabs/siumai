//! Tests for middleware hooks
//!
//! This test verifies that all middleware hooks are called correctly:
//! - transform_json_body
//! - on_stream_end
//! - on_stream_error

use siumai::error::LlmError;
use siumai::execution::middleware::language_model::LanguageModelMiddleware;
use siumai::streaming::ChatStreamEvent;
use siumai::types::{ChatRequest, ChatResponse, FinishReason, MessageContent};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Mock middleware that tracks hook calls
#[derive(Clone)]
struct TrackingMiddleware {
    transform_json_body_called: Arc<AtomicBool>,
    on_stream_end_called: Arc<AtomicBool>,
    on_stream_error_called: Arc<AtomicBool>,
    call_count: Arc<AtomicUsize>,
}

impl TrackingMiddleware {
    fn new() -> Self {
        Self {
            transform_json_body_called: Arc::new(AtomicBool::new(false)),
            on_stream_end_called: Arc::new(AtomicBool::new(false)),
            on_stream_error_called: Arc::new(AtomicBool::new(false)),
            call_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn was_transform_json_body_called(&self) -> bool {
        self.transform_json_body_called.load(Ordering::SeqCst)
    }

    fn was_on_stream_end_called(&self) -> bool {
        self.on_stream_end_called.load(Ordering::SeqCst)
    }

    fn was_on_stream_error_called(&self) -> bool {
        self.on_stream_error_called.load(Ordering::SeqCst)
    }

    fn get_call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }

    #[allow(dead_code)]
    fn reset(&self) {
        self.transform_json_body_called
            .store(false, Ordering::SeqCst);
        self.on_stream_end_called.store(false, Ordering::SeqCst);
        self.on_stream_error_called.store(false, Ordering::SeqCst);
        self.call_count.store(0, Ordering::SeqCst);
    }
}

impl LanguageModelMiddleware for TrackingMiddleware {
    fn transform_json_body(
        &self,
        _req: &ChatRequest,
        body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        self.transform_json_body_called
            .store(true, Ordering::SeqCst);
        self.call_count.fetch_add(1, Ordering::SeqCst);

        // Add a marker to the body to prove transformation happened
        body["_middleware_marker"] = serde_json::json!("transformed");

        Ok(())
    }

    fn on_stream_end(&self, _req: &ChatRequest, _response: &ChatResponse) -> Result<(), LlmError> {
        self.on_stream_end_called.store(true, Ordering::SeqCst);
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn on_stream_error(&self, _req: &ChatRequest, _error: &LlmError) -> Result<(), LlmError> {
        self.on_stream_error_called.store(true, Ordering::SeqCst);
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[test]
fn test_transform_json_body_hook() {
    let middleware = TrackingMiddleware::new();
    let req = ChatRequest::new(vec![]);
    let mut body = serde_json::json!({
        "model": "test-model",
        "messages": []
    });

    // Call the hook
    let result = middleware.transform_json_body(&req, &mut body);

    // Verify
    assert!(result.is_ok());
    assert!(middleware.was_transform_json_body_called());
    assert_eq!(middleware.get_call_count(), 1);
    assert_eq!(
        body.get("_middleware_marker").and_then(|v| v.as_str()),
        Some("transformed")
    );
}

#[test]
fn test_transform_json_body_chain() {
    let middleware1 = TrackingMiddleware::new();
    let middleware2 = TrackingMiddleware::new();

    let middlewares: Vec<Arc<dyn LanguageModelMiddleware>> =
        vec![Arc::new(middleware1.clone()), Arc::new(middleware2.clone())];

    let req = ChatRequest::new(vec![]);
    let mut body = serde_json::json!({
        "model": "test-model"
    });

    // Apply chain
    let result = siumai::execution::middleware::language_model::apply_json_body_transform_chain(
        &middlewares,
        &req,
        &mut body,
    );

    // Verify both middlewares were called
    assert!(result.is_ok());
    assert!(middleware1.was_transform_json_body_called());
    assert!(middleware2.was_transform_json_body_called());
    assert_eq!(middleware1.get_call_count(), 1);
    assert_eq!(middleware2.get_call_count(), 1);
}

#[test]
fn test_on_stream_end_hook() {
    let middleware = TrackingMiddleware::new();
    let req = ChatRequest::new(vec![]);
    let response = ChatResponse {
        id: Some("test-123".to_string()),
        content: MessageContent::Text("Hello".to_string()),
        model: Some("test-model".to_string()),
        usage: None,
        finish_reason: Some(FinishReason::Stop),
        provider_metadata: None,
        warnings: None,
        audio: None,
        system_fingerprint: None,
        service_tier: None,
    };

    // Call the hook
    let result = middleware.on_stream_end(&req, &response);

    // Verify
    assert!(result.is_ok());
    assert!(middleware.was_on_stream_end_called());
    assert_eq!(middleware.get_call_count(), 1);
}

#[test]
fn test_on_stream_error_hook() {
    let middleware = TrackingMiddleware::new();
    let req = ChatRequest::new(vec![]);
    let error = LlmError::InternalError("Test error".to_string());

    // Call the hook
    let result = middleware.on_stream_error(&req, &error);

    // Verify
    assert!(result.is_ok());
    assert!(middleware.was_on_stream_error_called());
    assert_eq!(middleware.get_call_count(), 1);
}

#[test]
fn test_apply_stream_event_chain_with_stream_end() {
    let middleware = TrackingMiddleware::new();
    let middlewares: Vec<Arc<dyn LanguageModelMiddleware>> = vec![Arc::new(middleware.clone())];

    let req = ChatRequest::new(vec![]);
    let response = ChatResponse {
        id: Some("test-123".to_string()),
        content: MessageContent::Text("Done".to_string()),
        model: Some("test-model".to_string()),
        usage: None,
        finish_reason: Some(FinishReason::Stop),
        provider_metadata: None,
        warnings: None,
        audio: None,
        system_fingerprint: None,
        service_tier: None,
    };

    let event = ChatStreamEvent::StreamEnd { response };

    // Apply chain
    let result = siumai::execution::middleware::language_model::apply_stream_event_chain(
        &middlewares,
        &req,
        event,
    );

    // Verify on_stream_end was called
    assert!(result.is_ok());
    assert!(middleware.was_on_stream_end_called());
}

#[test]
fn test_apply_stream_event_chain_with_error() {
    let middleware = TrackingMiddleware::new();
    let middlewares: Vec<Arc<dyn LanguageModelMiddleware>> = vec![Arc::new(middleware.clone())];

    let req = ChatRequest::new(vec![]);
    let event = ChatStreamEvent::Error {
        error: "Stream error".to_string(),
    };

    // Apply chain
    let result = siumai::execution::middleware::language_model::apply_stream_event_chain(
        &middlewares,
        &req,
        event,
    );

    // Verify on_stream_error was called
    assert!(result.is_ok());
    assert!(middleware.was_on_stream_error_called());
}

#[test]
fn test_middleware_hook_error_propagation() {
    /// Middleware that fails in transform_json_body
    #[derive(Clone)]
    struct FailingMiddleware;

    impl LanguageModelMiddleware for FailingMiddleware {
        fn transform_json_body(
            &self,
            _req: &ChatRequest,
            _body: &mut serde_json::Value,
        ) -> Result<(), LlmError> {
            Err(LlmError::InternalError("Transform failed".to_string()))
        }
    }

    let middleware = FailingMiddleware;
    let req = ChatRequest::new(vec![]);
    let mut body = serde_json::json!({});

    let result = middleware.transform_json_body(&req, &mut body);

    assert!(result.is_err());
    if let Err(LlmError::InternalError(msg)) = result {
        assert_eq!(msg, "Transform failed");
    } else {
        panic!("Expected InternalError");
    }
}

#[test]
fn test_multiple_middlewares_execution_order() {
    /// Middleware that appends to a field
    #[derive(Clone)]
    struct AppendingMiddleware {
        tag: String,
    }

    impl AppendingMiddleware {
        fn new(tag: impl Into<String>) -> Self {
            Self { tag: tag.into() }
        }
    }

    impl LanguageModelMiddleware for AppendingMiddleware {
        fn transform_json_body(
            &self,
            _req: &ChatRequest,
            body: &mut serde_json::Value,
        ) -> Result<(), LlmError> {
            let current = body
                .get("tags")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            body["tags"] = serde_json::Value::String(format!("{}{}", current, self.tag));
            Ok(())
        }
    }

    let mw1 = Arc::new(AppendingMiddleware::new("A"));
    let mw2 = Arc::new(AppendingMiddleware::new("B"));
    let mw3 = Arc::new(AppendingMiddleware::new("C"));

    let middlewares: Vec<Arc<dyn LanguageModelMiddleware>> = vec![mw1, mw2, mw3];

    let req = ChatRequest::new(vec![]);
    let mut body = serde_json::json!({});

    let result = siumai::execution::middleware::language_model::apply_json_body_transform_chain(
        &middlewares,
        &req,
        &mut body,
    );

    assert!(result.is_ok());
    assert_eq!(
        body.get("tags").and_then(|v| v.as_str()),
        Some("ABC"),
        "Middlewares should execute in order"
    );
}
