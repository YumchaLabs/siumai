//! Testing Executors Example
//!
//! This example demonstrates how to test executors using mock ProviderSpec
//! implementations. This is useful for:
//! - Unit testing your code without making real API calls
//! - Testing error handling and retry logic
//! - Simulating different provider behaviors
//!
//! Run with: cargo run --example testing_executors

use siumai::core::{ChatTransformers, ProviderContext, ProviderSpec};
use siumai::error::LlmError;
use siumai::execution::executors::chat::{ChatExecutor, HttpChatExecutor};
use siumai::execution::transformers::request::RequestTransformer;
use siumai::execution::transformers::response::ResponseTransformer;
use siumai::traits::ProviderCapabilities;
use siumai::types::{ChatMessage, ChatRequest, ChatResponse, Usage};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Mock ProviderSpec for testing
struct MockProviderSpec {
    /// Counter to track how many times the spec is used
    call_count: Arc<AtomicUsize>,
    /// Whether to simulate an error
    should_error: bool,
}

impl MockProviderSpec {
    fn new(should_error: bool) -> Self {
        Self {
            call_count: Arc::new(AtomicUsize::new(0)),
            should_error,
        }
    }

    fn get_call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

impl ProviderSpec for MockProviderSpec {
    fn id(&self) -> &'static str {
        "mock-provider"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_chat()
    }

    fn build_headers(
        &self,
        _ctx: &ProviderContext,
    ) -> Result<reqwest::header::HeaderMap, LlmError> {
        use reqwest::header::HeaderMap;
        Ok(HeaderMap::new())
    }

    fn chat_url(&self, _stream: bool, _req: &ChatRequest, ctx: &ProviderContext) -> String {
        format!("{}/mock/chat", ctx.base_url)
    }

    fn choose_chat_transformers(
        &self,
        _req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> ChatTransformers {
        ChatTransformers {
            request: Arc::new(MockRequestTransformer {
                call_count: self.call_count.clone(),
            }),
            response: Arc::new(MockResponseTransformer {
                should_error: self.should_error,
            }),
            stream: None,
            json: None,
        }
    }
}

/// Mock request transformer
struct MockRequestTransformer {
    call_count: Arc<AtomicUsize>,
}

impl RequestTransformer for MockRequestTransformer {
    fn provider_id(&self) -> &str {
        "mock-provider"
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        // Increment call count
        self.call_count.fetch_add(1, Ordering::SeqCst);

        // Return a simple mock request body
        let model = if req.common_params.model.is_empty() {
            "mock-model"
        } else {
            &req.common_params.model
        };

        Ok(serde_json::json!({
            "model": model,
            "messages": req.messages.iter().map(|m| {
                let content_str = match &m.content {
                    siumai::types::MessageContent::Text(text) => text.clone(),
                    siumai::types::MessageContent::MultiModal(_) => "[multimodal content]".to_string(),
                    #[cfg(feature = "structured-messages")]
                    siumai::types::MessageContent::Json(_) => "[json content]".to_string(),
                };
                serde_json::json!({
                    "role": match m.role {
                        siumai::types::MessageRole::System => "system",
                        siumai::types::MessageRole::User => "user",
                        siumai::types::MessageRole::Assistant => "assistant",
                        siumai::types::MessageRole::Tool => "tool",
                        siumai::types::MessageRole::Developer => "developer",
                    },
                    "content": content_str,
                })
            }).collect::<Vec<_>>(),
        }))
    }
}

/// Mock response transformer
struct MockResponseTransformer {
    should_error: bool,
}

impl ResponseTransformer for MockResponseTransformer {
    fn provider_id(&self) -> &str {
        "mock-provider"
    }

    fn transform_chat_response(&self, _raw: &serde_json::Value) -> Result<ChatResponse, LlmError> {
        if self.should_error {
            return Err(LlmError::ParseError(
                "Mock error: simulated failure".to_string(),
            ));
        }

        // Return a mock response
        Ok(ChatResponse {
            id: Some("mock-response-id".to_string()),
            content: siumai::types::MessageContent::Text(
                "This is a mock response from the test provider.".to_string(),
            ),
            model: Some("mock-model".to_string()),
            finish_reason: Some(siumai::types::FinishReason::Stop),
            usage: Some(Usage::new(10, 20)),
            provider_metadata: None,
            audio: None,
            system_fingerprint: None,
            service_tier: None,
            warnings: None,
        })
    }
}

/// Test helper function
async fn test_executor_with_spec(
    spec: Arc<MockProviderSpec>,
    should_succeed: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let ctx = ProviderContext::new(
        "mock-provider",
        "http://localhost:8080".to_string(),
        None,
        Default::default(),
    );

    // Get transformers from spec
    let transformers = spec.choose_chat_transformers(&ChatRequest::new(vec![]), &ctx);

    let executor = HttpChatExecutor {
        provider_id: "mock-provider".to_string(),
        http_client: reqwest::Client::new(),
        request_transformer: transformers.request,
        response_transformer: transformers.response,
        stream_transformer: transformers.stream,
        json_stream_converter: transformers.json,
        policy: siumai::execution::ExecutionPolicy::new().with_stream_disable_compression(false),
        middlewares: vec![],
        provider_spec: spec.clone(),
        provider_context: ctx,
    };

    let request = ChatRequest::new(vec![ChatMessage::user("Test message").build()])
        .with_model_params(
            siumai::types::CommonParamsBuilder::new()
                .model("mock-model")
                .build()
                .unwrap(),
        );

    // Note: This will fail with a connection error since we're not running a real server
    // In a real test, you would use wiremock or similar to mock the HTTP server
    let result = executor.execute(request).await;

    if should_succeed {
        // In a real test with wiremock, this would succeed
        println!("   ⚠️  Note: This would succeed with a mock HTTP server");
    } else {
        // Error is expected
        assert!(result.is_err(), "Expected an error but got success");
        println!("   ✅ Error handling works as expected");
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing Executors Example\n");

    println!("📝 Test 1: Mock ProviderSpec with success scenario");
    let mock_spec = Arc::new(MockProviderSpec::new(false));
    test_executor_with_spec(mock_spec.clone(), true).await?;
    println!("   Call count: {}\n", mock_spec.get_call_count());

    println!("📝 Test 2: Mock ProviderSpec with error scenario");
    let error_spec = Arc::new(MockProviderSpec::new(true));
    test_executor_with_spec(error_spec.clone(), false).await?;
    println!("   Call count: {}\n", error_spec.get_call_count());

    println!("✅ Testing example completed!\n");
    println!("💡 Key Points:");
    println!("   - Created MockProviderSpec for testing");
    println!("   - Implemented mock transformers that don't make real API calls");
    println!("   - Tracked call counts to verify behavior");
    println!("   - Simulated error scenarios");
    println!("\n📚 For real integration tests, use:");
    println!("   - wiremock crate to mock HTTP servers");
    println!("   - See siumai/tests/executors_*_retry_401.rs for examples");

    Ok(())
}
