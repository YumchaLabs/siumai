//! Groq streaming fixtures tests
//!
//! These tests verify Groq streaming responses using real API response fixtures.
//! All fixtures are based on official Groq API documentation:
//! https://console.groq.com/docs/api-reference
//!
//! Groq chat streaming now goes through the external core provider
//! (`siumai-provider-groq`) and the OpenAI Chat standard, bridged via
//! `GroqSpec` into aggregator `ChatStreamEvent`s. The tests construct
//! the same streaming transformer bundle as runtime code and drive it
//! with SSE fixtures.

use siumai::LlmError;
use siumai::core::{ProviderContext, ProviderSpec};
use siumai::execution::transformers::stream::StreamChunkTransformer;
use siumai::streaming::{ChatStreamEvent, SseEventConverter};
use siumai::types::{ChatMessage, ChatRequest, CommonParams};

use crate::support;

/// Thin adapter to reuse the runtime StreamChunkTransformer in tests.
#[derive(Clone)]
struct GroqStdEventConverter {
    inner: std::sync::Arc<dyn StreamChunkTransformer>,
}

impl GroqStdEventConverter {
    fn new() -> Self {
        // Build a minimal ChatRequest; content is irrelevant for streaming shape.
        let req = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hello").build()])
            .common_params(CommonParams {
                model: "llama-3.3-70b-versatile".to_string(),
                ..Default::default()
            })
            .build();

        // Provider context mirroring real Groq configuration.
        let ctx = ProviderContext::new(
            "groq",
            "https://api.groq.com/openai/v1".to_string(),
            None,
            std::collections::HashMap::new(),
        );

        let spec = siumai::providers::groq::spec::GroqSpec;
        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let stream = bundle
            .stream
            .expect("Groq should provide streaming transformers");

        Self { inner: stream }
    }
}

impl SseEventConverter for GroqStdEventConverter {
    fn convert_event(
        &self,
        event: eventsource_stream::Event,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Vec<Result<ChatStreamEvent, LlmError>>>
                + Send
                + Sync
                + '_,
        >,
    > {
        self.inner.convert_event(event)
    }

    fn handle_stream_end(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
        self.inner.handle_stream_end()
    }
}

fn make_groq_converter() -> GroqStdEventConverter {
    GroqStdEventConverter::new()
}

#[tokio::test]
async fn groq_simple_content_stop_fixture() {
    let bytes = support::load_sse_fixture_as_bytes("tests/fixtures/groq/simple_content_stop.sse")
        .expect("load fixture");

    let converter = make_groq_converter();
    let events = support::collect_sse_events(bytes, converter).await;

    let mut content = String::new();

    for e in events {
        if let ChatStreamEvent::ContentDelta { delta, .. } = e {
            content.push_str(&delta);
        }
    }

    assert_eq!(content, "Hello world");
    // Usage and explicit StreamEnd are now handled at the core layer;
    // here we only assert that streaming preserves content.
}
