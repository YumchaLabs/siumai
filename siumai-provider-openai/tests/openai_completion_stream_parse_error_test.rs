#![cfg(feature = "openai")]

use futures_util::StreamExt;
use siumai_provider_openai::providers::openai::{OpenAiClient, OpenAiConfig};
use siumai_provider_openai::traits::CompletionCapability;
use siumai_provider_openai::types::{ChatStreamEvent, ChatStreamPart, CompletionRequest};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn make_client(base_url: &str) -> OpenAiClient {
    OpenAiClient::new(
        OpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model("gpt-3.5-turbo-instruct"),
        reqwest::Client::new(),
    )
}

#[tokio::test]
async fn openai_completion_parse_error_emits_stream_start_before_error_without_raw_chunks() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw("data: not-json\n\n", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let client = make_client(&format!("{}/v1", server.uri()));
    let mut stream = client
        .complete_stream(CompletionRequest::new("hello"))
        .await
        .expect("completion stream");

    let mut events = Vec::new();
    while let Some(item) = stream.next().await {
        events.push(item);
    }

    assert_eq!(events.len(), 3);
    match events.first().expect("stream-start event") {
        Ok(ChatStreamEvent::StreamStart { metadata }) => {
            assert_eq!(metadata.provider, "openai");
        }
        other => panic!("expected stream-start event, got {other:?}"),
    }
    assert!(matches!(
        events.get(1),
        Some(Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::StreamStart { warnings }
        })) if warnings.is_empty()
    ));
    assert!(matches!(
        events.get(2),
        Some(Err(siumai_provider_openai::LlmError::ParseError(message)))
            if message.contains("Failed to parse completion stream event")
    ));
}
