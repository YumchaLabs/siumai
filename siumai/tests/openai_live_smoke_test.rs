#![cfg(feature = "openai")]

use futures::StreamExt;
use siumai::embedding::{self, BatchEmbeddingRequest, EmbedOptions};
use siumai::prelude::unified::*;

fn openai_live_enabled() -> bool {
    matches!(
        std::env::var("OPENAI_API_KEY"),
        Ok(value) if !value.trim().is_empty() && value != "demo-key"
    )
}

fn live_text_request(prompt: &str) -> ChatRequest {
    ChatRequest::builder()
        .message(user!(prompt))
        .temperature(0.0)
        .max_tokens(64)
        .build()
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and live network access"]
async fn openai_gpt_5_2_registry_text_and_stream_smoke() {
    if !openai_live_enabled() {
        eprintln!("[skip] OPENAI_API_KEY not set, skipping live OpenAI smoke");
        return;
    }

    let model = registry::global()
        .language_model("openai:gpt-5.2")
        .expect("registry model");

    let response = text::generate(
        &model,
        live_text_request("Reply with the exact string SIUMAI_OK and nothing else."),
        text::GenerateOptions::default(),
    )
    .await
    .expect("openai gpt-5.2 non-stream response");

    let content = response.content_text().unwrap_or_default();
    assert!(
        content.contains("SIUMAI_OK"),
        "expected SIUMAI_OK in response, got: {content}"
    );

    let mut stream = text::stream(
        &model,
        live_text_request("Reply with the exact string STREAM_OK and nothing else."),
        text::StreamOptions::default(),
    )
    .await
    .expect("openai gpt-5.2 stream");

    let mut collected = String::new();
    let mut saw_stream_end = false;

    while let Some(event) = stream.next().await {
        match event.expect("stream event") {
            ChatStreamEvent::ContentDelta { delta, .. } => collected.push_str(&delta),
            ChatStreamEvent::StreamEnd { .. } => {
                saw_stream_end = true;
                break;
            }
            _ => {}
        }
    }

    assert!(saw_stream_end, "expected StreamEnd from live stream");
    assert!(
        collected.contains("STREAM_OK"),
        "expected STREAM_OK in stream, got: {collected}"
    );
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and live network access"]
async fn openai_embedding_batch_registry_smoke() {
    if !openai_live_enabled() {
        eprintln!("[skip] OPENAI_API_KEY not set, skipping live OpenAI embedding smoke");
        return;
    }

    let model = registry::global()
        .embedding_model("openai:text-embedding-3-small")
        .expect("registry embedding model");

    let request = BatchEmbeddingRequest::new(vec![
        EmbeddingRequest::single("siumai live smoke request A"),
        EmbeddingRequest::single("siumai live smoke request B"),
    ]);

    let response = embedding::embed_many(&model, request, EmbedOptions::default())
        .await
        .expect("openai embedding batch response");

    assert_eq!(response.responses.len(), 2);

    for item in response.responses {
        let item = item.expect("embedding response");
        assert_eq!(item.embeddings.len(), 1);
        assert!(
            !item.embeddings[0].is_empty(),
            "expected non-empty embedding vector"
        );
    }
}
