#![cfg(feature = "xai")]

use siumai::prelude::unified::*;
use std::path::Path;

fn fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("xai")
        .join("responses-stream")
        .join("text")
}

fn read_fixture_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .expect("read fixture file")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.to_string())
        .collect()
}

fn run_converter(lines: Vec<String>) -> Vec<ChatStreamEvent> {
    let conv =
        siumai_provider_xai::standards::openai::responses_sse::OpenAiResponsesEventConverter::new()
            .with_stream_parts_style(
                siumai_provider_xai::standards::openai::responses_sse::StreamPartsStyle::Xai,
            );

    let mut events: Vec<ChatStreamEvent> = Vec::new();
    for (i, line) in lines.into_iter().enumerate() {
        let ev = eventsource_stream::Event {
            event: "".to_string(),
            data: line,
            id: i.to_string(),
            retry: None,
        };

        let out = futures::executor::block_on(conv.convert_event(ev));
        for item in out {
            match item {
                Ok(evt) => events.push(evt),
                Err(err) => panic!("failed to convert chunk: {err:?}"),
            }
        }
    }

    while let Some(item) = conv.handle_stream_end() {
        match item {
            Ok(evt) => events.push(evt),
            Err(err) => panic!("failed to finalize stream: {err:?}"),
        }
    }

    events
}

fn custom_events_by_type(events: &[ChatStreamEvent], ty: &str) -> Vec<serde_json::Value> {
    events
        .iter()
        .filter_map(|e| match e {
            ChatStreamEvent::Custom { data, .. }
                if data.get("type").and_then(|v| v.as_str()) == Some(ty) =>
            {
                Some(data.clone())
            }
            _ => None,
        })
        .collect()
}

fn assert_xai_text_stream_fixture(fixture_file: &str) {
    let path = fixtures_dir().join(fixture_file);
    assert!(path.exists(), "fixture missing: {:?}", path);

    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let events = run_converter(lines);

    let stream_starts = custom_events_by_type(&events, "stream-start");
    let metadata = custom_events_by_type(&events, "response-metadata");
    let reasoning_starts = custom_events_by_type(&events, "reasoning-start");
    let reasoning_deltas = custom_events_by_type(&events, "reasoning-delta");
    let reasoning_ends = custom_events_by_type(&events, "reasoning-end");
    let text_starts = custom_events_by_type(&events, "text-start");
    let text_deltas = custom_events_by_type(&events, "text-delta");
    let text_ends = custom_events_by_type(&events, "text-end");
    let finishes = custom_events_by_type(&events, "finish");

    assert_eq!(stream_starts.len(), 1, "expected exactly one stream-start");
    assert_eq!(metadata.len(), 1, "expected exactly one response-metadata");

    let resp_id = metadata[0]
        .get("id")
        .and_then(|v| v.as_str())
        .expect("response-metadata.id");
    let expected_reasoning_id = format!("reasoning-rs_{resp_id}");
    let expected_text_id = format!("text-msg_{resp_id}");

    assert_eq!(
        reasoning_starts[0].get("id").and_then(|v| v.as_str()),
        Some(expected_reasoning_id.as_str())
    );
    assert_eq!(
        reasoning_ends[0].get("id").and_then(|v| v.as_str()),
        Some(expected_reasoning_id.as_str())
    );
    for evt in reasoning_starts
        .iter()
        .chain(reasoning_deltas.iter())
        .chain(reasoning_ends.iter())
    {
        assert_eq!(
            evt.get("id").and_then(|v| v.as_str()),
            Some(expected_reasoning_id.as_str())
        );
        assert!(
            evt.get("providerMetadata").is_none(),
            "xAI reasoning stream parts should omit providerMetadata"
        );
    }

    assert_eq!(
        text_starts[0].get("id").and_then(|v| v.as_str()),
        Some(expected_text_id.as_str())
    );
    assert_eq!(
        text_ends[0].get("id").and_then(|v| v.as_str()),
        Some(expected_text_id.as_str())
    );
    for evt in text_starts
        .iter()
        .chain(text_deltas.iter())
        .chain(text_ends.iter())
    {
        assert_eq!(
            evt.get("id").and_then(|v| v.as_str()),
            Some(expected_text_id.as_str())
        );
        assert!(
            evt.get("providerMetadata").is_none(),
            "xAI text stream parts should omit providerMetadata"
        );
    }

    assert_eq!(finishes.len(), 1, "expected exactly one finish");
    assert!(
        finishes[0].get("providerMetadata").is_none(),
        "xAI finish stream part should omit providerMetadata"
    );
    assert_eq!(
        finishes[0]
            .get("finishReason")
            .and_then(|r| r.get("raw"))
            .and_then(|v| v.as_str()),
        Some("completed")
    );
    assert_eq!(
        finishes[0]
            .get("finishReason")
            .and_then(|r| r.get("unified"))
            .and_then(|v| v.as_str()),
        Some("stop")
    );
    assert!(
        finishes[0]
            .get("usage")
            .and_then(|u| u.get("inputTokens"))
            .and_then(|t| t.get("cacheWrite"))
            .is_none(),
        "xAI finish usage.inputTokens.cacheWrite should be omitted"
    );
}

#[test]
fn xai_responses_text_stream_emits_vercel_aligned_text_and_reasoning() {
    assert_xai_text_stream_fixture("xai-text-streaming.1.chunks.txt");
}

#[test]
fn xai_responses_text_with_reasoning_stream_emits_vercel_aligned_text_and_reasoning() {
    assert_xai_text_stream_fixture("xai-text-with-reasoning-streaming.1.chunks.txt");
}

#[test]
fn xai_responses_text_with_reasoning_store_false_stream_emits_vercel_aligned_text_and_reasoning() {
    assert_xai_text_stream_fixture("xai-text-with-reasoning-streaming-store-false.1.chunks.txt");
}
