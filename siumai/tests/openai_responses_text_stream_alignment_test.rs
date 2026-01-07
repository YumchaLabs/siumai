#![cfg(feature = "openai")]

use siumai::prelude::unified::*;
use std::path::Path;

fn fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
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
    let conv = siumai::provider_ext::openai::ext::OpenAiResponsesEventConverter::new();

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
                if data.get("type") == Some(&serde_json::Value::String(ty.to_string())) =>
            {
                Some(data.clone())
            }
            _ => None,
        })
        .collect()
}

#[test]
fn openai_responses_text_stream_emits_stream_start_metadata_text_and_finish() {
    let path = fixtures_dir().join("openai-text-deltas.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let events = run_converter(lines);

    let stream_starts = custom_events_by_type(&events, "stream-start");
    let metadata = custom_events_by_type(&events, "response-metadata");
    let text_starts = custom_events_by_type(&events, "text-start");
    let text_deltas = custom_events_by_type(&events, "text-delta");
    let text_ends = custom_events_by_type(&events, "text-end");
    let finishes = custom_events_by_type(&events, "finish");

    assert_eq!(stream_starts.len(), 1, "expected exactly one stream-start");
    assert_eq!(
        stream_starts[0].get("warnings"),
        Some(&serde_json::json!([]))
    );

    assert_eq!(metadata.len(), 1, "expected exactly one response-metadata");
    assert_eq!(
        metadata[0].get("id").and_then(|v| v.as_str()),
        Some("resp_67c9a81b6a048190a9ee441c5755a4e8")
    );
    assert_eq!(
        metadata[0].get("modelId").and_then(|v| v.as_str()),
        Some("gpt-4o-2024-07-18")
    );
    assert_eq!(
        metadata[0].get("timestamp").and_then(|v| v.as_str()),
        Some("2025-03-06T13:50:19.000Z")
    );

    assert_eq!(text_starts.len(), 1, "expected exactly one text-start");
    assert_eq!(
        text_starts[0].get("id").and_then(|v| v.as_str()),
        Some("msg_67c9a81dea8c8190b79651a2b3adf91e")
    );

    assert_eq!(text_deltas.len(), 2, "expected exactly two text-delta");
    assert_eq!(
        text_deltas[0].get("delta").and_then(|v| v.as_str()),
        Some("Hello,")
    );
    assert_eq!(
        text_deltas[1].get("delta").and_then(|v| v.as_str()),
        Some(" World!")
    );
    for delta in &text_deltas {
        assert_eq!(
            delta.get("id").and_then(|v| v.as_str()),
            Some("msg_67c9a81dea8c8190b79651a2b3adf91e")
        );
    }

    assert_eq!(text_ends.len(), 1, "expected exactly one text-end");
    assert_eq!(
        text_ends[0].get("id").and_then(|v| v.as_str()),
        Some("msg_67c9a8787f4c8190b49c858d4c1cf20c")
    );

    assert_eq!(finishes.len(), 1, "expected exactly one finish");
    assert_eq!(
        finishes[0]
            .get("providerMetadata")
            .and_then(|m| m.get("openai"))
            .and_then(|o| o.get("responseId"))
            .and_then(|v| v.as_str()),
        Some("resp_67c9a81b6a048190a9ee441c5755a4e8")
    );
    assert_eq!(
        finishes[0]
            .get("finishReason")
            .and_then(|r| r.get("unified"))
            .and_then(|v| v.as_str()),
        Some("stop")
    );
    assert_eq!(
        finishes[0]
            .get("usage")
            .and_then(|u| u.get("inputTokens"))
            .and_then(|u| u.get("total"))
            .and_then(|v| v.as_u64()),
        Some(543)
    );
    assert_eq!(
        finishes[0]
            .get("usage")
            .and_then(|u| u.get("outputTokens"))
            .and_then(|u| u.get("reasoning"))
            .and_then(|v| v.as_u64()),
        Some(123)
    );
}
