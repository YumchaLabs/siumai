#![cfg(feature = "openai")]

use siumai::prelude::unified::*;
use std::path::Path;

fn fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("responses-stream")
        .join("reasoning")
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
fn openai_responses_reasoning_stream_emits_reasoning_delta_with_encrypted_content_fixture() {
    let path = fixtures_dir().join("openai-reasoning-encrypted-content.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let events = run_converter(lines);

    let starts = custom_events_by_type(&events, "reasoning-start");
    let deltas = custom_events_by_type(&events, "reasoning-delta");
    let ends = custom_events_by_type(&events, "reasoning-end");

    assert!(!starts.is_empty(), "expected reasoning-start events");
    assert!(!deltas.is_empty(), "expected reasoning-delta events");
    assert!(!ends.is_empty(), "expected reasoning-end events");

    let start = &starts[0];
    let item_id = start
        .get("providerMetadata")
        .and_then(|m| m.get("openai"))
        .and_then(|o| o.get("itemId"))
        .and_then(|v| v.as_str())
        .expect("reasoning-start providerMetadata.openai.itemId");
    let expected_id = format!("{item_id}:0");
    assert_eq!(
        start.get("id").and_then(|v| v.as_str()),
        Some(expected_id.as_str())
    );

    let start_enc = start
        .get("providerMetadata")
        .and_then(|m| m.get("openai"))
        .and_then(|o| o.get("reasoningEncryptedContent"))
        .and_then(|v| v.as_str())
        .expect("reasoning-start providerMetadata.openai.reasoningEncryptedContent");
    assert!(!start_enc.is_empty(), "expected encrypted content on start");

    let end = &ends[0];
    assert_eq!(
        end.get("id").and_then(|v| v.as_str()),
        Some(expected_id.as_str())
    );
    let end_enc = end
        .get("providerMetadata")
        .and_then(|m| m.get("openai"))
        .and_then(|o| o.get("reasoningEncryptedContent"))
        .and_then(|v| v.as_str())
        .expect("reasoning-end providerMetadata.openai.reasoningEncryptedContent");
    assert!(!end_enc.is_empty(), "expected encrypted content on end");
}

#[test]
fn openai_responses_reasoning_stream_emits_start_and_end_for_empty_summary_encrypted() {
    let path = fixtures_dir().join("openai-reasoning-encrypted-empty-summary.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let events = run_converter(lines);

    let starts = custom_events_by_type(&events, "reasoning-start");
    let deltas = custom_events_by_type(&events, "reasoning-delta");
    let ends = custom_events_by_type(&events, "reasoning-end");

    assert_eq!(deltas.len(), 0, "expected no reasoning-delta events");
    assert_eq!(
        starts.len(),
        1,
        "expected exactly one reasoning-start event"
    );
    assert_eq!(ends.len(), 1, "expected exactly one reasoning-end event");

    let start = &starts[0];
    let item_id = start
        .get("providerMetadata")
        .and_then(|m| m.get("openai"))
        .and_then(|o| o.get("itemId"))
        .and_then(|v| v.as_str())
        .expect("reasoning-start providerMetadata.openai.itemId");
    let expected_id = format!("{item_id}:0");
    assert_eq!(
        start.get("id").and_then(|v| v.as_str()),
        Some(expected_id.as_str())
    );

    let start_enc = start
        .get("providerMetadata")
        .and_then(|m| m.get("openai"))
        .and_then(|o| o.get("reasoningEncryptedContent"))
        .and_then(|v| v.as_str())
        .expect("reasoning-start providerMetadata.openai.reasoningEncryptedContent");
    assert_eq!(start_enc, "encrypted_reasoning_data_abc123");

    let end = &ends[0];
    assert_eq!(
        end.get("id").and_then(|v| v.as_str()),
        Some(expected_id.as_str())
    );
    let end_enc = end
        .get("providerMetadata")
        .and_then(|m| m.get("openai"))
        .and_then(|o| o.get("reasoningEncryptedContent"))
        .and_then(|v| v.as_str())
        .expect("reasoning-end providerMetadata.openai.reasoningEncryptedContent");
    assert_eq!(end_enc, "encrypted_reasoning_data_final_def456");
}
