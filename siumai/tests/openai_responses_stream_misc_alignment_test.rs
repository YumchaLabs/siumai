#![cfg(feature = "openai")]

use siumai::prelude::unified::*;
use std::path::Path;

fn fixtures_dir(subdir: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("responses-stream")
        .join(subdir)
}

fn read_fixture_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .expect("read fixture file")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.to_string())
        .collect()
}

fn run_converter(
    conv: siumai::provider_ext::openai::ext::OpenAiResponsesEventConverter,
    lines: Vec<String>,
) -> Vec<ChatStreamEvent> {
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
fn openai_responses_incomplete_stream_emits_finish_reason_raw() {
    let path = fixtures_dir("misc").join("openai-incomplete-finish-reason.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let conv = siumai::provider_ext::openai::ext::OpenAiResponsesEventConverter::new();
    let events = run_converter(conv, lines);

    let finishes = custom_events_by_type(&events, "finish");
    assert_eq!(finishes.len(), 1, "expected exactly one finish");

    assert_eq!(
        finishes[0]
            .get("finishReason")
            .and_then(|r| r.get("raw"))
            .and_then(|v| v.as_str()),
        Some("max_output_tokens")
    );
    assert_eq!(
        finishes[0]
            .get("finishReason")
            .and_then(|r| r.get("unified"))
            .and_then(|v| v.as_str()),
        Some("length")
    );
}

#[test]
fn openai_responses_logprobs_stream_emits_finish_provider_metadata_logprobs() {
    let path = fixtures_dir("misc").join("openai-logprobs.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let conv = siumai::provider_ext::openai::ext::OpenAiResponsesEventConverter::new();
    let events = run_converter(conv, lines);

    let finishes = custom_events_by_type(&events, "finish");
    assert_eq!(finishes.len(), 1, "expected exactly one finish");

    let logprobs = finishes[0]
        .get("providerMetadata")
        .and_then(|m| m.get("openai"))
        .and_then(|o| o.get("logprobs"))
        .and_then(|v| v.as_array())
        .expect("expected providerMetadata.openai.logprobs array");
    assert_eq!(logprobs.len(), 1, "expected one output_text logprobs block");

    let first_block = logprobs[0]
        .as_array()
        .expect("expected inner logprobs array");
    assert_eq!(first_block.len(), 1, "expected one token logprobs entry");

    assert_eq!(
        first_block[0].get("token").and_then(|v| v.as_str()),
        Some("N")
    );

    let top = first_block[0]
        .get("top_logprobs")
        .and_then(|v| v.as_array())
        .expect("expected top_logprobs array");
    assert!(
        top.iter()
            .any(|t| t.get("token").and_then(|v| v.as_str()) == Some("Please")),
        "expected top_logprobs to include 'Please'"
    );
}

#[test]
fn openai_responses_stream_can_emit_azure_provider_metadata_key() {
    let path = fixtures_dir("text").join("openai-text-deltas.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let conv = siumai::provider_ext::openai::ext::OpenAiResponsesEventConverter::new()
        .with_provider_metadata_key("azure");
    let events = run_converter(conv, lines);

    let text_starts = custom_events_by_type(&events, "text-start");
    assert_eq!(text_starts.len(), 1, "expected exactly one text-start");

    let provider_meta = text_starts[0]
        .get("providerMetadata")
        .and_then(|v| v.as_object())
        .expect("expected providerMetadata object");
    assert!(provider_meta.contains_key("azure"));
    assert!(!provider_meta.contains_key("openai"));

    let finishes = custom_events_by_type(&events, "finish");
    assert_eq!(finishes.len(), 1, "expected exactly one finish");
    let finish_meta = finishes[0]
        .get("providerMetadata")
        .and_then(|v| v.as_object())
        .expect("expected finish providerMetadata object");
    assert!(finish_meta.contains_key("azure"));
    assert!(!finish_meta.contains_key("openai"));
}
