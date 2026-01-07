#![cfg(feature = "openai")]

use siumai::prelude::unified::*;
use std::path::Path;

fn read_fixture_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .expect("read fixture file")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.to_string())
        .collect()
}

fn run_converter(lines: Vec<String>, tools: Vec<Tool>) -> Vec<ChatStreamEvent> {
    let conv = siumai::provider_ext::openai::ext::OpenAiResponsesEventConverter::new()
        .with_request_tools(&tools);

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
fn openai_responses_shell_stream_emits_response_metadata_for_each_response() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("responses-stream")
        .join("shell")
        .join("openai-shell-tool.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let tools = vec![Tool::ProviderDefined(ProviderDefinedTool::new(
        "openai.shell",
        "shell",
    ))];

    let events = run_converter(read_fixture_lines(&path), tools);

    let stream_starts = custom_events_by_type(&events, "stream-start");
    let metadata = custom_events_by_type(&events, "response-metadata");

    assert_eq!(stream_starts.len(), 1, "expected exactly one stream-start");
    assert_eq!(
        metadata.len(),
        2,
        "expected response-metadata for each response.created"
    );

    let ids: Vec<&str> = metadata
        .iter()
        .filter_map(|m| m.get("id").and_then(|v| v.as_str()))
        .collect();

    assert!(
        ids.contains(&"resp_0434d6d64b12b08900692f639c40408195a50fd07b77ce08a7"),
        "expected first response id in response-metadata"
    );
    assert!(
        ids.contains(&"resp_0434d6d64b12b08900692f639d784481959af65f985b9c13e2"),
        "expected second response id in response-metadata"
    );
}
