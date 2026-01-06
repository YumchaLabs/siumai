#![cfg(feature = "anthropic")]

use siumai::prelude::unified::*;
use std::path::Path;

fn fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("anthropic")
        .join("messages-stream")
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
    use siumai::experimental::standards::anthropic::params::AnthropicParams;
    use siumai::experimental::standards::anthropic::streaming::AnthropicEventConverter;
    use siumai::prelude::unified::SseEventConverter;

    let conv = AnthropicEventConverter::new(AnthropicParams::default());

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

fn tool_events(events: &[ChatStreamEvent], kind: &str, tool_name: &str) -> Vec<serde_json::Value> {
    events
        .iter()
        .filter_map(|e| match e {
            ChatStreamEvent::Custom { data, .. }
                if data.get("type") == Some(&serde_json::json!(kind))
                    && data.get("toolName") == Some(&serde_json::json!(tool_name)) =>
            {
                Some(data.clone())
            }
            _ => None,
        })
        .collect()
}

#[test]
fn anthropic_stream_mcp_emits_tool_call_and_result() {
    let path = fixtures_dir().join("anthropic-mcp.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);
    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let events = run_converter(lines);

    let calls = tool_events(&events, "tool-call", "echo");
    let results = tool_events(&events, "tool-result", "echo");

    assert!(!calls.is_empty(), "expected tool-call for echo");
    assert!(!results.is_empty(), "expected tool-result for echo");

    for ev in calls.iter().chain(results.iter()) {
        assert!(ev.get("toolCallId").and_then(|v| v.as_str()).is_some());
    }
}

#[test]
fn anthropic_stream_web_search_emits_tool_call_and_result() {
    let path = fixtures_dir().join("anthropic-web-search-tool.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);
    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let events = run_converter(lines);

    let calls = tool_events(&events, "tool-call", "web_search");
    let results = tool_events(&events, "tool-result", "web_search");

    assert!(!calls.is_empty(), "expected tool-call for web_search");
    assert!(!results.is_empty(), "expected tool-result for web_search");
}

#[test]
fn anthropic_stream_web_fetch_emits_tool_call_and_result() {
    let path = fixtures_dir().join("anthropic-web-fetch-tool.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);
    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let events = run_converter(lines);

    let calls = tool_events(&events, "tool-call", "web_fetch");
    let results = tool_events(&events, "tool-result", "web_fetch");

    assert!(!calls.is_empty(), "expected tool-call for web_fetch");
    assert!(!results.is_empty(), "expected tool-result for web_fetch");
}
