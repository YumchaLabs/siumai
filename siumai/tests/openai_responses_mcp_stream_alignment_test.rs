#![cfg(feature = "openai")]

use siumai::prelude::unified::*;
use std::path::Path;

fn fixture_path(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("responses-stream")
        .join("mcp")
        .join(name)
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

#[test]
fn openai_responses_mcp_stream_emits_mcp_tool_calls_and_results() {
    let path = fixture_path("openai-mcp-tool.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);
    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let events = run_converter(lines);

    let tool_calls: Vec<serde_json::Value> = events
        .iter()
        .filter_map(|e| match e {
            ChatStreamEvent::Custom { data, .. }
                if data.get("type") == Some(&serde_json::json!("tool-call")) =>
            {
                Some(data.clone())
            }
            _ => None,
        })
        .collect();

    let tool_results: Vec<serde_json::Value> = events
        .iter()
        .filter_map(|e| match e {
            ChatStreamEvent::Custom { data, .. }
                if data.get("type") == Some(&serde_json::json!("tool-result")) =>
            {
                Some(data.clone())
            }
            _ => None,
        })
        .collect();

    assert!(tool_calls.len() >= 2, "expected >= 2 tool-call events");
    assert!(tool_results.len() >= 2, "expected >= 2 tool-result events");

    for ev in tool_calls.iter().chain(tool_results.iter()) {
        assert_eq!(
            ev.get("toolName").and_then(|v| v.as_str()),
            Some("mcp.web_search_exa")
        );
        assert!(ev.get("toolCallId").and_then(|v| v.as_str()).is_some());
    }
}

#[test]
fn openai_responses_mcp_stream_emits_tool_approval_request() {
    let path = fixture_path("openai-mcp-tool-approval.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);
    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let events = run_converter(lines);

    let tool_calls: Vec<serde_json::Value> = events
        .iter()
        .filter_map(|e| match e {
            ChatStreamEvent::Custom { data, .. }
                if data.get("type") == Some(&serde_json::json!("tool-call")) =>
            {
                Some(data.clone())
            }
            _ => None,
        })
        .collect();

    let approval_requests: Vec<serde_json::Value> = events
        .iter()
        .filter_map(|e| match e {
            ChatStreamEvent::Custom { data, .. }
                if data.get("type") == Some(&serde_json::json!("tool-approval-request")) =>
            {
                Some(data.clone())
            }
            _ => None,
        })
        .collect();

    assert!(!tool_calls.is_empty(), "expected tool-call event");
    assert!(
        !approval_requests.is_empty(),
        "expected tool-approval-request event"
    );

    let call = tool_calls
        .iter()
        .find(|v| v.get("toolName") == Some(&serde_json::json!("mcp.create_short_url")))
        .expect("mcp.create_short_url tool-call");
    assert_eq!(call.get("toolCallId"), Some(&serde_json::json!("id-0")));

    let approval = approval_requests
        .iter()
        .find(|v| v.get("toolCallId") == Some(&serde_json::json!("id-0")))
        .expect("tool-approval-request for id-0");
    assert!(
        approval
            .get("approvalId")
            .and_then(|v| v.as_str())
            .is_some()
    );
}
