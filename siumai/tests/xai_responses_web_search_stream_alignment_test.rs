#![cfg(feature = "xai")]

use siumai::prelude::unified::*;
use std::path::Path;

fn fixture_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("xai")
        .join("responses-stream")
        .join("web-search")
        .join("xai-web-search-tool.1.chunks.txt")
}

fn read_fixture_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .expect("read fixture file")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.to_string())
        .collect()
}

#[test]
fn xai_responses_web_search_stream_emits_vercel_aligned_tool_input() {
    let path = fixture_path();
    assert!(path.exists(), "fixture missing: {:?}", path);

    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let tools = vec![Tool::ProviderDefined(ProviderDefinedTool::new(
        "xai.web_search",
        "web_search",
    ))];

    let conv =
        siumai_provider_xai::standards::xai::responses_sse::XaiResponsesEventConverter::new()
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

    let tool_inputs_start: Vec<serde_json::Value> = events
        .iter()
        .filter_map(|e| match e {
            ChatStreamEvent::Custom { data, .. }
                if data.get("type") == Some(&serde_json::json!("tool-input-start")) =>
            {
                Some(data.clone())
            }
            _ => None,
        })
        .collect();
    let tool_inputs_delta: Vec<serde_json::Value> = events
        .iter()
        .filter_map(|e| match e {
            ChatStreamEvent::Custom { data, .. }
                if data.get("type") == Some(&serde_json::json!("tool-input-delta")) =>
            {
                Some(data.clone())
            }
            _ => None,
        })
        .collect();
    let tool_inputs_end: Vec<serde_json::Value> = events
        .iter()
        .filter_map(|e| match e {
            ChatStreamEvent::Custom { data, .. }
                if data.get("type") == Some(&serde_json::json!("tool-input-end")) =>
            {
                Some(data.clone())
            }
            _ => None,
        })
        .collect();
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

    assert!(!tool_inputs_start.is_empty(), "expected tool-input-start");
    assert!(!tool_inputs_delta.is_empty(), "expected tool-input-delta");
    assert!(!tool_inputs_end.is_empty(), "expected tool-input-end");
    assert!(!tool_calls.is_empty(), "expected tool-call");
    assert!(
        tool_results.is_empty(),
        "expected no tool-result for xAI web_search"
    );

    let start = tool_inputs_start
        .iter()
        .find(|s| s.get("toolName").and_then(|v| v.as_str()) == Some("web_search"))
        .expect("tool-input-start for web_search");
    let call_id = start
        .get("id")
        .and_then(|v| v.as_str())
        .expect("tool-input-start id");

    assert!(
        start.get("providerExecuted").is_none(),
        "xAI tool-input-start should not include providerExecuted"
    );

    let delta = tool_inputs_delta
        .iter()
        .find(|d| d.get("id").and_then(|v| v.as_str()) == Some(call_id))
        .and_then(|d| d.get("delta").and_then(|v| v.as_str()))
        .expect("tool-input-delta for web_search");
    assert!(
        delta.starts_with("{\"query\":\"what is xAI\""),
        "expected web_search tool input delta to include arguments"
    );

    assert!(
        tool_inputs_end
            .iter()
            .any(|e| e.get("id").and_then(|v| v.as_str()) == Some(call_id)),
        "expected tool-input-end for web_search"
    );

    let tool_call = tool_calls
        .iter()
        .find(|c| c.get("toolCallId").and_then(|v| v.as_str()) == Some(call_id))
        .expect("tool-call for web_search");
    assert_eq!(
        tool_call.get("toolName").and_then(|v| v.as_str()),
        Some("web_search")
    );
    assert_eq!(
        tool_call.get("providerExecuted").and_then(|v| v.as_bool()),
        Some(true)
    );
    assert_eq!(tool_call.get("input").and_then(|v| v.as_str()), Some(delta));
}
