#![cfg(feature = "xai")]

use siumai::prelude::unified::*;
use std::path::Path;

fn fixture_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("xai")
        .join("responses-stream")
        .join("x-search")
        .join("xai-x-search-tool.chunks.txt")
}

fn read_fixture_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .expect("read fixture file")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.to_string())
        .collect()
}

fn custom_events_by_type(events: &[ChatStreamEvent], kind: &str) -> Vec<serde_json::Value> {
    events
        .iter()
        .filter_map(|e| match e {
            ChatStreamEvent::Custom { data, .. }
                if data.get("type").and_then(|v| v.as_str()) == Some(kind) =>
            {
                Some(data.clone())
            }
            _ => None,
        })
        .collect()
}

#[test]
fn xai_responses_x_search_stream_emits_vercel_aligned_custom_tool_calls() {
    let path = fixture_path();
    assert!(path.exists(), "fixture missing: {:?}", path);

    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let tools = vec![Tool::ProviderDefined(ProviderDefinedTool::new(
        "xai.x_search",
        "x_search",
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

    let tool_inputs_start = custom_events_by_type(&events, "tool-input-start");
    let tool_inputs_delta = custom_events_by_type(&events, "tool-input-delta");
    let tool_inputs_end = custom_events_by_type(&events, "tool-input-end");
    let tool_calls = custom_events_by_type(&events, "tool-call");
    let tool_results = custom_events_by_type(&events, "tool-result");

    assert!(
        tool_results.is_empty(),
        "expected no tool-result events for xAI x_search/web_search stream"
    );

    let x_search_start = tool_inputs_start
        .iter()
        .find(|s| s.get("toolName").and_then(|v| v.as_str()) == Some("x_search"))
        .expect("tool-input-start for x_search");
    let x_search_id = x_search_start
        .get("id")
        .and_then(|v| v.as_str())
        .expect("x_search tool-input-start id");
    assert!(
        x_search_start.get("providerExecuted").is_none(),
        "x_search tool-input-start should not include providerExecuted"
    );

    let x_search_delta = tool_inputs_delta
        .iter()
        .find(|d| d.get("id").and_then(|v| v.as_str()) == Some(x_search_id))
        .and_then(|d| d.get("delta").and_then(|v| v.as_str()))
        .expect("tool-input-delta for x_search");
    assert!(
        x_search_delta.contains("\"query\":\"from:xai filter:media\""),
        "expected x_search tool input delta to contain query"
    );
    assert!(
        tool_inputs_end
            .iter()
            .any(|e| e.get("id").and_then(|v| v.as_str()) == Some(x_search_id)),
        "expected tool-input-end for x_search"
    );

    let x_search_call = tool_calls
        .iter()
        .find(|c| c.get("toolCallId").and_then(|v| v.as_str()) == Some(x_search_id))
        .expect("tool-call for x_search");
    assert_eq!(
        x_search_call.get("toolName").and_then(|v| v.as_str()),
        Some("x_search")
    );
    assert_eq!(
        x_search_call
            .get("providerExecuted")
            .and_then(|v| v.as_bool()),
        Some(true)
    );
    assert_eq!(
        x_search_call.get("input").and_then(|v| v.as_str()),
        Some(x_search_delta)
    );

    let view_video_start = tool_inputs_start
        .iter()
        .find(|s| s.get("toolName").and_then(|v| v.as_str()) == Some("view_x_video"))
        .expect("tool-input-start for view_x_video");
    let view_video_id = view_video_start
        .get("id")
        .and_then(|v| v.as_str())
        .expect("view_x_video tool-input-start id");
    let view_video_delta = tool_inputs_delta
        .iter()
        .find(|d| d.get("id").and_then(|v| v.as_str()) == Some(view_video_id))
        .and_then(|d| d.get("delta").and_then(|v| v.as_str()))
        .expect("tool-input-delta for view_x_video");
    assert!(
        view_video_delta.contains("\"video_url\":"),
        "expected view_x_video delta to contain video_url"
    );

    let view_video_call = tool_calls
        .iter()
        .find(|c| c.get("toolCallId").and_then(|v| v.as_str()) == Some(view_video_id))
        .expect("tool-call for view_x_video");
    assert_eq!(
        view_video_call.get("toolName").and_then(|v| v.as_str()),
        Some("view_x_video")
    );
    assert_eq!(
        view_video_call.get("input").and_then(|v| v.as_str()),
        Some(view_video_delta)
    );

    let web_search_starts: Vec<&serde_json::Value> = tool_inputs_start
        .iter()
        .filter(|s| s.get("toolName").and_then(|v| v.as_str()) == Some("web_search"))
        .collect();
    assert!(
        !web_search_starts.is_empty(),
        "expected web_search tool-input-start events"
    );

    let some_web_search_id = web_search_starts[0]
        .get("id")
        .and_then(|v| v.as_str())
        .expect("web_search tool-input-start id");
    let web_search_delta = tool_inputs_delta
        .iter()
        .find(|d| d.get("id").and_then(|v| v.as_str()) == Some(some_web_search_id))
        .and_then(|d| d.get("delta").and_then(|v| v.as_str()))
        .expect("tool-input-delta for web_search");
    assert_eq!(web_search_delta, "", "expected empty web_search delta");

    let web_search_call = tool_calls
        .iter()
        .find(|c| c.get("toolCallId").and_then(|v| v.as_str()) == Some(some_web_search_id))
        .expect("tool-call for web_search");
    assert_eq!(
        web_search_call.get("toolName").and_then(|v| v.as_str()),
        Some("web_search")
    );
    assert_eq!(
        web_search_call.get("input").and_then(|v| v.as_str()),
        Some("")
    );
}
