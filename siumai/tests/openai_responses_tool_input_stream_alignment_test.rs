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
fn openai_responses_apply_patch_stream_emits_tool_input_events() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("responses-stream")
        .join("apply-patch");

    let tools = vec![Tool::ProviderDefined(ProviderDefinedTool::new(
        "openai.apply_patch",
        "apply_patch",
    ))];

    // create_file with streaming diff
    let path = base.join("openai-apply-patch-tool.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);
    let lines = read_fixture_lines(&path);
    let events = run_converter(lines, tools.clone());

    let starts = custom_events_by_type(&events, "tool-input-start");
    let deltas = custom_events_by_type(&events, "tool-input-delta");
    let ends = custom_events_by_type(&events, "tool-input-end");
    let tool_calls = custom_events_by_type(&events, "tool-call");

    assert!(!starts.is_empty(), "expected tool-input-start events");
    assert!(!deltas.is_empty(), "expected tool-input-delta events");
    assert!(!ends.is_empty(), "expected tool-input-end events");
    assert!(!tool_calls.is_empty(), "expected tool-call events");

    let apply_patch_call = tool_calls
        .iter()
        .find(|c| c.get("toolName").and_then(|v| v.as_str()) == Some("apply_patch"))
        .expect("apply_patch tool-call");

    let call_id = apply_patch_call
        .get("toolCallId")
        .and_then(|v| v.as_str())
        .expect("toolCallId");

    let start = starts
        .iter()
        .find(|s| s.get("id").and_then(|v| v.as_str()) == Some(call_id))
        .expect("tool-input-start for apply_patch call id");
    assert_eq!(
        start.get("toolName").and_then(|v| v.as_str()),
        Some("apply_patch")
    );

    let prefix = deltas
        .iter()
        .find(|d| d.get("id").and_then(|v| v.as_str()) == Some(call_id))
        .and_then(|d| d.get("delta").and_then(|v| v.as_str()))
        .expect("first tool-input-delta prefix");
    assert!(
        prefix.starts_with(&format!(
            "{{\"callId\":\"{call_id}\",\"operation\":{{\"type\":\"create_file\",\"path\":\"shopping-checklist.md\",\"diff\":\""
        )),
        "unexpected apply_patch prefix: {prefix:?}"
    );

    let has_plus_delta = deltas.iter().any(|d| {
        d.get("id").and_then(|v| v.as_str()) == Some(call_id)
            && d.get("delta").and_then(|v| v.as_str()) == Some("+")
    });
    assert!(has_plus_delta, "expected apply_patch diff delta '+'");

    let has_suffix = deltas.iter().any(|d| {
        d.get("id").and_then(|v| v.as_str()) == Some(call_id)
            && d.get("delta").and_then(|v| v.as_str()) == Some("\"}}")
    });
    assert!(has_suffix, "expected apply_patch suffix delta '\"}}'");

    let has_end = ends
        .iter()
        .any(|e| e.get("id").and_then(|v| v.as_str()) == Some(call_id));
    assert!(has_end, "expected tool-input-end for apply_patch");

    // delete_file (single delta)
    let path = base.join("openai-apply-patch-tool-delete.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);
    let lines = read_fixture_lines(&path);
    let events = run_converter(lines, tools);

    let deltas = custom_events_by_type(&events, "tool-input-delta");
    let expected = "{\"callId\":\"call_delete_1\",\"operation\":{\"type\":\"delete_file\",\"path\":\"obsolete.txt\"}}";
    assert!(
        deltas
            .iter()
            .any(|d| d.get("delta").and_then(|v| v.as_str()) == Some(expected)),
        "expected delete_file tool-input-delta to match"
    );
}

#[test]
fn openai_responses_web_search_stream_emits_tool_input_start_end() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("responses-stream")
        .join("web-search")
        .join("openai-web-search-tool.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let tools = vec![Tool::ProviderDefined(ProviderDefinedTool::new(
        "openai.web_search",
        "webSearch",
    ))];

    let events = run_converter(read_fixture_lines(&path), tools);
    let starts = custom_events_by_type(&events, "tool-input-start");
    let ends = custom_events_by_type(&events, "tool-input-end");
    let tool_calls = custom_events_by_type(&events, "tool-call");

    let web_calls: Vec<_> = tool_calls
        .iter()
        .filter(|c| c.get("toolName").and_then(|v| v.as_str()) == Some("webSearch"))
        .collect();
    assert!(!web_calls.is_empty(), "expected webSearch tool-call events");

    for call in web_calls {
        let id = call
            .get("toolCallId")
            .and_then(|v| v.as_str())
            .expect("toolCallId");

        assert!(
            starts
                .iter()
                .any(|s| s.get("id").and_then(|v| v.as_str()) == Some(id)),
            "expected tool-input-start for {id}"
        );
        assert!(
            ends.iter()
                .any(|e| e.get("id").and_then(|v| v.as_str()) == Some(id)),
            "expected tool-input-end for {id}"
        );
    }
}

#[test]
fn openai_responses_code_interpreter_stream_emits_tool_input_events() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("responses-stream")
        .join("code-interpreter")
        .join("openai-code-interpreter-tool.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let tools = vec![Tool::ProviderDefined(ProviderDefinedTool::new(
        "openai.code_interpreter",
        "codeExecution",
    ))];

    let events = run_converter(read_fixture_lines(&path), tools);

    let starts = custom_events_by_type(&events, "tool-input-start");
    let deltas = custom_events_by_type(&events, "tool-input-delta");
    let ends = custom_events_by_type(&events, "tool-input-end");
    let tool_calls = custom_events_by_type(&events, "tool-call");

    let first_start = starts
        .iter()
        .find(|s| s.get("toolName").and_then(|v| v.as_str()) == Some("codeExecution"))
        .expect("codeExecution tool-input-start");
    let ci_id = first_start
        .get("id")
        .and_then(|v| v.as_str())
        .expect("tool-input-start id");

    assert!(
        deltas.iter().any(|d| {
            d.get("id").and_then(|v| v.as_str()) == Some(ci_id)
                && d.get("delta").and_then(|v| v.as_str()).is_some_and(|s| {
                    s.starts_with("{\"containerId\":\"") && s.ends_with("\"code\":\"")
                })
        }),
        "expected codeExecution prefix delta"
    );

    assert!(
        deltas.iter().any(|d| {
            d.get("id").and_then(|v| v.as_str()) == Some(ci_id)
                && d.get("delta").and_then(|v| v.as_str()) == Some("\"}")
        }),
        "expected codeExecution suffix delta"
    );

    assert!(
        ends.iter()
            .any(|e| e.get("id").and_then(|v| v.as_str()) == Some(ci_id)),
        "expected tool-input-end for codeExecution"
    );

    assert!(
        tool_calls.iter().any(|c| {
            c.get("toolCallId").and_then(|v| v.as_str()) == Some(ci_id)
                && c.get("toolName").and_then(|v| v.as_str()) == Some("codeExecution")
        }),
        "expected codeExecution tool-call for {ci_id}"
    );
}

#[test]
fn openai_responses_function_tool_stream_emits_tool_input_and_tool_call() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("responses-stream")
        .join("reasoning")
        .join("openai-reasoning-encrypted-content.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let events = run_converter(read_fixture_lines(&path), vec![]);

    let starts = custom_events_by_type(&events, "tool-input-start");
    let ends = custom_events_by_type(&events, "tool-input-end");
    let tool_calls = custom_events_by_type(&events, "tool-call");

    let call = tool_calls
        .iter()
        .find(|c| c.get("toolName").and_then(|v| v.as_str()) == Some("calculator"))
        .expect("calculator tool-call");
    let call_id = call
        .get("toolCallId")
        .and_then(|v| v.as_str())
        .expect("toolCallId");

    assert!(
        starts.iter().any(|s| {
            s.get("id").and_then(|v| v.as_str()) == Some(call_id)
                && s.get("toolName").and_then(|v| v.as_str()) == Some("calculator")
        }),
        "expected tool-input-start for {call_id}"
    );
    assert!(
        ends.iter()
            .any(|e| e.get("id").and_then(|v| v.as_str()) == Some(call_id)),
        "expected tool-input-end for {call_id}"
    );
    assert_eq!(
        call.get("input").and_then(|v| v.as_str()),
        Some("{\"a\":12,\"b\":7,\"op\":\"add\"}")
    );
}
