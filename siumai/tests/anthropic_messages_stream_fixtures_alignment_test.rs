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

#[test]
fn anthropic_stream_tool_search_emits_tool_call_and_result() {
    let path = fixtures_dir().join("anthropic-tool-search-regex.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);
    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let events = run_converter(lines);

    let calls = tool_events(&events, "tool-call", "tool_search");
    let results = tool_events(&events, "tool-result", "tool_search");

    assert!(!calls.is_empty(), "expected tool-call for tool_search");
    assert!(!results.is_empty(), "expected tool-result for tool_search");
}

#[test]
fn anthropic_stream_code_execution_emits_tool_call_and_result() {
    let path = fixtures_dir().join("anthropic-code-execution-20250825.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);
    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let events = run_converter(lines);

    let calls = tool_events(&events, "tool-call", "code_execution");
    let results = tool_events(&events, "tool-result", "code_execution");

    assert!(!calls.is_empty(), "expected tool-call for code_execution");
    assert!(
        !results.is_empty(),
        "expected tool-result for code_execution"
    );
}

#[test]
fn anthropic_stream_agent_skills_emits_code_execution_events() {
    let path = fixtures_dir().join("anthropic-code-execution-20250825.pptx-skill.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);
    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let events = run_converter(lines);

    let calls = tool_events(&events, "tool-call", "code_execution");
    let results = tool_events(&events, "tool-result", "code_execution");

    assert!(!calls.is_empty(), "expected tool-call for code_execution");
    assert!(
        !results.is_empty(),
        "expected tool-result for code_execution"
    );
}

#[test]
fn anthropic_stream_programmatic_tool_calling_emits_code_execution_events() {
    let path = fixtures_dir().join("anthropic-programmatic-tool-calling.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);
    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let events = run_converter(lines);

    let calls = tool_events(&events, "tool-call", "code_execution");
    let results = tool_events(&events, "tool-result", "code_execution");

    assert!(!calls.is_empty(), "expected tool-call for code_execution");
    assert!(
        !results.is_empty(),
        "expected tool-result for code_execution"
    );
}

#[test]
fn anthropic_stream_tool_no_args_emits_tool_call_delta() {
    let path = fixtures_dir().join("anthropic-tool-no-args.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);
    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let events = run_converter(lines);

    let calls: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            ChatStreamEvent::ToolCallDelta {
                id,
                function_name: Some(name),
                ..
            } => Some((id.clone(), name.clone())),
            _ => None,
        })
        .collect();

    assert!(
        calls.iter().any(|(_, name)| name == "updateIssueList"),
        "expected ToolCallDelta for updateIssueList"
    );
}

#[test]
fn anthropic_stream_json_tool_emits_tool_call_delta_and_args_delta() {
    let path = fixtures_dir().join("anthropic-json-tool.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);
    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let events = run_converter(lines);

    let has_json_tool_call_delta = events.iter().any(|e| {
        matches!(
            e,
            ChatStreamEvent::ToolCallDelta {
                function_name: Some(name),
                ..
            } if name == "json"
        )
    });
    assert!(
        !has_json_tool_call_delta,
        "did not expect ToolCallDelta for reserved json tool"
    );

    let has_json_text = events.iter().any(|e| {
        matches!(
            e,
            ChatStreamEvent::ContentDelta { delta, .. } if delta.contains("\"elements\"")
        )
    });
    assert!(has_json_text, "expected ContentDelta containing elements");

    let has_stop = events.iter().any(|e| {
        matches!(
            e,
            ChatStreamEvent::StreamEnd { response } if response.finish_reason == Some(FinishReason::Stop)
        )
    });
    assert!(has_stop, "expected StreamEnd with finish_reason=stop");
}

#[test]
fn anthropic_stream_json_tool_text_prefix_is_ignored_in_vercel_stream_parts() {
    use siumai::experimental::standards::anthropic::params::StructuredOutputMode;
    use siumai::experimental::standards::anthropic::streaming::AnthropicEventConverter;
    use siumai::prelude::unified::SseEventConverter;

    let path = fixtures_dir().join("anthropic-json-tool.2.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);
    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let conv = AnthropicEventConverter::new(
        siumai::experimental::standards::anthropic::params::AnthropicParams::default()
            .with_structured_output_mode(StructuredOutputMode::JsonTool),
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

    let stream_starts = custom_events_by_type(&events, "stream-start");
    let metadata = custom_events_by_type(&events, "response-metadata");
    let text_starts = custom_events_by_type(&events, "text-start");
    let text_deltas = custom_events_by_type(&events, "text-delta");
    let text_ends = custom_events_by_type(&events, "text-end");
    let finishes = custom_events_by_type(&events, "finish");

    assert_eq!(stream_starts.len(), 1, "expected exactly one stream-start");
    assert_eq!(metadata.len(), 1, "expected exactly one response-metadata");
    assert_eq!(text_starts.len(), 1, "expected exactly one text-start");
    assert_eq!(text_ends.len(), 1, "expected exactly one text-end");
    assert_eq!(finishes.len(), 1, "expected exactly one finish");

    assert_eq!(
        metadata[0].get("id").and_then(|v| v.as_str()),
        Some("msg_01K2JbSUMYhez5RHoK9ZCj9U")
    );
    assert_eq!(
        metadata[0].get("modelId").and_then(|v| v.as_str()),
        Some("claude-haiku-4-5-20251001")
    );

    assert_eq!(
        text_starts[0].get("id").and_then(|v| v.as_str()),
        Some("1"),
        "expected json tool content block index as text id"
    );

    let deltas_for_block_1: Vec<&str> = text_deltas
        .iter()
        .filter(|d| d.get("id").and_then(|v| v.as_str()) == Some("1"))
        .filter_map(|d| d.get("delta").and_then(|v| v.as_str()))
        .collect();
    assert_eq!(deltas_for_block_1.len(), 2, "expected two json text-delta");
    assert_eq!(
        deltas_for_block_1[0],
        "{\"elements\": [{\"location\": \"San Francisco\", \"temperature\": 58, \"condition\": \"sunny\"}]"
    );
    assert_eq!(deltas_for_block_1[1], "}");

    assert_eq!(text_ends[0].get("id").and_then(|v| v.as_str()), Some("1"));

    assert_eq!(
        finishes[0]
            .get("finishReason")
            .and_then(|r| r.get("raw"))
            .and_then(|v| v.as_str()),
        Some("tool_use")
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
        Some(849)
    );
    assert_eq!(
        finishes[0]
            .get("usage")
            .and_then(|u| u.get("outputTokens"))
            .and_then(|u| u.get("total"))
            .and_then(|v| v.as_u64()),
        Some(47)
    );
}
