#![cfg(feature = "google")]

use siumai::prelude::unified::ChatStreamEvent;
use std::path::Path;

fn fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("google")
        .join("generative-ai-stream")
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
    use siumai::prelude::unified::SseEventConverter;
    use siumai_provider_gemini::providers::gemini::streaming::GeminiEventConverter;
    use siumai_provider_gemini::providers::gemini::types::GeminiConfig;

    let conv = GeminiEventConverter::new(GeminiConfig::default());

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
fn google_stream_code_execution_emits_tool_call_and_result() {
    let path = fixtures_dir().join("google-code-execution.1.chunks.txt");
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

    let call_id = calls[0]
        .get("toolCallId")
        .and_then(|v| v.as_str())
        .expect("toolCallId missing")
        .to_string();
    assert!(
        results
            .iter()
            .any(|v| v.get("toolCallId").and_then(|id| id.as_str()) == Some(call_id.as_str())),
        "expected tool-result to share toolCallId with tool-call"
    );
}
