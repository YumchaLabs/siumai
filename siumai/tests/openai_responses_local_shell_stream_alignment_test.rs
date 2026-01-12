#![cfg(feature = "openai")]

use siumai::prelude::unified::*;
use std::path::Path;

fn fixture_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("responses-stream")
        .join("local-shell")
        .join("openai-local-shell-tool.1.chunks.txt")
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
fn openai_responses_local_shell_stream_emits_vercel_aligned_tool_call() {
    let path = fixture_path();
    assert!(path.exists(), "fixture missing: {:?}", path);

    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let tools = vec![Tool::ProviderDefined(ProviderDefinedTool::new(
        "openai.local_shell",
        "shell",
    ))];

    let conv = siumai::provider_ext::openai::ext::OpenAiResponsesEventConverter::new()
        .with_request_tools(&tools);

    let mut tool_calls: Vec<serde_json::Value> = Vec::new();
    for (i, line) in lines.into_iter().enumerate() {
        let ev = eventsource_stream::Event {
            event: "".to_string(),
            data: line,
            id: i.to_string(),
            retry: None,
        };

        let out = futures::executor::block_on(conv.convert_event(ev));
        for item in out {
            let evt = item.expect("convert chunk");
            if let ChatStreamEvent::Custom { data, .. } = evt
                && data.get("type") == Some(&serde_json::json!("tool-call"))
            {
                tool_calls.push(data);
            }
        }
    }

    assert_eq!(tool_calls.len(), 1, "expected exactly one tool-call");
    let call = &tool_calls[0];
    assert_eq!(call.get("toolName").and_then(|v| v.as_str()), Some("shell"));
    assert!(call.get("toolCallId").and_then(|v| v.as_str()).is_some());

    let input = call
        .get("input")
        .and_then(|v| v.as_str())
        .expect("tool-call input string");
    let parsed: serde_json::Value = serde_json::from_str(input).expect("input is json");
    assert_eq!(
        parsed
            .get("action")
            .and_then(|a| a.get("type"))
            .and_then(|v| v.as_str()),
        Some("exec")
    );
}
