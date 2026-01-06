#![cfg(feature = "openai")]

use siumai::prelude::unified::*;
use std::path::Path;

fn fixture_paths() -> Vec<std::path::PathBuf> {
    let base = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("responses-stream")
        .join("apply-patch");

    vec![
        base.join("openai-apply-patch-tool.1.chunks.txt"),
        base.join("openai-apply-patch-tool-delete.1.chunks.txt"),
    ]
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
fn openai_responses_apply_patch_stream_emits_tool_call() {
    let tools = vec![Tool::ProviderDefined(ProviderDefinedTool::new(
        "openai.apply_patch",
        "apply_patch",
    ))];

    let conv = siumai::provider_ext::openai::ext::OpenAiResponsesEventConverter::new()
        .with_request_tools(&tools);

    for path in fixture_paths() {
        assert!(path.exists(), "fixture missing: {:?}", path);
        let lines = read_fixture_lines(&path);
        assert!(!lines.is_empty(), "fixture empty: {:?}", path);

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

        assert_eq!(
            tool_calls.len(),
            1,
            "expected exactly one tool-call: {:?}",
            path
        );
        let call = &tool_calls[0];
        assert_eq!(
            call.get("toolName").and_then(|v| v.as_str()),
            Some("apply_patch")
        );
        let tool_call_id = call
            .get("toolCallId")
            .and_then(|v| v.as_str())
            .expect("toolCallId");

        let input = call
            .get("input")
            .and_then(|v| v.as_str())
            .expect("tool-call input string");
        let parsed: serde_json::Value = serde_json::from_str(input).expect("input is json");
        assert_eq!(
            parsed.get("callId").and_then(|v| v.as_str()),
            Some(tool_call_id)
        );
        assert!(parsed.get("operation").is_some());
    }
}
