#![cfg(feature = "openai")]

use siumai::experimental::streaming::SseEventConverter;
use siumai::prelude::unified::*;
use std::path::Path;

fn fixture_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("responses-stream")
        .join("image-generation")
        .join("openai-image-generation-tool.1.chunks.txt")
}

fn read_fixture_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .expect("read fixture file")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.to_string())
        .collect()
}

fn event_payloads_by_type(events: &[ChatStreamEvent], ty: &str) -> Vec<serde_json::Value> {
    let stable_parts: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            ChatStreamEvent::Part { part } | ChatStreamEvent::PartWithReplay { part, .. } => {
                Some(serde_json::to_value(part).expect("serialize stream part"))
            }
            _ => None,
        })
        .filter(|value| value.get("type").and_then(|v| v.as_str()) == Some(ty))
        .collect();
    if !stable_parts.is_empty() {
        return stable_parts;
    }

    events
        .iter()
        .filter_map(|event| match event {
            ChatStreamEvent::Custom { data, .. } => Some(data.clone()),
            _ => None,
        })
        .filter(|value| value.get("type").and_then(|v| v.as_str()) == Some(ty))
        .collect()
}

#[test]
fn openai_responses_image_generation_stream_emits_vercel_aligned_tool_names() {
    let path = fixture_path();
    assert!(path.exists(), "fixture missing: {:?}", path);

    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let tools = vec![Tool::ProviderDefined(ProviderDefinedTool::new(
        "openai.image_generation",
        "generateImage",
    ))];

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

    let tool_calls = event_payloads_by_type(&events, "tool-call");
    let tool_results = event_payloads_by_type(&events, "tool-result");

    assert!(!tool_calls.is_empty(), "expected tool-call events");
    assert!(!tool_results.is_empty(), "expected tool-result events");
    assert!(
        tool_results
            .iter()
            .any(|ev| ev.get("preliminary") == Some(&serde_json::json!(true))
                && ev
                    .get("result")
                    .and_then(|result| result.get("result"))
                    .and_then(|result| result.as_str())
                    .is_some()),
        "expected preliminary partial image tool-result"
    );

    for ev in tool_calls.iter().chain(tool_results.iter()) {
        assert_eq!(
            ev.get("toolName").and_then(|v| v.as_str()),
            Some("generateImage")
        );
        assert!(ev.get("toolCallId").and_then(|v| v.as_str()).is_some());
    }
}
