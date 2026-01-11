#![cfg(feature = "google")]

//! Gateway-only Gemini stream alignment tests.
//!
//! Gemini `functionResponse` parts are typically sent in the *next request*,
//! not emitted by the model in the same streaming response. For stream
//! transcoding gateways, it can be useful to replay tool results as
//! `functionResponse` frames.

use eventsource_stream::Event;
use siumai::prelude::unified::*;
use std::path::Path;

fn gemini_fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("gemini")
}

fn read_gemini_sse_data_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .expect("read fixture file")
        .lines()
        .filter_map(|l| {
            let l = l.trim();
            if l.is_empty() {
                return None;
            }
            Some(l.trim_start_matches("data: ").trim().to_string())
        })
        .collect()
}

fn parse_sse_json_frames(bytes: &[u8]) -> Vec<serde_json::Value> {
    let text = String::from_utf8_lossy(bytes);
    text.split("\n\n")
        .filter_map(|frame| {
            let frame = frame.trim();
            if frame.is_empty() {
                return None;
            }
            let data_line = frame
                .lines()
                .find(|l| l.starts_with("data: "))
                .map(|l| l.trim_start_matches("data: ").trim())?;
            if data_line.is_empty() || data_line == "[DONE]" {
                return None;
            }
            serde_json::from_str::<serde_json::Value>(data_line).ok()
        })
        .collect()
}

#[test]
fn gemini_function_response_frames_are_parsed_as_v3_tool_result_events() {
    let path = gemini_fixtures_dir().join("function_response_then_finish.sse");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let conv = siumai::protocol::gemini::streaming::GeminiEventConverter::new(
        siumai::protocol::gemini::types::GeminiConfig::default(),
    );

    let mut events: Vec<ChatStreamEvent> = Vec::new();
    for (i, line) in read_gemini_sse_data_lines(&path).into_iter().enumerate() {
        let ev = Event {
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

    assert_eq!(tool_results.len(), 1, "expected one tool-result event");
    let tr = &tool_results[0];
    assert_eq!(tr.get("toolCallId"), Some(&serde_json::json!("call_1")));
    assert_eq!(tr.get("toolName"), Some(&serde_json::json!("test-tool")));
    assert_eq!(tr.get("result"), Some(&serde_json::json!({"value":"ok"})));
}

#[test]
fn gemini_can_serialize_v3_tool_result_as_function_response_frame() {
    use siumai::prelude::unified::SseEventConverter;

    let conv = siumai::protocol::gemini::streaming::GeminiEventConverter::new(
        siumai::protocol::gemini::types::GeminiConfig::default(),
    )
    .with_emit_function_response_tool_results(true);

    let bytes = conv
        .serialize_event(&ChatStreamEvent::Custom {
            event_type: "openai:tool-result".to_string(),
            data: serde_json::json!({
                "type": "tool-result",
                "toolCallId": "call_123",
                "toolName": "test-tool",
                "result": { "value": 42 }
            }),
        })
        .expect("serialize tool-result");

    let frames = parse_sse_json_frames(&bytes);
    assert!(
        frames.iter().any(|v| {
            v.get("candidates")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("content"))
                .and_then(|c| c.get("parts"))
                .and_then(|p| p.get(0))
                .and_then(|p| p.get("functionResponse"))
                .and_then(|fr| fr.get("name"))
                .and_then(|n| n.as_str())
                == Some("test-tool")
        }),
        "expected functionResponse frame: {frames:?}"
    );

    let id = frames
        .iter()
        .find_map(|v| {
            v.get("siumai")?
                .get("toolCallId")?
                .as_str()
                .map(|s| s.to_string())
        })
        .expect("siumai.toolCallId");
    assert_eq!(id, "call_123");
}
