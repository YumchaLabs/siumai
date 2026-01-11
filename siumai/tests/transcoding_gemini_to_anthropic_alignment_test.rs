#![cfg(all(feature = "anthropic", feature = "google"))]

//! Cross-protocol streaming transcoding alignment tests (Gemini -> Anthropic).
//!
//! These tests validate the gateway/proxy pattern:
//! - Parse Gemini GenerateContent SSE into unified stream events, then
//! - Re-serialize into Anthropic Messages SSE.

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

            if data_line == "[DONE]" || data_line.is_empty() {
                return None;
            }

            serde_json::from_str::<serde_json::Value>(data_line).ok()
        })
        .collect()
}

fn run_gemini_converter(lines: Vec<String>) -> Vec<ChatStreamEvent> {
    let conv = siumai::protocol::gemini::streaming::GeminiEventConverter::new(
        siumai::protocol::gemini::types::GeminiConfig::default(),
    );

    let mut events: Vec<ChatStreamEvent> = Vec::new();
    for (i, line) in lines.into_iter().enumerate() {
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

    while let Some(item) = conv.handle_stream_end() {
        match item {
            Ok(evt) => events.push(evt),
            Err(err) => panic!("failed to finalize stream: {err:?}"),
        }
    }

    events
}

fn encode_anthropic_messages_sse(events: Vec<ChatStreamEvent>) -> Vec<u8> {
    use siumai::prelude::unified::SseEventConverter;
    use siumai::protocol::anthropic::params::AnthropicParams;
    use siumai::protocol::anthropic::streaming::AnthropicEventConverter;

    let conv = AnthropicEventConverter::new(AnthropicParams::default());

    let mut out = Vec::new();
    let has_stream_start = events
        .iter()
        .any(|e| matches!(e, ChatStreamEvent::StreamStart { .. }));
    if !has_stream_start {
        let start = conv
            .serialize_event(&ChatStreamEvent::StreamStart {
                metadata: ResponseMetadata {
                    id: Some("msg_siumai_0".to_string()),
                    model: Some("claude-siumai-proxy".to_string()),
                    created: None,
                    provider: "anthropic".to_string(),
                    request_id: None,
                },
            })
            .expect("serialize stream start");
        out.extend_from_slice(&start);
    }

    for ev in events {
        let chunk = conv
            .serialize_event(&ev)
            .expect("serialize Anthropic messages chunk");
        out.extend_from_slice(&chunk);
    }

    out
}

#[test]
fn gemini_thought_then_text_transcodes_to_anthropic_messages_sse() {
    let path = gemini_fixtures_dir().join("thought_then_text_stop.sse");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let upstream = run_gemini_converter(read_gemini_sse_data_lines(&path));
    assert!(!upstream.is_empty(), "fixture produced no events");

    let bytes = encode_anthropic_messages_sse(upstream);
    let frames = parse_sse_json_frames(&bytes);

    assert!(
        frames.iter().any(|v| v["type"] == "message_start"),
        "expected message_start frame: {frames:?}"
    );

    assert!(
        frames.iter().any(|v| {
            v["type"] == "content_block_delta"
                && v["delta"]["type"] == "thinking_delta"
                && v["delta"]["thinking"].as_str().is_some()
        }),
        "expected thinking_delta content_block_delta: {frames:?}"
    );

    assert!(
        frames.iter().any(|v| {
            v["type"] == "content_block_delta"
                && v["delta"]["type"] == "text_delta"
                && v["delta"]["text"]
                    .as_str()
                    .is_some_and(|s| s.contains("Final answer."))
        }),
        "expected text_delta containing 'Final answer.': {frames:?}"
    );

    assert!(
        frames.iter().any(|v| v["type"] == "message_stop"),
        "expected message_stop frame: {frames:?}"
    );
}

#[test]
fn gemini_function_call_transcodes_to_anthropic_messages_sse() {
    let path = gemini_fixtures_dir().join("function_call_then_finish.sse");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let upstream = run_gemini_converter(read_gemini_sse_data_lines(&path));
    assert!(!upstream.is_empty(), "fixture produced no events");

    let bytes = encode_anthropic_messages_sse(upstream);
    let frames = parse_sse_json_frames(&bytes);

    assert!(
        frames.iter().any(|v| v["type"] == "message_start"),
        "expected message_start frame: {frames:?}"
    );

    assert!(
        frames.iter().any(|v| {
            v["type"] == "content_block_start"
                && v["content_block"]["type"] == "tool_use"
                && v["content_block"]["name"] == "test-tool"
                && v["content_block"]["id"].as_str().is_some()
        }),
        "expected tool_use content_block_start for test-tool: {frames:?}"
    );

    assert!(
        frames.iter().any(|v| {
            v["type"] == "content_block_delta"
                && v["delta"]["type"] == "input_json_delta"
                && v["delta"]["partial_json"]
                    .as_str()
                    .is_some_and(|s| s.contains("example value"))
        }),
        "expected input_json_delta containing tool args: {frames:?}"
    );

    assert!(
        frames.iter().any(|v| v["type"] == "message_stop"),
        "expected message_stop frame: {frames:?}"
    );
}

#[test]
fn gemini_multi_function_calls_transcodes_to_anthropic_messages_sse() {
    let path = gemini_fixtures_dir().join("multi_function_calls_then_finish.sse");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let upstream = run_gemini_converter(read_gemini_sse_data_lines(&path));
    assert!(!upstream.is_empty(), "fixture produced no events");

    let bytes = encode_anthropic_messages_sse(upstream);
    let frames = parse_sse_json_frames(&bytes);

    let tool_use_names: Vec<_> = frames
        .iter()
        .filter(|v| v["type"] == "content_block_start" && v["content_block"]["type"] == "tool_use")
        .filter_map(|v| v["content_block"]["name"].as_str().map(|s| s.to_string()))
        .collect();

    assert!(
        tool_use_names.iter().any(|n| n == "tool_a")
            && tool_use_names.iter().any(|n| n == "tool_b"),
        "expected tool_a and tool_b tool_use blocks, got: {tool_use_names:?}"
    );
}
