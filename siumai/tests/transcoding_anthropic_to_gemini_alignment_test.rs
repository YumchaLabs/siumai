#![cfg(all(feature = "anthropic", feature = "google"))]

//! Cross-protocol streaming transcoding alignment tests (Anthropic -> Gemini).
//!
//! These tests validate the gateway/proxy pattern:
//! - Parse Anthropic Messages SSE chunks into unified stream events, then
//! - Re-serialize into Gemini GenerateContent SSE.

use eventsource_stream::Event;
use siumai::prelude::unified::*;
use std::path::Path;

fn anthropic_fixtures_dir() -> std::path::PathBuf {
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

fn run_anthropic_converter(lines: Vec<String>) -> Vec<ChatStreamEvent> {
    let conv = siumai::protocol::anthropic::streaming::AnthropicEventConverter::new(
        siumai::protocol::anthropic::params::AnthropicParams::default(),
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

fn encode_gemini_generate_content_sse(events: Vec<ChatStreamEvent>) -> Vec<u8> {
    use siumai::experimental::streaming::V3UnsupportedPartBehavior;
    use siumai::prelude::unified::SseEventConverter;
    use siumai::protocol::gemini::streaming::GeminiEventConverter;
    use siumai::protocol::gemini::types::GeminiConfig;

    let conv = GeminiEventConverter::new(GeminiConfig::default())
        .with_v3_unsupported_part_behavior(V3UnsupportedPartBehavior::AsText);

    let mut out = Vec::new();
    for ev in events {
        let chunk = conv
            .serialize_event(&ev)
            .expect("serialize Gemini generateContent chunk");
        out.extend_from_slice(&chunk);
    }
    out
}

#[test]
fn anthropic_web_search_transcodes_to_gemini_sse() {
    let path = anthropic_fixtures_dir().join("anthropic-web-search-tool.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let upstream = run_anthropic_converter(read_fixture_lines(&path));
    assert!(!upstream.is_empty(), "fixture produced no events");

    let bytes = encode_gemini_generate_content_sse(upstream);
    let frames = parse_sse_json_frames(&bytes);

    assert!(
        frames.iter().any(|v| {
            v.get("candidates")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("content"))
                .and_then(|c| c.get("parts"))
                .and_then(|p| p.get(0))
                .and_then(|p| p.get("functionCall"))
                .and_then(|fc| fc.get("name"))
                .and_then(|n| n.as_str())
                == Some("web_search")
        }),
        "expected Gemini functionCall name=web_search: {frames:?}"
    );

    assert!(
        frames.iter().any(|v| {
            v.get("candidates")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("content"))
                .and_then(|c| c.get("parts"))
                .and_then(|p| p.get(0))
                .and_then(|p| p.get("text"))
                .and_then(|t| t.as_str())
                .is_some_and(|s| s.contains("[tool-result]"))
        }),
        "expected lossy [tool-result] text part: {frames:?}"
    );

    assert!(
        frames.iter().any(|v| {
            v.get("candidates")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("finishReason"))
                .and_then(|r| r.as_str())
                .is_some()
        }),
        "expected finishReason frame: {frames:?}"
    );
}
