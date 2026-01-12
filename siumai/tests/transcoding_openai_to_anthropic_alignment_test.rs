#![cfg(all(feature = "anthropic", feature = "openai"))]

//! Cross-protocol streaming transcoding alignment tests (OpenAI -> Anthropic).
//!
//! These tests validate the gateway/proxy pattern:
//! - Parse OpenAI Responses SSE chunks into unified stream events, then
//! - Re-serialize into Anthropic Messages SSE.

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

fn decode_openai_responses(lines: Vec<String>, tools: Vec<Tool>) -> Vec<ChatStreamEvent> {
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
                Err(err) => panic!("failed to parse OpenAI responses chunk: {err:?}"),
            }
        }
    }

    while let Some(item) = conv.handle_stream_end() {
        match item {
            Ok(evt) => events.push(evt),
            Err(err) => panic!("failed to finalize OpenAI responses stream: {err:?}"),
        }
    }

    events
}

fn encode_anthropic_messages_sse(
    events: Vec<ChatStreamEvent>,
    behavior: siumai::experimental::streaming::V3UnsupportedPartBehavior,
) -> Vec<u8> {
    use siumai::prelude::unified::SseEventConverter;
    use siumai::protocol::anthropic::params::AnthropicParams;
    use siumai::protocol::anthropic::streaming::AnthropicEventConverter;

    let conv = AnthropicEventConverter::new(AnthropicParams::default())
        .with_v3_unsupported_part_behavior(behavior);

    let mut out = Vec::new();

    // Ensure the stream starts with a valid Anthropic message_start frame.
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

    for ev in events {
        let chunk = conv
            .serialize_event(&ev)
            .expect("serialize Anthropic messages chunk");
        out.extend_from_slice(&chunk);
    }

    out
}

#[test]
fn openai_responses_web_search_transcodes_to_anthropic_messages_sse() {
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

    let upstream = decode_openai_responses(read_fixture_lines(&path), tools);
    assert!(!upstream.is_empty(), "fixture produced no events");

    let bytes = encode_anthropic_messages_sse(
        upstream,
        siumai::experimental::streaming::V3UnsupportedPartBehavior::AsText,
    );
    let frames = parse_sse_json_frames(&bytes);

    assert!(
        frames.iter().any(|v| v["type"] == "message_start"),
        "expected message_start frame: {frames:?}"
    );

    assert!(
        frames.iter().any(|v| {
            v["type"] == "content_block_start"
                && v["content_block"]["type"] == "tool_use"
                && v["content_block"]["name"] == "webSearch"
                && v["content_block"]["id"].as_str().is_some()
        }),
        "expected tool_use content_block_start for webSearch: {frames:?}"
    );

    assert!(
        frames.iter().any(|v| {
            v["type"] == "content_block_delta"
                && v["delta"]["type"] == "text_delta"
                && v["delta"]["text"]
                    .as_str()
                    .is_some_and(|s| s.contains("[tool-result]"))
        }),
        "expected lossy [tool-result] text_delta: {frames:?}"
    );

    assert!(
        frames.iter().any(|v| v["type"] == "message_stop"),
        "expected message_stop frame: {frames:?}"
    );
}

#[test]
fn openai_responses_mcp_transcodes_to_anthropic_messages_sse() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("responses-stream")
        .join("mcp")
        .join("openai-mcp-tool.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let upstream = decode_openai_responses(read_fixture_lines(&path), Vec::new());
    assert!(!upstream.is_empty(), "fixture produced no events");

    let bytes = encode_anthropic_messages_sse(
        upstream,
        siumai::experimental::streaming::V3UnsupportedPartBehavior::AsText,
    );
    let frames = parse_sse_json_frames(&bytes);

    assert!(
        frames.iter().any(|v| v["type"] == "message_start"),
        "expected message_start frame: {frames:?}"
    );

    assert!(
        frames.iter().any(|v| {
            v["type"] == "content_block_start"
                && v["content_block"]["type"] == "tool_use"
                && v["content_block"]["name"] == "mcp.web_search_exa"
                && v["content_block"]["id"].as_str().is_some()
        }),
        "expected tool_use content_block_start for mcp.web_search_exa: {frames:?}"
    );

    assert!(
        frames.iter().any(|v| {
            v["type"] == "content_block_delta"
                && v["delta"]["type"] == "text_delta"
                && v["delta"]["text"]
                    .as_str()
                    .is_some_and(|s| s.contains("[tool-result]"))
        }),
        "expected lossy [tool-result] text_delta: {frames:?}"
    );

    assert!(
        frames.iter().any(|v| v["type"] == "message_stop"),
        "expected message_stop frame: {frames:?}"
    );
}

#[test]
fn openai_responses_mcp_tool_approval_request_is_dropped_in_strict_anthropic_transcoding() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("responses-stream")
        .join("mcp")
        .join("openai-mcp-tool-approval.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let upstream = decode_openai_responses(read_fixture_lines(&path), Vec::new());
    assert!(!upstream.is_empty(), "fixture produced no events");

    let bytes = encode_anthropic_messages_sse(
        upstream,
        siumai::experimental::streaming::V3UnsupportedPartBehavior::Drop,
    );
    let frames = parse_sse_json_frames(&bytes);

    assert!(
        frames.iter().any(|v| v["type"] == "message_start"),
        "expected message_start frame: {frames:?}"
    );

    assert!(
        frames.iter().any(|v| {
            v["type"] == "content_block_start"
                && v["content_block"]["type"] == "tool_use"
                && v["content_block"]["name"].as_str().is_some()
                && v["content_block"]["id"].as_str().is_some()
        }),
        "expected tool_use content_block_start: {frames:?}"
    );

    assert!(
        !frames.iter().any(|v| {
            v["type"] == "content_block_delta"
                && v["delta"]["type"] == "text_delta"
                && v["delta"]["text"]
                    .as_str()
                    .is_some_and(|s| s.contains("[tool-approval-request]"))
        }),
        "expected strict transcoding to drop tool-approval-request: {frames:?}"
    );

    assert!(
        frames.iter().any(|v| v["type"] == "message_stop"),
        "expected message_stop frame: {frames:?}"
    );
}

#[test]
fn openai_responses_mcp_tool_approval_request_is_downgraded_in_lossy_anthropic_transcoding() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("responses-stream")
        .join("mcp")
        .join("openai-mcp-tool-approval.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let upstream = decode_openai_responses(read_fixture_lines(&path), Vec::new());
    assert!(!upstream.is_empty(), "fixture produced no events");

    let bytes = encode_anthropic_messages_sse(
        upstream,
        siumai::experimental::streaming::V3UnsupportedPartBehavior::AsText,
    );
    let frames = parse_sse_json_frames(&bytes);

    assert!(
        frames.iter().any(|v| {
            v["type"] == "content_block_delta"
                && v["delta"]["type"] == "text_delta"
                && v["delta"]["text"]
                    .as_str()
                    .is_some_and(|s| s.contains("[tool-approval-request]"))
        }),
        "expected lossy transcoding to downgrade tool-approval-request into text: {frames:?}"
    );
}
