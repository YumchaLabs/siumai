#![cfg(feature = "openai")]

//! Cross-protocol streaming transcoding policy tests (OpenAI Responses -> OpenAI Chat Completions).
//!
//! This suite locks down how `tool-approval-request` v3 parts are handled when the target protocol
//! does not have a native representation (Chat Completions):
//! - `Drop` => the part is omitted
//! - `AsText` => the part is downgraded into a lossy assistant text delta

use siumai::prelude::unified::*;
use std::path::Path;
use std::sync::Arc;

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

fn decode_openai_responses(lines: Vec<String>) -> Vec<ChatStreamEvent> {
    let conv = siumai::provider_ext::openai::ext::OpenAiResponsesEventConverter::new();

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

fn encode_openai_chat_completions_sse(
    events: Vec<ChatStreamEvent>,
    behavior: siumai::experimental::streaming::V3UnsupportedPartBehavior,
) -> Vec<u8> {
    use siumai::protocol::openai::compat::openai_config::OpenAiCompatibleConfig;
    use siumai::protocol::openai::compat::provider_registry::{
        ConfigurableAdapter, ProviderConfig, ProviderFieldMappings,
    };
    use siumai::protocol::openai::compat::streaming::OpenAiCompatibleEventConverter;

    let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
        id: "openai".to_string(),
        name: "OpenAI".to_string(),
        base_url: "https://api.openai.com/v1".to_string(),
        field_mappings: ProviderFieldMappings::default(),
        capabilities: vec!["tools".to_string()],
        default_model: Some("gpt-4o-mini".to_string()),
        supports_reasoning: false,
        api_key_env: None,
        api_key_env_aliases: vec![],
    }));

    let cfg = OpenAiCompatibleConfig::new(
        "openai",
        "sk-siumai-encoding-only",
        "https://api.openai.com/v1",
        adapter.clone(),
    )
    .with_model("gpt-4o-mini");

    let encoder = OpenAiCompatibleEventConverter::new(cfg, adapter)
        .with_v3_unsupported_part_behavior(behavior);

    let mut out = Vec::new();
    for ev in events {
        let chunk = encoder
            .serialize_event(&ev)
            .expect("serialize OpenAI chat completions chunk");
        out.extend_from_slice(&chunk);
    }
    out
}

#[test]
fn openai_responses_mcp_tool_approval_request_is_dropped_in_strict_openai_chat_completions() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("responses-stream")
        .join("mcp")
        .join("openai-mcp-tool-approval.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let upstream = decode_openai_responses(read_fixture_lines(&path));
    assert!(!upstream.is_empty(), "fixture produced no events");

    let bytes = encode_openai_chat_completions_sse(
        upstream,
        siumai::experimental::streaming::V3UnsupportedPartBehavior::Drop,
    );
    let frames = parse_sse_json_frames(&bytes);

    assert!(
        !frames.iter().any(|v| {
            v.get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("delta"))
                .and_then(|d| d.get("content"))
                .and_then(|t| t.as_str())
                .is_some_and(|s| s.contains("[tool-approval-request]"))
        }),
        "expected strict transcoding to drop tool-approval-request: {frames:?}"
    );
}

#[test]
fn openai_responses_mcp_tool_approval_request_is_downgraded_in_lossy_openai_chat_completions() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("responses-stream")
        .join("mcp")
        .join("openai-mcp-tool-approval.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let upstream = decode_openai_responses(read_fixture_lines(&path));
    assert!(!upstream.is_empty(), "fixture produced no events");

    let bytes = encode_openai_chat_completions_sse(
        upstream,
        siumai::experimental::streaming::V3UnsupportedPartBehavior::AsText,
    );
    let frames = parse_sse_json_frames(&bytes);

    assert!(
        frames.iter().any(|v| {
            v.get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("delta"))
                .and_then(|d| d.get("content"))
                .and_then(|t| t.as_str())
                .is_some_and(|s| s.contains("[tool-approval-request]"))
        }),
        "expected lossy transcoding to downgrade tool-approval-request into text: {frames:?}"
    );
}
