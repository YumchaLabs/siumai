#![cfg(all(feature = "anthropic", feature = "openai"))]

//! Cross-protocol streaming transcoding alignment tests (Vercel-aligned v3 parts).
//!
//! These tests validate that we can:
//! 1) Parse Anthropic Messages SSE chunks into unified stream events, then
//! 2) Re-serialize the stream into OpenAI Responses SSE or OpenAI Chat Completions SSE, and
//! 3) Parse the resulting downstream stream without losing critical tool boundaries.

use eventsource_stream::Event;
use siumai::experimental::streaming::OpenAiResponsesStreamPartsBridge;
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

fn extract_sse_data_payload_lines(bytes: &[u8]) -> Vec<String> {
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
                .map(|l| l.trim_start_matches("data: ").trim());

            let data_line = data_line?;
            if data_line == "[DONE]" || data_line.is_empty() {
                return None;
            }
            Some(data_line.to_string())
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

fn v3_tool_parts(
    events: &[ChatStreamEvent],
    kind: &str,
    tool_name: &str,
) -> Vec<serde_json::Value> {
    events
        .iter()
        .filter_map(|e| match e {
            ChatStreamEvent::Custom { data, .. }
                if data.get("type") == Some(&serde_json::Value::String(kind.to_string()))
                    && data.get("toolName")
                        == Some(&serde_json::Value::String(tool_name.to_string())) =>
            {
                Some(data.clone())
            }
            _ => None,
        })
        .collect()
}

fn encode_openai_responses_with_bridge(events: Vec<ChatStreamEvent>) -> Vec<u8> {
    use siumai::prelude::unified::SseEventConverter;
    use siumai::protocol::openai::responses_sse::OpenAiResponsesEventConverter;

    let mut bridge = OpenAiResponsesStreamPartsBridge::new();
    let encoder = OpenAiResponsesEventConverter::new();

    let mut out = Vec::new();
    for ev in events {
        for bridged in bridge.bridge_event(ev) {
            let chunk = encoder
                .serialize_event(&bridged)
                .expect("serialize OpenAI responses chunk");
            out.extend_from_slice(&chunk);
        }
    }

    out
}

fn decode_openai_responses(bytes: &[u8]) -> Vec<ChatStreamEvent> {
    use siumai::protocol::openai::responses_sse::OpenAiResponsesEventConverter;

    let lines = extract_sse_data_payload_lines(bytes);
    let conv = OpenAiResponsesEventConverter::new();

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

fn encode_openai_chat_completions(
    events: Vec<ChatStreamEvent>,
    behavior: siumai::experimental::streaming::V3UnsupportedPartBehavior,
) -> Vec<u8> {
    use siumai::protocol::openai::compat::openai_config::OpenAiCompatibleConfig;
    use siumai::protocol::openai::compat::provider_registry::{
        ConfigurableAdapter, ProviderConfig, ProviderFieldMappings,
    };
    use siumai::protocol::openai::compat::streaming::OpenAiCompatibleEventConverter;

    let adapter = std::sync::Arc::new(ConfigurableAdapter::new(ProviderConfig {
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

fn decode_openai_chat_completions(bytes: &[u8]) -> Vec<ChatStreamEvent> {
    use siumai::protocol::openai::compat::openai_config::OpenAiCompatibleConfig;
    use siumai::protocol::openai::compat::provider_registry::{
        ConfigurableAdapter, ProviderConfig, ProviderFieldMappings,
    };
    use siumai::protocol::openai::compat::streaming::OpenAiCompatibleEventConverter;

    let adapter = std::sync::Arc::new(ConfigurableAdapter::new(ProviderConfig {
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
        "sk-siumai-decoding-only",
        "https://api.openai.com/v1",
        adapter.clone(),
    )
    .with_model("gpt-4o-mini");

    let conv = OpenAiCompatibleEventConverter::new(cfg, adapter);
    let lines = extract_sse_data_payload_lines(bytes);

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
                Err(err) => panic!("failed to parse OpenAI chat completion chunk: {err:?}"),
            }
        }
    }

    while let Some(item) = conv.handle_stream_end() {
        match item {
            Ok(evt) => events.push(evt),
            Err(err) => panic!("failed to finalize OpenAI chat completion stream: {err:?}"),
        }
    }

    events
}

#[test]
fn anthropic_web_fetch_transcodes_to_openai_responses_preserving_tool_call_and_result() {
    let path = anthropic_fixtures_dir().join("anthropic-web-fetch-tool.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let upstream = run_anthropic_converter(lines);
    let upstream_calls = v3_tool_parts(&upstream, "tool-call", "web_fetch");
    let upstream_results = v3_tool_parts(&upstream, "tool-result", "web_fetch");
    assert!(!upstream_calls.is_empty(), "expected upstream tool-call");
    assert!(
        !upstream_results.is_empty(),
        "expected upstream tool-result"
    );

    let tool_call_id = upstream_calls[0]
        .get("toolCallId")
        .and_then(|v| v.as_str())
        .expect("toolCallId");

    let bytes = encode_openai_responses_with_bridge(upstream);
    let downstream = decode_openai_responses(&bytes);

    let calls = v3_tool_parts(&downstream, "tool-call", "web_fetch");
    let results = v3_tool_parts(&downstream, "tool-result", "web_fetch");

    assert!(!calls.is_empty(), "expected downstream tool-call");
    assert!(!results.is_empty(), "expected downstream tool-result");
    assert!(
        calls
            .iter()
            .any(|v| v.get("toolCallId").and_then(|v| v.as_str()) == Some(tool_call_id)),
        "expected toolCallId to be preserved"
    );
}

#[test]
fn anthropic_mcp_transcodes_to_openai_chat_completions_with_lossy_tool_result_fallback() {
    let path = anthropic_fixtures_dir().join("anthropic-mcp.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let upstream = run_anthropic_converter(lines);
    let upstream_calls = v3_tool_parts(&upstream, "tool-call", "echo");
    let upstream_results = v3_tool_parts(&upstream, "tool-result", "echo");
    assert!(!upstream_calls.is_empty(), "expected upstream tool-call");
    assert!(
        !upstream_results.is_empty(),
        "expected upstream tool-result"
    );

    let bytes = encode_openai_chat_completions(
        upstream,
        siumai::experimental::streaming::V3UnsupportedPartBehavior::AsText,
    );
    let downstream = decode_openai_chat_completions(&bytes);

    let has_tool_call_delta = downstream.iter().any(|e| {
        matches!(
            e,
            ChatStreamEvent::ToolCallDelta {
                function_name: Some(name),
                ..
            } if name == "echo"
        )
    });
    assert!(has_tool_call_delta, "expected ToolCallDelta for echo");

    let content: String = downstream
        .iter()
        .filter_map(|e| match e {
            ChatStreamEvent::ContentDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        content.contains("[tool-result] echo"),
        "expected lossy fallback text for tool-result"
    );
}
