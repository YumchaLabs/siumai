#![cfg(all(feature = "google", feature = "openai"))]

//! Cross-protocol streaming transcoding alignment tests (Gemini -> OpenAI).
//!
//! This suite exercises the gateway/proxy pattern:
//! - Parse Gemini GenerateContent SSE into unified stream events, then
//! - Re-serialize into OpenAI Chat Completions SSE and OpenAI Responses SSE.

use eventsource_stream::Event;
use siumai::experimental::streaming::OpenAiResponsesStreamPartsBridge;
use siumai::prelude::unified::*;
use std::path::Path;
use std::sync::Arc;

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
                .map(|l| l.trim_start_matches("data: ").trim())?;

            if data_line == "[DONE]" || data_line.is_empty() {
                return None;
            }

            Some(data_line.to_string())
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

fn encode_openai_chat_completions(events: Vec<ChatStreamEvent>) -> Vec<u8> {
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

    let encoder = OpenAiCompatibleEventConverter::new(cfg, adapter);

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

fn custom_v3_tool_calls(events: &[ChatStreamEvent], tool_name: &str) -> Vec<serde_json::Value> {
    events
        .iter()
        .filter_map(|e| match e {
            ChatStreamEvent::Custom { data, .. }
                if data.get("type") == Some(&serde_json::json!("tool-call"))
                    && data.get("toolName") == Some(&serde_json::json!(tool_name)) =>
            {
                Some(data.clone())
            }
            _ => None,
        })
        .collect()
}

#[test]
fn gemini_simple_text_transcodes_to_openai_chat_completions_and_responses() {
    let path = gemini_fixtures_dir().join("simple_text_then_finish.sse");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let lines = read_gemini_sse_data_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let upstream = run_gemini_converter(lines);
    let upstream_text: String = upstream
        .iter()
        .filter_map(|e| match e {
            ChatStreamEvent::ContentDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(upstream_text, "Hello world");

    let chat_bytes = encode_openai_chat_completions(upstream.clone());
    let chat_events = decode_openai_chat_completions(&chat_bytes);
    let chat_text: String = chat_events
        .iter()
        .filter_map(|e| match e {
            ChatStreamEvent::ContentDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(chat_text, "Hello world");

    let responses_bytes = encode_openai_responses_with_bridge(upstream);
    let responses_events = decode_openai_responses(&responses_bytes);
    let has_text_delta = responses_events.iter().any(|e| match e {
        ChatStreamEvent::Custom { data, .. } => {
            data.get("type") == Some(&serde_json::json!("text-delta"))
        }
        _ => false,
    });
    assert!(
        has_text_delta,
        "expected OpenAI Responses v3 text-delta parts"
    );
}

#[test]
fn gemini_function_call_transcodes_to_openai_chat_completions_and_responses() {
    let path = gemini_fixtures_dir().join("function_call_then_finish.sse");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let upstream = run_gemini_converter(read_gemini_sse_data_lines(&path));
    assert!(!upstream.is_empty(), "fixture produced no events");

    assert!(
        upstream
            .iter()
            .any(|e| matches!(e, ChatStreamEvent::ContentDelta { .. })),
        "expected initial text content delta"
    );

    let has_tool_call = upstream.iter().any(|e| match e {
        ChatStreamEvent::ToolCallDelta {
            function_name,
            arguments_delta,
            ..
        } => {
            function_name.as_deref() == Some("test-tool")
                && arguments_delta
                    .as_deref()
                    .is_some_and(|s| s.contains("example value"))
        }
        _ => false,
    });
    assert!(has_tool_call, "expected ToolCallDelta for test-tool");

    let chat_bytes = encode_openai_chat_completions(upstream.clone());
    let chat_events = decode_openai_chat_completions(&chat_bytes);
    assert!(
        chat_events
            .iter()
            .any(|e| matches!(e, ChatStreamEvent::ToolCallDelta { .. })),
        "expected tool-call events in OpenAI chat completions stream"
    );

    let responses_bytes = encode_openai_responses_with_bridge(upstream);
    let responses_events = decode_openai_responses(&responses_bytes);

    let has_responses_tool_call = responses_events.iter().any(|e| match e {
        ChatStreamEvent::ToolCallDelta { function_name, .. } => {
            function_name.as_deref() == Some("test-tool")
        }
        ChatStreamEvent::Custom { data, .. } => {
            data.get("type") == Some(&serde_json::json!("tool-call"))
                && data.get("toolName").and_then(|v| v.as_str()) == Some("test-tool")
        }
        _ => false,
    });
    assert!(
        has_responses_tool_call,
        "expected tool call to be present after OpenAI Responses re-serialization"
    );
}

#[test]
fn gemini_multi_function_calls_transcodes_to_openai_chat_completions_and_responses() {
    let path = gemini_fixtures_dir().join("multi_function_calls_then_finish.sse");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let upstream = run_gemini_converter(read_gemini_sse_data_lines(&path));
    assert!(!upstream.is_empty(), "fixture produced no events");

    let tool_names: Vec<_> = upstream
        .iter()
        .filter_map(|e| match e {
            ChatStreamEvent::ToolCallDelta { function_name, .. } => function_name.clone(),
            _ => None,
        })
        .collect();
    assert!(
        tool_names.iter().any(|n| n == "tool_a") && tool_names.iter().any(|n| n == "tool_b"),
        "expected tool_a and tool_b tool calls, got: {tool_names:?}"
    );

    let chat_bytes = encode_openai_chat_completions(upstream.clone());
    let chat_events = decode_openai_chat_completions(&chat_bytes);
    let chat_tool_calls = chat_events
        .iter()
        .filter(|e| matches!(e, ChatStreamEvent::ToolCallDelta { .. }))
        .count();
    assert!(chat_tool_calls >= 2, "expected >=2 tool calls");

    let responses_bytes = encode_openai_responses_with_bridge(upstream);
    let responses_events = decode_openai_responses(&responses_bytes);
    let a = custom_v3_tool_calls(&responses_events, "tool_a");
    let b = custom_v3_tool_calls(&responses_events, "tool_b");
    assert!(!a.is_empty(), "expected tool-call for tool_a");
    assert!(!b.is_empty(), "expected tool-call for tool_b");
}

#[test]
fn gemini_function_response_transcodes_to_openai_responses_tool_result() {
    let path = gemini_fixtures_dir().join("function_response_then_finish.sse");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let upstream = run_gemini_converter(read_gemini_sse_data_lines(&path));
    assert!(!upstream.is_empty(), "fixture produced no events");

    let bytes = encode_openai_responses_with_bridge(upstream);
    let events = decode_openai_responses(&bytes);

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
fn gemini_function_call_thought_signature_can_be_exposed_as_v3_part() {
    let path = gemini_fixtures_dir().join("function_call_with_thought_signature.sse");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let conv = siumai::protocol::gemini::streaming::GeminiEventConverter::new(
        siumai::protocol::gemini::types::GeminiConfig::default(),
    )
    .with_emit_v3_tool_call_parts(true);

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

    let v3_calls = custom_v3_tool_calls(&events, "test-tool");
    assert!(!v3_calls.is_empty(), "expected v3 tool-call event");
    let pm = v3_calls[0]
        .get("providerMetadata")
        .and_then(|v| v.get("google"))
        .and_then(|v| v.get("thoughtSignature"))
        .and_then(|v| v.as_str());
    assert_eq!(pm, Some("sig_test_123"));
}
