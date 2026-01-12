#![cfg(feature = "google")]

use siumai::prelude::unified::ChatStreamEvent;
use std::path::Path;

fn fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("google")
        .join("generative-ai-stream")
}

fn read_fixture_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .expect("read fixture file")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.to_string())
        .collect()
}

fn run_converter(lines: Vec<String>) -> Vec<ChatStreamEvent> {
    use siumai::prelude::unified::SseEventConverter;
    use siumai_provider_gemini::providers::gemini::streaming::GeminiEventConverter;
    use siumai_provider_gemini::providers::gemini::types::GeminiConfig;

    let conv = GeminiEventConverter::new(GeminiConfig::default());

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

    events
}

fn run_converter_with_provider_id(provider_id: &str, lines: Vec<String>) -> Vec<ChatStreamEvent> {
    use siumai::experimental::core::{ProviderContext, ProviderSpec};
    use siumai::prelude::unified::{ChatRequest, CommonParams};
    use std::collections::HashMap;

    let spec = siumai_provider_gemini::providers::gemini::spec::GeminiSpec;
    let ctx = ProviderContext::new(
        provider_id,
        "https://generativelanguage.googleapis.com/v1beta",
        Some("test-api-key".to_string()),
        HashMap::new(),
    );
    let req = ChatRequest {
        stream: true,
        common_params: CommonParams {
            model: "gemini-pro".to_string(),
            ..Default::default()
        },
        ..Default::default()
    };

    let bundle = spec.choose_chat_transformers(&req, &ctx);
    let stream = bundle.stream.expect("expected stream transformer");

    let mut events: Vec<ChatStreamEvent> = Vec::new();
    for (i, line) in lines.into_iter().enumerate() {
        let ev = eventsource_stream::Event {
            event: "".to_string(),
            data: line,
            id: i.to_string(),
            retry: None,
        };

        let out = futures::executor::block_on(stream.convert_event(ev));
        for item in out {
            match item {
                Ok(evt) => events.push(evt),
                Err(err) => panic!("failed to convert chunk: {err:?}"),
            }
        }
    }

    while let Some(item) = stream.handle_stream_end() {
        match item {
            Ok(evt) => events.push(evt),
            Err(err) => panic!("failed to finalize stream: {err:?}"),
        }
    }

    events
}

fn tool_events(events: &[ChatStreamEvent], kind: &str, tool_name: &str) -> Vec<serde_json::Value> {
    events
        .iter()
        .filter_map(|e| match e {
            ChatStreamEvent::Custom { data, .. }
                if data.get("type") == Some(&serde_json::json!(kind))
                    && data.get("toolName") == Some(&serde_json::json!(tool_name)) =>
            {
                Some(data.clone())
            }
            _ => None,
        })
        .collect()
}

fn custom_events_by_type(events: &[ChatStreamEvent], ty: &str) -> Vec<serde_json::Value> {
    events
        .iter()
        .filter_map(|e| match e {
            ChatStreamEvent::Custom { data, .. }
                if data.get("type") == Some(&serde_json::Value::String(ty.to_string())) =>
            {
                Some(data.clone())
            }
            _ => None,
        })
        .collect()
}

#[test]
fn google_stream_code_execution_emits_tool_call_and_result() {
    let path = fixtures_dir().join("google-code-execution.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);
    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let events = run_converter(lines);

    let calls = tool_events(&events, "tool-call", "code_execution");
    let results = tool_events(&events, "tool-result", "code_execution");

    assert!(!calls.is_empty(), "expected tool-call for code_execution");
    assert!(
        !results.is_empty(),
        "expected tool-result for code_execution"
    );

    let call_id = calls[0]
        .get("toolCallId")
        .and_then(|v| v.as_str())
        .expect("toolCallId missing")
        .to_string();
    assert!(
        results
            .iter()
            .any(|v| v.get("toolCallId").and_then(|id| id.as_str()) == Some(call_id.as_str())),
        "expected tool-result to share toolCallId with tool-call"
    );
}

#[test]
fn google_stream_thought_signature_is_exposed_on_reasoning_events() {
    let path = fixtures_dir().join("google-thought-signature-reasoning.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);
    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let events = run_converter(lines);

    let starts = custom_events_by_type(&events, "reasoning-start");
    let deltas = custom_events_by_type(&events, "reasoning-delta");

    assert!(!starts.is_empty(), "expected reasoning-start event");
    assert!(!deltas.is_empty(), "expected reasoning-delta event");

    let start = &starts[0];
    assert_eq!(
        start
            .get("providerMetadata")
            .and_then(|m| m.get("google"))
            .and_then(|m| m.get("thoughtSignature"))
            .and_then(|v| v.as_str()),
        Some("stream_sig")
    );

    assert!(
        deltas.iter().any(|ev| {
            ev.get("providerMetadata")
                .and_then(|m| m.get("google"))
                .and_then(|m| m.get("thoughtSignature"))
                .and_then(|v| v.as_str())
                == Some("stream_sig")
        }),
        "expected reasoning-delta to include providerMetadata.google.thoughtSignature"
    );
}

#[test]
fn vertex_provider_id_stream_reasoning_events_use_vertex_provider_metadata_key() {
    let path = fixtures_dir().join("google-thought-signature-reasoning.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);
    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let events = run_converter_with_provider_id("vertex", lines);

    let starts = custom_events_by_type(&events, "reasoning-start");
    let deltas = custom_events_by_type(&events, "reasoning-delta");

    assert!(!starts.is_empty(), "expected reasoning-start event");
    assert!(!deltas.is_empty(), "expected reasoning-delta event");

    let start = &starts[0];
    assert_eq!(
        start
            .get("providerMetadata")
            .and_then(|m| m.get("vertex"))
            .and_then(|m| m.get("thoughtSignature"))
            .and_then(|v| v.as_str()),
        Some("stream_sig")
    );
    assert!(
        start
            .get("providerMetadata")
            .and_then(|m| m.get("google"))
            .is_none()
    );

    assert!(
        deltas.iter().any(|ev| {
            ev.get("providerMetadata")
                .and_then(|m| m.get("vertex"))
                .and_then(|m| m.get("thoughtSignature"))
                .and_then(|v| v.as_str())
                == Some("stream_sig")
        }),
        "expected reasoning-delta to include providerMetadata.vertex.thoughtSignature"
    );

    let end = events.iter().find_map(|e| match e {
        ChatStreamEvent::StreamEnd { response } => Some(response),
        _ => None,
    });
    let end = end.expect("expected StreamEnd event");
    let meta = end
        .provider_metadata
        .as_ref()
        .expect("expected provider_metadata on stream end response");
    assert!(meta.contains_key("vertex"));
    assert!(!meta.contains_key("google"));
}
