#![cfg(feature = "azure")]

use siumai::experimental::core::{ProviderContext, ProviderSpec};
use siumai::prelude::unified::*;
use std::path::Path;

fn fixture_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("azure")
        .join("responses")
        .join("code-interpreter")
        .join("azure-code-interpreter-tool.1.chunks.txt")
}

fn read_fixture_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .expect("read fixture file")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.to_string())
        .collect()
}

fn collect_custom_events(events: &[ChatStreamEvent], ty: &str) -> Vec<serde_json::Value> {
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
fn azure_code_interpreter_stream_emits_vercel_aligned_tool_names() {
    let path = fixture_path();
    assert!(path.exists(), "fixture missing: {:?}", path);

    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let spec =
        siumai::experimental::providers::azure::providers::azure_openai::AzureOpenAiSpec::default();
    let ctx = ProviderContext::new(
        "azure",
        "https://test-resource.openai.azure.com/openai",
        Some("test-api-key".to_string()),
        std::collections::HashMap::new(),
    );
    let req = ChatRequest {
        common_params: CommonParams {
            model: "test-deployment".to_string(),
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

    let tool_input_starts = collect_custom_events(&events, "tool-input-start");
    let tool_input_deltas = collect_custom_events(&events, "tool-input-delta");

    assert!(
        !tool_input_starts.is_empty(),
        "expected tool-input-start events"
    );
    assert!(
        !tool_input_deltas.is_empty(),
        "expected tool-input-delta events"
    );

    for ev in tool_input_starts.iter() {
        assert_eq!(
            ev.get("toolName").and_then(|v| v.as_str()),
            Some("code_interpreter")
        );
        assert_eq!(
            ev.get("providerExecuted").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert!(ev.get("id").and_then(|v| v.as_str()).is_some());
    }
}
