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
        .join("reasoning-encrypted-content")
        .join("azure-reasoning-encrypted-content.1.chunks.txt")
}

fn read_fixture_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .expect("read fixture file")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.to_string())
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
fn azure_reasoning_encrypted_content_stream_emits_azure_provider_metadata() {
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

    let reasoning_starts = custom_events_by_type(&events, "reasoning-start");
    let reasoning_deltas = custom_events_by_type(&events, "reasoning-delta");
    let reasoning_ends = custom_events_by_type(&events, "reasoning-end");

    assert!(
        !reasoning_starts.is_empty(),
        "expected reasoning-start events"
    );
    assert!(
        !reasoning_deltas.is_empty(),
        "expected reasoning-delta events"
    );
    assert!(!reasoning_ends.is_empty(), "expected reasoning-end events");

    for ev in reasoning_starts
        .iter()
        .chain(reasoning_deltas.iter())
        .chain(reasoning_ends.iter())
    {
        assert!(
            ev.get("providerMetadata")
                .and_then(|m| m.get("azure"))
                .is_some(),
            "expected providerMetadata.azure"
        );
        assert!(
            ev.get("providerMetadata")
                .and_then(|m| m.get("openai"))
                .is_none()
        );
    }
}
