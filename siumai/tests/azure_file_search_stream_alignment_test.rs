#![cfg(feature = "azure")]

use siumai::experimental::core::{ProviderContext, ProviderSpec};
use siumai::prelude::unified::*;
use std::path::Path;

fn fixture_paths() -> Vec<std::path::PathBuf> {
    let base = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("azure")
        .join("responses")
        .join("file-search");

    vec![
        base.join("openai-file-search-tool.1.chunks.txt"),
        base.join("openai-file-search-tool.2.chunks.txt"),
    ]
}

fn read_fixture_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .expect("read fixture file")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.to_string())
        .collect()
}

#[test]
fn azure_file_search_stream_emits_custom_tool_name_and_azure_metadata() {
    let tools = vec![Tool::ProviderDefined(ProviderDefinedTool::new(
        "openai.file_search",
        "file_search",
    ))];

    let spec =
        siumai::experimental::providers::azure::providers::azure_openai::AzureOpenAiSpec::default();
    let ctx = ProviderContext::new(
        "azure",
        "https://test-resource.openai.azure.com/openai",
        Some("test-api-key".to_string()),
        std::collections::HashMap::new(),
    );

    let req = ChatRequest {
        tools: Some(tools),
        common_params: CommonParams {
            model: "test-deployment".to_string(),
            ..Default::default()
        },
        ..Default::default()
    };
    let bundle = spec.choose_chat_transformers(&req, &ctx);
    let stream = bundle.stream.expect("expected stream transformer");

    for path in fixture_paths() {
        assert!(path.exists(), "fixture missing: {:?}", path);
        let lines = read_fixture_lines(&path);
        assert!(!lines.is_empty(), "fixture empty");

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

        let tool_calls: Vec<serde_json::Value> = events
            .iter()
            .filter_map(|e| match e {
                ChatStreamEvent::Custom { data, .. }
                    if data.get("type") == Some(&serde_json::json!("tool-call")) =>
                {
                    Some(data.clone())
                }
                _ => None,
            })
            .collect();

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

        assert!(
            !tool_calls.is_empty(),
            "expected tool-call events: {:?}",
            path
        );
        assert!(
            !tool_results.is_empty(),
            "expected tool-result events: {:?}",
            path
        );

        for ev in tool_calls.iter().chain(tool_results.iter()) {
            assert_eq!(
                ev.get("toolName").and_then(|v| v.as_str()),
                Some("file_search")
            );
            assert!(ev.get("toolCallId").and_then(|v| v.as_str()).is_some());
        }

        let has_azure_provider_metadata = events.iter().any(|e| match e {
            ChatStreamEvent::Custom { data, .. } => data
                .get("providerMetadata")
                .and_then(|m| m.get("azure"))
                .is_some(),
            _ => false,
        });
        assert!(
            has_azure_provider_metadata,
            "expected providerMetadata.azure in stream parts"
        );
    }
}
