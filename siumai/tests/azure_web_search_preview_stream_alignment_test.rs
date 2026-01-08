#![cfg(feature = "openai")]

use siumai::prelude::unified::*;
use std::path::Path;

fn fixture_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("azure")
        .join("responses")
        .join("web-search-preview")
        .join("azure-web-search-preview-tool.1.chunks.txt")
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
fn azure_web_search_preview_stream_emits_vercel_aligned_tool_names() {
    let path = fixture_path();
    assert!(path.exists(), "fixture missing: {:?}", path);

    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let conv = siumai::provider_ext::openai::ext::OpenAiResponsesEventConverter::new()
        .with_provider_metadata_key("azure");

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

    let text_starts: Vec<serde_json::Value> = events
        .iter()
        .filter_map(|e| match e {
            ChatStreamEvent::Custom { data, .. }
                if data.get("type") == Some(&serde_json::json!("text-start")) =>
            {
                Some(data.clone())
            }
            _ => None,
        })
        .collect();

    assert!(!tool_calls.is_empty(), "expected tool-call events");
    assert!(!tool_results.is_empty(), "expected tool-result events");
    assert!(!text_starts.is_empty(), "expected text-start events");

    for ev in tool_calls.iter().chain(tool_results.iter()) {
        assert_eq!(
            ev.get("toolName").and_then(|v| v.as_str()),
            Some("web_search_preview")
        );
        assert!(ev.get("toolCallId").and_then(|v| v.as_str()).is_some());
        assert!(ev.get("providerMetadata").is_none());
    }

    for ev in text_starts.iter() {
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

    let has_url_sources = events.iter().any(|e| match e {
        ChatStreamEvent::Custom { data, .. } => {
            data.get("type") == Some(&serde_json::json!("source"))
                && data.get("sourceType") == Some(&serde_json::json!("url"))
        }
        _ => false,
    });
    assert!(has_url_sources, "expected url citation sources");
}
