#![cfg(feature = "google-vertex")]

use serde::de::DeserializeOwned;
use serde_json::Value;
use siumai::experimental::core::{ProviderContext, ProviderSpec};
use siumai_core::types::{ContentPart, MessageContent};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("vertex")
        .join("chat")
}

fn case_dirs(root: &Path) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root)
        .expect("read fixture root dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .map(|e| e.path())
        .collect();

    dirs.sort_by(|a, b| {
        a.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .cmp(&b.file_name().unwrap_or_default().to_string_lossy())
    });

    dirs
}

fn read_text(path: impl AsRef<Path>) -> String {
    std::fs::read_to_string(path).expect("read fixture text")
}

fn read_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> T {
    let text = read_text(path);
    serde_json::from_str(&text).expect("parse fixture json")
}

fn normalize_json(value: &mut Value) {
    match value {
        Value::Object(map) => {
            for v in map.values_mut() {
                normalize_json(v);
            }

            let keys: Vec<String> = map
                .iter()
                .filter_map(|(k, v)| {
                    if v.is_null() {
                        return Some(k.clone());
                    }
                    if let Value::Object(obj) = v
                        && obj.is_empty()
                    {
                        return Some(k.clone());
                    }
                    None
                })
                .collect();
            for k in keys {
                map.remove(&k);
            }
        }
        Value::Array(arr) => {
            for v in arr.iter_mut() {
                normalize_json(v);
            }
        }
        _ => {}
    }
}

fn vertex_ctx_enterprise() -> ProviderContext {
    let base_url = "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/test-project/locations/us-central1/publishers/google";
    let mut extra: HashMap<String, String> = HashMap::new();
    extra.insert("Authorization".to_string(), "Bearer token".to_string());
    ProviderContext::new("vertex", base_url.to_string(), None, extra)
}

fn assert_vertex_thought_signature(
    provider_metadata: &Option<std::collections::HashMap<String, serde_json::Value>>,
    expected: &str,
) {
    let Some(meta) = provider_metadata.as_ref() else {
        panic!("expected providerMetadata");
    };
    let sig = meta
        .get("vertex")
        .and_then(|v| v.get("thoughtSignature"))
        .and_then(|v| v.as_str())
        .expect("expected providerMetadata.vertex.thoughtSignature");
    assert_eq!(sig, expected);
}

fn run_case(root: &Path) {
    let req: siumai::prelude::unified::ChatRequest = read_json(root.join("request.json"));
    let expected_body: Value = read_json(root.join("expected_body.json"));
    let expected_url = read_text(root.join("expected_url.txt")).trim().to_string();

    let case_name = root.file_name().unwrap_or_default().to_string_lossy();

    let ctx = vertex_ctx_enterprise();
    let spec = siumai::experimental::providers::google_vertex::standards::vertex_generative_ai::VertexGenerativeAiStandard::new()
        .create_spec("vertex");

    let url = spec.chat_url(req.stream, &req, &ctx);
    assert_eq!(url, expected_url, "fixture case: {}", root.display());

    let transformers = spec.choose_chat_transformers(&req, &ctx);
    let got_body = transformers
        .request
        .transform_chat(&req)
        .expect("transform");

    let mut got_value = got_body;
    let mut expected_value = expected_body;
    normalize_json(&mut got_value);
    normalize_json(&mut expected_value);
    assert_eq!(
        got_value,
        expected_value,
        "fixture case: {}",
        root.display()
    );

    let response_path = root.join("response.json");
    if response_path.exists() {
        let raw: Value = read_json(response_path);
        let resp = transformers
            .response
            .transform_chat_response(&raw)
            .expect("transform response");

        if let Some(meta) = resp.provider_metadata.as_ref() {
            assert!(
                meta.contains_key("vertex"),
                "expected provider_metadata.vertex (fixture case: {})",
                root.display()
            );
        }

        if case_name.contains("thought-signature-text-and-reasoning") {
            let MessageContent::MultiModal(parts) = resp.content else {
                panic!("expected multimodal content");
            };
            assert_eq!(parts.len(), 3);

            match &parts[0] {
                ContentPart::Text {
                    provider_metadata, ..
                } => assert_vertex_thought_signature(provider_metadata, "sig1"),
                other => panic!("expected first part to be text, got: {other:?}"),
            }

            match &parts[1] {
                ContentPart::Reasoning {
                    provider_metadata, ..
                } => assert_vertex_thought_signature(provider_metadata, "sig2"),
                other => panic!("expected second part to be reasoning, got: {other:?}"),
            }

            match &parts[2] {
                ContentPart::Text {
                    provider_metadata, ..
                } => assert_vertex_thought_signature(provider_metadata, "sig3"),
                other => panic!("expected third part to be text, got: {other:?}"),
            }
        } else if case_name.contains("thought-signature-tool-call") {
            let MessageContent::MultiModal(parts) = resp.content else {
                panic!("expected multimodal content");
            };

            let tool_meta = parts
                .iter()
                .find_map(|p| match p {
                    ContentPart::ToolCall {
                        tool_name,
                        provider_metadata,
                        ..
                    } if tool_name == "test-tool" => Some(provider_metadata),
                    _ => None,
                })
                .expect("expected tool-call part");

            assert_vertex_thought_signature(tool_meta, "tool_sig");
        }
    }
}

#[test]
fn vertex_chat_fixtures_match() {
    let roots = case_dirs(&fixtures_dir());
    assert!(!roots.is_empty(), "no vertex chat fixture cases found");
    for root in roots {
        run_case(&root);
    }
}
