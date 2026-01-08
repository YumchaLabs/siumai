#![cfg(feature = "google")]

use serde::de::DeserializeOwned;
use serde_json::Value;
use siumai::experimental::core::{ProviderContext, ProviderSpec};
use siumai_core::types::{ChatResponse, MessageContent, Warning};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("google")
        .join("generative-ai")
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

fn build_body(req: &siumai::prelude::unified::ChatRequest) -> Value {
    let spec = siumai_provider_gemini::providers::gemini::spec::GeminiSpec;
    let ctx = ProviderContext::new(
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta",
        Some("test-api-key".to_string()),
        HashMap::new(),
    );

    let transformers = spec.choose_chat_transformers(req, &ctx);
    let mut body = transformers.request.transform_chat(req).expect("transform");
    if let Some(cb) = spec.chat_before_send(req, &ctx) {
        body = cb(&body).expect("before_send");
    }
    body
}

fn run_case(root: &Path) {
    let req: siumai::prelude::unified::ChatRequest = read_json(root.join("request.json"));
    let expected_body: Value = read_json(root.join("expected_body.json"));

    let got_body = build_body(&req);

    let mut got_value = got_body;
    let mut expected_value = expected_body;
    normalize_json(&mut got_value);
    normalize_json(&mut expected_value);
    assert_eq!(got_value, expected_value);

    // Optional warning parity: compare provider middleware warnings when the fixture provides them.
    let expected_warnings_path = root.join("expected_warnings.json");
    if expected_warnings_path.exists() {
        use siumai_provider_gemini::execution::LanguageModelMiddleware;

        let expected_warnings: Vec<Warning> = read_json(expected_warnings_path);
        let mw = siumai_provider_gemini::providers::gemini::GeminiToolWarningsMiddleware::new();
        let base = ChatResponse::new(MessageContent::Text("ok".to_string()));
        let out = mw.post_generate(&req, base).expect("post_generate");
        assert_eq!(out.warnings.unwrap_or_default(), expected_warnings);
    }
}

#[test]
fn google_generative_ai_fixtures_match() {
    let roots = case_dirs(&fixtures_dir());
    assert!(!roots.is_empty(), "no fixture cases found");
    for root in roots {
        run_case(&root);
    }
}

#[test]
fn google_code_execution_response_emits_tool_call_and_result() {
    use siumai::prelude::unified::{ContentPart, MessageContent};
    use siumai_core::types::ToolResultOutput;

    let root = fixtures_dir().join("google-code-execution.1");
    let req: siumai::prelude::unified::ChatRequest = read_json(root.join("request.json"));
    let raw: Value = read_json(root.join("response.json"));

    let spec = siumai_provider_gemini::providers::gemini::spec::GeminiSpec;
    let transformers = spec.choose_chat_transformers(
        &req,
        &ProviderContext::new(
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta",
            Some("test-api-key".to_string()),
            HashMap::new(),
        ),
    );

    let resp = transformers
        .response
        .transform_chat_response(&raw)
        .expect("transform response");

    let MessageContent::MultiModal(parts) = resp.content else {
        panic!("expected multimodal content");
    };

    let mut call_id: Option<String> = None;
    let mut has_call = false;
    let mut has_result = false;

    for p in parts {
        match p {
            ContentPart::ToolCall {
                tool_call_id,
                tool_name,
                arguments,
                provider_executed: Some(true),
                ..
            } if tool_name == "code_execution" => {
                has_call = true;
                call_id = Some(tool_call_id);
                assert!(
                    arguments.get("language").and_then(|v| v.as_str()).is_some(),
                    "expected code_execution language"
                );
                assert!(
                    arguments.get("code").and_then(|v| v.as_str()).is_some(),
                    "expected code_execution code"
                );
            }
            ContentPart::ToolResult {
                tool_call_id,
                tool_name,
                output: ToolResultOutput::Json { value },
                provider_executed: Some(true),
                ..
            } if tool_name == "code_execution" => {
                has_result = true;
                assert_eq!(Some(tool_call_id), call_id);
                assert!(
                    value.get("outcome").and_then(|v| v.as_str()).is_some(),
                    "expected code_execution result outcome"
                );
            }
            _ => {}
        }
    }

    assert!(has_call, "expected code_execution tool call");
    assert!(has_result, "expected code_execution tool result");
}

fn assert_google_thought_signature(
    provider_metadata: &Option<std::collections::HashMap<String, serde_json::Value>>,
    expected: &str,
) {
    let Some(meta) = provider_metadata.as_ref() else {
        panic!("expected providerMetadata");
    };
    let sig = meta
        .get("google")
        .and_then(|v| v.get("thoughtSignature"))
        .and_then(|v| v.as_str())
        .expect("expected providerMetadata.google.thoughtSignature");
    assert_eq!(sig, expected);
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

#[test]
fn google_response_thought_signatures_are_exposed_on_text_and_reasoning_parts() {
    use siumai::prelude::unified::{ContentPart, MessageContent};

    let root = fixtures_dir().join("google-thought-signature-text-and-reasoning.1");
    let req: siumai::prelude::unified::ChatRequest = read_json(root.join("request.json"));
    let raw: Value = read_json(root.join("response.json"));

    let spec = siumai_provider_gemini::providers::gemini::spec::GeminiSpec;
    let transformers = spec.choose_chat_transformers(
        &req,
        &ProviderContext::new(
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta",
            Some("test-api-key".to_string()),
            HashMap::new(),
        ),
    );

    let resp = transformers
        .response
        .transform_chat_response(&raw)
        .expect("transform response");

    let MessageContent::MultiModal(parts) = resp.content else {
        panic!("expected multimodal content");
    };

    assert_eq!(parts.len(), 3);

    match &parts[0] {
        ContentPart::Text {
            text,
            provider_metadata,
        } => {
            assert_eq!(text, "Visible text part 1. ");
            assert_google_thought_signature(provider_metadata, "sig1");
        }
        other => panic!("expected first part to be text, got: {other:?}"),
    }

    match &parts[1] {
        ContentPart::Reasoning {
            text,
            provider_metadata,
        } => {
            assert_eq!(text, "This is a thought process.");
            assert_google_thought_signature(provider_metadata, "sig2");
        }
        other => panic!("expected second part to be reasoning, got: {other:?}"),
    }

    match &parts[2] {
        ContentPart::Text {
            text,
            provider_metadata,
        } => {
            assert_eq!(text, "Visible text part 2.");
            assert_google_thought_signature(provider_metadata, "sig3");
        }
        other => panic!("expected third part to be text, got: {other:?}"),
    }
}

#[test]
fn google_response_thought_signatures_are_exposed_on_tool_calls() {
    use siumai::prelude::unified::{ContentPart, MessageContent};

    let root = fixtures_dir().join("google-thought-signature-tool-call.1");
    let req: siumai::prelude::unified::ChatRequest = read_json(root.join("request.json"));
    let raw: Value = read_json(root.join("response.json"));

    let spec = siumai_provider_gemini::providers::gemini::spec::GeminiSpec;
    let transformers = spec.choose_chat_transformers(
        &req,
        &ProviderContext::new(
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta",
            Some("test-api-key".to_string()),
            HashMap::new(),
        ),
    );

    let resp = transformers
        .response
        .transform_chat_response(&raw)
        .expect("transform response");

    let MessageContent::MultiModal(parts) = resp.content else {
        panic!("expected multimodal content");
    };

    let tool = parts
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

    assert_google_thought_signature(tool, "tool_sig");
}

#[test]
fn vertex_provider_id_uses_vertex_key_for_thought_signature_parts_and_response_metadata() {
    use siumai::prelude::unified::{ContentPart, MessageContent};

    let root = fixtures_dir().join("google-thought-signature-text-and-reasoning.1");
    let req: siumai::prelude::unified::ChatRequest = read_json(root.join("request.json"));
    let raw: Value = read_json(root.join("response.json"));

    let spec = siumai_provider_gemini::providers::gemini::spec::GeminiSpec;
    let transformers = spec.choose_chat_transformers(
        &req,
        &ProviderContext::new(
            "vertex",
            "https://generativelanguage.googleapis.com/v1beta",
            Some("test-api-key".to_string()),
            HashMap::new(),
        ),
    );

    let resp = transformers
        .response
        .transform_chat_response(&raw)
        .expect("transform response");

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

    let meta = resp
        .provider_metadata
        .as_ref()
        .expect("expected provider_metadata");
    assert!(
        meta.contains_key("vertex"),
        "expected provider_metadata.vertex"
    );
    assert!(!meta.contains_key("google"));
}
