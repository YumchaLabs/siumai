#![cfg(feature = "google-vertex")]

use serde::de::DeserializeOwned;
use serde_json::Value;
use siumai::experimental::bridge::{
    BridgeMode, BridgeTarget, bridge_chat_response_to_gemini_generate_content_json_value,
};
use siumai::experimental::core::{ProviderContext, ProviderSpec};
use siumai::experimental::encoding::JsonEncodeOptions;
use siumai::prelude::unified::{ChatRequest, ChatResponse};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("vertex")
        .join("chat")
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
            for inner in map.values_mut() {
                normalize_json(inner);
            }

            let keys: Vec<String> = map
                .iter()
                .filter_map(|(key, inner)| {
                    if inner.is_null() {
                        return Some(key.clone());
                    }
                    if let Value::Object(obj) = inner
                        && obj.is_empty()
                    {
                        return Some(key.clone());
                    }
                    None
                })
                .collect();
            for key in keys {
                map.remove(&key);
            }
        }
        Value::Array(items) => {
            for inner in items.iter_mut() {
                normalize_json(inner);
            }
        }
        _ => {}
    }
}

fn strip_tool_call_ids(value: &mut Value) {
    match value {
        Value::Object(map) => {
            map.remove("toolCallId");
            map.remove("tool_call_id");
            for inner in map.values_mut() {
                strip_tool_call_ids(inner);
            }
        }
        Value::Array(items) => {
            for inner in items.iter_mut() {
                strip_tool_call_ids(inner);
            }
        }
        _ => {}
    }
}

fn normalize_response_json(response: &ChatResponse) -> Value {
    let mut value = serde_json::to_value(response).expect("serialize chat response");
    normalize_json(&mut value);
    value
}

fn projected_response_json(response: &ChatResponse) -> Value {
    let mut value = normalize_response_json(response);
    strip_tool_call_ids(&mut value);
    value
}

fn vertex_ctx_enterprise() -> ProviderContext {
    let base_url = "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/test-project/locations/us-central1/publishers/google";
    let mut extra: HashMap<String, String> = HashMap::new();
    extra.insert("Authorization".to_string(), "Bearer token".to_string());
    ProviderContext::new("vertex", base_url.to_string(), None, extra)
}

fn transform_vertex_provider_response(request: &ChatRequest, raw: &Value) -> ChatResponse {
    let ctx = vertex_ctx_enterprise();
    let spec = siumai::experimental::providers::google_vertex::standards::vertex_generative_ai::VertexGenerativeAiStandard::new()
        .create_spec("vertex");
    spec.choose_chat_transformers(request, &ctx)
        .response
        .transform_chat_response(raw)
        .expect("transform provider response")
}

fn roundtrip_provider_response(
    case: &str,
) -> (
    siumai::experimental::bridge::BridgeResult<Value>,
    Value,
    ChatResponse,
    ChatResponse,
) {
    let root = fixtures_dir().join(case);
    let request: ChatRequest = read_json(root.join("request.json"));
    let raw: Value = read_json(root.join("response.json"));
    let response = transform_vertex_provider_response(&request, &raw);

    let bridged = bridge_chat_response_to_gemini_generate_content_json_value(
        &response,
        Some(BridgeTarget::GeminiGenerateContent),
        BridgeMode::BestEffort,
        JsonEncodeOptions::default(),
    )
    .unwrap_or_else(|err| panic!("bridge response {}: {err:?}", root.display()));

    assert!(
        !bridged.is_rejected(),
        "bridge rejected fixture case {} with report {:?}",
        root.display(),
        bridged.report
    );

    let bridged_json = bridged.value.clone().expect("bridged response json");
    let roundtripped = transform_vertex_provider_response(&request, &bridged_json);

    (bridged, bridged_json, response, roundtripped)
}

#[test]
fn vertex_generate_content_response_bridge_roundtrip_reasoning_fixture_matches_exact_projection() {
    let case = "google-thought-signature-text-and-reasoning.1";
    let response_path = fixtures_dir().join(case).join("response.json");
    let (bridged, bridged_json, expected, roundtripped) = roundtrip_provider_response(case);

    assert!(
        bridged.report.is_exact(),
        "fixture case {} expected exact report, got {:?}",
        response_path.display(),
        bridged.report
    );
    assert_eq!(
        bridged_json["candidates"][0]["content"]["parts"][0]["thoughtSignature"],
        serde_json::json!("sig1"),
        "fixture case {} should preserve visible text thoughtSignature",
        response_path.display()
    );
    assert_eq!(
        bridged_json["candidates"][0]["content"]["parts"][1]["thought"],
        serde_json::json!(true),
        "fixture case {} should preserve reasoning marker",
        response_path.display()
    );
    assert_eq!(
        bridged_json["candidates"][0]["content"]["parts"][1]["thoughtSignature"],
        serde_json::json!("sig2"),
        "fixture case {} should preserve reasoning thoughtSignature",
        response_path.display()
    );
    assert_eq!(
        normalize_response_json(&roundtripped),
        normalize_response_json(&expected),
        "fixture case {} should roundtrip exactly",
        response_path.display()
    );
    assert!(
        normalize_response_json(&roundtripped)
            .pointer("/content/MultiModal/1/providerMetadata/vertex/thoughtSignature")
            .is_some(),
        "fixture case {} should keep vertex thoughtSignature metadata",
        response_path.display()
    );
}

#[test]
fn vertex_generate_content_response_bridge_roundtrip_tool_call_fixture_preserves_projection() {
    let case = "google-thought-signature-tool-call.1";
    let response_path = fixtures_dir().join(case).join("response.json");
    let (bridged, bridged_json, expected, roundtripped) = roundtrip_provider_response(case);

    assert!(
        bridged.report.is_lossy(),
        "fixture case {} expected lossy report, got {:?}",
        response_path.display(),
        bridged.report
    );
    assert!(
        bridged
            .report
            .lossy_fields
            .iter()
            .any(|field| field == "finish_reason"),
        "fixture case {} expected finish_reason projection, got {:?}",
        response_path.display(),
        bridged.report
    );
    assert_eq!(
        bridged_json["candidates"][0]["content"]["parts"][0]["functionCall"]["name"],
        serde_json::json!("test-tool"),
        "fixture case {} should preserve functionCall name",
        response_path.display()
    );
    assert_eq!(
        bridged_json["candidates"][0]["content"]["parts"][0]["thoughtSignature"],
        serde_json::json!("tool_sig"),
        "fixture case {} should preserve tool-call thoughtSignature",
        response_path.display()
    );
    assert_eq!(
        projected_response_json(&roundtripped),
        projected_response_json(&expected),
        "fixture case {} should preserve projected tool-call semantics",
        response_path.display()
    );
    assert!(
        projected_response_json(&roundtripped)
            .pointer("/content/MultiModal/0/providerMetadata/vertex/thoughtSignature")
            .is_some(),
        "fixture case {} should keep vertex tool-call thoughtSignature metadata",
        response_path.display()
    );
}

#[test]
fn vertex_generate_content_response_bridge_roundtrip_code_execution_fixture_matches_projection() {
    let case = "google-code-execution.1";
    let response_path = fixtures_dir().join(case).join("response.json");
    let (bridged, bridged_json, expected, roundtripped) = roundtrip_provider_response(case);

    assert!(
        bridged.report.is_exact(),
        "fixture case {} expected exact report, got {:?}",
        response_path.display(),
        bridged.report
    );
    assert_eq!(
        bridged_json["candidates"][0]["content"]["parts"][0]["executableCode"]["language"],
        serde_json::json!("PYTHON"),
        "fixture case {} should preserve executableCode language",
        response_path.display()
    );
    assert_eq!(
        bridged_json["candidates"][0]["content"]["parts"][1]["codeExecutionResult"]["outcome"],
        serde_json::json!("OUTCOME_OK"),
        "fixture case {} should preserve codeExecutionResult outcome",
        response_path.display()
    );
    assert_eq!(
        projected_response_json(&roundtripped),
        projected_response_json(&expected),
        "fixture case {} should preserve projected code-execution semantics",
        response_path.display()
    );
}
