#![cfg(feature = "google")]

use serde::de::DeserializeOwned;
use serde_json::Value;
use siumai::experimental::bridge::{
    BridgeMode, BridgeTarget, bridge_chat_response_to_gemini_generate_content_json_value,
};
use siumai::experimental::encoding::JsonEncodeOptions;
use siumai::experimental::execution::transformers::response::ResponseTransformer;
use siumai::experimental::standards::gemini::transformers::GeminiResponseTransformer;
use siumai::prelude::unified::ChatResponse;
use siumai::protocol::gemini::types::GeminiConfig;
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("google")
        .join("generative-ai")
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

fn roundtrip_provider_response(
    case: &str,
) -> (
    siumai::experimental::bridge::BridgeResult<Value>,
    Value,
    ChatResponse,
    ChatResponse,
) {
    let root = fixtures_dir().join(case);
    let raw: Value = read_json(root.join("response.json"));
    let tx = GeminiResponseTransformer {
        config: GeminiConfig::default(),
    };

    let response = tx
        .transform_chat_response(&raw)
        .unwrap_or_else(|err| panic!("transform raw response {}: {err:?}", root.display()));

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
    let roundtripped = tx
        .transform_chat_response(&bridged_json)
        .unwrap_or_else(|err| {
            panic!(
                "transform bridged response {} failed: {err:?}",
                root.display()
            )
        });

    (bridged, bridged_json, response, roundtripped)
}

#[test]
fn gemini_generate_content_response_bridge_roundtrip_reasoning_fixture_matches_exact_projection() {
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
        bridged_json["candidates"][0]["content"]["parts"][0]["text"],
        serde_json::json!("Visible text part 1. "),
        "fixture case {} should preserve first visible text part",
        response_path.display()
    );
    assert_eq!(
        bridged_json["candidates"][0]["content"]["parts"][0]["thoughtSignature"],
        serde_json::json!("sig1"),
        "fixture case {} should preserve first visible text thoughtSignature",
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
        bridged_json["candidates"][0]["content"]["parts"][2]["thoughtSignature"],
        serde_json::json!("sig3"),
        "fixture case {} should preserve trailing visible text thoughtSignature",
        response_path.display()
    );
    assert_eq!(
        bridged_json["usageMetadata"]["totalTokenCount"],
        serde_json::json!(30),
        "fixture case {} should preserve usage totals",
        response_path.display()
    );
    assert_eq!(
        bridged_json["modelVersion"],
        serde_json::json!("gemini-pro"),
        "fixture case {} should preserve modelVersion",
        response_path.display()
    );
    assert_eq!(
        normalize_response_json(&roundtripped),
        normalize_response_json(&expected),
        "fixture case {} should roundtrip exactly",
        response_path.display()
    );
}

#[test]
fn gemini_generate_content_response_bridge_roundtrip_tool_call_fixture_preserves_projection() {
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
    assert!(
        bridged.report.dropped_fields.is_empty(),
        "fixture case {} should not drop visible tool-call fields, got {:?}",
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
        bridged_json["usageMetadata"]["totalTokenCount"],
        serde_json::json!(30),
        "fixture case {} should preserve usage totals",
        response_path.display()
    );
    assert_eq!(
        bridged_json["modelVersion"],
        serde_json::json!("gemini-pro"),
        "fixture case {} should preserve modelVersion",
        response_path.display()
    );
    assert_eq!(
        projected_response_json(&roundtripped),
        projected_response_json(&expected),
        "fixture case {} should preserve projected tool-call semantics",
        response_path.display()
    );
}

#[test]
fn gemini_generate_content_response_bridge_roundtrip_code_execution_fixture_matches_projection() {
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
        bridged_json["candidates"][0]["content"]["parts"][0]["executableCode"]["code"],
        serde_json::json!("print(1+1)"),
        "fixture case {} should preserve executableCode body",
        response_path.display()
    );
    assert_eq!(
        bridged_json["candidates"][0]["content"]["parts"][1]["codeExecutionResult"]["outcome"],
        serde_json::json!("OUTCOME_OK"),
        "fixture case {} should preserve codeExecutionResult outcome",
        response_path.display()
    );
    assert_eq!(
        bridged_json["candidates"][0]["content"]["parts"][1]["codeExecutionResult"]["output"],
        serde_json::json!("2"),
        "fixture case {} should preserve codeExecutionResult output",
        response_path.display()
    );
    assert_eq!(
        projected_response_json(&roundtripped),
        projected_response_json(&expected),
        "fixture case {} should preserve projected code-execution semantics",
        response_path.display()
    );
}
