#![cfg(feature = "openai")]

use serde_json::Value;
use siumai::experimental::execution::transformers::response::ResponseTransformer;
use std::path::Path;

fn fixture_case_dir(case: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("responses")
        .join("response")
        .join(case)
}

fn read_json(path: impl AsRef<Path>) -> Value {
    let text = std::fs::read_to_string(path).expect("read fixture");
    serde_json::from_str(&text).expect("parse fixture json")
}

#[test]
fn openai_responses_response_can_emit_azure_provider_metadata_key() {
    let root = fixture_case_dir("basic-text");
    assert!(root.exists(), "fixture missing: {:?}", root);

    let response = read_json(root.join("response.json"));

    let tx = siumai::experimental::standards::openai::transformers::response::OpenAiResponsesResponseTransformer::new()
        .with_provider_metadata_key("azure");
    let resp = tx.transform_chat_response(&response).expect("transform");

    let provider_metadata = resp.provider_metadata.expect("expected provider_metadata");

    assert!(provider_metadata.contains_key("azure"));
    assert!(!provider_metadata.contains_key("openai"));
}
