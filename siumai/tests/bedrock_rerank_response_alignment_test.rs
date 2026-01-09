#![cfg(feature = "bedrock")]

//! Alignment tests for Vercel `@ai-sdk/amazon-bedrock` reranking response fixture.

use serde_json::Value;
use siumai::experimental::standards::bedrock::rerank::BedrockRerankStandard;
use std::path::Path;

fn fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("bedrock")
        .join("rerank")
}

fn read_json(path: impl AsRef<Path>) -> Value {
    let text = std::fs::read_to_string(path).expect("read fixture json");
    serde_json::from_str(&text).expect("parse fixture json")
}

#[test]
fn bedrock_rerank_response_maps_results() {
    let raw = read_json(fixtures_dir().join("bedrock-reranking.1.json"));
    let standard = BedrockRerankStandard::new();
    let tx = standard.create_transformers("bedrock");
    let resp = tx.response.transform(raw).expect("transform response");

    assert_eq!(resp.results.len(), 2);
    assert_eq!(resp.results[0].index, 0);
    assert!(resp.results[0].relevance_score > 0.5);
}
