#![cfg(feature = "openai")]

use siumai::prelude::unified::LlmError;
use siumai_protocol_openai::standards::openai::errors::classify_openai_compatible_http_error;
use std::path::Path;

fn fixture_text(path: impl AsRef<Path>) -> String {
    std::fs::read_to_string(path).expect("read fixture")
}

#[test]
fn openrouter_resource_exhausted_error_is_classified_as_rate_limit() {
    let body = fixture_text(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("openrouter")
            .join("errors")
            .join("openrouter-resource-exhausted.1.json"),
    );

    let err = classify_openai_compatible_http_error("openrouter", 429, &body)
        .expect("classified as openai-compatible");

    match err {
        LlmError::RateLimitError(msg) => assert!(msg.contains("RESOURCE_EXHAUSTED")),
        other => panic!("unexpected error variant: {other:?}"),
    }
}
