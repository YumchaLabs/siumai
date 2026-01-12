#![cfg(feature = "openai")]

use siumai::experimental::standards::openai::errors::classify_openai_compatible_http_error;
use siumai::prelude::unified::LlmError;
use std::path::Path;

fn fixture_text(path: impl AsRef<Path>) -> String {
    std::fs::read_to_string(path).expect("read fixture")
}

#[test]
fn openai_error_fixture_maps_message_losslessly() {
    let body = fixture_text(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("openai")
            .join("errors")
            .join("openai-error.1.json"),
    );

    let expected_message = serde_json::from_str::<serde_json::Value>(&body)
        .expect("parse fixture json")
        .get("error")
        .and_then(|v| v.get("message"))
        .and_then(|v| v.as_str())
        .expect("missing error.message")
        .to_string();

    let err = classify_openai_compatible_http_error("openai", 429, &body).expect("classified");

    match err {
        LlmError::QuotaExceededError(msg) => assert_eq!(msg, expected_message),
        other => panic!("unexpected error variant: {other:?}"),
    }
}
