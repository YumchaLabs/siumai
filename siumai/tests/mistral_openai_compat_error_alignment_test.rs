#![cfg(feature = "openai")]

use siumai::prelude::unified::LlmError;
use siumai_protocol_openai::standards::openai::errors::classify_openai_compatible_http_error;
use std::path::Path;

fn fixture_text(path: impl AsRef<Path>) -> String {
    std::fs::read_to_string(path).expect("read fixture")
}

#[test]
fn mistral_openai_compatible_authentication_error_is_parsed() {
    let body = fixture_text(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("mistral")
            .join("errors")
            .join("mistral-authentication-error.1.json"),
    );

    let err = classify_openai_compatible_http_error("mistral", 401, &body)
        .expect("classified as openai-compatible");

    match err {
        LlmError::AuthenticationError(msg) => assert!(msg.to_lowercase().contains("api key")),
        other => panic!("unexpected error variant: {other:?}"),
    }
}
