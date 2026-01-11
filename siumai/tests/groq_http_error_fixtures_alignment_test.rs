#![cfg(feature = "groq")]

use reqwest::header::HeaderMap;
use siumai::experimental::core::ProviderSpec;
use siumai::prelude::unified::LlmError;
use std::path::Path;

fn fixture_text(path: impl AsRef<Path>) -> String {
    std::fs::read_to_string(path).expect("read fixture")
}

#[test]
fn groq_error_fixture_preserves_message() {
    let body_text = fixture_text(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("groq")
            .join("errors")
            .join("groq-error.1.json"),
    );

    let spec = siumai_provider_groq::providers::groq::spec::GroqSpec;
    let err = spec
        .classify_http_error(401, &body_text, &HeaderMap::new())
        .expect("expected classified error");
    match err {
        LlmError::AuthenticationError(msg) => assert_eq!(msg, "Invalid API key"),
        other => panic!("unexpected error variant: {other:?}"),
    }
}
