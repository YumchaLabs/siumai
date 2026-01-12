#![cfg(feature = "togetherai")]

use reqwest::header::HeaderMap;
use siumai::experimental::core::ProviderSpec;
use siumai::prelude::unified::LlmError;
use std::path::Path;

fn fixture_text(path: impl AsRef<Path>) -> String {
    std::fs::read_to_string(path).expect("read fixture")
}

#[test]
fn togetherai_error_fixture_preserves_message() {
    let body = fixture_text(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("togetherai")
            .join("errors")
            .join("togetherai-error.1.json"),
    );

    let spec = siumai::experimental::standards::togetherai::rerank::TogetherAiRerankStandard::new()
        .create_spec("togetherai");
    let err = spec
        .classify_http_error(401, &body, &HeaderMap::new())
        .expect("classified");

    match err {
        LlmError::AuthenticationError(msg) => assert_eq!(msg, "Invalid API key"),
        other => panic!("unexpected error variant: {other:?}"),
    }
}
