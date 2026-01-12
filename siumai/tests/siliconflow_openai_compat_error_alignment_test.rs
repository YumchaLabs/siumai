#![cfg(feature = "openai")]

use siumai::prelude::unified::LlmError;
use siumai_protocol_openai::standards::openai::errors::classify_openai_compatible_http_error;
use std::path::Path;

fn fixture_text(path: impl AsRef<Path>) -> String {
    std::fs::read_to_string(path).expect("read fixture")
}

#[test]
fn siliconflow_openai_compatible_error_envelope_is_parsed() {
    let body = fixture_text(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("siliconflow")
            .join("errors")
            .join("siliconflow-authentication-error.1.json"),
    );

    let err = classify_openai_compatible_http_error("siliconflow", 401, &body)
        .expect("classified as openai-compatible");

    match err {
        LlmError::AuthenticationError(msg) => assert_eq!(msg, "invalid api token"),
        other => panic!("unexpected error variant: {other:?}"),
    }
}
