#![cfg(feature = "bedrock")]

use reqwest::header::HeaderMap;
use siumai::experimental::core::ProviderSpec;
use siumai::prelude::unified::LlmError;
use std::path::Path;

fn fixture_text(path: impl AsRef<Path>) -> String {
    std::fs::read_to_string(path).expect("read fixture")
}

#[test]
fn bedrock_error_fixture_preserves_message() {
    let body = fixture_text(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("bedrock")
            .join("errors")
            .join("bedrock-error.1.json"),
    );

    let standard = siumai::experimental::standards::bedrock::chat::BedrockChatStandard::new();
    let spec = standard.create_spec("bedrock");
    let err = spec
        .classify_http_error(403, &body, &HeaderMap::new())
        .expect("classified");

    match err {
        LlmError::AuthenticationError(msg) => {
            assert_eq!(
                msg,
                "The security token included in the request is invalid."
            )
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}
