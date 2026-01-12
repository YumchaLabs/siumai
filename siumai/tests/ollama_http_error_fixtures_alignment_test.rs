#![cfg(feature = "ollama")]

use reqwest::header::HeaderMap;
use siumai::experimental::core::{ProviderContext, ProviderSpec};
use siumai::prelude::unified::LlmError;
use siumai_provider_ollama::providers::ollama::config::OllamaParams;
use siumai_provider_ollama::providers::ollama::spec::OllamaSpec;
use std::collections::HashMap;
use std::path::Path;

fn fixture_text(path: impl AsRef<Path>) -> String {
    std::fs::read_to_string(path).expect("read fixture")
}

#[test]
fn ollama_error_fixture_preserves_message() {
    let body = fixture_text(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("ollama")
            .join("errors")
            .join("ollama-error.1.json"),
    );

    let spec = OllamaSpec::new(OllamaParams::default());
    let ctx = ProviderContext::new("ollama", "http://localhost:11434", None, HashMap::new());
    let err = spec
        .classify_http_error(404, &body, &HeaderMap::new())
        .or_else(|| {
            Some(siumai::retry_api::classify_http_error(
                "ollama",
                404,
                &body,
                &HeaderMap::new(),
                None,
            ))
        })
        .expect("classified");

    match err {
        LlmError::ApiError { code, message, .. } => {
            assert_eq!(code, 404);
            assert_eq!(
                message,
                "model 'invalid-model' not found, try pulling it first"
            );
        }
        other => panic!("unexpected error variant: {other:?}"),
    }

    // Smoke: build_headers should be fine without auth.
    let _headers = spec.build_headers(&ctx).expect("headers");
}
