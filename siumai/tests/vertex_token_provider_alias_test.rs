#![cfg(feature = "google-vertex")]
#![allow(deprecated)]

use std::sync::Arc;

use siumai::experimental::auth::StaticTokenProvider;
use siumai::prelude::unified::*;

#[tokio::test]
async fn build_anthropic_vertex_with_google_token_provider_alias() {
    let result = Siumai::builder()
        .anthropic_vertex()
        .base_url("https://us-central1-aiplatform.googleapis.com/v1/projects/demo/locations/us-central1/publishers/anthropic")
        .model("claude-3-5-sonnet-20241022")
        .with_google_token_provider(Arc::new(StaticTokenProvider::new("test-token")))
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Anthropic Vertex build should accept the neutral Google token-provider alias"
    );
}

#[tokio::test]
async fn build_anthropic_vertex_with_vertex_token_provider_alias() {
    let result = Siumai::builder()
        .anthropic_vertex()
        .base_url("https://us-central1-aiplatform.googleapis.com/v1/projects/demo/locations/us-central1/publishers/anthropic")
        .model("claude-3-5-sonnet-20241022")
        .with_vertex_token_provider(Arc::new(StaticTokenProvider::new("test-token")))
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Anthropic Vertex build should accept the Vertex-focused token-provider alias"
    );
}
