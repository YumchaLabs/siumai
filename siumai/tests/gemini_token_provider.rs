#![cfg(feature = "google")]
use std::sync::Arc;

use siumai::{auth::StaticTokenProvider, provider::SiumaiBuilder, types::ProviderType};

#[tokio::test]
async fn build_gemini_with_token_provider_without_api_key() {
    // Use a Vertex-style base URL (no network calls performed during build).
    let vertex_base = "https://aiplatform.googleapis.com/v1/projects/demo/locations/us-central1/publishers/google";

    let tp = Arc::new(StaticTokenProvider::new("test-token"));

    let result = SiumaiBuilder::new()
        .provider(ProviderType::Gemini)
        .model("gemini-1.5-flash")
        .base_url(vertex_base)
        .with_gemini_token_provider(tp)
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Gemini build should succeed when a token provider is supplied even without an API key"
    );
}
