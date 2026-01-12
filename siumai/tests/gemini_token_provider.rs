#![cfg(feature = "google")]
use std::sync::Arc;

use siumai::experimental::auth::StaticTokenProvider;
use siumai::prelude::unified::*;

#[tokio::test]
async fn build_gemini_with_token_provider_without_api_key() {
    // Use a Vertex-style base URL (no network calls performed during build).
    let vertex_base = "https://us-central1-aiplatform.googleapis.com/v1/projects/demo/locations/us-central1/publishers/google";

    let tp = Arc::new(StaticTokenProvider::new("test-token"));

    let result = Siumai::builder()
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
