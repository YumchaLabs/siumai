#![cfg(feature = "google")]
//! Validation: When using Authorization (Bearer) for Vertex AI auth,
//! Gemini should not be blocked by API key validation.

use siumai::{provider::SiumaiBuilder, types::ProviderType};

#[tokio::test]
async fn build_gemini_with_bearer_does_not_require_api_key() {
    // Simulate a Vertex AI base_url prefix (no real call, construction only)
    let vertex_base = "https://aiplatform.googleapis.com/v1/projects/test/locations/us-central1/publishers/google";

    let result = SiumaiBuilder::new()
        .provider(ProviderType::Gemini)
        .model("gemini-1.5-flash")
        .base_url(vertex_base)
        // Inject Bearer auth header to trigger relaxed validation
        .http_header("Authorization", "Bearer test-token")
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Gemini with Bearer should not require API key"
    );
}

#[tokio::test]
async fn build_gemini_without_key_and_without_bearer_should_fail() {
    // Regular GenAI base_url to validate the failure branch when both Key and Bearer are missing
    let genai_base = "https://generativelanguage.googleapis.com/v1beta";

    let result = SiumaiBuilder::new()
        .provider(ProviderType::Gemini)
        .model("gemini-1.5-flash")
        .base_url(genai_base)
        // No api_key and no Authorization header
        .build()
        .await;

    assert!(
        result.is_err(),
        "Gemini without API key and Authorization should fail"
    );
}
