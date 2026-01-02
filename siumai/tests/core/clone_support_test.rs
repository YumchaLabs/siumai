//! Clone Support Tests
//!
//! Tests to verify that all major types in the siumai library support Clone
//! for concurrent usage scenarios.

use siumai::experimental::client::{ClientWrapper, LlmClient};
use siumai::prelude::*;
use siumai::provider::Siumai;

/// Test that ClientWrapper implements Clone
#[test]
fn test_client_wrapper_clone() {
    // This test doesn't require async since we're just testing the Clone trait
    // We'll create mock clients for testing

    // Test that ClientWrapper implements Clone
    fn assert_clone<T: Clone>() {}
    assert_clone::<ClientWrapper>();

    // Test that Siumai implements Clone
    assert_clone::<Siumai>();
}

/// Test basic clone functionality
#[tokio::test]
async fn test_basic_clone_functionality() {
    // Test OpenAI client clone
    #[cfg(feature = "openai")]
    {
        let client = Provider::openai()
            .api_key("test-key")
            .model("gpt-4")
            .build()
            .await
            .expect("Failed to build OpenAI client");

        let cloned_client = client.clone();
        // Just test that clone works without errors
        assert_eq!(
            LlmClient::provider_id(&client),
            LlmClient::provider_id(&cloned_client)
        );
    }
}
