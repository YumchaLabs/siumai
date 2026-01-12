#![cfg(feature = "google")]
use siumai::Provider;

#[tokio::test]
async fn gemini_builder_sets_common_params() {
    let client = Provider::gemini()
        .api_key("test-key")
        .base_url("https://example.googleapis.com/v1beta")
        .model("gemini-1.5-flash")
        .temperature(0.5)
        .max_tokens(2048)
        .top_p(0.8)
        .stop_sequences(vec!["END".to_string()])
        .build()
        .await
        .expect("should build client");

    let cp = &client.common_params;
    assert_eq!(cp.model, "gemini-1.5-flash");
    assert_eq!(cp.temperature, Some(0.5));
    assert_eq!(cp.max_tokens, Some(2048));
    assert_eq!(cp.top_p, Some(0.8));
    assert_eq!(
        cp.stop_sequences.as_ref().unwrap(),
        &vec!["END".to_string()]
    );
}
