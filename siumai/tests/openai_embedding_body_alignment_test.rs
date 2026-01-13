#![cfg(feature = "openai")]

use siumai::experimental::core::{ProviderContext, ProviderSpec};
use std::collections::HashMap;

#[test]
fn openai_embedding_body_defaults_match_vercel() {
    let spec = siumai_provider_openai::providers::openai::spec::OpenAiSpec::new();
    let ctx = ProviderContext::new(
        "openai",
        "https://api.openai.com/v1",
        Some("test-api-key".to_string()),
        HashMap::new(),
    );

    let mut req = siumai::prelude::unified::EmbeddingRequest::single("hello")
        .with_model("text-embedding-3-small");

    // Vercel alignment: dimensions/user come from providerOptions.
    req.provider_options_map.insert(
        "openai",
        serde_json::json!({
            "dimensions": 512,
            "user": "end_user_1"
        }),
    );

    let tx = spec.choose_embedding_transformers(&req, &ctx);
    let body = tx.request.transform_embedding(&req).expect("transform");

    assert_eq!(body["model"], "text-embedding-3-small");
    assert_eq!(body["encoding_format"], "float");
    assert_eq!(body["dimensions"], 512);
    assert_eq!(body["user"], "end_user_1");
}
