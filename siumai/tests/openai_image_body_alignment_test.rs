#![cfg(feature = "openai")]

use siumai::experimental::core::{ProviderContext, ProviderSpec};
use std::collections::HashMap;

#[test]
fn openai_image_generation_body_merges_provider_options_and_defaults_response_format() {
    let spec = siumai_provider_openai::providers::openai::spec::OpenAiSpec::new();
    let ctx = ProviderContext::new(
        "openai",
        "https://api.openai.com/v1",
        Some("test-api-key".to_string()),
        HashMap::new(),
    );

    let mut req = siumai::prelude::unified::ImageGenerationRequest {
        prompt: "A cat".to_string(),
        count: 1,
        model: Some("dall-e-3".to_string()),
        ..Default::default()
    };
    req.provider_options_map.insert(
        "openai",
        serde_json::json!({
            "background": "transparent",
            "output_format": "png"
        }),
    );

    let tx = spec.choose_image_transformers(&req, &ctx);
    let body = tx.request.transform_image(&req).expect("transform");

    assert_eq!(body["model"], "dall-e-3");
    assert_eq!(body["prompt"], "A cat");
    assert_eq!(body["response_format"], "b64_json");
    assert_eq!(body["background"], "transparent");
    assert_eq!(body["output_format"], "png");
}

#[test]
fn openai_image_response_metadata_contains_openai_images() {
    let spec = siumai_provider_openai::providers::openai::spec::OpenAiSpec::new();
    let ctx = ProviderContext::new(
        "openai",
        "https://api.openai.com/v1",
        Some("test-api-key".to_string()),
        HashMap::new(),
    );

    let req = siumai::prelude::unified::ImageGenerationRequest {
        prompt: "A cat".to_string(),
        count: 1,
        model: Some("dall-e-3".to_string()),
        ..Default::default()
    };

    let tx = spec.choose_image_transformers(&req, &ctx);
    let raw = serde_json::json!({
        "created": 123,
        "size": "1024x1024",
        "quality": "high",
        "background": "transparent",
        "output_format": "png",
        "usage": { "input_tokens": 1, "output_tokens": 2, "total_tokens": 3 },
        "data": [
            { "b64_json": "abc", "revised_prompt": "A cute cat" }
        ]
    });

    let resp = tx
        .response
        .transform_image_response(&raw)
        .expect("transform");
    let openai = resp
        .metadata
        .get("openai")
        .and_then(|v| v.as_object())
        .expect("openai meta");
    let images = openai
        .get("images")
        .and_then(|v| v.as_array())
        .expect("images");
    assert_eq!(images.len(), 1);
    assert_eq!(
        images[0].get("outputFormat").and_then(|v| v.as_str()),
        Some("png")
    );
    assert_eq!(images[0].get("created").and_then(|v| v.as_u64()), Some(123));
    assert_eq!(
        images[0].get("revisedPrompt").and_then(|v| v.as_str()),
        Some("A cute cat")
    );
}
