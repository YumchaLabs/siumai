//! Real Image Generation Integration Tests
//!
//! These tests perform real image generation calls against providers.
//! They are ignored by default to avoid unintended costs.

use siumai::prelude::*;

/// OpenAI image generation (url format)
#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY and may incur costs"]
async fn test_openai_image_generation_integration() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable not set");

    // Build OpenAI client (images via images/generations)
    let client = Siumai::builder()
        .openai()
        .api_key(api_key)
        .model("gpt-image-1")
        .build()
        .await?;

    // Request a small, fast image
    let mut req = ImageGenerationRequest::default();
    req.prompt = "A small 128x128 pixel red square, minimalist".to_string();
    req.count = 1;
    req.size = Some("256x256".to_string());
    req.response_format = Some("url".to_string());

    let resp = client.generate_images(req).await?;
    assert_eq!(resp.images.len(), 1);
    let img = &resp.images[0];
    assert!(img.url.is_some(), "image url should be present");
    println!("🖼️ Image URL: {}", img.url.as_deref().unwrap());
    Ok(())
}
