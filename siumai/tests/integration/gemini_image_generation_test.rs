//! Real Gemini Image Generation Integration Test
//!
//! Ignored by default to avoid unintended costs.

use siumai::prelude::*;

#[tokio::test]
#[ignore = "Requires GEMINI_API_KEY and may incur costs"]
async fn test_gemini_image_generation_integration() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("GEMINI_API_KEY")
        .expect("GEMINI_API_KEY environment variable not set");

    // Choose a model that supports image generation via generateContent
    // Adjust if your key/model availability differs
    let client = LlmBuilder::new()
        .gemini()
        .api_key(api_key)
        .model("gemini-2.0-flash-exp")
        .build()
        .await?;

    let mut req = ImageGenerationRequest::default();
    req.prompt = "Generate a simple 256x256 blue circle on white background".to_string();
    req.count = 1;
    // Gemini may return base64 inline data; size hint may be ignored
    req.size = Some("256x256".to_string());

    let resp = client.generate_images(req).await?;
    assert!(!resp.images.is_empty());
    let img = &resp.images[0];
    let has_url = img.url.as_ref().map(|u| !u.is_empty()).unwrap_or(false);
    let has_b64 = img.b64_json.as_ref().map(|b| !b.is_empty()).unwrap_or(false);
    assert!(has_url || has_b64, "expected url or b64_json in image result");
    if has_url {
        println!("💎 Gemini image URL: {}", img.url.as_deref().unwrap());
    } else {
        println!("💎 Gemini image base64 length: {}", img.b64_json.as_deref().unwrap().len());
    }
    Ok(())
}

