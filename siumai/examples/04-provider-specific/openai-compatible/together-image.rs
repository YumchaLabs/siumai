//! Together image generation on the shared OpenAI-compatible runtime.
//!
//! Package tier:
//! - compat preset example on the shared OpenAI-compatible runtime
//! - preferred path in this file: config-first built-in compat construction
//! - not a dedicated provider-owned image package
//!
//! Credentials:
//! - reads `TOGETHER_API_KEY` from the environment
//!
//! Run:
//! ```bash
//! export TOGETHER_API_KEY="your-api-key-here"
//! cargo run --example together-image --features openai
//! ```

use siumai::prelude::unified::*;
use siumai::provider_ext::openai_compatible::OpenAiCompatibleClient;

const TOGETHER_IMAGE_MODEL: &str = "black-forest-labs/FLUX.1-schnell";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client =
        OpenAiCompatibleClient::from_builtin_env("together", Some(TOGETHER_IMAGE_MODEL)).await?;

    let request = ImageGenerationRequest {
        prompt: "A small blue delivery robot in a rainy alley, cinematic lighting".to_string(),
        negative_prompt: Some("blurry, low quality".to_string()),
        size: Some("1024x1024".to_string()),
        count: 1,
        model: Some(TOGETHER_IMAGE_MODEL.to_string()),
        response_format: Some("url".to_string()),
        ..Default::default()
    };

    let response = image::generate(&client, request, image::GenerateOptions::default()).await?;
    println!("Images returned: {}", response.images.len());

    if let Some(first) = response.images.first() {
        if let Some(url) = &first.url {
            println!("Image URL: {url}");
        } else if let Some(b64) = &first.b64_json {
            println!("Received base64 image payload ({} chars)", b64.len());
        }
    }

    Ok(())
}
