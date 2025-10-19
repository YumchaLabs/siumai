// Example: Using Registry image_model to generate an image
// Run with: cargo run --example image --features openai

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "openai"))]
    {
        eprintln!("Enable --features openai to run this example");
        return Ok(());
    }

    #[cfg(feature = "openai")]
    {
        if std::env::var("OPENAI_API_KEY").is_err() {
            eprintln!("Set OPENAI_API_KEY to run this example");
            return Ok(());
        }
        let reg = siumai::registry::helpers::create_registry_with_defaults();
        // OpenAI image generation model (adjust if needed)
        let im = reg.image_model("openai:gpt-image-1")?;
        let req = ImageGenerationRequest {
            prompt: "A cute robot drawing standing on a beach".into(),
            size: Some("1024x1024".into()),
            count: 1,
            ..Default::default()
        };
        let resp = im.generate_images(req).await?;
        println!("generated images: {}", resp.images.len());
    }
    Ok(())
}

