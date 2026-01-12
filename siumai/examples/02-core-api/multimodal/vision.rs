//! Multimodal Chat - Image understanding
//!
//! This example demonstrates how to send images to multimodal chat models.
//! Supports: OpenAI GPT-4o, Anthropic Claude 3.5, Google Gemini.
//!
//! ## Run
//! ```bash
//! cargo run --example vision --features openai
//! ```

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .openai()
        .api_key(&std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini") // Vision-capable model
        .build()
        .await?;

    // Create message with image using builder pattern
    let message = ChatMessage::user("What's in this image? Describe it in detail.")
        .with_image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Rust_programming_language_black_logo.svg/1200px-Rust_programming_language_black_logo.svg.png".to_string(),
            Some("high".to_string()),
        )
        .build();

    let response = client.chat(vec![message]).await?;

    println!("AI: {}", response.content_text().unwrap());

    Ok(())
}
