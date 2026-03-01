//! Simple Chat - Using `text::generate` (recommended)
//!
//! This example demonstrates the simplest text generation request.
//! Best for: Quick single-turn conversations without tools.
//!
//! ## Run
//! ```bash
//! cargo run --example simple-chat --features openai
//! ```

use siumai::prelude::unified::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .openai()
        .api_key(&std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini")
        .build()
        .await?;

    // Simple request - just messages
    let response = text::generate(
        &client,
        ChatRequest::new(vec![user!("What is Rust?")]),
        text::GenerateOptions::default(),
    )
    .await?;

    println!("AI: {}", response.content_text().unwrap_or_default());

    Ok(())
}
