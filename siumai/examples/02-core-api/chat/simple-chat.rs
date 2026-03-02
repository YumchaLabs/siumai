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
    // Recommended construction: resolve a model handle from the registry.
    // Note: API key is automatically read from `OPENAI_API_KEY`.
    let model = registry::global().language_model("openai:gpt-4o-mini")?;

    // Simple request - just messages
    let response = text::generate(
        &model,
        ChatRequest::new(vec![user!("What is Rust?")]),
        text::GenerateOptions::default(),
    )
    .await?;

    println!("AI: {}", response.content_text().unwrap_or_default());

    Ok(())
}
