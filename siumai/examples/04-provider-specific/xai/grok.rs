//! xAI Grok - Using Grok models
//!
//! This example demonstrates using xAI's Grok models.
//!
//! ## Run
//! ```bash
//! cargo run --example grok --features xai
//! ```

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .xai()
        .api_key(&std::env::var("XAI_API_KEY")?)
        .model("grok-beta")
        .build()
        .await?;

    println!("🤖 xAI Grok Example\n");

    let response = client
        .chat(vec![user!(
            "What makes Grok unique compared to other AI models?"
        )])
        .await?;

    println!("Grok: {}", response.content_text().unwrap());

    Ok(())
}
