//! Simple Chat - Using client.chat()
//!
//! This example demonstrates the simplest chat method.
//! Best for: Quick single-turn conversations without tools.
//!
//! ## Run
//! ```bash
//! cargo run --example simple-chat --features openai
//! ```

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .openai()
        .api_key(&std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini")
        .build()
        .await?;

    // Simple chat - just messages
    let response = client.chat(vec![user!("What is Rust?")]).await?;

    println!("AI: {}", response.content_text().unwrap());

    Ok(())
}
