//! Provider Switching - Unified interface across providers
//!
//! This example demonstrates how easy it is to switch between providers
//! using Siumai's unified interface. The same code works with any provider!
//!
//! ## Setup
//! ```bash
//! export OPENAI_API_KEY="your-key"
//! export ANTHROPIC_API_KEY="your-key"
//! export GOOGLE_API_KEY="your-key"
//! ```
//!
//! ## Run
//! ```bash
//! cargo run --example provider-switching --features "openai,anthropic,google"
//! ```

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Siumai Provider Switching Example\n");

    let prompt = "What is 2+2? Answer in one sentence.";

    // Try OpenAI
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        println!("🤖 OpenAI (GPT-4o-mini):");
        let client = Siumai::builder()
            .openai()
            .api_key(&api_key)
            .model("gpt-4o-mini")
            .build()
            .await?;

        let response = client.chat(vec![user!(prompt)]).await?;
        println!("   {}\n", response.content_text().unwrap());
    }

    // Try Anthropic - same code, just different builder!
    if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
        println!("🤖 Anthropic (Claude 3.5 Haiku):");
        let client = Siumai::builder()
            .anthropic()
            .api_key(&api_key)
            .model("claude-3-5-haiku-20241022")
            .build()
            .await?;

        let response = client.chat(vec![user!(prompt)]).await?;
        println!("   {}\n", response.content_text().unwrap());
    }

    // Try Google - same code again!
    if let Ok(api_key) = std::env::var("GOOGLE_API_KEY") {
        println!("🤖 Google (Gemini 2.0 Flash):");
        let client = Siumai::builder()
            .gemini()
            .api_key(&api_key)
            .model("gemini-2.0-flash-exp")
            .build()
            .await?;

        let response = client.chat(vec![user!(prompt)]).await?;
        println!("   {}\n", response.content_text().unwrap());
    }

    println!("✅ Provider switching completed!");
    println!("💡 Notice: The same client.chat() method works for all providers!");

    Ok(())
}
