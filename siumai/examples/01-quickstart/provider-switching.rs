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
        let cfg = siumai::providers::openai::OpenAiConfig::new(api_key).with_model("gpt-4o-mini");
        let client = siumai::providers::openai::OpenAiClient::from_config(cfg)?;

        let response = text::generate(
            &client,
            ChatRequest::new(vec![user!(prompt)]),
            text::GenerateOptions::default(),
        )
        .await?;
        println!("   {}\n", response.content_text().unwrap());
    }

    // Try Anthropic - same code, different provider config!
    if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
        println!("🤖 Anthropic (Claude 3.5 Haiku):");
        let cfg = siumai::providers::anthropic::AnthropicConfig::new(api_key)
            .with_model("claude-3-5-haiku-20241022");
        let client = siumai::providers::anthropic::AnthropicClient::from_config(cfg)?;

        let response = text::generate(
            &client,
            ChatRequest::new(vec![user!(prompt)]),
            text::GenerateOptions::default(),
        )
        .await?;
        println!("   {}\n", response.content_text().unwrap());
    }

    // Try Google - same code again!
    if let Ok(api_key) = std::env::var("GOOGLE_API_KEY") {
        println!("🤖 Google (Gemini 2.0 Flash):");
        let cfg = siumai::providers::gemini::GeminiConfig::new(api_key)
            .with_model("gemini-2.0-flash-exp".to_string());
        let client = siumai::providers::gemini::GeminiClient::from_config(cfg)?;

        let response = text::generate(
            &client,
            ChatRequest::new(vec![user!(prompt)]),
            text::GenerateOptions::default(),
        )
        .await?;
        println!("   {}\n", response.content_text().unwrap());
    }

    println!("✅ Provider switching completed!");
    println!("💡 Notice: The same text::generate() call works for all providers!");

    Ok(())
}
