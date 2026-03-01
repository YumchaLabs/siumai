//! Basic Registry - String-driven model resolution
//!
//! This example demonstrates using the Provider Registry for
//! string-driven model resolution with "provider:model" format.
//!
//! ## Run
//! ```bash
//! cargo run --example basic-registry --features "openai,anthropic"
//! ```

use siumai::prelude::unified::*;
use siumai::registry::helpers::create_registry_with_defaults;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📚 Provider Registry Example\n");

    // Create registry with default middlewares
    let registry = create_registry_with_defaults();

    // Use OpenAI via registry
    if std::env::var("OPENAI_API_KEY").is_ok() {
        println!("Using OpenAI via registry:");
        let lm = registry.language_model("openai:gpt-4o-mini")?;
        let response = text::generate(
            &lm,
            ChatRequest::new(vec![user!("Say hello!")]),
            text::GenerateOptions::default(),
        )
        .await?;
        println!("  {}\n", response.content_text().unwrap_or_default());
    }

    // Use Anthropic via registry
    if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        println!("Using Anthropic via registry:");
        let lm = registry.language_model("anthropic:claude-3-5-haiku-20241022")?;
        let response = text::generate(
            &lm,
            ChatRequest::new(vec![user!("Say hello!")]),
            text::GenerateOptions::default(),
        )
        .await?;
        println!("  {}\n", response.content_text().unwrap_or_default());
    }

    // Use Google via registry
    if std::env::var("GOOGLE_API_KEY").is_ok() {
        println!("Using Google via registry:");
        let lm = registry.language_model("google:gemini-2.0-flash-exp")?;
        let response = text::generate(
            &lm,
            ChatRequest::new(vec![user!("Say hello!")]),
            text::GenerateOptions::default(),
        )
        .await?;
        println!("  {}\n", response.content_text().unwrap_or_default());
    }

    println!("✅ Registry allows string-driven model selection!");

    Ok(())
}
