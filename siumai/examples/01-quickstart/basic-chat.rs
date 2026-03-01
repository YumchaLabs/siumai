//! Basic Chat - Simplest way to use Siumai
//!
//! This example shows the most basic chat usage.
//! Works with any provider (OpenAI, Anthropic, Google, Ollama).
//!
//! ## Setup
//! ```bash
//! export OPENAI_API_KEY="your-key"
//! # or ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.
//! ```
//!
//! ## Run
//! ```bash
//! cargo run --example basic-chat --features openai
//! ```

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Siumai Basic Chat Example\n");

    // Build client - change provider here
    let client = Siumai::builder()
        .openai() // or .anthropic() / .google() / .ollama()
        .api_key(&std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini")
        .build()
        .await?;

    // Recommended invocation style: model-family APIs.
    let response = text::generate(
        &client,
        ChatRequest::new(vec![user!("Hello! Introduce yourself in one sentence.")]),
        text::GenerateOptions::default(),
    )
    .await?;

    // Print the response
    println!("AI: {}\n", response.content_text().unwrap());

    // Show usage statistics
    if let Some(usage) = &response.usage {
        println!(
            "📊 Usage: {} tokens ({} prompt + {} completion)",
            usage.total_tokens, usage.prompt_tokens, usage.completion_tokens
        );
    }

    println!("\n✅ Example completed!");
    Ok(())
}
