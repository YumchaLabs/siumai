//! Ollama Local Models - Run models locally
//!
//! This example demonstrates using Ollama for local model inference.
//! No API key required!
//!
//! ## Setup
//! 1. Install Ollama: https://ollama.ai
//! 2. Pull a model: `ollama pull llama3.2`
//! 3. Run this example
//!
//! ## Run
//! ```bash
//! cargo run --example local-models --features ollama
//! ```

use siumai::prelude::unified::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏠 Ollama Local Models Example\n");

    // No API key needed for Ollama!
    let client = Siumai::builder()
        .ollama()
        .base_url("http://localhost:11434") // Default Ollama URL
        .model("llama3.2") // or "mistral", "codellama", etc.
        .build()
        .await?;

    println!("Using local model: llama3.2\n");

    let response = text::generate(
        &client,
        ChatRequest::new(vec![user!("Explain what Ollama is in one sentence")]),
        text::GenerateOptions::default(),
    )
    .await?;

    println!("AI: {}\n", response.content_text().unwrap_or_default());

    println!("✅ Running models locally - no API costs!");
    println!("💡 Try other models: ollama pull mistral");

    Ok(())
}
