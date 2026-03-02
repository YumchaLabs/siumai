//! Chat Request - Using `text::generate` + `ChatRequest` (Recommended ⭐)
//!
//! This example demonstrates the recommended way to do text generation in `0.11.0-beta.6+`.
//! `ChatRequest` supports all enhanced fields: provider options, http config, telemetry, etc.
//!
//! ## Why use `ChatRequest`?
//! - ✅ Full control over all request parameters
//! - ✅ Preserves provider-specific parameters
//! - ✅ Supports HTTP configuration and telemetry
//! - ✅ Future-proof API design
//!
//! ## Run
//! ```bash
//! cargo run --example chat-request --features openai
//! ```

use siumai::prelude::unified::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📝 Chat Request Example (Recommended API)\n");

    // Recommended construction: resolve a model handle from the registry.
    // Note: API key is automatically read from `OPENAI_API_KEY`.
    let model = registry::global().language_model("openai:gpt-4o-mini")?;

    // Method 1: Simple ChatRequest
    println!("Method 1: Simple ChatRequest");
    let request = ChatRequest::new(vec![user!("What is Rust?")]);
    let response = text::generate(&model, request, text::GenerateOptions::default()).await?;
    println!("AI: {}\n", response.content_text().unwrap_or_default());

    // Method 2: ChatRequestBuilder (for complex requests)
    println!("Method 2: ChatRequestBuilder");
    let request = ChatRequest::builder()
        .message(user!("Explain async/await in Rust"))
        .temperature(0.7)
        .max_tokens(500)
        .build();

    let response = text::generate(&model, request, text::GenerateOptions::default()).await?;
    println!("AI: {}\n", response.content_text().unwrap_or_default());

    // Method 3: With multiple parameters
    println!("Method 3: With multiple parameters");
    let request = ChatRequest::builder()
        .message(user!("Tell me a joke"))
        .temperature(0.9)
        .max_tokens(100)
        .build();

    let response = text::generate(&model, request, text::GenerateOptions::default()).await?;
    println!("AI: {}\n", response.content_text().unwrap_or_default());

    println!("✅ Example completed!");
    Ok(())
}
