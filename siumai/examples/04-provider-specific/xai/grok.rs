//! xAI Grok - Using Grok models (config-first)
//!
//! This example demonstrates using xAI's Grok models via the xAI provider module.
//!
//! ## Run
//! ```bash
//! # Set your xAI API key
//! export XAI_API_KEY="your-api-key-here"
//!
//! cargo run --example grok --features xai
//! ```

use siumai::models;
use siumai::prelude::unified::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("xAI Grok Example\n");

    // Config-first: read API key from env (XAI_API_KEY).
    // This client is a provider-owned wrapper over the shared OpenAI-compatible runtime.
    let client =
        siumai::providers::xai::XaiClient::from_builtin_env(Some(models::xai::grok_2::GROK_2_1212))
            .await?;

    let response = text::generate(
        &client,
        ChatRequest::new(vec![user!(
            "What makes Grok unique compared to other AI models?"
        )]),
        text::GenerateOptions::default(),
    )
    .await?;

    println!("Grok: {}", response.content_text().unwrap_or_default());

    Ok(())
}
