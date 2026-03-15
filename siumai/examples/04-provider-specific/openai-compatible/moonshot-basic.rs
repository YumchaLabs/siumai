//! Moonshot AI - Basic Chat Example
//!
//! This example demonstrates basic usage of Moonshot AI (Kimi) models.
//! Moonshot AI specializes in long-context understanding and Chinese language processing.
//!
//! Package tier:
//! - compat preset example on the shared OpenAI-compatible runtime
//! - preferred path in this file: config-first built-in compat construction
//! - use `moonshot-siumai-builder.rs` only when you specifically want a builder convenience comparison
//!
//! ## Features
//! - Latest Kimi K2 model with enhanced capabilities
//! - Support for extremely long context (up to 256K tokens)
//! - Excellent Chinese and English bilingual support
//!
//! ## Run
//! ```bash
//! # Set your Moonshot API key
//! export MOONSHOT_API_KEY="your-api-key-here"
//!
//! # Run the example
//! cargo run --example moonshot-basic --features openai
//! ```
//!
//! ## Get API Key
//! Visit https://platform.moonshot.cn/ to get your API key

use siumai::models;
use siumai::prelude::*;
use siumai::provider_ext::openai_compatible::OpenAiCompatibleClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build Moonshot client using the latest Kimi K2 model (config-first).
    // Note: API key is automatically read from `MOONSHOT_API_KEY`.
    let client = OpenAiCompatibleClient::from_builtin_env(
        "moonshot",
        Some(models::openai_compatible::moonshot::KIMI_K2_0905_PREVIEW),
    )
    .await?;

    // Example 1: Simple chat
    println!("Example 1: Basic chat");

    let response = text::generate(
        &client,
        ChatRequest::new(vec![user!(
            "你好！请用中文简单介绍一下 Moonshot AI 的特点。"
        )]),
        text::GenerateOptions::default(),
    )
    .await?;

    println!("Answer:\n{}\n", response.content_text().unwrap_or_default());

    // Example 2: Bilingual conversation
    println!("Example 2: Bilingual prompt");

    let response = text::generate(
        &client,
        ChatRequest::new(vec![user!(
            "What are the key advantages of Moonshot AI's models?"
        )]),
        text::GenerateOptions::default(),
    )
    .await?;

    println!("Answer:\n{}\n", response.content_text().unwrap_or_default());

    // Example 3: Using different context window models
    println!("Example 3: Context window variants");

    // For short conversations, use 8K model (more cost-effective)
    let client_8k = OpenAiCompatibleClient::from_builtin_env(
        "moonshot",
        Some(models::openai_compatible::moonshot::MOONSHOT_V1_8K),
    )
    .await?;

    let response = text::generate(
        &client_8k,
        ChatRequest::new(vec![user!("What is 2+2?")]),
        text::GenerateOptions::default(),
    )
    .await?;

    println!(
        "8K answer:\n{}\n",
        response.content_text().unwrap_or_default()
    );

    // For long documents, use 128K model
    let client_128k = OpenAiCompatibleClient::from_builtin_env(
        "moonshot",
        Some(models::openai_compatible::moonshot::MOONSHOT_V1_128K),
    )
    .await?;

    let response = text::generate(
        &client_128k,
        ChatRequest::new(vec![user!(
            "Explain the concept of long-context understanding in AI models."
        )]),
        text::GenerateOptions::default(),
    )
    .await?;

    println!(
        "128K answer:\n{}\n",
        response.content_text().unwrap_or_default()
    );
    println!("Notes:");
    println!("- Use `KIMI_K2_0905_PREVIEW` for the broadest feature set");
    println!("- Use `MOONSHOT_V1_8K` for short, cost-sensitive chats");
    println!("- Use `MOONSHOT_V1_128K` for longer documents");
    println!("- Moonshot is strong in bilingual Chinese/English workflows");

    Ok(())
}
