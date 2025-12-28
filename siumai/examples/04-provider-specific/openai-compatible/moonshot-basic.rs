//! Moonshot AI - Basic Chat Example
//!
//! This example demonstrates basic usage of Moonshot AI (Kimi) models.
//! Moonshot AI specializes in long-context understanding and Chinese language processing.
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

use siumai::prelude::*;
use siumai::models;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌙 Moonshot AI (Kimi) - Basic Chat Example\n");
    println!("===========================================\n");

    // Build Moonshot client using the latest Kimi K2 model
    // Note: API key is automatically read from MOONSHOT_API_KEY environment variable
    let client = Siumai::builder()
        .moonshot()
        .model(models::openai_compatible::moonshot::KIMI_K2_0905_PREVIEW) // Using model constant
        .build()
        .await?;

    // Example 1: Simple chat
    println!("📝 Example 1: Simple Chat\n");

    let response = client
        .chat(vec![user!(
            "你好！请用中文简单介绍一下 Moonshot AI 的特点。"
        )])
        .await?;

    println!("Kimi: {}\n", response.content_text().unwrap());

    // Example 2: Bilingual conversation
    println!("📝 Example 2: Bilingual Conversation\n");

    let response = client
        .chat(vec![user!(
            "What are the key advantages of Moonshot AI's models?"
        )])
        .await?;

    println!("Kimi: {}\n", response.content_text().unwrap());

    // Example 3: Using different context window models
    println!("📝 Example 3: Different Context Window Models\n");

    // For short conversations, use 8K model (more cost-effective)
    let client_8k = Siumai::builder()
        .moonshot()
        .model(models::openai_compatible::moonshot::MOONSHOT_V1_8K)
        .build()
        .await?;

    let response = client_8k.chat(vec![user!("What is 2+2?")]).await?;

    println!("Kimi (8K): {}\n", response.content_text().unwrap());

    // For long documents, use 128K model
    let client_128k = Siumai::builder()
        .moonshot()
        .model(models::openai_compatible::moonshot::MOONSHOT_V1_128K)
        .build()
        .await?;

    let response = client_128k
        .chat(vec![user!(
            "Explain the concept of long-context understanding in AI models."
        )])
        .await?;

    println!("Kimi (128K): {}\n", response.content_text().unwrap());

    println!("✅ Example completed successfully!");
    println!("\n💡 Tips:");
    println!("   - Use KIMI_K2_0905_PREVIEW for latest features");
    println!("   - Use MOONSHOT_V1_8K for cost-effective short chats");
    println!("   - Use MOONSHOT_V1_128K for long document processing");
    println!("   - Moonshot excels at Chinese language understanding");

    Ok(())
}
