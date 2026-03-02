//! Moonshot AI - Using `Siumai::builder()` (compatibility / unified interface)
#![allow(deprecated)]
//!
//! This example demonstrates using Moonshot AI through the unified `Siumai` interface.
//!
//! Note: `Siumai::builder()` is a *compatibility convenience* and is **not** the
//! recommended construction style for new code. Prefer config-first clients or the
//! registry handle when possible.
//!
//! ## Features
//! - Unified interface across all providers
//! - Automatic environment variable reading (MOONSHOT_API_KEY)
//! - Provider abstraction for flexible switching
//!
//! ## Run
//! ```bash
//! # Set your Moonshot API key
//! export MOONSHOT_API_KEY="your-api-key-here"
//!
//! # Run the example
//! cargo run --example moonshot-siumai-builder --features openai
//! ```
//!
//! ## Recommended alternative (config-first)
//!
//! ```rust,ignore
//! use siumai::models;
//! use siumai::providers::openai_compatible::OpenAiCompatibleClient;
//!
//! // Reads `MOONSHOT_API_KEY` by default.
//! let client = OpenAiCompatibleClient::from_builtin_env(
//!     "moonshot",
//!     Some(models::openai_compatible::moonshot::KIMI_K2_0905_PREVIEW),
//! )
//! .await?;
//! ```
//!
//! See also: `moonshot-basic.rs`, `moonshot-long-context.rs`, and `moonshot-tools.rs`.

use siumai::models;
use siumai::prelude::unified::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌙 Moonshot AI - Siumai::builder() Example\n");
    println!("===========================================\n");

    // Build Moonshot client using Siumai::builder() (unified interface)
    // Note: API key is automatically read from MOONSHOT_API_KEY environment variable
    let client = Siumai::builder()
        .moonshot()
        .model(models::openai_compatible::moonshot::KIMI_K2_0905_PREVIEW)
        .build()
        .await?;

    println!("✅ Successfully created Moonshot client using Siumai::builder()\n");
    println!("   This demonstrates that the unified interface now supports");
    println!("   automatic environment variable reading for OpenAI-compatible providers.\n");

    // Test basic chat
    println!("📝 Testing Basic Chat\n");

    let response = text::generate(
        &client,
        ChatRequest::new(vec![user!("你好！请用一句话介绍 Moonshot AI。")]),
        text::GenerateOptions::default(),
    )
    .await?;

    println!("Kimi: {}\n", response.content_text().unwrap_or_default());

    // Demonstrate provider abstraction
    println!("🔄 Provider Abstraction Benefits:\n");
    println!("   With Siumai::builder(), you can easily switch providers:");
    println!("   - Change .moonshot() to .deepseek() or .openrouter()");
    println!("   - The rest of your code remains unchanged");
    println!("   - Perfect for multi-provider applications\n");

    Ok(())
}
