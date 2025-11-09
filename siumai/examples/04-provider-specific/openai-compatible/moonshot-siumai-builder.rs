//! Moonshot AI - Using Siumai::builder() (Unified Interface)
//!
//! This example demonstrates using Moonshot AI through the unified Siumai interface.
//! This approach is useful when you need provider abstraction or want to switch
//! providers dynamically.
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
//! ## Comparison with LlmBuilder
//!
//! **Using Siumai::builder() (this example)**:
//! ```rust,ignore
//! let client = Siumai::builder()
//!     .moonshot()
//!     .model(moonshot::KIMI_K2_0905_PREVIEW)
//!     .build()
//!     .await?;
//! ```
//! - Returns: `Siumai` (unified interface)
//! - Use case: Provider abstraction, dynamic switching
//! - Environment variable: Automatically reads MOONSHOT_API_KEY
//!
//! **Using LlmBuilder::new() (recommended for direct usage)**:
//! ```rust,ignore
//! let client = LlmBuilder::new()
//!     .moonshot()
//!     .model(moonshot::KIMI_K2_0905_PREVIEW)
//!     .build()
//!     .await?;
//! ```
//! - Returns: `OpenAiCompatibleClient` (specific type)
//! - Use case: Direct provider usage, type-specific features
//! - Environment variable: Automatically reads MOONSHOT_API_KEY

use siumai::prelude::*;
use siumai::providers::openai_compatible::moonshot;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌙 Moonshot AI - Siumai::builder() Example\n");
    println!("===========================================\n");

    // Build Moonshot client using Siumai::builder() (unified interface)
    // Note: API key is automatically read from MOONSHOT_API_KEY environment variable
    let client = Siumai::builder()
        .moonshot()
        .model(moonshot::KIMI_K2_0905_PREVIEW)
        .build()
        .await?;

    println!("✅ Successfully created Moonshot client using Siumai::builder()\n");
    println!("   This demonstrates that the unified interface now supports");
    println!("   automatic environment variable reading for OpenAI-compatible providers.\n");

    // Test basic chat
    println!("📝 Testing Basic Chat\n");

    let response = client
        .chat(vec![user!("你好！请用一句话介绍 Moonshot AI。")])
        .await?;

    println!("Kimi: {}\n", response.content_text().unwrap());

    // Demonstrate provider abstraction
    println!("🔄 Provider Abstraction Benefits:\n");
    println!("   With Siumai::builder(), you can easily switch providers:");
    println!("   - Change .moonshot() to .deepseek() or .openrouter()");
    println!("   - The rest of your code remains unchanged");
    println!("   - Perfect for multi-provider applications\n");

    Ok(())
}
