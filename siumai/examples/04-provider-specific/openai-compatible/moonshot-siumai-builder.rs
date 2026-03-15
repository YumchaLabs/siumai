//! Moonshot AI - Using `Siumai::builder()` (compatibility / unified interface)
#![allow(deprecated)]
//!
//! This example demonstrates using Moonshot AI through the unified `Siumai` interface.
//!
//! Note: `Siumai::builder()` is a *compatibility convenience* and is **not** the
//! recommended construction style for new code. Prefer config-first clients or the
//! registry handle when possible.
//!
//! Package tier:
//! - compat preset example on the shared OpenAI-compatible runtime
//! - this file is intentionally a builder convenience demo
//! - compare it against `moonshot-basic.rs`, `moonshot-long-context.rs`, and `moonshot-tools.rs`
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
    // Build Moonshot client using Siumai::builder() (unified interface)
    // Note: API key is automatically read from `MOONSHOT_API_KEY`.
    let client = Siumai::builder()
        .moonshot()
        .model(models::openai_compatible::moonshot::KIMI_K2_0905_PREVIEW)
        .build()
        .await?;

    println!("Built a Moonshot client through `Siumai::builder()`.");
    println!(
        "This is useful for compatibility comparisons, but config-first remains the preferred path.\n"
    );

    // Test basic chat
    println!("Example: Basic chat through the compatibility builder");

    let response = text::generate(
        &client,
        ChatRequest::new(vec![user!("你好！请用一句话介绍 Moonshot AI。")]),
        text::GenerateOptions::default(),
    )
    .await?;

    println!("Answer:\n{}\n", response.content_text().unwrap_or_default());

    // Demonstrate provider abstraction
    println!("Why keep this demo:");
    println!("- You can switch `.moonshot()` to another compat/provider shortcut");
    println!("- The surrounding request code stays almost unchanged");
    println!("- It is useful when comparing migration or convenience flows");

    Ok(())
}
