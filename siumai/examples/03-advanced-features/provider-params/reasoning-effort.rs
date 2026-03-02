//! Reasoning Effort - Control thinking depth
//!
//! This example shows how to use type-safe provider options to control reasoning effort.
//! Supported by: OpenAI o1/o3 models, xAI Grok models.
//!
//! ## Run
//! ```bash
//! cargo run --example reasoning-effort --features openai
//! ```

use siumai::prelude::unified::*;
use siumai::provider_ext::openai::{OpenAiChatRequestExt, OpenAiOptions, ReasoningEffort};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Recommended construction: resolve a model handle from the registry.
    // Note: API key is automatically read from `OPENAI_API_KEY`.
    let model = registry::global().language_model("openai:o1-mini")?; // Reasoning model

    println!("🧠 Reasoning Effort Example\n");

    // ✅ New API: Use type-safe OpenAiOptions with reasoning_effort
    let request = ChatRequest::builder()
        .message(user!("Solve this logic puzzle: If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies?"))
        .build()
        .with_openai_options(
            OpenAiOptions::new().with_reasoning_effort(ReasoningEffort::High),
        );

    let response = text::generate(&model, request, text::GenerateOptions::default()).await?;

    println!("🤖 AI (with high reasoning effort):");
    println!("{}", response.content_text().unwrap_or_default());
    println!();

    println!("💡 Migration Note:");
    println!("   Old API: ProviderParams::new().with_param(\"reasoning_effort\", json!(\"high\"))");
    println!("   New API: OpenAiOptions::new().with_reasoning_effort(ReasoningEffort::High)");
    println!("   Benefits: Type-safe enum, compile-time validation, no typos!");

    Ok(())
}
