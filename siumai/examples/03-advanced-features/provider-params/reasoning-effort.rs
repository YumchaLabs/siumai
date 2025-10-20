//! Reasoning Effort - Control thinking depth
//!
//! This example shows how to use type-safe provider options to control reasoning effort.
//! Supported by: OpenAI o1/o3 models, xAI Grok models.
//!
//! ## Run
//! ```bash
//! cargo run --example reasoning-effort --features openai
//! ```

use siumai::prelude::*;
use siumai::types::{OpenAiOptions, ReasoningEffort};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .openai()
        .api_key(&std::env::var("OPENAI_API_KEY")?)
        .model("o1-mini") // Reasoning model
        .build()
        .await?;

    println!("🧠 Reasoning Effort Example\n");

    // ✅ New API: Use type-safe OpenAiOptions with reasoning_effort
    let request = ChatRequest::builder()
        .message(user!("Solve this logic puzzle: If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies?"))
        .openai_options(
            OpenAiOptions::new()
                .with_reasoning_effort(ReasoningEffort::High)
        )
        .build();

    let response = client.chat_request(request).await?;

    println!("🤖 AI (with high reasoning effort):");
    println!("{}", response.content_text().unwrap());
    println!();

    println!("💡 Migration Note:");
    println!("   Old API: ProviderParams::new().with_param(\"reasoning_effort\", json!(\"high\"))");
    println!("   New API: OpenAiOptions::new().with_reasoning_effort(ReasoningEffort::High)");
    println!("   Benefits: Type-safe enum, compile-time validation, no typos!");

    Ok(())
}
