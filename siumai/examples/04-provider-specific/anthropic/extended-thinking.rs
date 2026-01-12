//! Anthropic Extended Thinking - Deep reasoning
//!
//! This example demonstrates Anthropic's extended thinking capability
//! for complex reasoning tasks using type-safe provider options.
//!
//! ## Run
//! ```bash
//! cargo run --example extended-thinking --features anthropic
//! ```

use siumai::prelude::*;
use siumai::provider_ext::anthropic::{
    AnthropicChatRequestExt, AnthropicOptions, ThinkingModeConfig,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .anthropic()
        .api_key(&std::env::var("ANTHROPIC_API_KEY")?)
        .model("claude-3-7-sonnet-20250219")
        .build()
        .await?;

    println!("🧠 Anthropic Extended Thinking Example\n");

    // ✅ New API: Use type-safe AnthropicOptions with ThinkingModeConfig
    let request = ChatRequest::builder()
        .message(user!("Solve this complex problem: A farmer has 17 sheep. All but 9 die. How many are left? Explain your reasoning step by step."))
        .build()
        .with_anthropic_options(AnthropicOptions::new().with_thinking_mode(ThinkingModeConfig {
            enabled: true,
            thinking_budget: Some(5000),
        }));

    let response = client.chat_request(request).await?;

    // The response will include thinking process
    let reasoning = response.reasoning();
    if !reasoning.is_empty() {
        println!("💭 Thinking process:");
        for r in reasoning {
            println!("{}\n", r);
        }
    }

    println!("📝 Final answer:");
    println!("{}", response.content_text().unwrap());
    println!();

    println!("💡 Migration Note:");
    println!("   Old API: ProviderParams::new().with_param(\"thinking\", json!({{...}}))");
    println!(
        "   New API: AnthropicOptions::new().with_thinking_mode(ThinkingModeConfig {{ enabled: true, thinking_budget: Some(5000) }})"
    );
    println!("   Benefits: Type-safe struct, no JSON serialization, compile-time validation");

    Ok(())
}
