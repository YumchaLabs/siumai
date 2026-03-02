//! Anthropic Extended Thinking - Deep reasoning
//!
//! This example demonstrates Anthropic's extended thinking capability
//! for complex reasoning tasks using type-safe provider options.
//!
//! ## Run
//! ```bash
//! cargo run --example extended-thinking --features anthropic
//! ```

use siumai::prelude::unified::*;
use siumai::provider_ext::anthropic::{
    AnthropicChatRequestExt, AnthropicOptions, ThinkingModeConfig,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Recommended construction: resolve a model handle from the registry.
    // Note: API key is automatically read from `ANTHROPIC_API_KEY`.
    let model = registry::global().language_model("anthropic:claude-3-7-sonnet-20250219")?;

    println!("🧠 Anthropic Extended Thinking Example\n");

    // ✅ New API: Use type-safe AnthropicOptions with ThinkingModeConfig
    let request = ChatRequest::builder()
        .message(user!("Solve this complex problem: A farmer has 17 sheep. All but 9 die. How many are left? Explain your reasoning step by step."))
        .build()
        .with_anthropic_options(AnthropicOptions::new().with_thinking_mode(ThinkingModeConfig {
            enabled: true,
            thinking_budget: Some(5000),
        }));

    let response = text::generate(&model, request, text::GenerateOptions::default()).await?;

    // The response will include thinking process
    let reasoning = response.reasoning();
    if !reasoning.is_empty() {
        println!("💭 Thinking process:");
        for r in reasoning {
            println!("{}\n", r);
        }
    }

    println!("📝 Final answer:");
    println!("{}", response.content_text().unwrap_or_default());
    println!();

    println!("💡 Migration Note:");
    println!("   Old API: ProviderParams::new().with_param(\"thinking\", json!({{...}}))");
    println!(
        "   New API: AnthropicOptions::new().with_thinking_mode(ThinkingModeConfig {{ enabled: true, thinking_budget: Some(5000) }})"
    );
    println!("   Benefits: Type-safe struct, no JSON serialization, compile-time validation");

    Ok(())
}
