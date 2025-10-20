//! Anthropic Extended Thinking - Deep reasoning
//!
//! This example demonstrates Anthropic's extended thinking capability
//! for complex reasoning tasks.
//!
//! ## Run
//! ```bash
//! cargo run --example extended-thinking --features anthropic
//! ```

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .anthropic()
        .api_key(&std::env::var("ANTHROPIC_API_KEY")?)
        .model("claude-3-7-sonnet-20250219")
        .build()
        .await?;

    println!("🧠 Anthropic Extended Thinking Example\n");

    // Enable extended thinking via provider params
    let provider_params = ProviderParams::new().with_param(
        "thinking",
        serde_json::json!({
            "type": "enabled",
            "budget_tokens": 5000
        }),
    );

    let request = ChatRequest::builder()
        .message(user!("Solve this complex problem: A farmer has 17 sheep. All but 9 die. How many are left? Explain your reasoning step by step."))
        .provider_params(provider_params)
        .build();

    let response = client.chat_request(request).await?;

    // The response will include thinking process
    if let Some(thinking) = response.thinking {
        println!("💭 Thinking process:");
        println!("{}\n", thinking);
    }

    println!("📝 Final answer:");
    println!("{}", response.content_text().unwrap());

    Ok(())
}
