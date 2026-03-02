//! Anthropic Extensions - Extended Thinking (explicit extension API)
//!
//! This example demonstrates how to use Anthropic-specific helpers via `siumai::provider_ext`,
//! while still constructing the client through the unified `Siumai` interface.
//!
//! ## Run
//! ```bash
//! cargo run --example extended-thinking-ext --features anthropic
//! ```

use siumai::prelude::unified::*;
use siumai::provider_ext::anthropic::ThinkingModeConfig;
use siumai::provider_ext::anthropic::ext::{structured_output, thinking};
use siumai::user;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Recommended construction: resolve a model handle from the registry.
    // Note: API key is automatically read from `ANTHROPIC_API_KEY`.
    let client = registry::global().language_model("anthropic:claude-3-7-sonnet-20250219")?;

    // Extension API: enable Anthropic thinking mode for this request.
    let request = ChatRequest::builder()
        .message(user!(
            "A farmer has 17 sheep. All but 9 die. How many are left? Explain your reasoning."
        ))
        .build();

    let response = thinking::chat_with_thinking(
        &client,
        request,
        ThinkingModeConfig {
            enabled: true,
            thinking_budget: Some(5_000),
        },
    )
    .await?;

    println!("Thinking blocks: {}", response.reasoning().len());
    println!("Answer: {}", response.content_text().unwrap_or_default());

    // Extension API: request a JSON object response for structured output.
    let request = ChatRequest::builder()
        .message(user!(
            "Return a JSON object with keys: answer (string), confidence (number)."
        ))
        .build();

    let response = structured_output::chat_with_json_object(&client, request).await?;
    println!(
        "Structured output: {}",
        response.content_text().unwrap_or_default()
    );

    Ok(())
}
