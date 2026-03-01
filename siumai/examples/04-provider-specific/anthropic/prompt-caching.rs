//! Anthropic Prompt Caching - Reduce costs for repeated context
//!
//! This example demonstrates Anthropic's prompt caching to reduce costs
//! when using the same context repeatedly.
//!
//! ## Run
//! ```bash
//! cargo run --example prompt-caching --features anthropic
//! ```

use siumai::prelude::unified::*;
use siumai::provider_ext::anthropic::AnthropicChatResponseExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .anthropic()
        .api_key(&std::env::var("ANTHROPIC_API_KEY")?)
        .model("claude-3-5-sonnet-20241022")
        .build()
        .await?;

    println!("💾 Anthropic Prompt Caching Example\n");

    // Large context that we want to cache
    let large_context = "This is a large document that will be cached...".repeat(100);

    // Mark system message for caching
    let system_msg = ChatMessage::system(&large_context)
        .cache_control(CacheControl::Ephemeral)
        .build();

    // First request - will create cache
    println!("First request (creating cache)...");
    let response1 = text::generate(
        &client,
        ChatRequest::new(vec![system_msg.clone(), user!("Summarize the document")]),
        text::GenerateOptions::default(),
    )
    .await?;

    println!("AI: {}\n", response1.content_text().unwrap_or_default());
    if let Some(usage) = &response1.usage {
        println!("Usage: {:?}", usage);
        if let Some(anthropic_meta) = response1.anthropic_metadata()
            && let Some(cache_creation) = anthropic_meta.cache_creation_input_tokens
        {
            println!("Cache creation tokens: {:?}\n", cache_creation);
        }
    }

    // Second request - will use cache
    println!("Second request (using cache)...");
    let response2 = text::generate(
        &client,
        ChatRequest::new(vec![system_msg, user!("What are the key points?")]),
        text::GenerateOptions::default(),
    )
    .await?;

    println!("AI: {}\n", response2.content_text().unwrap_or_default());
    if let Some(usage) = &response2.usage {
        println!("Usage: {:?}", usage);
        if let Some(anthropic_meta) = response2.anthropic_metadata()
            && let Some(cache_read) = anthropic_meta.cache_read_input_tokens
        {
            println!("Cache read tokens: {:?}", cache_read);
            println!("💰 Cost savings from caching!");
        }
    }

    Ok(())
}
