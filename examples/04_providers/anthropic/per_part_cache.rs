//! Anthropic per-part Prompt Cache example
//!
//! Run with:
//!   ANTHROPIC_API_KEY=... cargo run --example anthropic_per_part_cache --features anthropic
//!
//! This example demonstrates how to:
//! - Build an Anthropic client
//! - Create a multimodal message (text + image)
//! - Apply message-level and per-part cache_control
//! - Send the chat request and inspect cached token usage

use siumai::prelude::*;
use siumai::types::{CacheControl, ChatMessage};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build Anthropic client
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("set ANTHROPIC_API_KEY to run this example");

    let client = Provider::anthropic()
        .api_key(api_key)
        .model("claude-3-5-sonnet-20241022")
        .build()
        .await?;

    // Message-level cache: cache a heavy system prompt
    let sys = ChatMessage::system("You are a helpful assistant. Long system prompt...")
        .cache_control(CacheControl::Ephemeral)
        .build();

    // Content-level cache: mark specific content parts as cached
    // Compose a multimodal user message: text + image
    let user = ChatMessage::user("Please describe the attached image.")
        .with_image("data:image/jpeg;base64,....".into(), None)
        // Cache the first part (the text prompt itself) to optimize repeated queries
        .cache_control_for_part(0, CacheControl::Ephemeral)
        .build();

    let resp = client.chat(vec![sys, user]).await?;
    println!("Assistant: {}", resp.content_text().unwrap_or("<no content>"));
    if let Some(usage) = resp.usage {
        println!(
            "Usage: prompt={} completion={} total={} cached_tokens={:?}",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens, usage.cached_tokens
        );
    }

    Ok(())
}

