//! OpenAI Responses API with prompt_cache_key example
//!
//! Run:
//!   OPENAI_API_KEY=... cargo run --example responses_prompt_cache --features openai
//!
//! This demonstrates how to route to Responses API and set `prompt_cache_key`
//! for caching repeated prompts.

use siumai::providers::openai::{OpenAiClient, OpenAiConfig};
use siumai::params::OpenAiParams;
use siumai::traits::ChatCapability;
use siumai::types::ChatMessage;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    siumai::tracing::init_default_tracing().ok();
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    println!("🚀 OpenAI Responses API + prompt_cache_key\n");

    // Enable Responses API and set a prompt cache key via OpenAiParams
    let mut cfg = OpenAiConfig::new(api_key)
        .with_base_url("https://api.openai.com/v1")
        .with_model("gpt-4o-mini")
        .with_responses_api(true);

    cfg.openai_params = OpenAiParams::builder()
        .prompt_cache_key("demo:responses:sys-prompt:v1")
        .build();

    let client = OpenAiClient::new(cfg, reqwest::Client::new());

    // The same request may benefit from server-side prompt cache when repeated.
    let msg = ChatMessage::user("Give me a short summary of Rust.").build();

    let resp1 = client.chat(vec![msg.clone()]).await?;
    println!("First response: {}\n", resp1.content_text().unwrap_or("<none>"));

    // Repeat to demonstrate cache usage on provider side (cannot assert here,
    // but you can observe latency/usage via provider console or tracing).
    let resp2 = client.chat(vec![msg]).await?;
    println!("Second response: {}\n", resp2.content_text().unwrap_or("<none>"));

    println!("✅ Done");
    Ok(())
}

