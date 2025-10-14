//! OpenAI Responses API via Unified Siumai Interface
//!
//! This example demonstrates calling the OpenAI Responses path through the
//! unified Siumai interface (no separate Responses client).
//! Key point: when `responses_api` is enabled, `chat/chat_stream` routes to `/responses`.
//!
//! Run: OPENAI_API_KEY=your_key cargo run --example openai_responses_api

use siumai::prelude::*;
use siumai::types::OpenAiBuiltInTool;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    siumai::tracing::init_default_tracing().ok();
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    println!("🚀 OpenAI Responses via Unified Interface\n");

    // Use unified client + enable Responses route
    let config = siumai::providers::openai::OpenAiConfig::new(&api_key)
        .with_model("gpt-4o")
        .with_responses_api(true)
        .with_built_in_tool(OpenAiBuiltInTool::WebSearch);
    let client = siumai::providers::openai::OpenAiClient::new(config, reqwest::Client::new());

    // 1) Basic chat (Responses path)
    println!("1️⃣ Basic Chat (Responses API)");
    let resp = client
        .chat_with_tools(
            vec![user!("What's the weather like today in San Francisco?")],
            None,
        )
        .await?;
    println!("Response: {}\n", resp.content.all_text());

    // 2) Streaming chat (Responses path)
    println!("2️⃣ Streaming (Responses API)");
    let stream = client
        .chat_stream(vec![user!("Give me a 3-line haiku about Rust.")], None)
        .await?;
    let final_resp = siumai::stream::collect_stream_response(stream).await?;
    println!("Final: {}\n", final_resp.content.all_text());

    println!("✅ Done");
    Ok(())
}
