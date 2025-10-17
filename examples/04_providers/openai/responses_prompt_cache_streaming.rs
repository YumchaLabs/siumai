//! OpenAI Responses API streaming with prompt_cache_key and tools
//!
//! Run:
//!   OPENAI_API_KEY=... cargo run --example responses_prompt_cache_streaming --features openai
//!
//! Demonstrates:
//! - Route to `/v1/responses` (Responses API)
//! - Set `prompt_cache_key`
//! - Provide a function tool signature
//! - Consume SSE stream and print content deltas

use futures::StreamExt;
use siumai::providers::openai::{OpenAiClient, OpenAiConfig};
use siumai::params::OpenAiParams;
use siumai::stream::ChatStreamEvent;
use siumai::traits::ChatCapability;
use siumai::types::{ChatMessage, Tool, ToolFunction};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    siumai::tracing::init_default_tracing().ok();
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    println!("🚀 OpenAI Responses API — streaming + prompt_cache_key + tools\n");

    // Enable Responses API and set a prompt cache key
    let mut cfg = OpenAiConfig::new(api_key)
        .with_base_url("https://api.openai.com/v1")
        .with_model("gpt-4o-mini")
        .with_responses_api(true);
    cfg.openai_params = OpenAiParams::builder()
        .prompt_cache_key("demo:responses:stream:prompt:v1")
        .build();

    let client = OpenAiClient::new(cfg, reqwest::Client::new());

    // Provide a simple function tool signature
    let tools = vec![Tool {
        r#type: "function".into(),
        function: ToolFunction {
            name: "search".into(),
            description: Some("Search the web".into()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "q": { "type": "string" } },
                "required": ["q"]
            }),
        },
    }];

    // Start streaming
    let mut stream = client
        .chat_stream(vec![ChatMessage::user("Find 2 facts about Rust.").build()], Some(tools))
        .await?;

    // Consume stream (print deltas)
    let mut final_text = String::new();
    while let Some(ev) = stream.next().await {
        match ev? {
            ChatStreamEvent::ContentDelta { delta, .. } => {
                print!("{}", delta);
                final_text.push_str(&delta);
            }
            ChatStreamEvent::StreamEnd { .. } => break,
            _ => {}
        }
    }
    println!("\n\nFinal: {}\n", final_text);

    println!("✅ Done");
    Ok(())
}

