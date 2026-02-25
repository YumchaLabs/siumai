//! OpenAI Responses - WebSocket incremental session
//!
//! This example demonstrates an agentic-friendly pattern:
//! - keep a persistent WebSocket connection for streaming `/responses`
//! - cache instructions/tools on the connection (warm-up with `generate=false`)
//! - send only incremental messages per step
//!
//! ## Run
//! ```bash
//! cargo run --example openai-responses-websocket-incremental --features openai-websocket
//! ```

use futures_util::StreamExt;
use siumai::prelude::*;
use siumai::provider_ext::openai::{
    OpenAiConfig, OpenAiIncrementalWebSocketSession, OpenAiWebSocketSession,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")?;

    // Create a WebSocket session from a plain config (no unified builder required).
    let config = OpenAiConfig::new(api_key)
        .with_base_url("https://api.openai.com/v1")
        .with_model("gpt-4.1-mini");
    let session = OpenAiWebSocketSession::from_config(config, reqwest::Client::new())?;

    // Cache defaults on the connection so each step can send only incremental messages.
    let mut inc = OpenAiIncrementalWebSocketSession::new(session)
        .cache_defaults_on_connection(None, Some("You are a concise assistant.".to_string()))
        .await?;

    for prompt in [
        "Hello! What can you do?",
        "Summarize your answer in 1 sentence.",
    ] {
        inc.push_user_text(prompt);
        let ChatStreamHandle {
            mut stream,
            cancel: _,
        } = inc.stream_next_with_cancel().await?;

        while let Some(item) = stream.next().await {
            let ev = item?;
            if let ChatStreamEvent::ContentDelta { delta, .. } = ev {
                print!("{delta}");
            }
        }
        println!("\n---");
    }

    Ok(())
}
