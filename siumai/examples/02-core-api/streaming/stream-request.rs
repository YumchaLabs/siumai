//! Stream Request - Using `text::stream` + `ChatRequest` (Recommended ⭐)
//!
//! This example demonstrates the recommended streaming API in `0.11.0-beta.6+`.
//! Like `ChatRequest`, it preserves all enhanced fields (provider options, http config, etc.).
//!
//! ## Run
//! ```bash
//! cargo run --example stream-request --features openai
//! ```

use futures::StreamExt;
use siumai::prelude::unified::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .openai()
        .api_key(&std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini")
        .build()
        .await?;

    // Build request with ChatRequestBuilder
    let request = ChatRequest::builder()
        .message(user!("Write a haiku about programming"))
        .temperature(0.8)
        .max_tokens(100)
        .build();

    println!("AI: ");
    let mut stream = text::stream(&client, request, text::StreamOptions::default()).await?;

    while let Some(event) = stream.next().await {
        match event? {
            ChatStreamEvent::ContentDelta { delta, .. } => {
                print!("{}", delta);
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            ChatStreamEvent::StreamEnd { response } => {
                println!("\n\n✅ Stream completed!");
                if let Some(usage) = &response.usage {
                    println!("📊 Tokens: {}", usage.total_tokens);
                }
            }
            _ => {}
        }
    }

    Ok(())
}
