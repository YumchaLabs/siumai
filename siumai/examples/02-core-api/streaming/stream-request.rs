//! Stream Request - Using client.chat_stream_request() (Recommended ⭐)
//!
//! This example demonstrates the recommended way to use streaming in 0.11.0+.
//! Similar to chat_request, this preserves all enhanced fields.
//!
//! ## Run
//! ```bash
//! cargo run --example stream-request --features openai
//! ```

use futures::StreamExt;
use siumai::prelude::*;

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

    // Stream with chat_stream_request
    println!("AI: ");
    let mut stream = client.chat_stream_request(request).await?;

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
