//! Streaming Chat - Real-time response streaming
//!
//! This example demonstrates streaming responses for better user experience.
//! Works with any provider that supports streaming.
//!
//! ## Setup
//! ```bash
//! export OPENAI_API_KEY="your-key"
//! ```
//!
//! ## Run
//! ```bash
//! cargo run --example streaming --features openai
//! ```

use futures::StreamExt;
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌊 Siumai Streaming Example\n");

    // Build client
    let client = Siumai::builder()
        .openai()
        .api_key(&std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini")
        .build()
        .await?;

    // Start streaming
    println!("AI: ");
    let mut stream = client
        .chat_stream(
            vec![user!("Write a short poem about Rust programming")],
            None,
        )
        .await?;

    // Process stream events
    while let Some(event) = stream.next().await {
        match event? {
            ChatStreamEvent::ContentDelta { delta, .. } => {
                print!("{}", delta);
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            ChatStreamEvent::StreamEnd { response } => {
                println!("\n");
                if let Some(usage) = &response.usage {
                    println!(
                        "📊 Usage: {} tokens ({} prompt + {} completion)",
                        usage.total_tokens, usage.prompt_tokens, usage.completion_tokens
                    );
                }
            }
            _ => {}
        }
    }

    println!("\n✅ Streaming completed!");
    Ok(())
}
