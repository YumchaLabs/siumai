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

    // Resolve a model via the registry - change provider/model here
    let reg = registry::global();
    let model = reg.language_model("openai:gpt-4o-mini")?;

    // Start streaming
    println!("AI: ");
    let mut stream = text::stream(
        &model,
        ChatRequest::new(vec![user!("Write a short poem about Rust programming")]),
        text::StreamOptions::default(),
    )
    .await?;
    // Process stream events
    while let Some(event) = stream.next().await {
        let event = event?;
        if let Some(delta) = event.text_delta() {
            print!("{}", delta);
            std::io::Write::flush(&mut std::io::stdout())?;
            continue;
        }

        if let ChatStreamEvent::StreamEnd { response } = event {
            println!("\n");
            if let Some(usage) = &response.usage {
                println!(
                    "📊 Usage: {} tokens ({} prompt + {} completion)",
                    usage.total_tokens().unwrap_or(0),
                    usage.prompt_tokens().unwrap_or(0),
                    usage.completion_tokens().unwrap_or(0)
                );
            }
        }
    }

    println!("\n✅ Streaming completed!");
    Ok(())
}
