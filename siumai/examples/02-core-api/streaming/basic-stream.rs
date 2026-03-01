//! Basic Stream - Using `text::stream` (recommended)
//!
//! This example demonstrates basic streaming for real-time responses.
//!
//! ## Run
//! ```bash
//! cargo run --example basic-stream --features openai
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

    println!("AI: ");
    let mut stream = text::stream(
        &client,
        ChatRequest::new(vec![user!("Count from 1 to 10")]),
        text::StreamOptions::default(),
    )
    .await?;

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
