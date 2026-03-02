//! Stream with Cancellation - Using `text::stream_with_cancel` (recommended)
//!
//! This example demonstrates how to cancel a stream early.
//! Useful for: User interruptions, timeouts, or conditional stopping.
//!
//! ## Run
//! ```bash
//! cargo run --example stream-with-cancel --features openai
//! ```

use futures::StreamExt;
use siumai::prelude::unified::*;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Recommended construction: resolve a model handle from the registry.
    // Note: API key is automatically read from `OPENAI_API_KEY`.
    let model = registry::global().language_model("openai:gpt-4o-mini")?;

    let handle = text::stream_with_cancel(
        &model,
        ChatRequest::new(vec![user!("Write a very long story about a dragon")]),
        text::StreamOptions::default(),
    )
    .await?;

    println!("AI: ");
    println!("(Will cancel after 2 seconds)\n");

    // Race between stream processing and timeout
    tokio::select! {
        _ = async {
            let stream = handle.stream;
            futures::pin_mut!(stream);
            while let Some(event) = stream.next().await {
                if let Ok(ChatStreamEvent::ContentDelta { delta, .. }) = event {
                    print!("{}", delta);
                    std::io::Write::flush(&mut std::io::stdout()).ok();
                }
            }
        } => {
            println!("\n\n✅ Stream completed naturally");
        }
        _ = tokio::time::sleep(Duration::from_secs(2)) => {
            handle.cancel.cancel();
            println!("\n\n⏹️  Stream cancelled after 2 seconds!");
        }
    }

    Ok(())
}
