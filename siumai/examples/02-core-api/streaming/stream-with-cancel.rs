//! Stream with Cancellation - Using client.chat_stream_with_cancel()
//!
//! This example demonstrates how to cancel a stream early.
//! Useful for: User interruptions, timeouts, or conditional stopping.
//!
//! ## Run
//! ```bash
//! cargo run --example stream-with-cancel --features openai
//! ```

use futures::StreamExt;
use siumai::prelude::*;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .openai()
        .api_key(&std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini")
        .build()
        .await?;

    // Get cancellable stream handle
    let handle = client
        .chat_stream_with_cancel(vec![user!("Write a very long story about a dragon")], None)
        .await?;

    println!("AI: ");
    println!("(Will cancel after 2 seconds)\n");

    // Race between stream processing and timeout
    tokio::select! {
        _ = async {
            futures::pin_mut!(handle.stream);
            while let Some(event) = handle.stream.next().await {
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
