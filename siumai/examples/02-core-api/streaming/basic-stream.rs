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

fn stream_text_delta(event: &ChatStreamEvent) -> Option<&str> {
    match event {
        ChatStreamEvent::ContentDelta { delta, .. } => Some(delta.as_str()),
        ChatStreamEvent::Part {
            part: ChatStreamPart::TextDelta { delta, .. },
        }
        | ChatStreamEvent::PartWithReplay {
            part: ChatStreamPart::TextDelta { delta, .. },
            ..
        } => Some(delta.as_str()),
        _ => None,
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Recommended construction: resolve a model handle from the registry.
    // Note: API key is automatically read from `OPENAI_API_KEY`.
    let model = registry::global().language_model("openai:gpt-4o-mini")?;

    println!("AI: ");
    let mut stream = text::stream(
        &model,
        ChatRequest::new(vec![user!("Count from 1 to 10")]),
        text::StreamOptions::default(),
    )
    .await?;

    while let Some(event) = stream.next().await {
        let event = event?;
        if let Some(delta) = stream_text_delta(&event) {
            print!("{}", delta);
            std::io::Write::flush(&mut std::io::stdout())?;
            continue;
        }

        match event {
            ChatStreamEvent::StreamEnd { response } => {
                println!("\n\n✅ Stream completed!");
                if let Some(usage) = &response.usage {
                    println!("📊 Tokens: {}", usage.total_tokens().unwrap_or(0));
                }
            }
            _ => {}
        }
    }

    Ok(())
}
