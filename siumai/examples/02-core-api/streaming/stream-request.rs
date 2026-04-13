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

    // Build request with ChatRequestBuilder
    let request = ChatRequest::builder()
        .message(user!("Write a haiku about programming"))
        .temperature(0.8)
        .max_tokens(100)
        .build();

    println!("AI: ");
    let mut stream = text::stream(&model, request, text::StreamOptions::default()).await?;

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
