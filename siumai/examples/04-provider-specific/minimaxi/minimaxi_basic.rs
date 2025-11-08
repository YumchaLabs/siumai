//! MiniMaxi Basic Example
//!
//! Run with:
//!   cargo run -p siumai --example minimaxi_basic --features minimaxi
//!
//! Env:
//!   set MINIMAXI_API_KEY=your-key

use futures::StreamExt;
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("MiniMaxi Basic Chat Example\n");

    let api_key = std::env::var("MINIMAXI_API_KEY")
        .expect("Please set MINIMAXI_API_KEY environment variable");

    // Build MiniMaxi client (now with SiumaiBuilder::minimaxi())
    let client = Siumai::builder()
        .minimaxi()
        .api_key(&api_key)
        .model("MiniMax-M2")
        .build()
        .await?;

    // Send a simple message
    let resp = client
        .chat(vec![user!("Hello MiniMaxi! Give me one fun fact.")])
        .await?;

    // Optionally show extracted reasoning (default: hidden)
    let show_reasoning = std::env::var("SIUMAI_SHOW_THINKING").ok().as_deref() == Some("1");
    if show_reasoning && resp.has_reasoning() {
        println!("Thinking:\n{}\n", resp.reasoning().join("\n"));
    }

    // Print final answer text (without reasoning tags)
    println!("AI: {}", resp.content_text().unwrap_or_default());

    // --- Streaming demo: prints ThinkingDelta and ContentDelta ---
    println!("\nStreaming (answer only):\n");
    let mut stream = client
        .chat_stream(
            vec![user!("Give me a short fun fact and think step by step.")],
            None,
        )
        .await?;

    let mut printed_any_content = false;

    // Toggle to display streamed thinking via env var
    let show_stream_thinking = std::env::var("SIUMAI_SHOW_THINKING").ok().as_deref() == Some("1");

    while let Some(event) = stream.next().await {
        match event? {
            ChatStreamEvent::ThinkingDelta { delta } => {
                if show_stream_thinking {
                    if !printed_any_content {
                        println!("Thinking: ");
                    }
                    print!("{}", delta);
                    std::io::Write::flush(&mut std::io::stdout())?;
                }
            }
            ChatStreamEvent::ContentDelta { delta, .. } => {
                if !printed_any_content {
                    println!("\n\nAI: ");
                    printed_any_content = true;
                }
                print!("{}", delta);
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            ChatStreamEvent::StreamStart { .. }
            | ChatStreamEvent::UsageUpdate { .. }
            | ChatStreamEvent::ToolCallDelta { .. }
            | ChatStreamEvent::Custom { .. } => {}
            ChatStreamEvent::StreamEnd { response } => {
                if !printed_any_content {
                    // If no deltas were printed (rare), show the final content snapshot
                    if let Some(text) = response.content_text() {
                        println!("\nAI: {}", text);
                    }
                } else {
                    println!();
                }
                if let Some(usage) = &response.usage {
                    println!(
                        "\nUsage: total={} prompt={} completion={}",
                        usage.total_tokens, usage.prompt_tokens, usage.completion_tokens
                    );
                }
            }
            ChatStreamEvent::Error { error } => {
                eprintln!("\n[stream error] {}", error);
            }
        }
    }

    if let Some(usage) = &resp.usage {
        println!(
            "\nUsage: total={} prompt={} completion={}",
            usage.total_tokens, usage.prompt_tokens, usage.completion_tokens
        );
    }

    Ok(())
}
