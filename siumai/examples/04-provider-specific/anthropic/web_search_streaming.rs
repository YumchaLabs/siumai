//! Anthropic - Web Search (provider-hosted tool) with streaming observability
//!
//! This example demonstrates how to observe provider-hosted tool execution in streaming mode:
//! - Anthropic server tools emit `ChatStreamEvent::Custom` events (`anthropic:tool-call` / `anthropic:tool-result`)
//! - Use `siumai::provider_ext::anthropic::ext::tools::AnthropicCustomEvent` to parse them
//!
//! Run:
//! ```bash
//! cargo run --example anthropic-web-search-streaming --features anthropic
//! ```

use futures_util::StreamExt;
use siumai::prelude::unified::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Recommended construction: resolve a model handle from the registry.
    // Note: API key is automatically read from `ANTHROPIC_API_KEY`.
    let model = registry::global().language_model("anthropic:claude-3-7-sonnet-20250219")?;

    let tool = siumai::hosted_tools::anthropic::web_search_20250305()
        .with_max_uses(1)
        .build();

    let request =
        ChatRequest::new(vec![user!("Search the web: What is Rust 1.85?")]).with_tools(vec![tool]);

    let mut stream = text::stream(&model, request, text::StreamOptions::default()).await?;

    while let Some(ev) = stream.next().await {
        let ev = ev?;

        if let Some(custom) =
            siumai::provider_ext::anthropic::ext::tools::AnthropicCustomEvent::from_stream_event(
                &ev,
            )
        {
            match custom {
                siumai::provider_ext::anthropic::ext::tools::AnthropicCustomEvent::ProviderToolCall(e) => {
                    println!("provider tool call: {} ({})", e.tool_name, e.tool_call_id);
                }
                siumai::provider_ext::anthropic::ext::tools::AnthropicCustomEvent::ProviderToolResult(e) => {
                    println!("provider tool result: {} ({})", e.tool_name, e.tool_call_id);
                }
                siumai::provider_ext::anthropic::ext::tools::AnthropicCustomEvent::Source(e) => {
                    println!("source: {} ({})", e.url, e.id);
                }
            }
            continue;
        }

        match ev {
            ChatStreamEvent::ContentDelta { delta, .. } => print!("{delta}"),
            ChatStreamEvent::StreamEnd { .. } => break,
            _ => {}
        }
    }

    Ok(())
}
