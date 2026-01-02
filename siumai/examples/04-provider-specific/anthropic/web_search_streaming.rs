//! Anthropic - Web Search (provider-hosted tool) with streaming observability
//!
//! This example demonstrates how to observe provider-hosted tool execution in streaming mode:
//! - Anthropic server tools emit `ChatStreamEvent::Custom` events (`anthropic:tool-call` / `anthropic:tool-result`)
//! - Use `siumai::provider_ext::anthropic::tools::AnthropicCustomEvent` to parse them
//!
//! Run:
//! ```bash
//! cargo run --example anthropic-web-search-streaming --features anthropic
//! ```

use futures_util::StreamExt;
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .anthropic()
        .api_key(&std::env::var("ANTHROPIC_API_KEY")?)
        .model("claude-3-7-sonnet-20250219")
        .build()
        .await?;

    let tool = siumai::hosted_tools::anthropic::web_search_20250305()
        .with_max_uses(1)
        .build();

    let mut request =
        ChatRequest::new(vec![user!("Search the web: What is Rust 1.85?")]).with_tools(vec![tool]);
    request.stream = true;

    let mut stream = client.chat_stream_request(request).await?;

    while let Some(ev) = stream.next().await {
        let ev = ev?;

        if let Some(custom) =
            siumai::provider_ext::anthropic::tools::AnthropicCustomEvent::from_stream_event(&ev)
        {
            match custom {
                siumai::provider_ext::anthropic::tools::AnthropicCustomEvent::ProviderToolCall(e) => {
                    println!("provider tool call: {} ({})", e.tool_name, e.tool_call_id);
                }
                siumai::provider_ext::anthropic::tools::AnthropicCustomEvent::ProviderToolResult(e) => {
                    println!("provider tool result: {} ({})", e.tool_name, e.tool_call_id);
                }
                siumai::provider_ext::anthropic::tools::AnthropicCustomEvent::Source(e) => {
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
