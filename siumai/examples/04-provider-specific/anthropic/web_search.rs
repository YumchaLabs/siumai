//! Anthropic - Web Search (provider-defined tool)
//!
//! This example demonstrates Vercel-aligned usage of provider-defined tools:
//! - Attach provider-defined tools via `ChatRequest::with_tools`
//! - Use `hosted_tools::anthropic::web_search_20250305` to construct the tool
//!
//! Run:
//! ```bash
//! cargo run --example anthropic-web-search --features anthropic
//! ```

use siumai::prelude::*;
use siumai::provider_ext::anthropic::AnthropicChatResponseExt;

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

    let request =
        ChatRequest::new(vec![user!("Search the web: What is Rust 1.85?")]).with_tools(vec![tool]);

    let response = client.chat_request(request).await?;
    println!("{}", response.content_text().unwrap_or_default());

    if let Some(meta) = response.anthropic_metadata() {
        if let Some(stu) = meta.server_tool_use {
            println!(
                "server_tool_use: web_search_requests={:?}, web_fetch_requests={:?}",
                stu.web_search_requests, stu.web_fetch_requests
            );
        }
        if let Some(citations) = meta.citations {
            println!("citations blocks = {}", citations.len());
        }
        if let Some(sources) = meta.sources {
            println!("sources = {}", sources.len());
            if let Some(first) = sources.first() {
                println!(
                    "first source url = {}",
                    first.url.as_deref().unwrap_or("<none>")
                );
            }
        }
    }

    Ok(())
}
