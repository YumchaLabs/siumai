//! Gemini - Enterprise Web Search (provider-defined tool)
//!
//! This example shows how to attach Gemini 2.0+ Enterprise Web Search via provider-defined tools.
//! (Actual availability depends on your Google Cloud / Vertex setup and model support.)
//!
//! Run:
//! ```bash
//! cargo run --example google-enterprise-web-search --features google
//! ```

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .gemini()
        .api_key(&std::env::var("GOOGLE_API_KEY")?)
        .model("gemini-2.0-flash-exp")
        .build()
        .await?;

    let request = ChatRequest::builder()
        .message(user!(
            "Use enterprise web search to find and summarize the latest Rust release notes."
        ))
        .tools(vec![siumai::hosted_tools::google::enterprise_web_search()])
        .build();

    let response = client.chat_request(request).await?;
    println!("{}", response.content_text().unwrap_or_default());
    Ok(())
}
