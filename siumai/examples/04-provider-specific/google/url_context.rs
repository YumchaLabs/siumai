//! Gemini - URL Context (provider-defined tool)
//!
//! This example shows how to enable Gemini 2.0+ URL context retrieval via provider-defined tools.
//!
//! Run:
//! ```bash
//! cargo run --example google-url-context --features google
//! ```

use siumai::prelude::*;
use siumai::provider_ext::gemini::GeminiChatResponseExt;

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
            "Using URL context, summarize this page in 5 bullet points: https://www.rust-lang.org/"
        ))
        .tools(vec![siumai::hosted_tools::google::url_context()])
        .build();

    let response = client.chat_request(request).await?;
    println!("{}", response.content_text().unwrap_or_default());

    if let Some(meta) = response.gemini_metadata()
        && let Some(url_ctx) = &meta.url_context_metadata
    {
        println!("\nURL context metadata:");
        println!("{}", serde_json::to_string_pretty(url_ctx).unwrap());
    }

    Ok(())
}
