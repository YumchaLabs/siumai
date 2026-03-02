//! Gemini - URL Context (provider-defined tool)
//!
//! This example shows how to enable Gemini 2.0+ URL context retrieval via provider-defined tools.
//!
//! Run:
//! ```bash
//! cargo run --example google-url-context --features google
//! ```

use siumai::prelude::unified::*;
use siumai::provider_ext::gemini::GeminiChatResponseExt;

fn read_api_key() -> Result<String, Box<dyn std::error::Error>> {
    if let Ok(k) = std::env::var("GOOGLE_API_KEY") {
        return Ok(k);
    }
    if let Ok(k) = std::env::var("GEMINI_API_KEY") {
        return Ok(k);
    }
    Err("Missing GOOGLE_API_KEY or GEMINI_API_KEY".into())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = read_api_key()?;

    // Recommended construction: provider config-first (no unified builder required).
    let client = siumai::providers::gemini::GeminiClient::from_config(
        siumai::providers::gemini::GeminiConfig::new(api_key)
            .with_model("gemini-2.0-flash-exp".to_string()),
    )?;

    let request = ChatRequest::builder()
        .message(user!(
            "Using URL context, summarize this page in 5 bullet points: https://www.rust-lang.org/"
        ))
        .tools(vec![siumai::hosted_tools::google::url_context()])
        .build();

    let response = text::generate(&client, request, text::GenerateOptions::default()).await?;
    println!("{}", response.content_text().unwrap_or_default());

    if let Some(meta) = response.gemini_metadata()
        && let Some(url_ctx) = &meta.url_context_metadata
    {
        println!("\nURL context metadata:");
        println!("{}", serde_json::to_string_pretty(url_ctx).unwrap());
    }

    Ok(())
}
