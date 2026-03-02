//! Gemini - Enterprise Web Search (provider-defined tool)
//!
//! This example shows how to attach Gemini 2.0+ Enterprise Web Search via provider-defined tools.
//! (Actual availability depends on your Google Cloud / Vertex setup and model support.)
//!
//! Run:
//! ```bash
//! cargo run --example google-enterprise-web-search --features google
//! ```

use siumai::prelude::unified::*;

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
            "Use enterprise web search to find and summarize the latest Rust release notes."
        ))
        .tools(vec![siumai::hosted_tools::google::enterprise_web_search()])
        .build();

    let response = text::generate(&client, request, text::GenerateOptions::default()).await?;
    println!("{}", response.content_text().unwrap_or_default());
    Ok(())
}
