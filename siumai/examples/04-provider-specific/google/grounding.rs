//! Google Grounding - Web search integration
//!
//! This example demonstrates Google's grounding feature for
//! real-time web search integration.
//!
//! ## Run
//! ```bash
//! cargo run --example grounding --features google
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

    println!("🌐 Google Grounding Example\n");

    // Enable web search grounding via provider-defined tools
    let request = ChatRequest::builder()
        .message(user!("What are the latest developments in Rust 1.85?"))
        .tools(vec![
            siumai::hosted_tools::google::google_search()
                .with_mode("MODE_DYNAMIC")
                .with_dynamic_threshold(0.3)
                .build(),
        ])
        .build();

    let response = text::generate(&client, request, text::GenerateOptions::default()).await?;

    println!("AI (with web search):");
    println!("{}\n", response.content_text().unwrap_or_default());

    // Check for grounding metadata and normalized sources via typed helper
    if let Some(meta) = response.gemini_metadata() {
        if let Some(grounding) = &meta.grounding_metadata {
            println!("🔍 Grounding metadata:");
            println!("{}", serde_json::to_string_pretty(grounding).unwrap());
        }

        if let Some(sources) = &meta.sources {
            println!("\nSources (Vercel-aligned):");
            println!("{}", serde_json::to_string_pretty(sources).unwrap());
        }
    }

    Ok(())
}
