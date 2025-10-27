//! Google Grounding - Web search integration
//!
//! This example demonstrates Google's grounding feature for
//! real-time web search integration.
//!
//! ## Run
//! ```bash
//! cargo run --example grounding --features google
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

    println!("🌐 Google Grounding Example\n");

    // Enable web search grounding via provider_options
    let gemini_options = GeminiOptions::new().with_search_grounding(SearchGroundingConfig {
        enabled: true,
        dynamic_retrieval_config: Some(DynamicRetrievalConfig {
            mode: DynamicRetrievalMode::ModeDynamic,
            dynamic_threshold: Some(0.3),
        }),
    });

    let request = ChatRequest::builder()
        .message(user!("What are the latest developments in Rust 1.85?"))
        .gemini_options(gemini_options)
        .build();

    let response = client.chat_request(request).await?;

    println!("AI (with web search):");
    println!("{}\n", response.content_text().unwrap());

    // Check for grounding metadata via typed helper
    if let Some(meta) = response.gemini_metadata() {
        if let Some(grounding) = &meta.grounding_metadata {
            println!("🔍 Grounding metadata:");
            println!("{}", serde_json::to_string_pretty(grounding).unwrap());
        }
    }

    Ok(())
}
