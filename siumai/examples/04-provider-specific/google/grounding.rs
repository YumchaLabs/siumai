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
        .google()
        .api_key(&std::env::var("GOOGLE_API_KEY")?)
        .model("gemini-2.0-flash-exp")
        .build()
        .await?;

    println!("🌐 Google Grounding Example\n");

    // Enable web search grounding
    let web_search = WebSearchConfig {
        enabled: true,
        dynamic_retrieval_threshold: Some(0.3),
    };

    let request = ChatRequest::builder()
        .message(user!("What are the latest developments in Rust 1.85?"))
        .web_search(web_search)
        .build();

    let response = client.chat_request(request).await?;

    println!("AI (with web search):");
    println!("{}\n", response.content_text().unwrap());

    // Check for grounding metadata
    if let Some(metadata) = response.grounding_metadata {
        println!("🔍 Grounding sources:");
        for source in metadata.sources {
            println!("  - {}", source.url);
        }
    }

    Ok(())
}
