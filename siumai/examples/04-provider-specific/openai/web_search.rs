//! OpenAI Responses API - Web Search (provider-defined tool)
//!
//! This example demonstrates Vercel-aligned usage:
//! - Enable the Responses API via `OpenAiOptions::with_responses_api`
//! - Attach provider-defined tools via `ChatRequest::with_tools`
//! - Use `hosted_tools::openai::web_search` to construct the tool
//!
//! Run:
//! ```bash
//! cargo run --example openai-web-search --features openai
//! ```

use siumai::prelude::*;
use siumai::types::{ChatRequest, OpenAiOptions, ResponsesApiConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .openai()
        .api_key(&std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini")
        .build()
        .await?;

    let tool = siumai::hosted_tools::openai::web_search()
        .with_search_context_size("high")
        .with_user_location(
            siumai::hosted_tools::openai::UserLocation::new("approximate")
                .with_country("US")
                .with_city("San Francisco")
                .with_timezone("America/Los_Angeles"),
        )
        .build();

    let request = ChatRequest::new(vec![user!("What are the latest Rust 1.85 developments?")])
        .with_tools(vec![tool])
        .with_openai_options(OpenAiOptions::new().with_responses_api(ResponsesApiConfig::new()));

    let response = client.chat_request(request).await?;
    println!("{}", response.content_text().unwrap_or_default());

    Ok(())
}
