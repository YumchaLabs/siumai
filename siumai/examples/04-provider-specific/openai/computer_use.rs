//! OpenAI Responses API - Computer Use (provider-defined tool)
//!
//! This example demonstrates Vercel-aligned usage:
//! - Enable the Responses API via `OpenAiOptions::with_responses_api`
//! - Attach provider-defined tools via `ChatRequest::with_tools`
//! - Use `hosted_tools::openai::computer_use` to construct the tool
//!
//! Run:
//! ```bash
//! cargo run --example openai-computer-use --features openai
//! ```

use siumai::prelude::*;
use siumai::provider_ext::openai::{OpenAiOptions, ResponsesApiConfig};
use siumai::types::ChatRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .openai()
        .api_key(&std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini")
        .build()
        .await?;

    let tool = siumai::hosted_tools::openai::computer_use(1920, 1080, "headless");

    let request = ChatRequest::new(vec![user!(
        "Use the computer tool to complete a simple task, then summarize what you did."
    )])
    .with_tools(vec![tool])
    .with_openai_options(OpenAiOptions::new().with_responses_api(ResponsesApiConfig::new()));

    let response = client.chat_request(request).await?;
    println!("{}", response.content_text().unwrap_or_default());

    Ok(())
}
