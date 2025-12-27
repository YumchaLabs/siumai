//! OpenAI Responses API - File Search (provider-defined tool)
//!
//! This example demonstrates Vercel-aligned usage:
//! - Enable the Responses API via `OpenAiOptions::with_responses_api`
//! - Attach provider-defined tools via `ChatRequest::with_tools`
//! - Use `hosted_tools::openai::file_search` to construct the tool
//!
//! Prerequisites:
//! - `OPENAI_API_KEY`
//! - `OPENAI_VECTOR_STORE_ID` (a vector store id, e.g. `vs_...`)
//!
//! Run:
//! ```bash
//! cargo run --example openai-file-search --features openai
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

    let vector_store_id = std::env::var("OPENAI_VECTOR_STORE_ID")?;

    let tool = siumai::hosted_tools::openai::file_search()
        .with_vector_store_ids(vec![vector_store_id])
        .with_max_num_results(8)
        .build();

    let request = ChatRequest::new(vec![user!(
        "Answer using only the documents in the attached vector store. Include short citations."
    )])
    .with_tools(vec![tool])
    .with_openai_options(OpenAiOptions::new().with_responses_api(ResponsesApiConfig::new()));

    let response = client.chat_request(request).await?;
    println!("{}", response.content_text().unwrap_or_default());

    Ok(())
}
