//! OpenAI Responses API - File Search (include results)
//!
//! This example shows how to request structured file search results via:
//! `ResponsesApiConfig::with_include(vec!["file_search_call.results"])`.
//!
//! Prerequisites:
//! - `OPENAI_API_KEY`
//! - `OPENAI_VECTOR_STORE_ID` (a vector store id, e.g. `vs_...`)
//!
//! Run:
//! ```bash
//! cargo run --example openai-file-search-results --features openai
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

    let vector_store_id = std::env::var("OPENAI_VECTOR_STORE_ID")?;

    let tool = siumai::hosted_tools::openai::file_search()
        .with_vector_store_ids(vec![vector_store_id])
        .with_max_num_results(5)
        .build();

    let request = ChatRequest::new(vec![user!(
        "Using only the documents in the vector store, define what an embedding model is."
    )])
    .with_tools(vec![tool])
    .with_openai_options(OpenAiOptions::new().with_responses_api(
        ResponsesApiConfig::new().with_include(vec!["file_search_call.results".to_string()]),
    ));

    let response = client.chat_request(request).await?;

    println!("Answer:\n{}\n", response.content_text().unwrap_or_default());
    println!(
        "Raw provider metadata (if any): {}",
        serde_json::to_string_pretty(&response.provider_metadata).unwrap()
    );

    // If you want structured tool results, parse tool-result parts from multimodal content:
    if let Some(parts) = response.content.as_multimodal() {
        for part in parts {
            if let Some(info) = part.as_tool_result() {
                println!(
                    "\nToolResult: tool_name={}, provider_executed={:?}\n{}",
                    info.tool_name,
                    info.provider_executed,
                    info.output.to_string_lossy()
                );
            }
        }
    }

    Ok(())
}
