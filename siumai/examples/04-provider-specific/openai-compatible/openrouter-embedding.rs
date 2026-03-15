//! OpenRouter embeddings on the shared OpenAI-compatible runtime.
//!
//! Package tier:
//! - compat vendor view
//! - preferred public story: built-in compat client + Stable `embedding::embed`
//! - not a dedicated provider-owned client package
//!
//! Credentials:
//! - reads `OPENROUTER_API_KEY` from the environment
//!
//! Run:
//! ```bash
//! export OPENROUTER_API_KEY="your-api-key-here"
//! cargo run --example openrouter-embedding --features openai
//! ```

use siumai::prelude::unified::*;
use siumai::provider_ext::openai_compatible::OpenAiCompatibleClient;
use siumai_core::types::EmbeddingFormat;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = OpenAiCompatibleClient::from_builtin_env(
        "openrouter",
        Some("openai/text-embedding-3-small"),
    )
    .await?;

    let request = EmbeddingRequest::single("Rust SDK design values explicit contracts.")
        .with_model("openai/text-embedding-3-small")
        .with_dimensions(512)
        .with_encoding_format(EmbeddingFormat::Float)
        .with_user("docs-openrouter-embedding");

    let response = embedding::embed(&client, request, embedding::EmbedOptions::default()).await?;

    println!("Embeddings returned: {}", response.embeddings.len());
    if let Some(first) = response.embeddings.first() {
        println!("First vector dimensions: {}", first.len());
        println!("First 5 values: {:?}", &first[..first.len().min(5)]);
    }

    Ok(())
}
