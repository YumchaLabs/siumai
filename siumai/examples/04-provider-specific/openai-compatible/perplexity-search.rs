//! Perplexity typed hosted-search request options and typed response metadata
//! on the generic OpenAI-compatible client.
//!
//! Package tier:
//! - compat vendor view
//! - preferred public story: typed vendor helpers + typed metadata on `OpenAiCompatibleClient`
//! - not a dedicated provider-owned client package
//!
//! Credentials:
//! - reads `PERPLEXITY_API_KEY` from the environment
//!
//! Run:
//! ```bash
//! export PERPLEXITY_API_KEY="your-api-key-here"
//! cargo run --example perplexity-search --features openai
//! ```

use siumai::prelude::unified::*;
use siumai::provider_ext::openai_compatible::OpenAiCompatibleClient;
use siumai::provider_ext::perplexity::{
    PerplexityChatRequestExt, PerplexityChatResponseExt, PerplexityOptions,
    PerplexitySearchContextSize, PerplexitySearchMode, PerplexitySearchRecencyFilter,
    PerplexityUserLocation,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = OpenAiCompatibleClient::from_builtin_env("perplexity", Some("sonar")).await?;

    let request = ChatRequest::new(vec![user!(
        "Find two recent Rust ecosystem developments and summarize each in one short sentence."
    )])
    .with_perplexity_options(
        PerplexityOptions::new()
            .with_search_mode(PerplexitySearchMode::Web)
            .with_search_recency_filter(PerplexitySearchRecencyFilter::Month)
            .with_search_context_size(PerplexitySearchContextSize::High)
            .with_return_images(true)
            .with_return_related_questions(true)
            .with_user_location(PerplexityUserLocation::new().with_country("US")),
    );

    let response = text::generate(&client, request, text::GenerateOptions::default()).await?;

    println!("Answer:\n{}", response.content_text().unwrap_or_default());
    println!("\nPerplexity hosted-search behavior was configured through typed provider options.");

    if let Some(metadata) = response.perplexity_metadata() {
        if let Some(usage) = metadata.usage.as_ref() {
            println!("\nPerplexity metadata:");
            println!("- citation_tokens: {:?}", usage.citation_tokens);
            println!("- num_search_queries: {:?}", usage.num_search_queries);
        }
        println!(
            "- images returned: {}",
            metadata.images.as_ref().map(Vec::len).unwrap_or(0)
        );
    }

    Ok(())
}
