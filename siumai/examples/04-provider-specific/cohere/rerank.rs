//! Cohere config-first rerank example.
//!
//! This example demonstrates the provider-owned Cohere rerank surface:
//! - config-first construction via `CohereConfig` / `CohereClient`
//! - stable family execution via `rerank::rerank(...)`
//! - request-level typed options via `CohereRerankOptions`
//!
//! Run:
//! ```bash
//! cargo run --example cohere-rerank --features cohere
//! ```
//!
//! Common environment variables:
//! - `COHERE_API_KEY`
//! - `COHERE_RERANK_MODEL`
//! - `COHERE_RERANK_QUERY`

use siumai::prelude::unified::*;
use siumai::provider_ext::cohere::{
    CohereClient, CohereConfig, CohereRerankOptions, CohereRerankRequestExt,
};

fn read_non_empty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = read_non_empty_env("COHERE_RERANK_MODEL")
        .unwrap_or_else(|| CohereConfig::DEFAULT_MODEL.to_string());
    let query = read_non_empty_env("COHERE_RERANK_QUERY")
        .unwrap_or_else(|| "Which document best explains Rust SDK design?".to_string());

    let client = CohereClient::from_config(CohereConfig::from_env()?.with_model(model.clone()))?;

    let request = RerankRequest::new(
        model,
        query,
        vec![
            "A cookbook chapter about broth and noodles.".to_string(),
            "A Rust SDK note about provider-owned configuration and typed extensions.".to_string(),
            "A gardening log for spring tomatoes.".to_string(),
        ],
    )
    .with_top_n(2)
    .with_cohere_options(
        CohereRerankOptions::new()
            .with_max_tokens_per_doc(256)
            .with_priority(1),
    );

    let response = rerank::rerank(&client, request, rerank::RerankOptions::default()).await?;

    println!("sorted indices: {:?}", response.sorted_indices());
    for result in &response.results {
        println!("doc[{}] => {:.3}", result.index, result.relevance_score);
    }

    Ok(())
}
