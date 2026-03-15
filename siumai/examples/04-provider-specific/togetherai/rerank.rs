//! TogetherAI config-first rerank example.
//!
//! This example demonstrates the provider-owned TogetherAI rerank surface:
//! - config-first construction via `TogetherAiConfig` / `TogetherAiClient`
//! - stable family execution via `rerank::rerank(...)`
//! - request-level typed options via `TogetherAiRerankOptions`
//! - structured documents with `rankFields`
//!
//! Run:
//! ```bash
//! cargo run --example togetherai-rerank --features togetherai
//! ```
//!
//! Common environment variables:
//! - `TOGETHER_API_KEY`
//! - `TOGETHERAI_RERANK_MODEL`
//! - `TOGETHERAI_RERANK_QUERY`

use siumai::prelude::unified::*;
use siumai::provider_ext::togetherai::{
    TogetherAiClient, TogetherAiConfig, TogetherAiRerankOptions, TogetherAiRerankRequestExt,
};

fn read_non_empty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = read_non_empty_env("TOGETHERAI_RERANK_MODEL")
        .unwrap_or_else(|| TogetherAiConfig::DEFAULT_MODEL.to_string());
    let query = read_non_empty_env("TOGETHERAI_RERANK_QUERY")
        .unwrap_or_else(|| "Which record is about Rust SDK architecture?".to_string());

    let client =
        TogetherAiClient::from_config(TogetherAiConfig::from_env()?.with_model(model.clone()))?;

    let request = RerankRequest::new_object_documents(
        model,
        query,
        vec![
            serde_json::json!({
                "title": "Crawler tutorial",
                "body": "A Python article on scraping websites."
            }),
            serde_json::json!({
                "title": "Rust SDK architecture",
                "body": "Provider-owned config and client surfaces keep vendor complexity local."
            }),
            serde_json::json!({
                "title": "Cooking notes",
                "body": "Dumpling wrappers and chili crisp pairing ideas."
            }),
        ],
    )
    .with_top_n(2)
    .with_togetherai_options(
        TogetherAiRerankOptions::new()
            .with_rank_fields(vec!["title".to_string(), "body".to_string()]),
    );

    let response = rerank::rerank(&client, request, rerank::RerankOptions::default()).await?;

    println!("top result index: {:?}", response.top_result_index());
    for result in &response.results {
        println!("doc[{}] => {:.3}", result.index, result.relevance_score);
    }

    Ok(())
}
