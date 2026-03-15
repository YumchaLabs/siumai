//! Registry-first rerank example.
//!
//! This example demonstrates the stable rerank family surface:
//! - `registry::global().reranking_model("provider:model")`
//! - provider-agnostic `RerankRequest`
//! - `rerank::rerank(...)` as the recommended family entry point
//!
//! Run:
//! ```bash
//! cargo run --example registry-rerank --features cohere
//! ```
//!
//! Common environment variables:
//! - `COHERE_API_KEY` (default provider path)
//! - `SIUMAI_RERANK_MODEL` (optional `provider:model`, default `cohere:rerank-english-v3.0`)
//! - `SIUMAI_RERANK_QUERY`

use siumai::prelude::unified::*;

fn read_non_empty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let registry_id = read_non_empty_env("SIUMAI_RERANK_MODEL")
        .unwrap_or_else(|| "cohere:rerank-english-v3.0".to_string());
    let Some((_, request_model)) = registry_id.split_once(':') else {
        eprintln!(
            "SIUMAI_RERANK_MODEL must use provider:model form, for example cohere:rerank-english-v3.0"
        );
        std::process::exit(2);
    };

    let query = read_non_empty_env("SIUMAI_RERANK_QUERY")
        .unwrap_or_else(|| "Which document is most relevant to Rust SDK architecture?".to_string());

    let model = registry::global().reranking_model(&registry_id)?;
    let request = RerankRequest::new(
        request_model.to_string(),
        query,
        vec![
            "A Python web crawler tutorial for beginners.".to_string(),
            "A Rust SDK architecture guide with provider-owned config and client surfaces."
                .to_string(),
            "A recipe for steamed dumplings and chili oil.".to_string(),
        ],
    )
    .with_top_n(2);

    let response = rerank::rerank(&model, request, rerank::RerankOptions::default()).await?;

    println!("top result index: {:?}", response.top_result_index());
    for result in &response.results {
        println!("doc[{}] => {:.3}", result.index, result.relevance_score);
    }

    Ok(())
}
