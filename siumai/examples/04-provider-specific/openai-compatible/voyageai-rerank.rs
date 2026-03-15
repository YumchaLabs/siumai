//! VoyageAI rerank on the shared OpenAI-compatible runtime.
//!
//! Package tier:
//! - compat preset example on the shared OpenAI-compatible runtime
//! - preferred path in this file: config-first built-in compat construction
//! - VoyageAI remains a non-chat embedding/rerank preset on this public story
//!
//! Credentials:
//! - reads `VOYAGEAI_API_KEY` from the environment
//!
//! Run:
//! ```bash
//! export VOYAGEAI_API_KEY="your-api-key-here"
//! cargo run --example voyageai-rerank --features openai
//! ```

use siumai::prelude::unified::*;
use siumai::provider_ext::openai_compatible::OpenAiCompatibleClient;

const VOYAGEAI_RERANK_MODEL: &str = "rerank-2";

fn read_non_empty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = read_non_empty_env("VOYAGEAI_RERANK_MODEL")
        .unwrap_or_else(|| VOYAGEAI_RERANK_MODEL.to_string());
    let query = read_non_empty_env("VOYAGEAI_RERANK_QUERY")
        .unwrap_or_else(|| "Which document explains evaluation signals for reranking?".to_string());

    let client = OpenAiCompatibleClient::from_builtin_env("voyageai", Some(&model)).await?;

    let request = RerankRequest::new(
        model,
        query,
        vec![
            "A coffee tasting note with citrus and caramel.".to_string(),
            "A note on evaluation signals for semantic reranking systems.".to_string(),
            "A packing list for a two-day conference trip.".to_string(),
        ],
    )
    .with_top_n(2);

    let response = rerank::rerank(&client, request, rerank::RerankOptions::default()).await?;

    println!("top result index: {:?}", response.top_result_index());
    for result in &response.results {
        println!("doc[{}] => {:.3}", result.index, result.relevance_score);
    }

    Ok(())
}
