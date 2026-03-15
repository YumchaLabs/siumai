//! Jina rerank on the shared OpenAI-compatible runtime.
//!
//! Package tier:
//! - compat preset example on the shared OpenAI-compatible runtime
//! - preferred path in this file: config-first built-in compat construction
//! - Jina remains a non-chat embedding/rerank preset on this public story
//!
//! Credentials:
//! - reads `JINA_API_KEY` from the environment
//!
//! Run:
//! ```bash
//! export JINA_API_KEY="your-api-key-here"
//! cargo run --example jina-rerank --features openai
//! ```

use siumai::prelude::unified::*;
use siumai::provider_ext::openai_compatible::OpenAiCompatibleClient;

const JINA_RERANK_MODEL: &str = "jina-reranker-m0";

fn read_non_empty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model =
        read_non_empty_env("JINA_RERANK_MODEL").unwrap_or_else(|| JINA_RERANK_MODEL.to_string());
    let query = read_non_empty_env("JINA_RERANK_QUERY")
        .unwrap_or_else(|| "Which note is about multilingual retrieval quality?".to_string());

    let client = OpenAiCompatibleClient::from_builtin_env("jina", Some(&model)).await?;

    let request = RerankRequest::new(
        model,
        query,
        vec![
            "A short note on gardening and tomato seedlings.".to_string(),
            "A note on multilingual retrieval quality and reranker evaluation.".to_string(),
            "A checklist for assembling a standing desk.".to_string(),
        ],
    )
    .with_top_n(2);

    let response = rerank::rerank(&client, request, rerank::RerankOptions::default()).await?;

    println!("sorted indices: {:?}", response.sorted_indices());
    for result in &response.results {
        println!("doc[{}] => {:.3}", result.index, result.relevance_score);
    }

    Ok(())
}
