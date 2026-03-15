//! SiliconFlow rerank on the shared OpenAI-compatible runtime.
//!
//! Package tier:
//! - compat preset example on the shared OpenAI-compatible runtime
//! - preferred path in this file: config-first built-in compat construction
//! - not a dedicated provider-owned rerank package
//!
//! Credentials:
//! - reads `SILICONFLOW_API_KEY` from the environment
//!
//! Run:
//! ```bash
//! export SILICONFLOW_API_KEY="your-api-key-here"
//! cargo run --example siliconflow-rerank --features openai
//! ```

use siumai::prelude::unified::*;
use siumai::provider_ext::openai_compatible::OpenAiCompatibleClient;

const SILICONFLOW_RERANK_MODEL: &str = "BAAI/bge-reranker-v2-m3";

fn read_non_empty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = read_non_empty_env("SILICONFLOW_RERANK_MODEL")
        .unwrap_or_else(|| SILICONFLOW_RERANK_MODEL.to_string());
    let query = read_non_empty_env("SILICONFLOW_RERANK_QUERY")
        .unwrap_or_else(|| "Which note is about Rust SDK architecture?".to_string());

    let client = OpenAiCompatibleClient::from_builtin_env("siliconflow", Some(&model)).await?;

    let request = RerankRequest::new(
        model,
        query,
        vec![
            "A cookbook note about dumplings and chili oil.".to_string(),
            "A Rust SDK note about provider-owned config and client surfaces.".to_string(),
            "A travel checklist for a weekend train ride.".to_string(),
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
