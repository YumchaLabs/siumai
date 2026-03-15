//! xAI reasoning with config-first provider setup.
//!
//! Package tier:
//! - provider-owned wrapper package
//! - preferred path in this example: config-first `XaiConfig` / `XaiClient`
//!
//! Credentials:
//! - set `XAI_API_KEY` before running this example
//!
//! Run:
//! ```bash
//! cargo run --example xai-reasoning --features xai
//! ```

use siumai::models;
use siumai::prelude::unified::*;
use siumai::provider_ext::xai::{XaiChatRequestExt, XaiClient, XaiConfig, XaiOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = XaiClient::from_config(
        XaiConfig::new(std::env::var("XAI_API_KEY")?)
            .with_model(models::xai::grok_4::GROK_4_LATEST)
            .with_reasoning(true)
            .with_reasoning_budget(2048),
    )
    .await?;

    let request = ChatRequest::new(vec![user!(
        "Solve this step by step: A service handles 320 requests per second and traffic grows by 15%. What is the new throughput?"
    )])
    .with_xai_options(XaiOptions::new().with_reasoning_effort("high"));

    let response = text::generate(&client, request, text::GenerateOptions::default()).await?;

    if response.has_reasoning() {
        println!("Reasoning:\n{}\n", response.reasoning().join("\n\n"));
    }

    println!("Answer:\n{}", response.content_text().unwrap_or_default());

    Ok(())
}
