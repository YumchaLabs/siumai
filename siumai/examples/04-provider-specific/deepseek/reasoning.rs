//! DeepSeek reasoning with typed provider options.
//!
//! Package tier:
//! - provider-owned wrapper package
//! - preferred path in this directory: config-first `DeepSeekClient`
//!
//! Credentials:
//! - set `DEEPSEEK_API_KEY` before running this example
//!
//! Run:
//! ```bash
//! cargo run --example deepseek-reasoning --features deepseek
//! ```

use siumai::models;
use siumai::prelude::unified::*;
use siumai::provider_ext::deepseek::{DeepSeekChatRequestExt, DeepSeekClient, DeepSeekOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = DeepSeekClient::from_builtin_env(Some(models::deepseek::REASONER)).await?;

    let request = ChatRequest::new(vec![user!(
        "Solve this step by step: If a service handles 240 requests per second and traffic grows by 25%, what is the new throughput?"
    )])
    .with_deepseek_options(DeepSeekOptions::new().with_reasoning_budget(2048));

    let response = text::generate(&client, request, text::GenerateOptions::default()).await?;

    if response.has_reasoning() {
        println!("Reasoning:\n{}\n", response.reasoning().join("\n\n"));
    }

    println!("Answer:\n{}", response.content_text().unwrap_or_default());

    Ok(())
}
