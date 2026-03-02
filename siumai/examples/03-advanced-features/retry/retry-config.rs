//! Retry Configuration - Handle transient failures
//!
//! This example demonstrates configuring retry behavior for resilience.
//!
//! ## Run
//! ```bash
//! cargo run --example retry-config --features openai
//! ```

use siumai::prelude::unified::*;
use siumai::retry_api::{RetryOptions, RetryPolicy};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure retry behavior using RetryPolicy
    let retry_policy = RetryPolicy::default()
        .with_max_attempts(3)
        .with_initial_delay(Duration::from_millis(1000))
        .with_max_delay(Duration::from_secs(10))
        .with_backoff_multiplier(2.0)
        .with_jitter(true);

    let retry_options = RetryOptions {
        backend: siumai::retry_api::RetryBackend::Policy,
        backoff_executor: None,
        policy: Some(retry_policy),
        retry_401: false,
        idempotent: true,
    };

    // Recommended construction: resolve a model handle from the registry.
    // Note: API key is automatically read from `OPENAI_API_KEY`.
    let model = registry::global().language_model("openai:gpt-4o-mini")?;

    println!("Sending request with retry configuration...");
    println!("(Will retry up to 3 times on transient failures)\n");

    let response = text::generate(
        &model,
        ChatRequest::new(vec![user!("Hello!")]),
        text::GenerateOptions {
            retry: Some(retry_options),
            ..Default::default()
        },
    )
    .await?;

    println!("AI: {}", response.content_text().unwrap_or_default());
    println!("\n✅ Request succeeded!");

    Ok(())
}
