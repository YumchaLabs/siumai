//! Retry Configuration - Handle transient failures
//!
//! This example demonstrates configuring retry behavior for resilience.
//!
//! ## Run
//! ```bash
//! cargo run --example retry-config --features openai
//! ```

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure retry behavior
    let retry_config = RetryConfig {
        max_retries: 3,
        initial_delay_ms: 1000,
        max_delay_ms: 10000,
        exponential_base: 2.0,
        jitter: true,
    };

    let client = Siumai::builder()
        .openai()
        .api_key(&std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini")
        .retry_config(retry_config)
        .build()
        .await?;

    println!("Sending request with retry configuration...");
    println!("(Will retry up to 3 times on transient failures)\n");

    let response = client.chat(vec![user!("Hello!")]).await?;

    println!("AI: {}", response.content_text().unwrap());
    println!("\n✅ Request succeeded!");

    Ok(())
}
