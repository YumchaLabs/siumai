//! Registry with LRU Cache - Efficient client reuse
//!
//! This example demonstrates the built-in LRU cache for client reuse.
//! Clients are cached and reused automatically, reducing overhead.
//!
//! ## Run
//! ```bash
//! cargo run --example registry-with-cache --features openai
//! ```

use siumai::prelude::unified::*;
use siumai::registry::{RegistryOptions, create_provider_registry};
use std::collections::HashMap;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("💾 Registry with LRU Cache Example\n");

    // Create registry with custom cache settings
    let registry = create_provider_registry(
        HashMap::new(), // Use default providers
        Some(RegistryOptions {
            separator: ':',
            language_model_middleware: vec![],
            http_interceptors: Vec::new(),
            http_client: None,
            http_transport: None,
            http_config: None,
            api_key: None,
            base_url: None,
            reasoning_enabled: None,
            reasoning_budget: None,
            provider_build_overrides: HashMap::new(),
            retry_options: None,
            max_cache_entries: Some(10), // Cache up to 10 clients
            client_ttl: Some(Duration::from_secs(300)), // 5 minute TTL
            auto_middleware: true,
        }),
    );

    if std::env::var("OPENAI_API_KEY").is_ok() {
        println!("First call (creates client):");
        let lm1 = registry.language_model("openai:gpt-4o-mini")?;
        let response1 = text::generate(
            &lm1,
            ChatRequest::new(vec![user!("What is 2+2?")]),
            text::GenerateOptions::default(),
        )
        .await?;
        println!("  {}\n", response1.content_text().unwrap_or_default());

        println!("Second call (uses cached client):");
        let lm2 = registry.language_model("openai:gpt-4o-mini")?;
        let response2 = text::generate(
            &lm2,
            ChatRequest::new(vec![user!("What is 3+3?")]),
            text::GenerateOptions::default(),
        )
        .await?;
        println!("  {}\n", response2.content_text().unwrap_or_default());

        println!("✅ Second call reused cached client!");
        println!("💡 Cache reduces overhead for repeated model access");
    }

    Ok(())
}
