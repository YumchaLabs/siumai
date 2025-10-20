//! Registry with LRU Cache - Efficient client reuse
//!
//! This example demonstrates the built-in LRU cache for client reuse.
//! Clients are cached and reused automatically, reducing overhead.
//!
//! ## Run
//! ```bash
//! cargo run --example registry-with-cache --features openai
//! ```

use siumai::prelude::*;
use siumai::registry::{RegistryOptions, create_provider_registry};
use std::collections::HashMap;
use std::sync::Arc;
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
            max_cache_entries: Some(10), // Cache up to 10 clients
            client_ttl: Some(Duration::from_secs(300)), // 5 minute TTL
        }),
    );

    if std::env::var("OPENAI_API_KEY").is_ok() {
        println!("First call (creates client):");
        let lm1 = registry.language_model("openai:gpt-4o-mini")?;
        let response1 = lm1.chat(vec![user!("What is 2+2?")]).await?;
        println!("  {}\n", response1.content_text().unwrap());

        println!("Second call (uses cached client):");
        let lm2 = registry.language_model("openai:gpt-4o-mini")?;
        let response2 = lm2.chat(vec![user!("What is 3+3?")]).await?;
        println!("  {}\n", response2.content_text().unwrap());

        println!("✅ Second call reused cached client!");
        println!("💡 Cache reduces overhead for repeated model access");
    }

    Ok(())
}
