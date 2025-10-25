// This ignored test documents how to use the registry handle with
// OpenAI-compatible providers. It is ignored by default because it requires
// network access and valid API keys.

#![cfg(feature = "openai")]

use siumai::execution::middleware::samples::chain_default_and_clamp;
use siumai::registry::entry::{RegistryOptions, create_provider_registry};
use std::collections::HashMap;

#[tokio::test]
#[ignore]
async fn registry_openrouter_smoke() -> Result<(), Box<dyn std::error::Error>> {
    // Skip when env var is not set
    if std::env::var("OPENROUTER_API_KEY").is_err() {
        return Ok(());
    }
    let reg = create_provider_registry(
        HashMap::new(),
        Some(RegistryOptions {
            separator: ':',
            language_model_middleware: chain_default_and_clamp(),
            max_cache_entries: None,
            client_ttl: None,
            auto_middleware: true,
        }),
    );
    // Example model id for OpenRouter
    let lm = reg.language_model("openrouter:openai/gpt-4o-mini")?;
    // Not executing network call by default (ignored). Uncomment to try:
    // let resp = lm.chat(vec![siumai::user!("ping")], None).await?;
    // assert!(resp.content_text().is_some());
    Ok(())
}
