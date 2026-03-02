//! Complex Request - Using ChatRequestBuilder
//!
//! This example demonstrates building complex requests with all features.
//!
//! ## Run
//! ```bash
//! cargo run --example complex-request --features openai
//! ```

use serde_json::json;
use siumai::prelude::unified::*;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Recommended construction: resolve a model handle from the registry.
    // Note: API key is automatically read from `OPENAI_API_KEY`.
    let model = registry::global().language_model("openai:gpt-4o-mini")?;

    // Build complex request with all features
    let mut http_config = HttpConfig::default();
    http_config
        .headers
        .insert("X-Custom-Header".to_string(), "custom-value".to_string());
    http_config.timeout = Some(Duration::from_secs(30));

    let tools = vec![Tool::function(
        "search".to_string(),
        "Search for information".to_string(),
        json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }),
    )];

    // ✅ Build complex request with all features
    let request = ChatRequest::builder()
        .message(system!("You are a helpful assistant"))
        .message(user!("What's the latest news about Rust?"))
        .temperature(0.7)
        .max_tokens(1000)
        .tools(tools)
        .http_config(http_config)
        .build();

    let response = text::generate(&model, request, text::GenerateOptions::default()).await?;

    if response.has_tool_calls() {
        let tool_calls = response.tool_calls();
        println!("🔧 Tool calls: {:#?}", tool_calls);
    } else {
        println!("AI: {}", response.content_text().unwrap_or_default());
    }

    Ok(())
}
