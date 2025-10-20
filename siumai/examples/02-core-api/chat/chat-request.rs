//! Chat Request - Using client.chat_request() (Recommended ⭐)
//!
//! This example demonstrates the recommended way to use chat API in 0.11.0+.
//! ChatRequest preserves all enhanced fields: provider_params, http_config,
//! web_search, and telemetry.
//!
//! ## Why use chat_request?
//! - ✅ Full control over all request parameters
//! - ✅ Preserves provider-specific parameters
//! - ✅ Supports HTTP configuration and telemetry
//! - ✅ Future-proof API design
//!
//! ## Run
//! ```bash
//! cargo run --example chat-request --features openai
//! ```

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📝 Chat Request Example (Recommended API)\n");

    let client = Siumai::builder()
        .openai()
        .api_key(&std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini")
        .build()
        .await?;

    // Method 1: Simple ChatRequest
    println!("Method 1: Simple ChatRequest");
    let request = ChatRequest::new(vec![user!("What is Rust?")]);
    let response = client.chat_request(request).await?;
    println!("AI: {}\n", response.content_text().unwrap());

    // Method 2: ChatRequestBuilder (for complex requests)
    println!("Method 2: ChatRequestBuilder");
    let request = ChatRequest::builder()
        .message(user!("Explain async/await in Rust"))
        .model("gpt-4o-mini")
        .temperature(0.7)
        .max_tokens(500)
        .build();

    let response = client.chat_request(request).await?;
    println!("AI: {}\n", response.content_text().unwrap());

    // Method 3: With provider-specific parameters
    println!("Method 3: With provider params");
    let provider_params =
        ProviderParams::new().with_param("frequency_penalty", serde_json::json!(0.5));

    let request = ChatRequest::builder()
        .message(user!("Tell me a joke"))
        .provider_params(provider_params)
        .build();

    let response = client.chat_request(request).await?;
    println!("AI: {}\n", response.content_text().unwrap());

    println!("✅ Example completed!");
    Ok(())
}
