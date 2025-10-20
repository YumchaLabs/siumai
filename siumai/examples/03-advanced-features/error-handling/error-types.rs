//! Error Handling - Understanding LlmError types
//!
//! This example demonstrates different error types and how to handle them.
//!
//! ## Run
//! ```bash
//! cargo run --example error-types --features openai
//! ```

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example 1: Invalid API key
    println!("Example 1: Invalid API key");
    let result = Siumai::builder()
        .openai()
        .api_key("invalid-key")
        .model("gpt-4o-mini")
        .build()
        .await;

    match result {
        Ok(client) => {
            // Try to use it
            match client.chat(vec![user!("Hello")]).await {
                Ok(_) => println!("  Unexpected success"),
                Err(e) => println!("  Error: {:?}", e),
            }
        }
        Err(e) => println!("  Build error: {:?}", e),
    }

    // Example 2: Invalid model
    println!("\nExample 2: Invalid model");
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        let client = Siumai::builder()
            .openai()
            .api_key(&api_key)
            .model("nonexistent-model")
            .build()
            .await?;

        match client.chat(vec![user!("Hello")]).await {
            Ok(_) => println!("  Unexpected success"),
            Err(e) => match e {
                LlmError::ApiError { status, message } => {
                    println!("  API Error - Status: {}, Message: {}", status, message);
                }
                _ => println!("  Other error: {:?}", e),
            },
        }
    }

    // Example 3: Proper error handling
    println!("\nExample 3: Proper error handling");
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        let client = Siumai::builder()
            .openai()
            .api_key(&api_key)
            .model("gpt-4o-mini")
            .build()
            .await?;

        match client.chat(vec![user!("Hello!")]).await {
            Ok(response) => {
                println!("  ✅ Success: {}", response.content_text().unwrap());
            }
            Err(e) => match e {
                LlmError::ApiError { status, message } => {
                    println!("  ❌ API Error: {} - {}", status, message);
                }
                LlmError::NetworkError(msg) => {
                    println!("  ❌ Network Error: {}", msg);
                }
                LlmError::RateLimitError { retry_after } => {
                    println!("  ❌ Rate Limited. Retry after: {:?}", retry_after);
                }
                _ => {
                    println!("  ❌ Other Error: {:?}", e);
                }
            },
        }
    }

    Ok(())
}
