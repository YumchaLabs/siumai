//! Error Handling - Understanding LlmError types
//!
//! This example demonstrates different error types and how to handle them.
//!
//! ## Run
//! ```bash
//! cargo run --example error-types --features openai
//! ```

use siumai::prelude::unified::*;
use siumai::providers::openai::{OpenAiClient, OpenAiConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example 1: Invalid API key
    println!("Example 1: Invalid API key");
    let result =
        OpenAiClient::from_config(OpenAiConfig::new("invalid-key").with_model("gpt-4o-mini"));

    match result {
        Ok(client) => {
            // Try to use it
            match text::generate(
                &client,
                ChatRequest::new(vec![user!("Hello")]),
                text::GenerateOptions::default(),
            )
            .await
            {
                Ok(_) => println!("  Unexpected success"),
                Err(e) => println!("  Error: {:?}", e),
            }
        }
        Err(e) => println!("  Build error: {:?}", e),
    }

    // Example 2: Invalid model
    println!("\nExample 2: Invalid model");
    if std::env::var("OPENAI_API_KEY").is_ok() {
        let model = registry::global().language_model("openai:nonexistent-model")?;

        match text::generate(
            &model,
            ChatRequest::new(vec![user!("Hello")]),
            text::GenerateOptions::default(),
        )
        .await
        {
            Ok(_) => println!("  Unexpected success"),
            Err(e) => match e {
                LlmError::ApiError { code, message, .. } => {
                    println!("  API Error - Code: {}, Message: {}", code, message);
                }
                _ => println!("  Other error: {:?}", e),
            },
        }
    }

    // Example 3: Proper error handling
    println!("\nExample 3: Proper error handling");
    if std::env::var("OPENAI_API_KEY").is_ok() {
        let model = registry::global().language_model("openai:gpt-4o-mini")?;

        match text::generate(
            &model,
            ChatRequest::new(vec![user!("Hello!")]),
            text::GenerateOptions::default(),
        )
        .await
        {
            Ok(response) => {
                println!(
                    "  ✅ Success: {}",
                    response.content_text().unwrap_or_default()
                );
            }
            Err(e) => match e {
                LlmError::ApiError { code, message, .. } => {
                    println!("  ❌ API Error: {} - {}", code, message);
                }
                LlmError::HttpError(msg) => {
                    println!("  ❌ HTTP Error: {}", msg);
                }
                LlmError::RateLimitError(msg) => {
                    println!("  ❌ Rate Limited: {}", msg);
                }
                _ => {
                    println!("  ❌ Other Error: {:?}", e);
                }
            },
        }
    }

    Ok(())
}
