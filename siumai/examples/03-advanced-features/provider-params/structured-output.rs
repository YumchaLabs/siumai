//! Structured Output - Cross-provider JSON schema support
//!
//! This example demonstrates using ProviderParams for structured output.
//! Works across OpenAI, Anthropic, and Google with unified API.
//!
//! ## Run
//! ```bash
//! cargo run --example structured-output --features openai
//! ```

use serde_json::json;
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .openai()
        .api_key(&std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini")
        .build()
        .await?;

    // Define JSON schema
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "hobbies": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["name", "age", "hobbies"]
    });

    // Use ProviderParams for structured output
    let provider_params = ProviderParams::new().with_structured_output(schema, "person_info");

    let request = ChatRequest::builder()
        .message(user!(
            "Generate a profile for a fictional software engineer"
        ))
        .provider_params(provider_params)
        .build();

    let response = client.chat_request(request).await?;

    println!("Structured Output:");
    println!("{}", response.content_text().unwrap());

    Ok(())
}
