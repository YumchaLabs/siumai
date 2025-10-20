//! Structured Output - Cross-provider JSON schema support
//!
//! This example demonstrates using type-safe provider options for structured output.
//! Works across OpenAI, Anthropic, and Google with unified API.
//!
//! ## Run
//! ```bash
//! cargo run --example structured-output --features openai
//! ```

use serde_json::json;
use siumai::prelude::*;
use siumai::types::{OpenAiOptions, ResponsesApiConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .openai()
        .api_key(&std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini")
        .build()
        .await?;

    println!("📝 Structured Output Example\n");

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
        "required": ["name", "age", "hobbies"],
        "additionalProperties": false
    });

    // ✅ New API: Use type-safe OpenAiOptions with ResponsesApiConfig
    let response_format = json!({
        "type": "json_schema",
        "json_schema": {
            "name": "person_info",
            "strict": true,
            "schema": schema
        }
    });

    let request =
        ChatRequest::builder()
            .message(user!(
                "Generate a profile for a fictional software engineer"
            ))
            .openai_options(OpenAiOptions::new().with_responses_api(
                ResponsesApiConfig::new().with_response_format(response_format),
            ))
            .build();

    let response = client.chat_request(request).await?;

    println!("✅ Structured Output:");
    println!("{}", response.content_text().unwrap());
    println!();

    println!("💡 Migration Note:");
    println!("   Old API: ProviderParams::new().with_structured_output(schema, name)");
    println!("   New API: OpenAiOptions::new().with_response_format(ResponseFormat {{ ... }})");
    println!("   Benefits: Type safety, IDE autocomplete, compile-time validation");

    Ok(())
}
