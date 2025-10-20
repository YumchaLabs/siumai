//! OpenAI Responses API - Advanced response configuration
//!
//! This example demonstrates using OpenAI's Responses API with
//! structured output and advanced configuration.
//!
//! Note: Session management (create_response_session, chat_with_session)
//! is not yet implemented. This example shows the current supported features.
//!
//! ## Run
//! ```bash
//! cargo run --example responses-api --features openai
//! ```

use siumai::prelude::*;
use siumai::types::{ChatRequest, OpenAiOptions, ReasoningEffort, ResponsesApiConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .openai()
        .api_key(&std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini")
        .build()
        .await?;

    println!("🔄 OpenAI Responses API Example\n");

    // Example 1: Using Responses API with structured output
    println!("📝 Example 1: Structured JSON Output\n");

    let response_format = serde_json::json!({
        "type": "json_schema",
        "json_schema": {
            "name": "math_response",
            "strict": true,
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "number"},
                    "explanation": {"type": "string"}
                },
                "required": ["answer", "explanation"],
                "additionalProperties": false
            }
        }
    });

    let request = ChatRequest::new(vec![user!("What is 2+2? Respond in JSON format.")])
        .with_openai_options(
            OpenAiOptions::new().with_responses_api(
                ResponsesApiConfig::new().with_response_format(response_format),
            ),
        );

    let response = client.chat_request(request).await?;
    println!("AI Response: {}\n", response.content_text().unwrap());

    // Example 2: Using Responses API with reasoning effort
    println!("📝 Example 2: With Reasoning Effort\n");

    let request2 = ChatRequest::new(vec![user!(
        "Explain the concept of recursion in programming."
    )])
    .with_openai_options(
        OpenAiOptions::new()
            .with_responses_api(ResponsesApiConfig::new())
            .with_reasoning_effort(ReasoningEffort::Medium),
    );

    let response2 = client.chat_request(request2).await?;
    println!("AI Response: {}\n", response2.content_text().unwrap());

    println!("✅ Responses API examples completed!");
    println!("\n💡 Note: Session management features (create_response_session, chat_with_session)");
    println!("   are planned for future implementation.");

    Ok(())
}
