//! OpenAI Responses API - Advanced response configuration
//!
//! This example demonstrates using OpenAI's Responses API with
//! structured output and advanced configuration.
//!
//! ## Key Features
//!
//! - Structured JSON output with schema validation
//! - Advanced configuration (text verbosity, truncation, etc.)
//! - Multi-turn conversations (see responses-multi-turn.rs example)
//!
//! ## Note on Session Management
//!
//! OpenAI Responses API is **stateless by design**. There are no server-side
//! session management APIs (create/list/delete sessions). Instead, you chain
//! conversations using `previous_response_id`. See the `responses-multi-turn.rs`
//! example for multi-turn conversation patterns.
//!
//! ## Run
//! ```bash
//! cargo run --example responses-api --features openai
//! ```

use siumai::prelude::*;
use siumai::provider_ext::openai::{
    OpenAiChatRequestExt, OpenAiOptions, ReasoningEffort, ResponsesApiConfig, TextVerbosity,
    Truncation,
};

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

    // Example 3: Advanced Responses API configuration
    println!("📝 Example 3: Advanced Configuration\n");

    let mut metadata = std::collections::HashMap::new();
    metadata.insert("user_id".to_string(), "demo_user_123".to_string());
    metadata.insert("session_type".to_string(), "example".to_string());

    let request3 = ChatRequest::new(vec![user!("Write a short poem about Rust programming.")])
        .with_openai_options(
            OpenAiOptions::new().with_responses_api(
                ResponsesApiConfig::new()
                    .with_instructions("You are a creative poet who loves programming.".to_string())
                    .with_text_verbosity(TextVerbosity::Medium)
                    .with_truncation(Truncation::Auto)
                    .with_max_tool_calls(5)
                    .with_store(false) // Don't store this response
                    .with_parallel_tool_calls(true)
                    .with_metadata(metadata),
            ),
        );

    let response3 = client.chat_request(request3).await?;
    println!("AI Response: {}\n", response3.content_text().unwrap());

    // Example 4: Multi-turn conversation with previous_response_id
    println!("📝 Example 4: Multi-turn Conversation (Simulated)\n");
    println!("Note: In a real scenario, you would use the response_id from the previous response");

    let request4 = ChatRequest::new(vec![user!("Tell me about Rust's ownership system.")])
        .with_openai_options(
            OpenAiOptions::new().with_responses_api(
                ResponsesApiConfig::new()
                    // In a real scenario: .with_previous_response(previous_response_id)
                    .with_include(vec![
                        "file_search_call.results".to_string(),
                        "reasoning.encrypted_content".to_string(),
                    ]),
            ),
        );

    let response4 = client.chat_request(request4).await?;
    println!("AI Response: {}\n", response4.content_text().unwrap());

    println!("✅ Responses API examples completed!");
    println!("\n💡 New features demonstrated:");
    println!("   - instructions: Custom system instructions");
    println!("   - text_verbosity: Control response detail level");
    println!("   - truncation: Context window management");
    println!("   - max_tool_calls: Limit tool usage");
    println!("   - store: Control response storage");
    println!("   - parallel_tool_calls: Enable parallel tool execution");
    println!("   - metadata: Attach custom metadata");
    println!("   - include: Request additional output data");
    println!("   - previous_response_id: Multi-turn conversations");
    println!("\n💡 Note: Session management features (create_response_session, chat_with_session)");
    println!("   are planned for future implementation (Phase 3).");

    Ok(())
}
