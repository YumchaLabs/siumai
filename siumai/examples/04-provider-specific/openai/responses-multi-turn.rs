//! Multi-turn Conversation with OpenAI Responses API
//!
//! This example demonstrates how to maintain conversation context across multiple turns
//! using the Responses API's `previous_response_id` parameter.
//!
//! ## Key Concepts
//!
//! - **Stateless Design**: Unlike Assistants API, Responses API doesn't maintain server-side sessions
//! - **Response ID Chain**: Each response has a unique ID that can be used to link conversations
//! - **Automatic Context**: OpenAI automatically loads previous context when you provide `previous_response_id`
//! - **Store Parameter**: Controls whether OpenAI stores the response for later retrieval
//!
//! ## Run
//! ```bash
//! cargo run --example responses-multi-turn --features openai
//! ```

use siumai::prelude::*;
use siumai::types::{ChatRequest, OpenAiOptions, ResponsesApiConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Multi-turn Conversation with Responses API\n");
    println!("This example shows how to chain conversations using previous_response_id\n");

    let client = Siumai::builder()
        .openai()
        .api_key(&std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini")
        .build()
        .await?;

    // ========================================
    // Turn 1: Initial Question
    // ========================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Turn 1: Initial Question");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let request1 = ChatRequest::new(vec![user!("What is Rust programming language?")])
        .with_openai_options(
            OpenAiOptions::new().with_responses_api(
                ResponsesApiConfig::new()
                    .with_store(true) // Store for later reference
                    .with_instructions(
                        "You are a helpful programming tutor. Keep responses concise."
                            .to_string(),
                    ),
            ),
        );

    let response1 = client.chat_request(request1).await?;
    println!("User: What is Rust programming language?");
    println!("AI: {}\n", response1.content_text().unwrap());

    // Extract response ID for next turn
    let response_id_1 = response1
        .response_id()
        .expect("Response ID not found")
        .to_string();

    println!("📝 Response ID: {}\n", response_id_1);

    // ========================================
    // Turn 2: Follow-up Question
    // ========================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Turn 2: Follow-up Question (with context)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let request2 = ChatRequest::new(vec![user!("Can you give me a simple code example?")])
        .with_openai_options(
            OpenAiOptions::new().with_responses_api(
                ResponsesApiConfig::new()
                    .with_previous_response(response_id_1.clone()) // Link to Turn 1
                    .with_store(true),
            ),
        );

    let response2 = client.chat_request(request2).await?;
    println!("User: Can you give me a simple code example?");
    println!("AI: {}\n", response2.content_text().unwrap());

    let response_id_2 = response2
        .response_id()
        .expect("Response ID not found")
        .to_string();

    println!("📝 Response ID: {}\n", response_id_2);

    // ========================================
    // Turn 3: Another Follow-up
    // ========================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Turn 3: Deep Dive (with full context)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let request3 = ChatRequest::new(vec![user!("Explain ownership in that example")])
        .with_openai_options(
            OpenAiOptions::new().with_responses_api(
                ResponsesApiConfig::new()
                    .with_previous_response(response_id_2.clone()) // Link to Turn 2
                    .with_store(true)
                    .with_max_tool_calls(5),
            ),
        );

    let response3 = client.chat_request(request3).await?;
    println!("User: Explain ownership in that example");
    println!("AI: {}\n", response3.content_text().unwrap());

    let response_id_3 = response3
        .response_id()
        .expect("Response ID not found")
        .to_string();

    println!("📝 Response ID: {}\n", response_id_3);

    // ========================================
    // Turn 4: Stateless Mode (No Storage)
    // ========================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Turn 4: Stateless Mode (store=false)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let request4 = ChatRequest::new(vec![user!("What are the main benefits?")])
        .with_openai_options(
            OpenAiOptions::new().with_responses_api(
                ResponsesApiConfig::new()
                    .with_previous_response(response_id_3) // Link to Turn 3
                    .with_store(false) // Don't store this response
                    .with_include(vec!["reasoning.encrypted_content".to_string()]),
            ),
        );

    let response4 = client.chat_request(request4).await?;
    println!("User: What are the main benefits?");
    println!("AI: {}\n", response4.content_text().unwrap());

    if let Some(response_id_4) = response4.response_id() {
        println!("📝 Response ID: {} (not stored on server)\n", response_id_4);
    }

    // ========================================
    // Summary
    // ========================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("✅ Multi-turn Conversation Completed!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("💡 Key Takeaways:");
    println!("   1. Each response has a unique ID accessible via response.response_id()");
    println!("   2. Use previous_response_id to link conversations and maintain context");
    println!("   3. Set store=true to enable server-side context storage");
    println!("   4. Set store=false for stateless mode (no data retention)");
    println!("   5. The AI automatically understands context from previous turns");
    println!("\n📚 Design Philosophy:");
    println!("   - Responses API is stateless by design (unlike Assistants API)");
    println!("   - No server-side 'session' objects or management APIs");
    println!("   - Client is responsible for storing and passing response IDs");
    println!("   - Context is managed via previous_response_id chain");

    Ok(())
}

