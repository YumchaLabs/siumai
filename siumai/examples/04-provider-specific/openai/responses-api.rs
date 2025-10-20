//! OpenAI Responses API - Stateful conversations
//!
//! This example demonstrates OpenAI's Responses API for managing
//! stateful conversations with automatic context management.
//!
//! ## Run
//! ```bash
//! cargo run --example responses-api --features openai
//! ```

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .openai()
        .api_key(&std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini")
        .build()
        .await?;

    println!("🔄 OpenAI Responses API Example\n");

    // Create a response session
    let session_id = client.create_response_session().await?;
    println!("Created session: {}\n", session_id);

    // First message
    println!("User: What is 2+2?");
    let response1 = client
        .chat_with_session(&session_id, vec![user!("What is 2+2?")])
        .await?;
    println!("AI: {}\n", response1.content_text().unwrap());

    // Follow-up (context is maintained automatically)
    println!("User: What about 3+3?");
    let response2 = client
        .chat_with_session(&session_id, vec![user!("What about 3+3?")])
        .await?;
    println!("AI: {}\n", response2.content_text().unwrap());

    // Clean up
    client.delete_response_session(&session_id).await?;
    println!("✅ Session deleted");

    Ok(())
}
