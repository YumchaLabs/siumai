//! Interactive Chatbot - Complete conversational AI
//!
//! This example demonstrates building a complete chatbot with:
//! - Conversation memory and context management
//! - Streaming responses for better UX
//! - Command system for bot control
//! - Multi-provider support
//!
//! ## Run
//! ```bash
//! cargo run --example chatbot --features openai
//! ```
//!
//! ## Learn More
//! See `siumai/examples/05_use_cases/simple_chatbot.rs` for the complete
//! implementation with all features.

use futures::StreamExt;
use siumai::prelude::*;
use std::collections::VecDeque;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 Interactive Chatbot\n");

    let client = Siumai::builder()
        .openai()
        .api_key(&std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini")
        .build()
        .await?;

    let mut conversation = VecDeque::new();
    conversation.push_back(system!(
        "You are a helpful AI assistant. Keep responses concise."
    ));

    println!("Type 'quit' to exit, 'clear' to clear history\n");

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        match input {
            "quit" => break,
            "clear" => {
                conversation.clear();
                conversation.push_back(system!(
                    "You are a helpful AI assistant. Keep responses concise."
                ));
                println!("🗑️  Conversation cleared\n");
                continue;
            }
            _ => {}
        }

        // Add user message
        conversation.push_back(user!(input));

        // Keep only last 20 messages
        while conversation.len() > 20 {
            conversation.pop_front();
        }

        // Stream response
        print!("AI: ");
        io::stdout().flush()?;

        let messages: Vec<ChatMessage> = conversation.iter().cloned().collect();
        let mut stream = client.chat_stream(messages, None).await?;

        let mut full_response = String::new();
        while let Some(event) = stream.next().await {
            if let ChatStreamEvent::ContentDelta { delta, .. } = event? {
                print!("{}", delta);
                io::stdout().flush()?;
                full_response.push_str(&delta);
            }
        }
        println!("\n");

        // Add assistant response to conversation
        conversation.push_back(assistant!(&full_response));
    }

    println!("👋 Goodbye!");
    Ok(())
}
