//! Basic MiniMaxi Provider Example
//!
//! This example demonstrates how to use the MiniMaxi provider for chat completions.
//!
//! ## Setup
//!
//! Set your MiniMaxi API key:
//! ```bash
//! export MINIMAXI_API_KEY=your-api-key-here
//! ```
//!
//! ## Run
//!
//! ```bash
//! cargo run --example minimaxi_basic --features minimaxi
//! ```

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for debugging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🚀 MiniMaxi Provider Example\n");

    // Create a MiniMaxi client
    // API key will be read from MINIMAXI_API_KEY environment variable
    let client = LlmBuilder::new()
        .minimaxi()
        .model("MiniMax-M2")
        .build()
        .await?;

    println!("✅ MiniMaxi client created successfully\n");

    // Example 1: Simple chat
    println!("📝 Example 1: Simple Chat");
    println!("─────────────────────────");
    
    let messages = vec![
        user!("你好！请用中文介绍一下 MiniMaxi 公司。"),
    ];

    println!("Sending request...");
    let response = client.chat(messages).await?;
    
    println!("Response: {}", response.content);
    println!();

    // Example 2: Chat with tools (function calling)
    println!("📝 Example 2: Chat with Tools");
    println!("─────────────────────────────");
    
    let messages = vec![
        user!("What's the weather like in Beijing today?"),
    ];

    let tools = vec![
        Tool::new(
            "get_weather",
            "Get the current weather for a location",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g., Beijing"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }),
        ),
    ];

    println!("Sending request with tools...");
    let response = client.chat_with_tools(messages, Some(tools)).await?;
    
    println!("Response: {}", response.content);
    if let Some(tool_calls) = response.tool_calls {
        println!("Tool calls: {} call(s)", tool_calls.len());
        for call in tool_calls {
            println!("  - Function: {}", call.function.name);
            println!("    Arguments: {}", call.function.arguments);
        }
    }
    println!();

    // Example 3: Streaming chat
    println!("📝 Example 3: Streaming Chat");
    println!("────────────────────────────");
    
    let messages = vec![
        user!("请用中文写一首关于人工智能的短诗。"),
    ];

    println!("Streaming response:");
    print!("  ");
    
    let mut stream = client.chat_stream(messages, None).await?;
    
    while let Some(event) = stream.next().await {
        match event {
            Ok(event) => {
                use siumai::streaming::ChatStreamEvent;
                match event {
                    ChatStreamEvent::ContentDelta { delta, .. } => {
                        print!("{}", delta);
                        use std::io::Write;
                        std::io::stdout().flush()?;
                    }
                    ChatStreamEvent::Done { .. } => {
                        println!("\n");
                    }
                    _ => {}
                }
            }
            Err(e) => {
                eprintln!("Stream error: {}", e);
                break;
            }
        }
    }

    println!("✅ All examples completed successfully!");

    Ok(())
}

