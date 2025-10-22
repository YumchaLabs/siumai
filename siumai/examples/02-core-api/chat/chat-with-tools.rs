//! Chat with Tools - Using client.chat_with_tools()
//!
//! This example demonstrates chat with function calling.
//! Best for: Conversations that need tool/function calling.
//!
//! ## Run
//! ```bash
//! cargo run --example chat-with-tools --features openai
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

    // Define tools using Tool::function()
    let tools = vec![Tool::function(
        "get_weather".to_string(),
        "Get the current weather for a location".to_string(),
        json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. San Francisco"
                }
            },
            "required": ["location"]
        }),
    )];

    // Chat with tools
    let response = client
        .chat_with_tools(vec![user!("What's the weather in Tokyo?")], Some(tools))
        .await?;

    // Check for tool calls
    if response.has_tool_calls() {
        println!("🔧 Tool calls:");
        for call in response.tool_calls() {
            if let siumai::types::ContentPart::ToolCall {
                tool_name,
                arguments,
                ..
            } = call
            {
                println!("  - {}: {}", tool_name, arguments);
            }
        }
    } else {
        println!("AI: {}", response.content_text().unwrap_or_default());
    }

    Ok(())
}
