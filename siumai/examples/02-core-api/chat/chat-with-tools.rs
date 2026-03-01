//! Chat with Tools - Using `text::generate` + tools
//!
//! This example demonstrates chat with function calling.
//! Best for: Conversations that need tool/function calling.
//!
//! ## Run
//! ```bash
//! cargo run --example chat-with-tools --features openai
//! ```

use serde_json::json;
use siumai::prelude::unified::*;

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

    let request = ChatRequest::new(vec![user!("What's the weather in Tokyo?")]).with_tools(tools);
    let response = text::generate(&client, request, text::GenerateOptions::default()).await?;

    // Check for tool calls
    if response.has_tool_calls() {
        println!("🔧 Tool calls:");
        for call in response.tool_calls() {
            if let ContentPart::ToolCall {
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
