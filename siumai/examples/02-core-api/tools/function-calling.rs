//! Function Calling - Basic tool usage
//!
//! This example demonstrates how to define and use tools/functions.
//!
//! ## Run
//! ```bash
//! cargo run --example function-calling --features openai
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

    // Define tools
    let tools = vec![
        Tool::function(
            "get_weather".to_string(),
            "Get current weather for a location".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }),
        ),
        Tool::function(
            "calculate".to_string(),
            "Perform mathematical calculations".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            }),
        ),
    ];

    // Request with tools
    let request = ChatRequest::builder()
        .message(user!("What's the weather in Tokyo and what's 15 * 23?"))
        .tools(tools)
        .build();

    let response = client.chat_request(request).await?;

    // Handle tool calls
    if let Some(tool_calls) = &response.tool_calls {
        println!("🔧 AI wants to call {} tool(s):\n", tool_calls.len());
        for call in tool_calls {
            if let Some(function) = &call.function {
                println!("  Function: {}", function.name);
                println!("  Arguments: {}", function.arguments);
                println!();
            }
        }
    } else {
        println!("AI: {}", response.content_text().unwrap_or_default());
    }

    Ok(())
}
