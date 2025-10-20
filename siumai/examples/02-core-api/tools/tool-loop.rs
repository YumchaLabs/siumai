//! Tool Loop - Complete tool execution cycle
//!
//! This example shows a complete tool calling loop:
//! 1. AI requests tool calls
//! 2. Execute tools
//! 3. Send results back
//! 4. Get final response
//!
//! ## Run
//! ```bash
//! cargo run --example tool-loop --features openai
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

    let tools = vec![Tool::function(
        "get_weather".to_string(),
        "Get weather for a location".to_string(),
        json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }),
    )];

    // Initial request
    let mut messages = vec![user!("What's the weather in Paris?")];

    let request = ChatRequest::builder()
        .messages(messages.clone())
        .tools(tools.clone())
        .build();

    let response = client.chat_request(request).await?;

    // Check for tool calls
    if let Some(tool_calls) = &response.tool_calls {
        if let Some(function) = &tool_calls[0].function {
            println!("🔧 AI requested tool call: {}\n", function.name);
        }

        // Add assistant message with tool call
        messages.push(
            ChatMessage::assistant("")
                .with_tool_calls(tool_calls.clone())
                .build(),
        );

        // Execute tool (simulated)
        let tool_result = json!({"temperature": 18, "condition": "Sunny"});
        println!("🌤️  Tool result: {}\n", tool_result);

        // Add tool result message
        messages.push(ChatMessage::tool(tool_result.to_string(), tool_calls[0].id.clone()).build());

        // Get final response
        let request = ChatRequest::builder()
            .messages(messages)
            .tools(tools)
            .build();

        let final_response = client.chat_request(request).await?;
        println!("AI: {}", final_response.content_text().unwrap());
    }

    Ok(())
}
