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
    if response.has_tool_calls() {
        // Add assistant message (with tool calls) to conversation history
        messages.extend(response.to_messages());

        // Execute tools and add results
        for tool_call in response.tool_calls() {
            // Use as_tool_call() for convenient access
            if let Some(info) = tool_call.as_tool_call() {
                println!("🔧 AI requested tool call: {}\n", info.tool_name);

                // Execute tool (simulated)
                let tool_result = json!({"temperature": 18, "condition": "Sunny"});
                println!("🌤️  Tool result: {}\n", tool_result);

                // Add tool result message
                messages.push(
                    ChatMessage::tool_result_json(info.tool_call_id, info.tool_name, tool_result)
                        .build(),
                );
            }
        }

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
