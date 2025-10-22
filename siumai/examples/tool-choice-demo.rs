//! Tool Choice Demo
//!
//! This example demonstrates how to use the `tool_choice` parameter to control
//! how the model uses tools.
//!
//! # Tool Choice Strategies
//!
//! - `Auto` (default): Model decides whether to call tools
//! - `Required`: Model must call at least one tool
//! - `None`: Model cannot call any tools
//! - `Tool { name }`: Model must call the specified tool
//!
//! # Usage
//!
//! ```bash
//! # Set your API key
//! export OPENAI_API_KEY=your-api-key
//!
//! # Run the example
//! cargo run --example tool-choice-demo --features openai
//! ```

use siumai::prelude::*;
use siumai::types::{Tool, ToolChoice};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the client
    let client = LlmBuilder::new()
        .openai()
        .model("gpt-4o-mini")
        .build()
        .await?;

    // Define tools
    let tools = vec![
        Tool::function(
            "get_weather".to_string(),
            "Get the current weather for a location".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit"
                    }
                },
                "required": ["location"]
            }),
        ),
        Tool::function(
            "get_time".to_string(),
            "Get the current time for a timezone".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "The timezone, e.g. America/New_York"
                    }
                },
                "required": ["timezone"]
            }),
        ),
    ];

    println!("🎯 Tool Choice Demo\n");
    println!("{}", "=".repeat(80));

    // Example 1: Auto (default) - Model decides
    println!("\n📌 Example 1: Auto (default) - Model decides whether to call tools\n");
    let request = ChatRequest::new(vec![
        ChatMessage::user("What's the weather like in San Francisco?").build(),
    ])
    .with_tools(tools.clone())
    .with_tool_choice(ToolChoice::Auto); // Explicit, but this is the default

    let response = client.chat_request(request).await?;
    println!("Response: {:?}", response.content);
    if response.has_tool_calls() {
        let tool_calls = response.tool_calls();
        println!("Tool calls: {} tool(s) called", tool_calls.len());
        for call in tool_calls {
            if let siumai::types::ContentPart::ToolCall {
                tool_name,
                arguments,
                ..
            } = call
            {
                println!("  - {}: {}", tool_name, arguments);
            }
        }
    }

    // Example 2: Required - Model must call at least one tool
    println!("\n📌 Example 2: Required - Model must call at least one tool\n");
    let request = ChatRequest::new(vec![ChatMessage::user("I need some information").build()])
        .with_tools(tools.clone())
        .with_tool_choice(ToolChoice::Required);

    let response = client.chat_request(request).await?;
    println!("Response: {:?}", response.content);
    if response.has_tool_calls() {
        let tool_calls = response.tool_calls();
        println!("Tool calls: {} tool(s) called (required)", tool_calls.len());
        for call in tool_calls {
            if let siumai::types::ContentPart::ToolCall {
                tool_name,
                arguments,
                ..
            } = call
            {
                println!("  - {}: {}", tool_name, arguments);
            }
        }
    }

    // Example 3: None - Model cannot call any tools
    println!("\n📌 Example 3: None - Model cannot call any tools\n");
    let request = ChatRequest::new(vec![
        ChatMessage::user("What's the weather like in San Francisco?").build(),
    ])
    .with_tools(tools.clone())
    .with_tool_choice(ToolChoice::None);

    let response = client.chat_request(request).await?;
    println!("Response: {:?}", response.content);
    println!(
        "Tool calls: {}",
        if response.has_tool_calls() {
            "Some (unexpected!)"
        } else {
            "None (as expected)"
        }
    );

    // Example 4: Tool - Model must call a specific tool
    println!("\n📌 Example 4: Tool - Model must call a specific tool\n");
    let request = ChatRequest::new(vec![ChatMessage::user("I need some information").build()])
        .with_tools(tools.clone())
        .with_tool_choice(ToolChoice::tool("get_weather"));

    let response = client.chat_request(request).await?;
    println!("Response: {:?}", response.content);
    if response.has_tool_calls() {
        let tool_calls = response.tool_calls();
        println!(
            "Tool calls: {} tool(s) called (forced to get_weather)",
            tool_calls.len()
        );
        for call in tool_calls {
            if let siumai::types::ContentPart::ToolCall {
                tool_name,
                arguments,
                ..
            } = call
            {
                println!("  - {}: {}", tool_name, arguments);
            }
        }
    }

    // Example 5: Using builder pattern
    println!("\n📌 Example 5: Using ChatRequestBuilder\n");
    let request = ChatRequestBuilder::new()
        .message(ChatMessage::user("What time is it in New York?").build())
        .tools(tools.clone())
        .tool_choice(ToolChoice::tool("get_time"))
        .build();

    let response = client.chat_request(request).await?;
    println!("Response: {:?}", response.content);
    if response.has_tool_calls() {
        let tool_calls = response.tool_calls();
        println!(
            "Tool calls: {} tool(s) called (forced to get_time)",
            tool_calls.len()
        );
        for call in tool_calls {
            if let siumai::types::ContentPart::ToolCall {
                tool_name,
                arguments,
                ..
            } = call
            {
                println!("  - {}: {}", tool_name, arguments);
            }
        }
    }

    println!("\n{}", "=".repeat(80));
    println!("\n✅ Tool Choice Demo Complete!\n");
    println!("Key Takeaways:");
    println!("  - Auto: Model decides (default behavior)");
    println!("  - Required: Model must call at least one tool");
    println!("  - None: Model cannot call any tools");
    println!("  - Tool {{ name }}: Model must call the specified tool");
    println!("\nNote: Different providers may have different support levels:");
    println!("  - OpenAI: Full support for all modes");
    println!("  - Anthropic: 'None' removes tools from request");
    println!("  - Gemini: Uses tool_config with function_calling_config");

    Ok(())
}
