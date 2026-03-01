//! Moonshot AI - Function Calling (Tools) Example
//!
//! This example demonstrates Moonshot AI's function calling capabilities.
//! Reference: https://platform.moonshot.cn/docs/guide/use-kimi-api-to-complete-tool-calls
//!
//! ## Features
//! - Function calling with Kimi models
//! - Multi-turn conversations with tool usage
//! - Automatic tool execution
//!
//! ## Run
//! ```bash
//! export MOONSHOT_API_KEY="your-api-key-here"
//! cargo run --example moonshot-tools --features openai
//! ```

use serde_json::json;
use siumai::models;
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌙 Moonshot AI - Function Calling Example\n");
    println!("==========================================\n");

    // Build Moonshot client (config-first).
    // Note: API key is automatically read from `MOONSHOT_API_KEY`.
    let client = siumai::providers::openai_compatible::OpenAiCompatibleClient::from_builtin_env(
        "moonshot",
        Some(models::openai_compatible::moonshot::KIMI_K2_0905_PREVIEW),
    )
    .await?;

    // Define tools
    let get_weather_tool = Tool::function(
        "get_weather".to_string(),
        "Get the current weather for a location".to_string(),
        json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g., Beijing, Shanghai"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }),
    );

    let search_info_tool = Tool::function(
        "search_info".to_string(),
        "Search for information on a topic".to_string(),
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "language": {
                    "type": "string",
                    "enum": ["zh", "en"],
                    "description": "Preferred language for results"
                }
            },
            "required": ["query"]
        }),
    );

    // Example 1: Single tool call
    println!("📝 Example 1: Single Tool Call\n");

    let request = ChatRequest::builder()
        .message(user!("北京今天天气怎么样？"))
        .tools(vec![get_weather_tool.clone()])
        .build();

    let response = text::generate(&client, request, text::GenerateOptions::default()).await?;

    if response.has_tool_calls() {
        let tool_calls = response.tool_calls();
        println!("🔧 Tool calls requested:");
        for call in tool_calls {
            if let Some(info) = call.as_tool_call() {
                println!("   - Function: {}", info.tool_name);
                println!("     Arguments: {}", info.arguments);
            }
        }
    }

    println!();

    // Example 2: Multiple tools available
    println!("📝 Example 2: Multiple Tools Available\n");

    let request = ChatRequest::builder()
        .message(user!(
            "帮我查一下上海的天气，然后搜索一下关于人工智能的最新信息"
        ))
        .tools(vec![get_weather_tool.clone(), search_info_tool.clone()])
        .build();

    let response = text::generate(&client, request, text::GenerateOptions::default()).await?;

    if response.has_tool_calls() {
        let tool_calls = response.tool_calls();
        println!("🔧 Tool calls requested:");
        for call in tool_calls {
            if let Some(info) = call.as_tool_call() {
                println!("   - Function: {}", info.tool_name);
                println!("     Arguments: {}", info.arguments);
            }
        }
    } else {
        println!("Response: {}", response.content_text().unwrap_or_default());
    }

    println!();

    // Example 3: Multi-turn conversation with tool execution
    println!("📝 Example 3: Multi-turn Conversation with Tool Execution\n");

    let mut messages = vec![user!(
        "What's the weather like in San Francisco? Please use celsius."
    )];

    let request = ChatRequest::builder()
        .messages(messages.clone())
        .tools(vec![get_weather_tool.clone()])
        .build();

    let response = text::generate(&client, request, text::GenerateOptions::default()).await?;

    // Add assistant's response to conversation
    messages.extend(response.to_messages());

    if response.has_tool_calls() {
        let tool_calls = response.tool_calls();
        println!("🔧 Assistant requested tool calls:");

        for call in tool_calls {
            if let Some(info) = call.as_tool_call() {
                println!("   - Function: {}", info.tool_name);
                println!("     Arguments: {}", info.arguments);

                // Simulate tool execution
                let tool_result = json!({
                    "location": "San Francisco",
                    "temperature": 18,
                    "unit": "celsius",
                    "condition": "Partly cloudy",
                    "humidity": 65
                });

                // Add tool result to conversation
                messages.push(
                    ChatMessage::tool_result_text(
                        info.tool_call_id,
                        info.tool_name,
                        tool_result.to_string(),
                    )
                    .build(),
                );
            }
        }

        println!("\n📤 Sending tool results back to Kimi...\n");

        // Continue conversation with tool results
        let final_request = ChatRequest::builder()
            .messages(messages)
            .tools(vec![get_weather_tool])
            .build();

        let final_response =
            text::generate(&client, final_request, text::GenerateOptions::default()).await?;

        println!("🌙 Kimi's final response:");
        println!("{}\n", final_response.content_text().unwrap_or_default());
    }

    println!("✅ Example completed successfully!");
    println!("\n💡 Tips:");
    println!("   - Moonshot supports OpenAI-compatible function calling");
    println!("   - Tools can be called multiple times in a conversation");
    println!("   - Kimi excels at understanding tool requirements in Chinese");
    println!("   - Use multi-turn conversations for complex tool interactions");

    Ok(())
}
