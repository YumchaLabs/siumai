//! Using the ToolLoopAgent for reusable multi-step agents.
//!
//! This example demonstrates:
//! - Creating a reusable agent with ToolLoopAgent
//! - Using the agent for multiple conversations
//! - Configuring agent behavior with stop conditions
//! - Agent callbacks for monitoring
//!
//! Run with: cargo run --example agent-pattern --features openai

use siumai::orchestrator::{ToolLoopAgent, ToolResolver, step_count_is};
use siumai::prelude::*;
use siumai::types::{Tool, ToolFunction};

// Simple tool resolver
struct WeatherResolver;

#[async_trait::async_trait]
impl ToolResolver for WeatherResolver {
    async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, siumai::error::LlmError> {
        match name {
            "get_weather" => {
                let location = arguments
                    .get("location")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                Ok(serde_json::json!({
                    "location": location,
                    "temperature": 72 + (rand::random::<i32>() % 20) - 10,
                    "condition": "sunny"
                }))
            }
            "get_forecast" => {
                let location = arguments
                    .get("location")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let days = arguments.get("days").and_then(|v| v.as_i64()).unwrap_or(3);
                Ok(serde_json::json!({
                    "location": location,
                    "days": days,
                    "forecast": "Mostly sunny with occasional clouds"
                }))
            }
            _ => Err(siumai::error::LlmError::InternalError(format!(
                "Unknown tool: {}",
                name
            ))),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize client
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
            "Get the current weather for a location".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name"
                    }
                },
                "required": ["location"]
            }),
        ),
        Tool::function(
            "get_forecast".to_string(),
            "Get the weather forecast for a location".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name"
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days to forecast"
                    }
                },
                "required": ["location", "days"]
            }),
        ),
    ];

    // Create a reusable agent
    let agent = ToolLoopAgent::new(client, tools, vec![step_count_is(10)])
        .with_system(
            "You are a helpful weather assistant. Always provide detailed weather information.",
        )
        .with_id("weather-agent")
        .on_step_finish(std::sync::Arc::new(|step| {
            println!(
                "  → Step completed with {} tool calls",
                step.tool_calls.len()
            );
        }))
        .on_finish(std::sync::Arc::new(|steps| {
            println!("  ✓ Agent finished in {} steps", steps.len());
        }));

    // Create resolver
    let resolver = WeatherResolver;

    // Use the agent for multiple conversations
    println!("=== Conversation 1 ===");
    let messages1 = vec![ChatMessage::user("What's the weather in Tokyo?").build()];
    let (response1, _) = agent.generate(messages1, &resolver).await?;
    println!("Response: {}\n", response1.content_text().unwrap_or(""));

    println!("=== Conversation 2 ===");
    let messages2 = vec![ChatMessage::user("Give me the 5-day forecast for London").build()];
    let (response2, _) = agent.generate(messages2, &resolver).await?;
    println!("Response: {}\n", response2.content_text().unwrap_or(""));

    println!("=== Conversation 3 ===");
    let messages3 =
        vec![ChatMessage::user("Compare the weather in New York and Los Angeles").build()];
    let (response3, _) = agent.generate(messages3, &resolver).await?;
    println!("Response: {}\n", response3.content_text().unwrap_or(""));

    println!("Agent ID: {:?}", agent.id());
    println!("Agent has {} tools", agent.tools().len());

    Ok(())
}
