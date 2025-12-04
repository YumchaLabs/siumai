//! Basic multi-step tool calling with orchestrator.
//!
//! This example demonstrates:
//! - Using the orchestrator for multi-step tool calling
//! - Implementing a simple ToolResolver
//! - Using stop conditions
//! - Tracking step results
//!
//! Run with: cargo run --example basic-orchestrator --features openai

use siumai::prelude::*;
use siumai::types::Tool;
use siumai_extras::orchestrator::{
    OrchestratorOptions, StepResult, ToolResolver, generate, step_count_is,
};
use std::sync::Arc;

// Simple tool resolver that implements weather and calculator tools
struct MyToolResolver;

#[async_trait::async_trait]
impl ToolResolver for MyToolResolver {
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
                    "temperature": 72,
                    "condition": "sunny"
                }))
            }
            "calculate" => {
                let a = arguments.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let b = arguments.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let op = arguments
                    .get("operation")
                    .and_then(|v| v.as_str())
                    .unwrap_or("add");

                let result = match op {
                    "add" => a + b,
                    "subtract" => a - b,
                    "multiply" => a * b,
                    "divide" if b != 0.0 => a / b,
                    _ => 0.0,
                };

                Ok(serde_json::json!({ "result": result }))
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
            "calculate".to_string(),
            "Perform a mathematical calculation".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "a": { "type": "number" },
                    "b": { "type": "number" },
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"]
                    }
                },
                "required": ["a", "b", "operation"]
            }),
        ),
    ];

    // Create messages
    let messages = vec![
        ChatMessage::user("What's the weather in San Francisco? Then calculate 25 * 4.").build(),
    ];

    // Create resolver
    let resolver = MyToolResolver;

    // Set up orchestrator options with callbacks
    let options = OrchestratorOptions {
        max_steps: 10,
        on_step_finish: Some(Arc::new(|step: &StepResult| {
            println!("\n=== Step Finished ===");
            println!("Tool calls: {}", step.tool_calls.len());
            println!("Messages: {}", step.messages.len());
            if let Some(usage) = &step.usage {
                println!(
                    "Usage: {} prompt + {} completion = {} total tokens",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                );
            }
        })),
        on_finish: Some(Arc::new(|steps: &[StepResult]| {
            println!("\n=== Orchestration Complete ===");
            println!("Total steps: {}", steps.len());
            if let Some(total_usage) = StepResult::merge_usage(steps) {
                println!(
                    "Total usage: {} prompt + {} completion = {} total tokens",
                    total_usage.prompt_tokens,
                    total_usage.completion_tokens,
                    total_usage.total_tokens
                );
            }
        })),
        ..Default::default()
    };

    // Run orchestrator
    println!("Starting orchestration...\n");
    let stop_condition = step_count_is(10);
    let (response, steps) = generate(
        &client,
        messages,
        Some(tools),
        Some(&resolver),
        &[&*stop_condition],
        options,
    )
    .await?;

    // Print final response
    println!("\n=== Final Response ===");
    println!("{}", response.content_text().unwrap_or("No text content"));

    println!("\n=== Step Summary ===");
    for (i, step) in steps.iter().enumerate() {
        println!("Step {}: {} tool calls", i + 1, step.tool_calls.len());
    }

    Ok(())
}
