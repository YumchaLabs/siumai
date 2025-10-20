//! Advanced stop conditions for orchestrator.
//!
//! This example demonstrates:
//! - Using built-in stop conditions (step_count_is, has_tool_call, has_text_response)
//! - Combining multiple stop conditions with any_of
//! - Creating custom stop conditions
//!
//! Run with: cargo run --example stop-conditions --features openai

use siumai::orchestrator::{
    OrchestratorOptions, ToolResolver, any_of, custom_condition, generate, has_text_response,
    has_tool_call, step_count_is,
};
use siumai::prelude::*;
use siumai::types::{Tool, ToolFunction};

// Tool resolver with a "finalAnswer" tool
struct ResearchResolver;

#[async_trait::async_trait]
impl ToolResolver for ResearchResolver {
    async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, siumai::error::LlmError> {
        match name {
            "search" => {
                let query = arguments
                    .get("query")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                Ok(serde_json::json!({
                    "results": format!("Search results for: {}", query),
                    "count": 5
                }))
            }
            "analyze" => {
                let data = arguments.get("data").and_then(|v| v.as_str()).unwrap_or("");
                Ok(serde_json::json!({
                    "analysis": format!("Analysis of: {}", data),
                    "confidence": 0.95
                }))
            }
            "finalAnswer" => {
                let answer = arguments
                    .get("answer")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                Ok(serde_json::json!({
                    "answer": answer,
                    "final": true
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
    let client = Siumai::builder()
        .openai()
        .api_key(&std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini")
        .build()
        .await?;

    let tools = vec![
        Tool::function(
            "search".to_string(),
            "Search for information".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            }),
        ),
        Tool::function(
            "analyze".to_string(),
            "Analyze data".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "data": { "type": "string" }
                },
                "required": ["data"]
            }),
        ),
        Tool::function(
            "finalAnswer".to_string(),
            "Provide the final answer when research is complete".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "answer": { "type": "string" }
                },
                "required": ["answer"]
            }),
        ),
    ];

    let resolver = ResearchResolver;

    // Example 1: Stop when finalAnswer is called OR after 10 steps
    println!("=== Example 1: Stop on finalAnswer OR 10 steps ===\n");
    let messages1 = vec![
        ChatMessage::user(
            "Research the benefits of Rust programming language and provide a final answer.",
        )
        .build(),
    ];

    let stop_condition_1 = any_of(vec![step_count_is(10), has_tool_call("finalAnswer")]);

    let (response1, steps1) = generate(
        &client,
        messages1,
        Some(tools.clone()),
        Some(&resolver),
        &[&*stop_condition_1],
        OrchestratorOptions::default(),
    )
    .await?;

    println!("Response: {}", response1.content_text().unwrap_or(""));
    println!("Completed in {} steps\n", steps1.len());

    // Example 2: Stop when model generates text (no tool calls)
    println!("=== Example 2: Stop on text response ===\n");
    let messages2 = vec![ChatMessage::user("What is 2+2? Just answer directly.").build()];

    let stop_condition_2 = has_text_response();

    let (response2, steps2) = generate(
        &client,
        messages2,
        Some(tools.clone()),
        Some(&resolver),
        &[&*stop_condition_2],
        OrchestratorOptions::default(),
    )
    .await?;

    println!("Response: {}", response2.content_text().unwrap_or(""));
    println!("Completed in {} steps\n", steps2.len());

    // Example 3: Custom stop condition - stop if more than 2 tool calls in any step
    println!("=== Example 3: Custom stop condition ===\n");
    let messages3 =
        vec![ChatMessage::user("Search for information about AI and analyze the results.").build()];

    let custom_stop = custom_condition(|steps| steps.iter().any(|s| s.tool_calls.len() > 2));

    let stop_condition_3 = any_of(vec![step_count_is(10), custom_stop]);

    let (response3, steps3) = generate(
        &client,
        messages3,
        Some(tools.clone()),
        Some(&resolver),
        &[&*stop_condition_3],
        OrchestratorOptions::default(),
    )
    .await?;

    println!("Response: {}", response3.content_text().unwrap_or(""));
    println!("Completed in {} steps", steps3.len());
    for (i, step) in steps3.iter().enumerate() {
        println!("  Step {}: {} tool calls", i + 1, step.tool_calls.len());
    }

    Ok(())
}
