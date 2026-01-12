//! Example: Streaming Tool Execution with Preliminary Results
//!
//! This example demonstrates how to implement tools that return intermediate results
//! during execution, which is useful for long-running operations that can provide
//! progress updates.
//!
//! Run with:
//! ```bash
//! cargo run -p siumai-extras --example streaming-tool-execution
//! ```

use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;

use siumai::prelude::*;
use siumai_extras::orchestrator::{
    OrchestratorOptions, ToolExecutionResult, ToolResolver, generate, step_count_is,
};

/// A tool resolver that supports streaming tool execution
struct StreamingToolResolver;

#[async_trait]
impl ToolResolver for StreamingToolResolver {
    /// Simple non-streaming tool execution
    async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value, LlmError> {
        match name {
            "get_weather" => {
                let city = arguments
                    .get("city")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown");
                Ok(json!({
                    "city": city,
                    "temperature": 72,
                    "condition": "sunny"
                }))
            }
            _ => Err(LlmError::InternalError(format!("Unknown tool: {}", name))),
        }
    }

    /// Streaming tool execution with preliminary results
    async fn call_tool_stream(
        &self,
        name: &str,
        arguments: Value,
    ) -> Result<BoxStream<'static, Result<ToolExecutionResult, LlmError>>, LlmError> {
        match name {
            // Simple tool - use default implementation (wraps call_tool)
            "get_weather" => {
                let result = self.call_tool(name, arguments).await?;
                Ok(Box::pin(stream::once(async move {
                    Ok(ToolExecutionResult::final_result(result))
                })))
            }

            // Long-running tool with progress updates
            "analyze_data" => {
                let dataset = arguments
                    .get("dataset")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();

                // Create a stream that emits progress updates
                Ok(Box::pin(stream::unfold(0, move |step| {
                    let dataset = dataset.clone();
                    async move {
                        if step < 5 {
                            // Simulate work
                            tokio::time::sleep(Duration::from_millis(500)).await;

                            let progress = (step + 1) * 20;
                            let result = if step < 4 {
                                // Preliminary results
                                ToolExecutionResult::preliminary(json!({
                                    "status": "processing",
                                    "dataset": dataset,
                                    "progress": progress,
                                    "current_step": format!("Analyzing chunk {}/5", step + 1)
                                }))
                            } else {
                                // Final result
                                ToolExecutionResult::final_result(json!({
                                    "status": "complete",
                                    "dataset": dataset,
                                    "progress": 100,
                                    "results": {
                                        "total_records": 10000,
                                        "anomalies_found": 42,
                                        "confidence": 0.95
                                    }
                                }))
                            };

                            Some((Ok(result), step + 1))
                        } else {
                            None
                        }
                    }
                })))
            }

            // Another streaming tool - file processing
            "process_file" => {
                let filename = arguments
                    .get("filename")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown.txt")
                    .to_string();

                Ok(Box::pin(stream::unfold(0, move |step| {
                    let filename = filename.clone();
                    async move {
                        if step < 3 {
                            tokio::time::sleep(Duration::from_millis(300)).await;

                            let result = if step < 2 {
                                ToolExecutionResult::preliminary(json!({
                                    "status": "processing",
                                    "filename": filename,
                                    "stage": match step {
                                        0 => "reading",
                                        1 => "parsing",
                                        _ => "unknown"
                                    }
                                }))
                            } else {
                                ToolExecutionResult::final_result(json!({
                                    "status": "complete",
                                    "filename": filename,
                                    "lines_processed": 1000,
                                    "errors": 0
                                }))
                            };

                            Some((Ok(result), step + 1))
                        } else {
                            None
                        }
                    }
                })))
            }

            _ => Err(LlmError::InternalError(format!("Unknown tool: {}", name))),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");

    // Create client
    let client = Siumai::builder()
        .openai()
        .api_key(&api_key)
        .model("gpt-4o-mini")
        .build()
        .await?;

    // Define tools
    let tools = vec![
        Tool::function(
            "get_weather".to_string(),
            "Get current weather for a city".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["city"]
            }),
        ),
        Tool::function(
            "analyze_data".to_string(),
            "Analyze a dataset (long-running operation with progress updates)".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Name of the dataset to analyze"
                    }
                },
                "required": ["dataset"]
            }),
        ),
        Tool::function(
            "process_file".to_string(),
            "Process a file (streaming operation)".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file to process"
                    }
                },
                "required": ["filename"]
            }),
        ),
    ];

    // Create resolver
    let resolver = StreamingToolResolver;

    // Initial messages
    let messages = vec![
        ChatMessage::user("Please analyze the 'sales_2024' dataset and tell me what you find.")
            .build(),
    ];

    println!("🚀 Starting orchestration with streaming tool execution...\n");

    // Set up options with preliminary result callback
    let options = OrchestratorOptions {
        max_steps: 5,
        on_preliminary_tool_result: Some(Arc::new(|tool_name, tool_call_id, output| {
            println!(
                "📊 [Preliminary] Tool: {} (ID: {})",
                tool_name, tool_call_id
            );
            println!(
                "   Output: {}\n",
                serde_json::to_string_pretty(output).unwrap()
            );
        })),
        on_step_finish: Some(Arc::new(|step| {
            println!("✅ Step finished with {} messages", step.messages.len());
        })),
        ..Default::default()
    };

    // Run orchestration
    let (response, steps) = generate(
        &client,
        messages,
        Some(tools),
        Some(&resolver),
        &[&*step_count_is(5)],
        options,
    )
    .await?;

    // Print results
    println!("\n🎉 Orchestration complete!");
    println!("Total steps: {}", steps.len());
    println!("\nFinal response:");
    if let Some(text) = response.content_text() {
        println!("{}", text);
    }

    Ok(())
}
