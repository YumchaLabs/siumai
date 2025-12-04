//! Streaming multi-step orchestration.
//!
//! This example demonstrates:
//! - Using generate_stream_owned for streaming orchestration
//! - Consuming the stream in real-time
//! - Receiving step results asynchronously
//! - Using the agent pattern with streaming
//!
//! Run with: cargo run --example streaming-orchestrator --features openai

use futures::StreamExt;
use siumai::prelude::*;
use siumai::types::{ChatStreamEvent, ContentPart, Tool};
use siumai_extras::orchestrator::{ToolLoopAgent, ToolResolver, step_count_is};

// Simple tool resolver
struct NewsResolver;

#[async_trait::async_trait]
impl ToolResolver for NewsResolver {
    async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, siumai::error::LlmError> {
        match name {
            "get_headlines" => {
                let category = arguments
                    .get("category")
                    .and_then(|v| v.as_str())
                    .unwrap_or("general");
                Ok(serde_json::json!({
                    "headlines": [
                        format!("{} news headline 1", category),
                        format!("{} news headline 2", category),
                        format!("{} news headline 3", category),
                    ]
                }))
            }
            "get_article" => {
                let headline = arguments
                    .get("headline")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                Ok(serde_json::json!({
                    "title": headline,
                    "content": format!("Full article content for: {}", headline),
                    "author": "AI Reporter"
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
            "get_headlines".to_string(),
            "Get news headlines for a category".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["technology", "sports", "business", "general"]
                    }
                },
                "required": ["category"]
            }),
        ),
        Tool::function(
            "get_article".to_string(),
            "Get the full article for a headline".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "headline": { "type": "string" }
                },
                "required": ["headline"]
            }),
        ),
    ];

    // Create agent
    let agent = ToolLoopAgent::new(client, tools, vec![step_count_is(10)])
        .with_system("You are a news assistant. Fetch headlines and provide summaries.")
        .with_id("news-agent")
        .on_step_finish(std::sync::Arc::new(|step| {
            println!("\n  [Step completed: {} tool calls]", step.tool_calls.len());
        }));

    let resolver = NewsResolver;

    println!("=== Streaming Orchestration ===\n");
    let messages = vec![
        ChatMessage::user("Get the latest technology headlines and summarize the first one.")
            .build(),
    ];

    // Start streaming
    let orchestration = agent.stream(messages, resolver).await?;

    println!("Streaming response:\n");

    // Consume the stream
    let mut stream = orchestration.stream;
    let mut current_text = String::new();

    while let Some(event) = stream.next().await {
        match event {
            Ok(ChatStreamEvent::ContentDelta { delta, .. }) => {
                print!("{}", delta);
                current_text.push_str(&delta);
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Ok(ChatStreamEvent::ToolCallDelta {
                function_name: Some(name),
                ..
            }) => {
                println!("\n  [Tool call: {}]", name);
            }
            Ok(ChatStreamEvent::StreamEnd { .. }) => {
                println!("\n  [Stream complete]");
            }
            Err(e) => {
                eprintln!("\nStream error: {}", e);
                break;
            }
            _ => {}
        }
    }

    // Wait for all steps to complete
    println!("\n\nWaiting for orchestration to complete...");
    let steps = orchestration.steps.await?;

    println!("\n=== Orchestration Complete ===");
    println!("Total steps: {}", steps.len());
    println!("Final text length: {} characters", current_text.len());

    // Print step summary
    for (i, step) in steps.iter().enumerate() {
        println!("\nStep {}:", i + 1);
        println!("  Tool calls: {}", step.tool_calls.len());
        for tc in &step.tool_calls {
            if let ContentPart::ToolCall { tool_name, .. } = tc {
                println!("    - {}", tool_name);
            }
        }
        if let Some(usage) = &step.usage {
            println!(
                "  Usage: {} prompt + {} completion = {} total",
                usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
            );
        }
    }

    Ok(())
}
