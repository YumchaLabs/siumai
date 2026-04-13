//! Streaming multi-step orchestration.
//!
//! This example demonstrates:
//! - Using generate_stream_owned for streaming orchestration
//! - Consuming the stream in real-time
//! - Receiving step results asynchronously
//! - Using the agent pattern with streaming
//!
//! Run with: cargo run -p siumai-extras --example streaming-orchestrator

use futures::StreamExt;
use siumai::prelude::unified::*;
use siumai_extras::orchestrator::{ToolLoopAgent, ToolResolver, step_count_is};

fn stream_text_delta(event: &ChatStreamEvent) -> Option<&str> {
    match event {
        ChatStreamEvent::ContentDelta { delta, .. } => Some(delta.as_str()),
        ChatStreamEvent::Part {
            part: ChatStreamPart::TextDelta { delta, .. },
        }
        | ChatStreamEvent::PartWithReplay {
            part: ChatStreamPart::TextDelta { delta, .. },
            ..
        } => Some(delta.as_str()),
        _ => None,
    }
}

// Simple tool resolver
struct NewsResolver;

#[async_trait::async_trait]
impl ToolResolver for NewsResolver {
    async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, LlmError> {
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
            _ => Err(LlmError::InternalError(format!("Unknown tool: {}", name))),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let reg = registry::global();
    let client = reg.language_model("openai:gpt-4o-mini")?;

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
            Ok(event) => {
                if let Some(delta) = stream_text_delta(&event) {
                    print!("{}", delta);
                    current_text.push_str(delta);
                    std::io::Write::flush(&mut std::io::stdout())?;
                    continue;
                }

                match event {
                    ChatStreamEvent::ToolCallDelta {
                        function_name: Some(name),
                        ..
                    } => {
                        println!("\n  [Tool call: {}]", name);
                    }
                    ChatStreamEvent::Part {
                        part: ChatStreamPart::ToolInputStart { tool_name, .. },
                    } => {
                        println!("\n  [Tool input start: {}]", tool_name);
                    }
                    ChatStreamEvent::Part {
                        part: ChatStreamPart::ToolCall(call),
                    } => {
                        println!("\n  [Tool call ready: {}]", call.tool_name);
                    }
                    ChatStreamEvent::StreamEnd { .. } => {
                        println!("\n  [Stream complete]");
                    }
                    _ => {}
                }
            }
            Err(e) => {
                eprintln!("\nStream error: {}", e);
                break;
            }
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
                usage.prompt_tokens().unwrap_or(0),
                usage.completion_tokens().unwrap_or(0),
                usage.total_tokens().unwrap_or(0)
            );
        }
    }

    Ok(())
}
