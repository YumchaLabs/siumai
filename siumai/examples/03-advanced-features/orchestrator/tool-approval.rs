//! Tool approval workflow for orchestrator.
//!
//! This example demonstrates:
//! - Approving tool calls before execution
//! - Modifying tool arguments
//! - Denying dangerous tool calls
//! - Using ToolApproval enum
//!
//! Run with: cargo run --example tool-approval --features openai

use siumai::orchestrator::{
    OrchestratorOptions, ToolApproval, ToolResolver, generate, step_count_is,
};
use siumai::prelude::*;
use siumai::types::Tool;
use std::sync::Arc;

// Tool resolver with potentially dangerous operations
struct SystemResolver;

#[async_trait::async_trait]
impl ToolResolver for SystemResolver {
    async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, siumai::error::LlmError> {
        match name {
            "read_file" => {
                let path = arguments.get("path").and_then(|v| v.as_str()).unwrap_or("");
                Ok(serde_json::json!({
                    "content": format!("Contents of {}", path),
                    "size": 1024
                }))
            }
            "write_file" => {
                let path = arguments.get("path").and_then(|v| v.as_str()).unwrap_or("");
                let content = arguments
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                Ok(serde_json::json!({
                    "success": true,
                    "path": path,
                    "bytes_written": content.len()
                }))
            }
            "delete_file" => {
                let path = arguments.get("path").and_then(|v| v.as_str()).unwrap_or("");
                Ok(serde_json::json!({
                    "success": true,
                    "deleted": path
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
            "read_file".to_string(),
            "Read a file from the filesystem".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"]
            }),
        ),
        Tool::function(
            "write_file".to_string(),
            "Write content to a file".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "content": { "type": "string" }
                },
                "required": ["path", "content"]
            }),
        ),
        Tool::function(
            "delete_file".to_string(),
            "Delete a file from the filesystem".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"]
            }),
        ),
    ];

    let messages = vec![
        ChatMessage::user("Read the file /etc/passwd, then write a backup to /tmp/backup.txt")
            .build(),
    ];

    let resolver = SystemResolver;

    // Set up approval callback
    let options = OrchestratorOptions {
        max_steps: 10,
        on_tool_approval: Some(Arc::new(|tool_name: &str, args: &serde_json::Value| {
            println!("\n🔍 Tool approval request:");
            println!("  Tool: {}", tool_name);
            println!("  Args: {}", serde_json::to_string_pretty(args).unwrap());

            match tool_name {
                // Deny dangerous operations
                "delete_file" => {
                    println!("  ❌ DENIED: Delete operations are not allowed");
                    ToolApproval::Deny {
                        reason: "Delete operations are not allowed for safety".to_string(),
                    }
                }

                // Modify sensitive file reads
                "read_file" => {
                    if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
                        if path.contains("/etc/") || path.contains("passwd") {
                            println!("  ⚠️  MODIFIED: Redirecting sensitive file read");
                            let mut modified = args.clone();
                            modified["path"] = serde_json::json!("/tmp/safe_file.txt");
                            return ToolApproval::Modify(modified);
                        }
                    }
                    println!("  ✅ APPROVED");
                    ToolApproval::Approve(args.clone())
                }

                // Approve write operations with logging
                "write_file" => {
                    if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
                        if path.starts_with("/tmp/") {
                            println!("  ✅ APPROVED: Writing to /tmp is safe");
                            ToolApproval::Approve(args.clone())
                        } else {
                            println!("  ⚠️  MODIFIED: Redirecting write to /tmp");
                            let mut modified = args.clone();
                            let filename = path.split('/').last().unwrap_or("output.txt");
                            modified["path"] = serde_json::json!(format!("/tmp/{}", filename));
                            ToolApproval::Modify(modified)
                        }
                    } else {
                        println!("  ✅ APPROVED");
                        ToolApproval::Approve(args.clone())
                    }
                }

                _ => {
                    println!("  ✅ APPROVED");
                    ToolApproval::Approve(args.clone())
                }
            }
        })),
        ..Default::default()
    };

    println!("Starting orchestration with tool approval...\n");
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

    println!("\n=== Final Response ===");
    println!("{}", response.content_text().unwrap_or("No text content"));

    println!("\n=== Summary ===");
    println!("Total steps: {}", steps.len());
    let total_tool_calls: usize = steps.iter().map(|s| s.tool_calls.len()).sum();
    println!("Total tool calls: {}", total_tool_calls);

    Ok(())
}
