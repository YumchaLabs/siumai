//! Complete MCP Stdio Integration Example
//!
//! This example demonstrates how to integrate MCP servers with Siumai
//! using the `siumai-extras` MCP integration.
//!
//! ## What is MCP?
//!
//! MCP (Model Context Protocol) is a standardized protocol for connecting
//! AI models to external tools and data sources. It allows:
//! - Dynamic tool discovery
//! - Standardized tool execution
//! - Multiple transport options (stdio, SSE, HTTP)
//!
//! ## Setup
//!
//! 1. Add dependencies to your Cargo.toml:
//!    ```toml
//!    [dependencies]
//!    siumai = { version = "0.11", features = ["openai"] }
//!    siumai-extras = { version = "0.11", features = ["mcp"] }
//!    tokio = { version = "1.0", features = ["full"] }
//!    ```
//!
//! 2. Create a simple MCP server (Node.js example):
//!    ```javascript
//!    // mcp-server.js
//!    const { McpServer } = require('@modelcontextprotocol/sdk/server/mcp.js');
//!    const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
//!    
//!    const server = new McpServer({ name: 'example', version: '1.0.0' });
//!    
//!    server.tool('get_weather', 'Get weather for a city', { city: 'string' },
//!      async ({ city }) => ({
//!        content: [{ type: 'text', text: `Weather in ${city}: Sunny, 72°F` }]
//!      })
//!    );
//!    
//!    const transport = new StdioServerTransport();
//!    server.connect(transport);
//!    ```
//!
//! 3. Run this example:
//!    ```bash
//!    cargo run --example complete-stdio-example --features openai
//!    ```
//!
//! ## Features Demonstrated
//!
//! - Connecting to MCP server via stdio
//! - Automatic tool discovery
//! - Tool execution through Siumai orchestrator
//! - Multi-step tool calling
//! - Usage tracking

use siumai::orchestrator::{OrchestratorOptions, StepResult, generate, step_count_is};
use siumai::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔌 MCP Stdio Integration Example\n");

    println!("❌ This example requires siumai-extras with the 'mcp' feature");
    println!("\nTo use this example:");
    println!("1. Add to your Cargo.toml:");
    println!("   siumai-extras = {{ version = \"0.11\", features = [\"mcp\"] }}");
    println!("\n2. Import and use:");
    println!("   use siumai_extras::mcp::mcp_tools_from_stdio;");
    println!("\n3. Connect to MCP server:");
    println!("   let (tools, resolver) = mcp_tools_from_stdio(\"node mcp-server.js\").await?;");
    println!("\nSee siumai/docs/guides/MCP_INTEGRATION.md for complete guide.");

    Ok(())

    // Uncomment below when siumai-extras mcp feature is available:
    /*
    use siumai_extras::mcp::mcp_tools_from_stdio;

    println!("Step 1: Connecting to MCP server via stdio...");

    // Connect to MCP server and get available tools
    // Replace "node mcp-server.js" with your actual MCP server command
    let mcp_command = std::env::var("MCP_SERVER_COMMAND")
        .unwrap_or_else(|_| "node mcp-server.js".to_string());

    println!("  Command: {}", mcp_command);

    let (tools, resolver) = match mcp_tools_from_stdio(&mcp_command).await {
        Ok(result) => result,
        Err(e) => {
            println!("\n❌ Failed to connect to MCP server: {}", e);
            println!("\nMake sure:");
            println!("1. Your MCP server is available");
            println!("2. The command is correct (set MCP_SERVER_COMMAND env var)");
            println!("3. Node.js and required packages are installed");
            return Err(e.into());
        }
    };

    println!("✅ Connected to MCP server");
    println!("\nStep 2: Discovered {} tools:", tools.len());
    for tool in &tools {
        println!("  - {} : {}", tool.name, tool.description.as_deref().unwrap_or("No description"));
    }

    // Get API key
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable not set");

    println!("\nStep 3: Creating Siumai OpenAI client...");
    let client = Siumai::builder()
        .openai()
        .api_key(&api_key)
        .model("gpt-4o-mini")
        .build()
        .await?;

    println!("✅ Client created");

    println!("\nStep 4: Setting up orchestrator with MCP tools...");

    // Create messages
    let messages = vec![
        user!("Use the available tools to help me. What tools do you have access to?")
    ];

    // Set up orchestrator options with callbacks
    let options = OrchestratorOptions {
        max_steps: 10,
        on_step_finish: Some(Arc::new(|step: &StepResult| {
            println!("\n  📊 Step finished:");
            println!("     - Tool calls: {}", step.tool_calls.len());
            if let Some(usage) = &step.usage {
                println!(
                    "     - Tokens: {} prompt + {} completion = {} total",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                );
            }
        })),
        on_finish: Some(Arc::new(|steps: &[StepResult]| {
            println!("\n  ✅ Orchestration complete!");
            println!("     - Total steps: {}", steps.len());
            if let Some(total_usage) = StepResult::merge_usage(steps) {
                println!(
                    "     - Total tokens: {} prompt + {} completion = {} total",
                    total_usage.prompt_tokens,
                    total_usage.completion_tokens,
                    total_usage.total_tokens
                );
            }
        })),
        ..Default::default()
    };

    println!("✅ Orchestrator configured");
    println!("\nStep 5: Running orchestration...");
    println!("─────────────────────────────────────");

    // Run orchestration
    let (response, steps) = generate(
        &client,
        messages,
        Some(tools),
        Some(&resolver),
        vec![step_count_is(10)],
        options,
    )
    .await?;

    println!("─────────────────────────────────────");
    println!("\n📝 Final Response:");
    println!("{}", response.content_text().unwrap_or("No text content"));

    println!("\n📈 Summary:");
    println!("  - Total steps: {}", steps.len());
    println!("  - Total tool calls: {}",
        steps.iter().map(|s| s.tool_calls.len()).sum::<usize>());

    if let Some(total_usage) = StepResult::merge_usage(&steps) {
        println!("  - Total tokens: {}", total_usage.total_tokens);
    }

    println!("\n✅ Example completed successfully!");
    */
}
