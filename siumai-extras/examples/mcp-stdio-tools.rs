//! MCP stdio tool discovery example.
//!
//! Run:
//! ```bash
//! MCP_SERVER_COMMAND="node mcp-server.js" \
//! cargo run -p siumai-extras --example mcp-stdio-tools --features "mcp,openai"
//! ```

use siumai::prelude::unified::*;
use siumai_extras::mcp::mcp_tools_from_stdio;

fn tool_label(tool: &Tool) -> (&str, &str) {
    match tool {
        Tool::Function { function } => (&function.name, &function.description),
        Tool::ProviderDefined(tool) => (&tool.name, tool.title.as_deref().unwrap_or(&tool.id)),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let command =
        std::env::var("MCP_SERVER_COMMAND").unwrap_or_else(|_| "node mcp-server.js".to_string());

    let (tools, resolver) = mcp_tools_from_stdio(&command).await?;

    println!("MCP stdio command: {command}");
    println!("Discovered {} tools:", tools.len());
    for tool in &tools {
        let (name, description) = tool_label(tool);
        println!("- {name}: {description}");
    }

    if std::env::var("OPENAI_API_KEY").is_err() {
        println!("\nOPENAI_API_KEY is missing, so the example stops after discovery.");
        return Ok(());
    }

    let reg = registry::global();
    let model = reg.language_model("openai:gpt-4o-mini")?;
    let messages = vec![user!(
        "Use the available MCP tools to summarize what capabilities you discovered."
    )];

    let (response, steps) = siumai_extras::orchestrator::generate(
        &model,
        messages,
        Some(tools),
        Some(&resolver),
        &[&*siumai_extras::orchestrator::step_count_is(6)],
        Default::default(),
    )
    .await?;

    println!("\nResponse:");
    println!("{}", response.content_text().unwrap_or_default());
    println!("Steps: {}", steps.len());

    Ok(())
}
