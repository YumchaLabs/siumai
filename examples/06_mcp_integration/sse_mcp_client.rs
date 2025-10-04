//! SSE MCP Client Example (rmcp 0.8.0)
//!
//! This example demonstrates how to connect to an SSE-capable MCP server
//! using rmcp's `SseClientTransport` and perform basic operations:
//! - initialize
//! - list tools
//! - call a tool
//!
//! Prerequisites:
//! - An SSE-capable MCP server running (endpoint like `http://127.0.0.1:3000/sse`)
//! - Enable feature `transport-sse-client-reqwest` (configured in Cargo.toml)
//!
//! Run:
//! ```bash
//! cargo run --example sse_mcp_client
//! ```

use rmcp::{
    ServiceExt,
    model::{CallToolRequestParam, ClientCapabilities, ClientInfo, Implementation},
    transport::SseClientTransport,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Basic logging to stdout
    siumai::tracing::init_default_tracing().ok();

    // Change this to your SSE endpoint
    let sse_url = std::env::var("MCP_SSE_ENDPOINT")
        .unwrap_or_else(|_| "http://127.0.0.1:3000/sse".to_string());

    println!("Connecting to SSE endpoint: {sse_url}");
    let transport = SseClientTransport::start(sse_url).await?;

    let client_info = ClientInfo {
        protocol_version: Default::default(),
        capabilities: ClientCapabilities::default(),
        client_info: Implementation {
            name: "siumai-sse-mcp-client".to_string(),
            title: None,
            version: "0.1.0".to_string(),
            website_url: None,
            icons: None,
        },
    };

    // Create the running client service over SSE transport
    let client = client_info.serve(transport).await?;

    // Initialize handshake done by serve(); show server info
    let server_info = client.peer_info();
    println!("Connected to server: {server_info:#?}");

    // List tools
    let tools = client.list_tools(Default::default()).await?;
    println!("Available tools: {}", tools.tools.len());
    for t in tools.tools.iter() {
        println!(
            " - {}: {}",
            t.name,
            t.description.as_deref().unwrap_or("(no description)")
        );
    }

    // Optionally call a tool if available (example: a tool named "add" without args schema)
    if tools
        .tools
        .iter()
        .any(|t| t.name == "increment" || t.name == "add")
    {
        let tool_name = if tools.tools.iter().any(|t| t.name == "increment") {
            "increment"
        } else {
            "add"
        };
        println!("Calling tool: {tool_name}");
        let result = client
            .call_tool(CallToolRequestParam {
                name: tool_name.into(),
                arguments: serde_json::json!({}).as_object().cloned(),
            })
            .await?;
        println!("Tool result: {result:#?}");
    }

    // Graceful shutdown
    client.cancel().await?;
    Ok(())
}
