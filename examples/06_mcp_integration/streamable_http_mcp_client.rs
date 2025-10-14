//! Streamable HTTP MCP Client Example (rmcp 0.8.0)
//!
//! Demonstrates using rmcp's StreamableHttpClientTransport (reqwest backend)
//! to connect to an MCP server over a streamable HTTP endpoint.
//!
//! Run:
//! ```bash
//! cargo run --example streamable_http_mcp_client
//! ```
//! Endpoint can be configured via env var MCP_HTTP_ENDPOINT (default: http://127.0.0.1:3000/mcp)

use rmcp::{
    ServiceExt,
    model::{CallToolRequestParam, ClientCapabilities, ClientInfo, Implementation},
    transport::StreamableHttpClientTransport,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    siumai::tracing::init_default_tracing().ok();

    let http_url = std::env::var("MCP_HTTP_ENDPOINT")
        .unwrap_or_else(|_| "http://127.0.0.1:3000/mcp".to_string());
    println!("Connecting to streamable HTTP endpoint: {http_url}");

    // Create transport from URI
    let transport = StreamableHttpClientTransport::from_uri(http_url);

    // Prepare client info (handshake)
    let client_info = ClientInfo {
        protocol_version: Default::default(),
        capabilities: ClientCapabilities::default(),
        client_info: Implementation {
            name: "siumai-streamable-http-mcp-client".to_string(),
            title: None,
            version: "0.1.0".to_string(),
            website_url: None,
            icons: None,
        },
    };

    // Serve over transport
    let client = client_info.serve(transport).await?;
    println!("Connected to server: {:#?}", client.peer_info());

    // List tools
    let tools = client.list_tools(Default::default()).await?;
    println!("Available tools: {}", tools.tools.len());

    // Try calling a simple tool if available
    if let Some(first_tool) = tools.tools.first() {
        println!("Calling first tool: {}", first_tool.name);
        let result = client
            .call_tool(CallToolRequestParam {
                name: first_tool.name.clone(),
                arguments: None,
            })
            .await?;
        println!("Tool result: {result:#?}");
    }

    client.cancel().await?;
    Ok(())
}
