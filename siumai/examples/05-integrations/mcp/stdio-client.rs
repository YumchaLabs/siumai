//! MCP Stdio Client - Connect to MCP servers
//!
//! This example demonstrates connecting to an MCP server via stdio
//! and using its tools with Siumai.
//!
//! MCP (Model Context Protocol) allows LLMs to access external tools
//! and data sources in a standardized way.
//!
//! ## Setup
//! 1. Build the MCP server: `cargo build --example stdio_mcp_server`
//! 2. Run this example: `cargo run --example stdio-client --features "openai,mcp"`
//!
//! ## Learn More
//! See `siumai/examples/06_mcp_integration/` for complete MCP examples:
//! - stdio, HTTP, SSE transports
//! - Client and server implementations
//! - Tool discovery and execution

// prelude not needed for this stub example

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔌 MCP Stdio Client Example\n");
    println!("This is a simplified example.");
    println!("For complete MCP integration examples, see:");
    println!("  - siumai/examples/06_mcp_integration/stdio_mcp_client.rs");
    println!("  - siumai/examples/06_mcp_integration/http_mcp_client.rs");
    println!("  - siumai/examples/06_mcp_integration/sse_mcp_client.rs");
    println!("\nMCP enables:");
    println!("  ✅ Standardized tool protocol");
    println!("  ✅ Multiple transport options (stdio, HTTP, SSE)");
    println!("  ✅ Tool discovery and execution");
    println!("  ✅ Integration with any LLM provider");

    Ok(())
}
