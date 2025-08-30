//! Stdio MCP Server Example
//!
//! This example demonstrates how to create a stdio MCP server using rmcp crate
//! that provides simple tools like add and get_time. The server communicates
//! via standard input/output, making it suitable for process-based integration.
//!
//! Features:
//! - Stdio transport (standard input/output)
//! - Simple mathematical operations (add)
//! - Time utilities (get_time)
//! - Real MCP protocol implementation
//! - JSON-RPC over stdio
//!
//! To run this server:
//! ```bash
//! cargo run --example stdio_mcp_server
//! ```
//!
//! To test with MCP inspector:
//! ```bash
//! npx @modelcontextprotocol/inspector cargo run --example stdio_mcp_server
//! ```

use rmcp::{
    ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::*,
    schemars, tool, tool_handler, tool_router,
    transport::stdio,
};
use serde::Deserialize;

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct AddRequest {
    #[schemars(description = "First number to add")]
    pub a: f64,
    #[schemars(description = "Second number to add")]
    pub b: f64,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetTimeRequest {
    #[schemars(description = "Timezone (optional, defaults to UTC)")]
    pub timezone: Option<String>,
}

/// Stdio MCP Server implementation with tools
#[derive(Clone)]
pub struct StdioMcpServer {
    tool_router: ToolRouter<StdioMcpServer>,
}

#[tool_router]
impl StdioMcpServer {
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    /// Add two numbers together
    #[tool(description = "Add two numbers together")]
    async fn add(
        &self,
        Parameters(AddRequest { a, b }): Parameters<AddRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        let result = a + b;
        eprintln!("🔧 Tool 'add' called: {a} + {b} = {result}");
        Ok(CallToolResult::success(vec![Content::text(format!(
            "{a} + {b} = {result}"
        ))]))
    }

    /// Get current time
    #[tool(description = "Get current date and time")]
    async fn get_time(
        &self,
        Parameters(GetTimeRequest { timezone }): Parameters<GetTimeRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        let now = std::time::SystemTime::now();
        let duration = now.duration_since(std::time::UNIX_EPOCH).unwrap();
        let timestamp = duration.as_secs();

        // Convert to readable format
        let datetime =
            chrono::DateTime::from_timestamp(timestamp as i64, 0).unwrap_or_else(chrono::Utc::now);

        let time_str = match timezone.as_deref() {
            Some("local") => {
                let local_time = datetime.with_timezone(&chrono::Local);
                format!(
                    "Current local time: {}",
                    local_time.format("%Y-%m-%d %H:%M:%S %Z")
                )
            }
            Some("UTC") | None => {
                format!(
                    "Current UTC time: {}",
                    datetime.format("%Y-%m-%d %H:%M:%S UTC")
                )
            }
            Some(tz) => {
                format!(
                    "Current time in {tz}: {}",
                    datetime.format("%Y-%m-%d %H:%M:%S UTC")
                )
            }
        };

        eprintln!("🔧 Tool 'get_time' called: {time_str}");
        Ok(CallToolResult::success(vec![Content::text(time_str)]))
    }
}

impl Default for StdioMcpServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tool_handler]
impl ServerHandler for StdioMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            server_info: Implementation {
                name: "siumai-stdio-mcp-server".into(),
                version: "1.0.0".into(),
            },
            instructions: Some("Stdio MCP Server providing simple tools like add and get_time for siumai LLM integration examples.".to_string()),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging to stderr (stdout is used for MCP communication)
    siumai::tracing::init_default_tracing().ok();

    eprintln!("🌐 Stdio MCP Server - Starting server with simple tools");
    eprintln!("📋 Available tools:");
    eprintln!("   • add - Add two numbers together");
    eprintln!("   • get_time - Get current date and time");
    eprintln!("📡 Communication: Standard Input/Output (JSON-RPC)");
    eprintln!("⏹️  Press Ctrl+C to stop");
    eprintln!();

    // Create MCP server instance
    let server = StdioMcpServer::new();

    // Start stdio MCP server
    let service = server.serve(stdio()).await?;

    // Wait for the service to complete
    service.waiting().await?;

    eprintln!("✅ Stdio MCP Server stopped");
    Ok(())
}
