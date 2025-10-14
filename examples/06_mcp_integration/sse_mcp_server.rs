//! SSE MCP Server Example (rmcp 0.8.0)
//!
//! Starts an SSE-capable MCP server that exposes simple tools like `add` and `get_time`.
//! Endpoint: http://127.0.0.1:3001/sse (default)
//!
//! Run:
//! ```bash
//! cargo run --example sse_mcp_server
//! ```
//! Configure bind via env: `SSE_BIND=127.0.0.1:3001`

use axum::Router;
use rmcp::{
    ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::*,
    schemars, tool, tool_handler, tool_router,
    transport::sse_server::{SseServer, SseServerConfig},
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

#[derive(Clone)]
pub struct SseMcpServer {
    tool_router: ToolRouter<SseMcpServer>,
}

#[tool_router]
impl SseMcpServer {
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
        Ok(CallToolResult::success(vec![Content::text(format!(
            "{a} + {b} = {result}"
        ))]))
    }

    /// Get current time (UTC/local)
    #[tool(description = "Get current date and time")]
    async fn get_time(
        &self,
        Parameters(GetTimeRequest { timezone }): Parameters<GetTimeRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        let now = std::time::SystemTime::now();
        let duration = now.duration_since(std::time::UNIX_EPOCH).unwrap();
        let timestamp = duration.as_secs();
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
            Some(tz) => format!(
                "Current time in {tz}: {}",
                datetime.format("%Y-%m-%d %H:%M:%S UTC")
            ),
        };
        Ok(CallToolResult::success(vec![Content::text(time_str)]))
    }
}

impl Default for SseMcpServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tool_handler]
impl ServerHandler for SseMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "siumai-sse-mcp-server".into(),
                version: "1.0.0".into(),
                ..Default::default()
            },
            instructions: Some(
                "SSE MCP Server providing simple tools like add and get_time for siumai LLM integration examples.".to_string(),
            ),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    siumai::tracing::init_default_tracing().ok();

    let bind = std::env::var("SSE_BIND").unwrap_or_else(|_| "127.0.0.1:3001".to_string());
    let config = SseServerConfig {
        bind: bind.parse()?,
        sse_path: "/sse".to_string(),
        post_path: "/message".to_string(),
        ct: tokio_util::sync::CancellationToken::new(),
        sse_keep_alive: None,
    };

    let (sse_server, router): (SseServer, Router) = SseServer::new(config);

    // Serve the HTTP endpoints for SSE
    let listener = tokio::net::TcpListener::bind(sse_server.config.bind).await?;
    let ct = sse_server.config.ct.child_token();
    let http_server = axum::serve(listener, router).with_graceful_shutdown(async move {
        ct.cancelled().await;
    });
    tokio::spawn(async move {
        if let Err(e) = http_server.await {
            eprintln!("SSE HTTP server error: {e}");
        }
    });

    // Start serving our MCP service over SSE transports
    let cancel_token = sse_server.with_service(SseMcpServer::new);
    println!("SSE MCP server running at http://{}/sse", bind);
    tokio::signal::ctrl_c().await?;
    cancel_token.cancel();
    Ok(())
}
