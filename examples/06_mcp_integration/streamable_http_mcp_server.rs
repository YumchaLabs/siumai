//! Streamable HTTP MCP Server Example (rmcp 0.8.0)
//!
//! Starts a streamable-HTTP-capable MCP server exposing `add` and `get_time` tools.
//! Endpoint: http://127.0.0.1:3000/mcp (default)
//!
//! Run:
//! ```bash
//! cargo run --example streamable_http_mcp_server
//! ```
//! Configure bind via env: `MCP_HTTP_BIND=127.0.0.1:3000`

use axum::Router;
use rmcp::{
    ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::*,
    schemars, tool, tool_handler, tool_router,
    transport::streamable_http_server::{
        StreamableHttpService, session::local::LocalSessionManager,
    },
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
pub struct StreamableHttpMcpServer {
    tool_router: ToolRouter<StreamableHttpMcpServer>,
}

#[tool_router]
impl StreamableHttpMcpServer {
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

impl Default for StreamableHttpMcpServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tool_handler]
impl ServerHandler for StreamableHttpMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "siumai-streamable-http-mcp-server".into(),
                version: "1.0.0".into(),
                ..Default::default()
            },
            instructions: Some(
                "Streamable HTTP MCP Server providing add/get_time tools for siumai LLM integration examples.".to_string(),
            ),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    siumai::tracing::init_default_tracing().ok();

    let bind = std::env::var("MCP_HTTP_BIND").unwrap_or_else(|_| "127.0.0.1:3000".to_string());
    let service = StreamableHttpService::new(
        || Ok(StreamableHttpMcpServer::new()),
        LocalSessionManager::default().into(),
        Default::default(),
    );

    let router = Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    println!("Streamable HTTP MCP server running at http://{bind}/mcp");
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
