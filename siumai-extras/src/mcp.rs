//! MCP (Model Context Protocol) integration for Siumai.
//!
//! This module provides convenient integration between Siumai and MCP servers
//! using the `rmcp` library. It allows you to:
//!
//! - Connect to MCP servers via stdio or streamable HTTP transports
//! - Automatically discover tools from MCP servers
//! - Execute MCP tools through Siumai's orchestrator
//! - Use MCP tools with any Siumai-supported LLM provider
//!
//! # Features
//!
//! - **Multiple Transports**: stdio and streamable HTTP
//! - **Automatic Tool Discovery**: Fetch tools from MCP servers
//! - **Seamless Integration**: Works with Siumai's orchestrator and agents
//! - **Type-Safe**: Leverages Rust's type system for safety
//!
//! # Example
//!
//! ```rust,ignore
//! use siumai::prelude::unified::*;
//! use siumai_extras::mcp::{McpToolResolver, mcp_tools_from_stdio};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Connect to MCP server and get tools
//!     let (tools, resolver) = mcp_tools_from_stdio("node mcp-server.js").await?;
//!     
//!     // Create Siumai model
//!     let reg = registry::global();
//!     let model = reg.language_model("openai:gpt-4o-mini")?;
//!     
//!     // Use with orchestrator
//!     let messages = vec![user!("Use the available tools to help me")];
//!     let (response, _) = siumai_extras::orchestrator::generate(
//!         &model,
//!         messages,
//!         Some(tools),
//!         Some(&resolver),
//!         vec![siumai_extras::orchestrator::step_count_is(10)],
//!         Default::default(),
//!     ).await?;
//!     
//!     println!("Response: {}", response.content_text().unwrap());
//!     Ok(())
//! }
//! ```
//!
//! # Transports
//!
//! ## Stdio Transport
//!
//! Connect to a local MCP server via stdio:
//!
//! ```rust,ignore
//! let (tools, resolver) = mcp_tools_from_stdio("node server.js").await?;
//! ```
//!
//! ## Streamable HTTP Transport
//!
//! Connect to an MCP server via streamable HTTP:
//!
//! ```rust,ignore
//! let (tools, resolver) = mcp_tools_from_http("http://localhost:3000/mcp").await?;
//! ```
//!
//! The older standalone SSE transport helpers remain as deprecated compatibility aliases because
//! upstream `rmcp` removed that client transport in favor of streamable HTTP.

use crate::orchestrator::ToolResolver;
use async_trait::async_trait;
use rmcp::{
    model::CallToolRequestParams,
    service::{Peer, RoleClient, RunningService, Service, ServiceExt},
    transport::{StreamableHttpClientTransport, TokioChildProcess},
};
use serde_json::Value;
use siumai::prelude::unified::{LlmError, Tool};
use std::sync::Arc;

/// MCP tool resolver that executes tools via an MCP service.
///
/// This struct implements Siumai's `ToolResolver` trait, allowing MCP tools
/// to be used seamlessly with Siumai's orchestrator and agents.
///
/// # Example
///
/// ```rust,ignore
/// use siumai_extras::mcp::McpToolResolver;
/// use rmcp::{ServiceExt, transport::{TokioChildProcess, ConfigureCommandExt}};
/// use tokio::process::Command;
///
/// let service = ()
///     .serve(TokioChildProcess::new(Command::new("uvx").configure(|cmd| {
///         cmd.arg("mcp-server-git");
///     }))?)
///     .await?;
/// let resolver = McpToolResolver::new(service);
/// ```
pub struct McpToolResolver {
    service: Arc<Peer<RoleClient>>,
}

impl McpToolResolver {
    /// Create a new MCP tool resolver from an existing MCP service.
    ///
    /// # Arguments
    ///
    /// * `service` - An initialized MCP service (RunningService or Peer)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let service = ().serve(transport).await?;
    /// let resolver = McpToolResolver::new(service);
    /// ```
    pub fn new<S: Service<RoleClient>>(service: RunningService<RoleClient, S>) -> Self {
        Self {
            service: Arc::new(service.peer().clone()),
        }
    }

    /// Create a new MCP tool resolver from a stdio command.
    ///
    /// This is a convenience method that creates a stdio transport and client.
    ///
    /// # Arguments
    ///
    /// * `command` - The command to execute (e.g., "node server.js")
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let resolver = McpToolResolver::from_stdio("node server.js").await?;
    /// ```
    /// Create a new MCP tool resolver from a stdio command.
    ///
    /// This is a convenience method that creates a stdio transport and service.
    ///
    /// # Arguments
    ///
    /// * `command` - The command to execute (e.g., "node server.js" or "uvx mcp-server-git")
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let resolver = McpToolResolver::from_stdio("uvx mcp-server-git").await?;
    /// ```
    pub async fn from_stdio(command: &str) -> Result<Self, LlmError> {
        // Parse command into program and args
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return Err(LlmError::InvalidInput("Empty command".to_string()));
        }

        let program = parts[0];
        let args = &parts[1..];

        let mut cmd = tokio::process::Command::new(program);
        cmd.args(args);

        let transport = TokioChildProcess::new(cmd).map_err(|e| {
            LlmError::InternalError(format!("Failed to create stdio transport: {}", e))
        })?;

        let service = ()
            .serve(transport)
            .await
            .map_err(|e| LlmError::InternalError(format!("Failed to create MCP service: {}", e)))?;

        Ok(Self::new(service))
    }

    /// Create a new MCP tool resolver from a streamable HTTP endpoint.
    ///
    /// # Arguments
    ///
    /// * `url` - The streamable HTTP endpoint URL (e.g., "http://localhost:3000/mcp")
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let resolver = McpToolResolver::from_http("http://localhost:3000/mcp").await?;
    /// ```
    pub async fn from_http(url: &str) -> Result<Self, LlmError> {
        let transport = StreamableHttpClientTransport::from_uri(url);

        let service = ()
            .serve(transport)
            .await
            .map_err(|e| LlmError::InternalError(format!("Failed to create MCP service: {}", e)))?;

        Ok(Self::new(service))
    }

    /// Create a new MCP tool resolver from a legacy SSE endpoint.
    ///
    /// The upstream `rmcp` SDK removed the standalone SSE client transport in favor of streamable
    /// HTTP. Prefer [`Self::from_http`] for new code.
    #[deprecated(
        since = "0.11.0",
        note = "rmcp removed standalone SSE transport; use from_http with a streamable HTTP MCP endpoint"
    )]
    pub async fn from_sse(url: &str) -> Result<Self, LlmError> {
        Self::from_http(url).await
    }

    /// Get the list of available tools from the MCP server.
    ///
    /// # Returns
    ///
    /// A vector of Siumai `Tool` objects that can be used with the orchestrator.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let resolver = McpToolResolver::from_stdio("node server.js").await?;
    /// let tools = resolver.list_tools().await?;
    /// ```
    pub async fn list_tools(&self) -> Result<Vec<Tool>, LlmError> {
        let mcp_tools = self
            .service
            .list_tools(Default::default())
            .await
            .map_err(|e| LlmError::InternalError(format!("Failed to list MCP tools: {}", e)))?;

        let tools = mcp_tools
            .tools
            .into_iter()
            .map(|t| {
                Tool::function(
                    t.name.to_string(),
                    t.description.map(|d| d.to_string()).unwrap_or_default(),
                    serde_json::to_value(&*t.input_schema).unwrap_or(serde_json::json!({})),
                )
            })
            .collect();

        Ok(tools)
    }
}

#[async_trait]
impl ToolResolver for McpToolResolver {
    async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value, LlmError> {
        let result = self
            .service
            .call_tool(match arguments.as_object().cloned() {
                Some(arguments) => {
                    CallToolRequestParams::new(name.to_string()).with_arguments(arguments)
                }
                None => CallToolRequestParams::new(name.to_string()),
            })
            .await
            .map_err(|e| LlmError::InternalError(format!("MCP tool execution failed: {}", e)))?;

        // Convert MCP result to JSON value
        // MCP returns a CallToolResult with content array
        Ok(serde_json::to_value(result).map_err(|e| {
            LlmError::InternalError(format!("Failed to serialize MCP result: {}", e))
        })?)
    }
}

/// Convenience function to create tools and resolver from a stdio command.
///
/// This is the easiest way to get started with MCP integration.
///
/// # Arguments
///
/// * `command` - The command to execute (e.g., "node server.js")
///
/// # Returns
///
/// A tuple of (tools, resolver) ready to use with Siumai's orchestrator.
///
/// # Example
///
/// ```rust,ignore
/// let (tools, resolver) = mcp_tools_from_stdio("node server.js").await?;
/// ```
pub async fn mcp_tools_from_stdio(command: &str) -> Result<(Vec<Tool>, McpToolResolver), LlmError> {
    let resolver = McpToolResolver::from_stdio(command).await?;
    let tools = resolver.list_tools().await?;
    Ok((tools, resolver))
}

/// Convenience function to create tools and resolver from a legacy SSE endpoint.
///
/// The upstream `rmcp` SDK removed the standalone SSE client transport in favor of streamable HTTP.
/// Prefer [`mcp_tools_from_http`] for new code.
///
/// # Arguments
///
/// * `url` - The streamable HTTP endpoint URL (e.g., "http://localhost:3000/mcp")
///
/// # Returns
///
/// A tuple of (tools, resolver) ready to use with Siumai's orchestrator.
///
/// # Example
///
/// ```rust,ignore
/// let (tools, resolver) = mcp_tools_from_http("http://localhost:3000/mcp").await?;
/// ```
#[deprecated(
    since = "0.11.0",
    note = "rmcp removed standalone SSE transport; use mcp_tools_from_http with a streamable HTTP MCP endpoint"
)]
pub async fn mcp_tools_from_sse(url: &str) -> Result<(Vec<Tool>, McpToolResolver), LlmError> {
    let resolver = McpToolResolver::from_http(url).await?;
    let tools = resolver.list_tools().await?;
    Ok((tools, resolver))
}

/// Convenience function to create tools and resolver from an HTTP endpoint.
///
/// # Arguments
///
/// * `url` - The HTTP endpoint URL (e.g., "http://localhost:3000/mcp")
///
/// # Returns
///
/// A tuple of (tools, resolver) ready to use with Siumai's orchestrator.
///
/// # Example
///
/// ```rust,ignore
/// let (tools, resolver) = mcp_tools_from_http("http://localhost:3000/mcp").await?;
/// ```
pub async fn mcp_tools_from_http(url: &str) -> Result<(Vec<Tool>, McpToolResolver), LlmError> {
    let resolver = McpToolResolver::from_http(url).await?;
    let tools = resolver.list_tools().await?;
    Ok((tools, resolver))
}
