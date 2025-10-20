//! MCP (Model Context Protocol) integration for Siumai.
//!
//! This module provides convenient integration between Siumai and MCP servers
//! using the `rmcp` library. It allows you to:
//!
//! - Connect to MCP servers via stdio, SSE, or HTTP transports
//! - Automatically discover tools from MCP servers
//! - Execute MCP tools through Siumai's orchestrator
//! - Use MCP tools with any Siumai-supported LLM provider
//!
//! # Features
//!
//! - **Multiple Transports**: stdio, SSE, HTTP
//! - **Automatic Tool Discovery**: Fetch tools from MCP servers
//! - **Seamless Integration**: Works with Siumai's orchestrator and agents
//! - **Type-Safe**: Leverages Rust's type system for safety
//!
//! # Example
//!
//! ```rust,ignore
//! use siumai::prelude::*;
//! use siumai_extras::mcp::{McpToolResolver, mcp_tools_from_stdio};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Connect to MCP server and get tools
//!     let (tools, resolver) = mcp_tools_from_stdio("node mcp-server.js").await?;
//!     
//!     // Create Siumai model
//!     let model = Siumai::builder()
//!         .openai()
//!         .api_key(std::env::var("OPENAI_API_KEY")?)
//!         .model("gpt-4o-mini")
//!         .build()
//!         .await?;
//!     
//!     // Use with orchestrator
//!     let messages = vec![user!("Use the available tools to help me")];
//!     let (response, _) = siumai::orchestrator::generate(
//!         &model,
//!         messages,
//!         Some(tools),
//!         Some(&resolver),
//!         vec![siumai::orchestrator::step_count_is(10)],
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
//! ## SSE Transport
//!
//! Connect to an MCP server via Server-Sent Events:
//!
//! ```rust,ignore
//! let (tools, resolver) = mcp_tools_from_sse("http://localhost:8080/sse").await?;
//! ```
//!
//! ## HTTP Transport
//!
//! Connect to an MCP server via HTTP:
//!
//! ```rust,ignore
//! let (tools, resolver) = mcp_tools_from_http("http://localhost:3000/mcp").await?;
//! ```

use async_trait::async_trait;
use rmcp::client::{Client, ClientBuilder};
use rmcp::transport::{ChildProcessTransport, SseClientTransport, StreamableHttpClientTransport};
use serde_json::Value;
use siumai::error::LlmError;
use siumai::orchestrator::ToolResolver;
use siumai::types::Tool;

/// MCP tool resolver that executes tools via an MCP client.
///
/// This struct implements Siumai's `ToolResolver` trait, allowing MCP tools
/// to be used seamlessly with Siumai's orchestrator and agents.
///
/// # Example
///
/// ```rust,ignore
/// use siumai_extras::mcp::McpToolResolver;
/// use rmcp::client::ClientBuilder;
/// use rmcp::transport::ChildProcessTransport;
///
/// let transport = ChildProcessTransport::new("node server.js")?;
/// let client = ClientBuilder::new(transport).build().await?;
/// let resolver = McpToolResolver::new(client);
/// ```
pub struct McpToolResolver {
    client: Client,
}

impl McpToolResolver {
    /// Create a new MCP tool resolver from an existing MCP client.
    ///
    /// # Arguments
    ///
    /// * `client` - An initialized MCP client
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let client = ClientBuilder::new(transport).build().await?;
    /// let resolver = McpToolResolver::new(client);
    /// ```
    pub fn new(client: Client) -> Self {
        Self { client }
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
    pub async fn from_stdio(command: &str) -> Result<Self, LlmError> {
        let transport = ChildProcessTransport::new(command).map_err(|e| {
            LlmError::InternalError(format!("Failed to create stdio transport: {}", e))
        })?;

        let client = ClientBuilder::new(transport)
            .build()
            .await
            .map_err(|e| LlmError::InternalError(format!("Failed to create MCP client: {}", e)))?;

        Ok(Self::new(client))
    }

    /// Create a new MCP tool resolver from an SSE endpoint.
    ///
    /// # Arguments
    ///
    /// * `url` - The SSE endpoint URL (e.g., "http://localhost:8080/sse")
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let resolver = McpToolResolver::from_sse("http://localhost:8080/sse").await?;
    /// ```
    pub async fn from_sse(url: &str) -> Result<Self, LlmError> {
        let transport = SseClientTransport::new(url).map_err(|e| {
            LlmError::InternalError(format!("Failed to create SSE transport: {}", e))
        })?;

        let client = ClientBuilder::new(transport)
            .build()
            .await
            .map_err(|e| LlmError::InternalError(format!("Failed to create MCP client: {}", e)))?;

        Ok(Self::new(client))
    }

    /// Create a new MCP tool resolver from an HTTP endpoint.
    ///
    /// # Arguments
    ///
    /// * `url` - The HTTP endpoint URL (e.g., "http://localhost:3000/mcp")
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let resolver = McpToolResolver::from_http("http://localhost:3000/mcp").await?;
    /// ```
    pub async fn from_http(url: &str) -> Result<Self, LlmError> {
        let transport = StreamableHttpClientTransport::new(url).map_err(|e| {
            LlmError::InternalError(format!("Failed to create HTTP transport: {}", e))
        })?;

        let client = ClientBuilder::new(transport)
            .build()
            .await
            .map_err(|e| LlmError::InternalError(format!("Failed to create MCP client: {}", e)))?;

        Ok(Self::new(client))
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
            .client
            .list_tools()
            .await
            .map_err(|e| LlmError::InternalError(format!("Failed to list MCP tools: {}", e)))?;

        let tools = mcp_tools
            .into_iter()
            .map(|t| Tool {
                name: t.name,
                description: Some(t.description.unwrap_or_default()),
                parameters: t.input_schema,
            })
            .collect();

        Ok(tools)
    }
}

#[async_trait]
impl ToolResolver for McpToolResolver {
    async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value, LlmError> {
        let result = self.client.call_tool(name, arguments).await.map_err(|e| {
            LlmError::ToolExecutionError(format!("MCP tool execution failed: {}", e))
        })?;

        // Convert MCP result to JSON value
        // MCP returns a ToolResult with content array
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

/// Convenience function to create tools and resolver from an SSE endpoint.
///
/// # Arguments
///
/// * `url` - The SSE endpoint URL (e.g., "http://localhost:8080/sse")
///
/// # Returns
///
/// A tuple of (tools, resolver) ready to use with Siumai's orchestrator.
///
/// # Example
///
/// ```rust,ignore
/// let (tools, resolver) = mcp_tools_from_sse("http://localhost:8080/sse").await?;
/// ```
pub async fn mcp_tools_from_sse(url: &str) -> Result<(Vec<Tool>, McpToolResolver), LlmError> {
    let resolver = McpToolResolver::from_sse(url).await?;
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
