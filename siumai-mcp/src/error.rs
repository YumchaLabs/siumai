use thiserror::Error;

/// MCP connection, discovery, lifecycle, or projection failure.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum McpError {
    #[error("MCP limit `{name}` must be greater than zero")]
    InvalidLimit { name: &'static str },
    #[error("MCP namespace must be 1..=48 portable tool-name characters")]
    InvalidNamespace,
    #[error("MCP remote tool name must not be empty")]
    InvalidRemoteToolName,
    #[error("MCP HTTP endpoint is invalid")]
    InvalidEndpoint,
    #[error("MCP HTTP endpoint is not allowed by the configured policy")]
    EndpointNotAllowed,
    #[error("failed to establish the MCP service: {0}")]
    Connect(String),
    #[error("the MCP service is closed")]
    Closed,
    #[error("the MCP notification limit was exceeded")]
    NotificationLimitExceeded,
    #[error("the MCP tool catalog changed; discover a fresh catalog before execution")]
    CatalogStale,
    #[error("the MCP tool catalog fingerprint no longer matches this binding")]
    CatalogFingerprintMismatch,
    #[error("failed to list MCP tools: {0}")]
    ListTools(String),
    #[error("MCP tool discovery exceeded {maximum} pages")]
    PageLimitExceeded { maximum: usize },
    #[error("MCP tool discovery repeated cursor `{cursor}`")]
    RepeatedCursor { cursor: String },
    #[error("MCP tool discovery exceeded {maximum} tools")]
    ToolLimitExceeded { maximum: usize },
    #[error("MCP tool `{tool}` definition exceeded {maximum} bytes")]
    SchemaLimitExceeded { tool: String, maximum: usize },
    #[error("MCP tool `{tool}` cannot be projected: {message}")]
    InvalidToolDefinition { tool: String, message: String },
    #[error("MCP tool catalog contains conflicting model-visible name `{name}`")]
    ToolNameConflict { name: String },
    #[error("MCP tool call failed: {0}")]
    CallTool(String),
    #[error("MCP tool result exceeded {maximum} bytes")]
    ResultLimitExceeded { maximum: usize },
    #[error("failed to encode an MCP protocol value: {0}")]
    Encoding(String),
    #[error("failed to close the MCP service: {0}")]
    Close(String),
    #[error("the MCP service did not close before the configured deadline")]
    CloseTimeout,
}
