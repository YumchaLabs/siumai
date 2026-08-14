use std::error::Error as StdError;
use std::fmt;
use std::sync::Arc;

use siumai_core::SensitiveErrorSource;
use thiserror::Error;

#[derive(Default)]
pub(crate) struct McpSensitiveDetails {
    pub(crate) endpoint: Option<Arc<str>>,
    pub(crate) response_body: Option<Arc<[u8]>>,
    pub(crate) auth_challenge: Option<Arc<str>>,
}

pub(crate) struct McpBackendSource {
    source: Box<dyn StdError + Send + Sync + 'static>,
    details: McpSensitiveDetails,
}

impl McpBackendSource {
    pub(crate) fn new(
        source: impl StdError + Send + Sync + 'static,
        details: McpSensitiveDetails,
    ) -> Self {
        Self {
            source: Box::new(source),
            details,
        }
    }
}

impl fmt::Debug for McpBackendSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("McpBackendSource")
            .field("source", &"[REDACTED]")
            .field("has_endpoint", &self.details.endpoint.is_some())
            .field(
                "response_body_bytes",
                &self.details.response_body.as_ref().map(|body| body.len()),
            )
            .field("has_auth_challenge", &self.details.auth_challenge.is_some())
            .finish()
    }
}

impl fmt::Display for McpBackendSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.source.fmt(formatter)
    }
}

impl StdError for McpBackendSource {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(self.source.as_ref())
    }
}

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
    #[error("failed to establish the MCP service")]
    Connect(#[source] siumai_core::Error),
    #[error("the MCP service is closed")]
    Closed,
    #[error("the MCP tool catalog changed; discover a fresh catalog before execution")]
    CatalogStale,
    #[error("the MCP tool catalog fingerprint no longer matches this binding")]
    CatalogFingerprintMismatch,
    #[error("failed to list MCP tools")]
    ListTools(#[source] siumai_core::Error),
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
    #[error("MCP tool result exceeded {maximum} bytes")]
    ResultLimitExceeded { maximum: usize },
    #[error("failed to encode an MCP protocol value: {0}")]
    Encoding(String),
    #[error("failed to close the MCP service")]
    Close(#[source] siumai_core::Error),
    #[error("the MCP service did not close before the configured deadline")]
    CloseTimeout,
}

impl McpError {
    /// Explicitly access a backend source that is redacted from default diagnostics.
    pub fn sensitive_source(&self) -> Option<&SensitiveErrorSource> {
        match self {
            Self::Connect(error) | Self::ListTools(error) | Self::Close(error) => {
                error.sensitive_source()
            }
            _ => None,
        }
    }

    /// Explicitly access a bounded HTTP error body retained by the MCP transport.
    pub fn sensitive_response_body(&self) -> Option<&[u8]> {
        self.backend_source()?.details.response_body.as_deref()
    }

    /// Explicitly access the endpoint retained for an HTTP status failure.
    pub fn sensitive_endpoint(&self) -> Option<&str> {
        self.backend_source()?.details.endpoint.as_deref()
    }

    /// Explicitly access an HTTP authentication challenge retained by the transport.
    pub fn sensitive_auth_challenge(&self) -> Option<&str> {
        self.backend_source()?.details.auth_challenge.as_deref()
    }

    fn backend_source(&self) -> Option<&McpBackendSource> {
        self.sensitive_source()?.expose().downcast_ref()
    }
}
