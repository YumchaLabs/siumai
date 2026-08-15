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
#[derive(Error)]
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
    #[error("MCP tool discovery repeated a cursor")]
    RepeatedCursor { cursor: String },
    #[error("MCP tool discovery exceeded {maximum} tools")]
    ToolLimitExceeded { maximum: usize },
    #[error("MCP tool definition exceeded the configured byte limit")]
    SchemaLimitExceeded { tool: String, maximum: usize },
    #[error("MCP tool definition cannot be projected")]
    InvalidToolDefinition { tool: String, message: String },
    #[error("MCP tool catalog contains a conflicting model-visible name")]
    ToolNameConflict { name: String },
    #[error("MCP tool result exceeded {maximum} bytes")]
    ResultLimitExceeded { maximum: usize },
    #[error("failed to encode an MCP protocol value")]
    Encoding(String),
    #[error("failed to close the MCP service")]
    Close(#[source] siumai_core::Error),
    #[error("the MCP service did not close before the configured deadline")]
    CloseTimeout,
}

impl fmt::Debug for McpError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = formatter.debug_struct("McpError");
        match self {
            Self::InvalidLimit { name } => debug.field("kind", &"InvalidLimit").field("name", name),
            Self::InvalidNamespace => debug.field("kind", &"InvalidNamespace"),
            Self::InvalidRemoteToolName => debug.field("kind", &"InvalidRemoteToolName"),
            Self::InvalidEndpoint => debug.field("kind", &"InvalidEndpoint"),
            Self::EndpointNotAllowed => debug.field("kind", &"EndpointNotAllowed"),
            Self::Connect(_) => debug
                .field("kind", &"Connect")
                .field("source", &"[REDACTED]"),
            Self::Closed => debug.field("kind", &"Closed"),
            Self::CatalogStale => debug.field("kind", &"CatalogStale"),
            Self::CatalogFingerprintMismatch => debug.field("kind", &"CatalogFingerprintMismatch"),
            Self::ListTools(_) => debug
                .field("kind", &"ListTools")
                .field("source", &"[REDACTED]"),
            Self::PageLimitExceeded { maximum } => debug
                .field("kind", &"PageLimitExceeded")
                .field("maximum", maximum),
            Self::RepeatedCursor { cursor } => debug
                .field("kind", &"RepeatedCursor")
                .field("cursor_bytes", &cursor.len()),
            Self::ToolLimitExceeded { maximum } => debug
                .field("kind", &"ToolLimitExceeded")
                .field("maximum", maximum),
            Self::SchemaLimitExceeded { tool, maximum } => debug
                .field("kind", &"SchemaLimitExceeded")
                .field("tool_bytes", &tool.len())
                .field("maximum", maximum),
            Self::InvalidToolDefinition { tool, message } => debug
                .field("kind", &"InvalidToolDefinition")
                .field("tool_bytes", &tool.len())
                .field("message_bytes", &message.len()),
            Self::ToolNameConflict { name } => debug
                .field("kind", &"ToolNameConflict")
                .field("name_bytes", &name.len()),
            Self::ResultLimitExceeded { maximum } => debug
                .field("kind", &"ResultLimitExceeded")
                .field("maximum", maximum),
            Self::Encoding(message) => debug
                .field("kind", &"Encoding")
                .field("message_bytes", &message.len()),
            Self::Close(_) => debug.field("kind", &"Close").field("source", &"[REDACTED]"),
            Self::CloseTimeout => debug.field("kind", &"CloseTimeout"),
        }
        .finish()
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_error_values_are_redacted_from_default_diagnostics() {
        let errors = [
            McpError::RepeatedCursor {
                cursor: "cursor-canary".to_string(),
            },
            McpError::SchemaLimitExceeded {
                tool: "remote-tool-canary".to_string(),
                maximum: 64,
            },
            McpError::InvalidToolDefinition {
                tool: "invalid-tool-canary".to_string(),
                message: "schema-message-canary".to_string(),
            },
            McpError::Encoding("encoding-message-canary".to_string()),
        ];

        for error in errors {
            let diagnostics = format!("{error:?} | {error}");
            for canary in [
                "cursor-canary",
                "remote-tool-canary",
                "invalid-tool-canary",
                "schema-message-canary",
                "encoding-message-canary",
            ] {
                assert!(!diagnostics.contains(canary));
            }
        }
    }
}
