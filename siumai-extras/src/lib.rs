//! # Siumai Extras
//!
//! Optional utilities for the `siumai` LLM library, including:
//!
//! - **Schema Validation** (`schema` feature): JSON Schema validation for structured outputs
//! - **Telemetry** (`telemetry` feature): Advanced tracing and logging with `tracing-subscriber`
//! - **OpenTelemetry** (`opentelemetry` feature): Full observability with distributed tracing and metrics
//! - **Server Adapters** (`server` feature): Axum integration for streaming responses
//! - **MCP Integration** (`mcp` feature): Model Context Protocol integration for dynamic tool discovery
//!
//! ## Features
//!
//! - `schema` - Enable JSON Schema validation utilities
//! - `telemetry` - Enable tracing subscriber and logging utilities
//! - `opentelemetry` - Enable OpenTelemetry distributed tracing and metrics
//! - `server` - Enable server adapter utilities (Axum)
//! - `mcp` - Enable MCP (Model Context Protocol) integration
//! - `all` - Enable all features
//!
//! ## Example
//!
//! ```toml
//! [dependencies]
//! siumai = "0.11"
//! siumai-extras = { version = "0.11", features = ["schema", "telemetry", "mcp"] }
//! ```
//!
//! ## MCP Integration
//!
//! The `mcp` feature provides seamless integration with MCP servers:
//!
//! ```rust,ignore
//! use siumai::prelude::*;
//! use siumai_extras::mcp::mcp_tools_from_stdio;
//!
//! // Connect to MCP server and get tools
//! let (tools, resolver) = mcp_tools_from_stdio("node mcp-server.js").await?;
//!
//! // Use with any Siumai model
//! let model = Siumai::builder().openai().build().await?;
//! let (response, _) = siumai_extras::orchestrator::generate(
//!     &model,
//!     messages,
//!     Some(tools),
//!     Some(&resolver),
//!     vec![siumai_extras::orchestrator::step_count_is(10)],
//!     Default::default(),
//! ).await?;
//! ```

#![warn(missing_docs)]
#![deny(unsafe_code)]

// Re-export core siumai types that are commonly used with extras
pub use siumai;

/// Schema validation utilities
#[cfg(feature = "schema")]
pub mod schema;

/// Telemetry and tracing utilities
#[cfg(feature = "telemetry")]
pub mod telemetry;

/// OpenTelemetry integration
#[cfg(feature = "opentelemetry")]
pub mod otel;

/// Metrics collection
#[cfg(feature = "opentelemetry")]
pub mod metrics;

/// OpenTelemetry middleware
#[cfg(feature = "opentelemetry")]
pub mod otel_middleware;

/// Server adapter utilities
#[cfg(feature = "server")]
pub mod server;

/// MCP (Model Context Protocol) integration
#[cfg(feature = "mcp")]
pub mod mcp;

/// Error types for siumai-extras
pub mod error;

// Internal helpers for structured output (shared by highlevel + orchestrator).
mod structured_output;

/// High-level structured object helpers (provider-agnostic).
pub mod highlevel;

/// Orchestrator and agent utilities for multi-step tool calling.
pub mod orchestrator;

/// Thinking/analysis utilities for model reasoning content.
pub mod analysis;

/// Performance metrics helpers and in-process monitoring.
pub mod performance;

/// Provider-hosted tools (web search, file search, code execution, etc.).
pub mod hosted_tools;

/// Client utilities such as client pools and managers.
pub mod client;
