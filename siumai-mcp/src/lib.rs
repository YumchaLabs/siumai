//! Lifecycle-safe Model Context Protocol tool integration.
//!
//! MCP servers are untrusted tool-definition and execution peers. This crate
//! keeps their service alive, bounds discovery and results, and projects tools
//! into Siumai's explicit [`siumai_runtime::tool::ToolBinding`] contract.
//! Remote annotations never grant execution permissions.

#![doc = include_str!("../README.md")]
#![deny(unsafe_code)]

mod catalog;
mod client;
mod config;
mod error;
mod transport;

pub use catalog::{McpCatalogFingerprint, McpToolCatalog, McpToolDefinition};
pub use client::{McpClient, McpProgress};
pub use config::{McpClientConfig, McpHttpEndpointPolicy, McpLimits, McpToolPolicy};
pub use error::McpError;
pub use siumai_transport::{HttpTransportRoute, ProxyBasicCredential, ProxyEndpoint};
