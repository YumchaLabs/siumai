//! Metadata types for chat messages

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::content::ToolResultOutput;

/// Tool call information (borrowed view)
///
/// Provides convenient access to tool call fields without pattern matching.
#[derive(Debug, Clone, Copy)]
pub struct ToolCallInfo<'a> {
    /// The tool call ID
    pub tool_call_id: &'a str,
    /// The tool name
    pub tool_name: &'a str,
    /// The tool input (AI SDK-aligned alias).
    pub input: &'a serde_json::Value,
    /// The tool arguments (JSON)
    pub arguments: &'a serde_json::Value,
    /// Whether the tool was executed by the provider
    pub provider_executed: Option<&'a bool>,
    /// Whether the tool is dynamic/runtime-defined.
    pub dynamic: Option<&'a bool>,
    /// Whether the tool call is invalid.
    pub invalid: Option<&'a bool>,
    /// Optional invalidity/error payload.
    pub error: Option<&'a serde_json::Value>,
    /// Optional human-readable tool title.
    pub title: Option<&'a str>,
}

/// Tool result information (borrowed view)
///
/// Provides convenient access to tool result fields without pattern matching.
#[derive(Debug, Clone, Copy)]
pub struct ToolResultInfo<'a> {
    /// The tool call ID
    pub tool_call_id: &'a str,
    /// The tool name
    pub tool_name: &'a str,
    /// Optional resolved tool input.
    pub input: Option<&'a serde_json::Value>,
    /// The tool output
    pub output: &'a ToolResultOutput,
    /// Whether the tool was executed by the provider
    pub provider_executed: Option<&'a bool>,
    /// Whether the tool is dynamic/runtime-defined.
    pub dynamic: Option<&'a bool>,
    /// Whether the result is preliminary.
    pub preliminary: Option<&'a bool>,
    /// Optional human-readable tool title.
    pub title: Option<&'a str>,
}

/// Cache control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheControl {
    /// Ephemeral cache
    Ephemeral,
    /// Persistent cache
    Persistent { ttl: Option<std::time::Duration> },
}

/// Message metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MessageMetadata {
    /// Message ID
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Timestamp
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
    /// Cache control (Anthropic-specific)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
    /// Custom metadata
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub custom: HashMap<String, serde_json::Value>,
}
