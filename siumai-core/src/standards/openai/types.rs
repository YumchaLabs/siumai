//! OpenAI(-compatible) wire format types.
//!
//! These types represent a subset of the OpenAI Chat Completions wire schema that is
//! required by the OpenAI-compatible protocol layer (e.g. message conversion).
#![deny(unsafe_code)]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// OpenAI message format (chat completions).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    /// Provider-specific extra fields (e.g. OpenAI-compatible vendor extensions).
    #[serde(default, flatten, skip_serializing_if = "HashMap::is_empty")]
    pub extra: HashMap<String, serde_json::Value>,
}

/// OpenAI tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiToolCall {
    pub id: String,
    pub r#type: String,
    pub function: Option<OpenAiFunction>,

    /// Provider-specific extra fields for tool calls (Vercel-aligned).
    #[serde(default, flatten, skip_serializing_if = "HashMap::is_empty")]
    pub extra: HashMap<String, serde_json::Value>,
}

/// OpenAI function call payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiFunction {
    pub name: String,
    pub arguments: String,
}
