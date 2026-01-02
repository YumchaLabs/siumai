//! OpenAI(-compatible) wire format types.
//!
//! These types represent a subset of the OpenAI Chat Completions wire schema that is
//! required by the OpenAI-compatible protocol layer (e.g. message conversion).
#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};

/// OpenAI message format (chat completions).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiMessage {
    pub role: String,
    pub content: Option<serde_json::Value>,
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
    pub tool_call_id: Option<String>,
}

/// OpenAI tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiToolCall {
    pub id: String,
    pub r#type: String,
    pub function: Option<OpenAiFunction>,
}

/// OpenAI function call payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiFunction {
    pub name: String,
    pub arguments: String,
}
