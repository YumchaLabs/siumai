//! OpenAI-compatible protocol types
//!
//! These types represent the OpenAI(-compatible) wire format and are intended
//! for reuse across multiple providers that implement OpenAI-style APIs.

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

/// OpenAI Chat Completions response
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAiChoice>,
    pub usage: Option<OpenAiUsage>,
}

/// OpenAI Chat Completions choice
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiChoice {
    pub index: u32,
    pub message: OpenAiMessage,
    pub finish_reason: Option<String>,
}

/// OpenAI usage summary
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiUsage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

/// OpenAI model metadata
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiModel {
    pub id: String,
    pub object: String,
    pub created: Option<u64>,
    pub owned_by: String,
    pub permission: Option<Vec<serde_json::Value>>,
    pub root: Option<String>,
    pub parent: Option<String>,
}

/// OpenAI models endpoint response
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiModelsResponse {
    pub object: String,
    pub data: Vec<OpenAiModel>,
}
