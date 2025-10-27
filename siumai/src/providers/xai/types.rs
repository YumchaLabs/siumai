//! `xAI` Type Definitions
//!
//! This module contains type definitions specific to the `xAI` API.

use serde::{Deserialize, Serialize};

/// `xAI` Chat Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XaiChatResponse {
    /// Response ID
    pub id: String,
    /// Object type
    pub object: String,
    /// Creation timestamp
    pub created: u64,
    /// Model used
    pub model: String,
    /// Response choices
    pub choices: Vec<XaiChoice>,
    /// Token usage information
    pub usage: Option<XaiUsage>,
    /// System fingerprint
    pub system_fingerprint: Option<String>,
}

/// `xAI` Choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XaiChoice {
    /// Choice index
    pub index: u32,
    /// Message content
    pub message: XaiMessage,
    /// Finish reason
    pub finish_reason: Option<String>,
}

/// `xAI` Message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XaiMessage {
    /// Message role
    pub role: String,
    /// Message content
    pub content: Option<serde_json::Value>,
    /// Tool calls
    pub tool_calls: Option<Vec<XaiToolCall>>,
    /// Reasoning content (for thinking models)
    pub reasoning_content: Option<String>,
}

/// `xAI` Tool Call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XaiToolCall {
    /// Tool call ID
    pub id: String,
    /// Tool type
    pub r#type: String,
    /// Function call details
    pub function: Option<XaiFunctionCall>,
}

/// `xAI` Function Call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XaiFunctionCall {
    /// Function name
    pub name: String,
    /// Function arguments
    pub arguments: String,
}

/// `xAI` Usage Information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XaiUsage {
    /// Prompt tokens
    pub prompt_tokens: Option<u32>,
    /// Completion tokens
    pub completion_tokens: Option<u32>,
    /// Total tokens
    pub total_tokens: Option<u32>,
    /// Reasoning tokens (for thinking models)
    pub reasoning_tokens: Option<u32>,
    /// Prompt tokens details
    pub prompt_tokens_details: Option<XaiPromptTokensDetails>,
    /// Completion tokens details
    pub completion_tokens_details: Option<XaiCompletionTokensDetails>,
}

/// `xAI` Prompt Tokens Details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XaiPromptTokensDetails {
    /// Text tokens
    pub text_tokens: Option<u32>,
    /// Audio tokens
    pub audio_tokens: Option<u32>,
    /// Image tokens
    pub image_tokens: Option<u32>,
    /// Cached tokens
    pub cached_tokens: Option<u32>,
}

/// `xAI` Completion Tokens Details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XaiCompletionTokensDetails {
    /// Reasoning tokens
    pub reasoning_tokens: Option<u32>,
}

/// `xAI` Stream Chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XaiStreamChunk {
    /// Chunk ID
    pub id: String,
    /// Object type
    pub object: String,
    /// Creation timestamp
    pub created: u64,
    /// Model used
    pub model: String,
    /// Stream choices
    pub choices: Vec<XaiStreamChoice>,
    /// Usage information (only in final chunk)
    pub usage: Option<XaiUsage>,
    /// System fingerprint
    pub system_fingerprint: Option<String>,
}

/// `xAI` Stream Choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XaiStreamChoice {
    /// Choice index
    pub index: u32,
    /// Delta content
    pub delta: XaiDelta,
    /// Finish reason
    pub finish_reason: Option<String>,
}

/// `xAI` Delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XaiDelta {
    /// Role (only in first chunk)
    pub role: Option<String>,
    /// Content delta
    pub content: Option<String>,
    /// Tool calls delta
    pub tool_calls: Option<Vec<XaiToolCallDelta>>,
    /// Reasoning content delta (for thinking models)
    pub reasoning_content: Option<String>,
}

/// `xAI` Tool Call Delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XaiToolCallDelta {
    /// Tool call index
    pub index: u32,
    /// Tool call ID
    pub id: Option<String>,
    /// Tool type
    pub r#type: Option<String>,
    /// Function call delta
    pub function: Option<XaiFunctionCallDelta>,
}

/// `xAI` Function Call Delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XaiFunctionCallDelta {
    /// Function name
    pub name: Option<String>,
    /// Function arguments delta
    pub arguments: Option<String>,
}

/// `xAI` Error Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XaiErrorResponse {
    /// Error details
    pub error: XaiError,
}

/// `xAI` Error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XaiError {
    /// Error message
    pub message: String,
    /// Error type
    pub r#type: Option<String>,
    /// Error code
    pub code: Option<String>,
}

/// `xAI` specific parameters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct XaiParams {
    /// Reasoning effort level (for thinking models)
    pub reasoning_effort: Option<String>,
    /// Whether to enable deferred completion
    pub deferred: Option<bool>,
    /// Parallel function calling setting
    pub parallel_function_calling: Option<bool>,
}

impl XaiParams {
    /// Create new xAI parameters
    pub const fn new() -> Self {
        Self {
            reasoning_effort: None,
            deferred: None,
            parallel_function_calling: None,
        }
    }

    /// Set reasoning effort level
    pub fn with_reasoning_effort<S: Into<String>>(mut self, effort: S) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }

    /// Enable deferred completion
    pub const fn with_deferred(mut self, deferred: bool) -> Self {
        self.deferred = Some(deferred);
        self
    }

    /// Set parallel function calling
    pub const fn with_parallel_function_calling(mut self, parallel: bool) -> Self {
        self.parallel_function_calling = Some(parallel);
        self
    }
}

/// xAI Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XaiModel {
    /// Model ID
    pub id: String,
    /// Object type
    pub object: String,
    /// Creation timestamp
    pub created: u64,
    /// Model owner
    pub owned_by: String,
}

/// xAI Models API response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XaiModelsResponse {
    /// Object type
    pub object: String,
    /// List of models
    pub data: Vec<XaiModel>,
}
