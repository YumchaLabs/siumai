//! `Groq` Type Definitions
//!
//! Type definitions specific to the Groq API.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Groq Chat Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<GroqChoice>,
    pub usage: Option<GroqUsage>,
    pub system_fingerprint: Option<String>,
    pub x_groq: Option<GroqMetadata>,
}

/// Groq Choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqChoice {
    pub index: u32,
    pub message: GroqMessage,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: Option<String>,
}

/// Groq Message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqMessage {
    pub role: String,
    pub content: Option<serde_json::Value>,
    pub tool_calls: Option<Vec<GroqToolCall>>,
}

/// Groq Tool Call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqToolCall {
    pub id: String,
    pub r#type: String,
    pub function: Option<GroqFunction>,
}

/// Groq Function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqFunction {
    pub name: String,
    pub arguments: String,
}

/// Groq Usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqUsage {
    pub queue_time: Option<f64>,
    pub prompt_tokens: Option<u32>,
    pub prompt_time: Option<f64>,
    pub completion_tokens: Option<u32>,
    pub completion_time: Option<f64>,
    pub total_tokens: Option<u32>,
    pub total_time: Option<f64>,
}

/// Groq Metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqMetadata {
    pub id: String,
}

/// Groq Chat Stream Chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqChatStreamChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<GroqStreamChoice>,
    pub usage: Option<GroqUsage>,
    pub system_fingerprint: Option<String>,
    pub x_groq: Option<GroqMetadata>,
}

/// Groq Stream Choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqStreamChoice {
    pub index: u32,
    pub delta: GroqDelta,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: Option<String>,
}

/// Groq Delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqDelta {
    pub role: Option<String>,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<GroqToolCallDelta>>,
}

/// Groq Tool Call Delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqToolCallDelta {
    pub index: Option<u32>,
    pub id: Option<String>,
    pub r#type: Option<String>,
    pub function: Option<GroqFunctionDelta>,
}

/// Groq Function Delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqFunctionDelta {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

/// Groq Model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqModel {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    pub active: bool,
    pub context_window: u32,
    pub public_apps: Option<serde_json::Value>,
    pub max_completion_tokens: Option<u32>,
}

/// Groq Models Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqModelsResponse {
    pub object: String,
    pub data: Vec<GroqModel>,
}

/// Groq Audio Transcription Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqTranscriptionResponse {
    pub text: String,
    pub x_groq: Option<GroqMetadata>,
}

/// Groq Audio Translation Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqTranslationResponse {
    pub text: String,
    pub x_groq: Option<GroqMetadata>,
}

/// Groq File
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqFile {
    pub id: String,
    pub object: String,
    pub bytes: u64,
    pub created_at: u64,
    pub filename: String,
    pub purpose: String,
}

/// Groq Files Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqFilesResponse {
    pub object: String,
    pub data: Vec<GroqFile>,
}

/// Groq Delete File Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqDeleteFileResponse {
    pub id: String,
    pub object: String,
    pub deleted: bool,
}

/// Groq Batch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqBatch {
    pub id: String,
    pub object: String,
    pub endpoint: String,
    pub errors: Option<serde_json::Value>,
    pub input_file_id: String,
    pub completion_window: String,
    pub status: String,
    pub output_file_id: Option<String>,
    pub error_file_id: Option<String>,
    pub finalizing_at: Option<u64>,
    pub failed_at: Option<u64>,
    pub expired_at: Option<u64>,
    pub cancelled_at: Option<u64>,
    pub request_counts: GroqRequestCounts,
    pub metadata: Option<serde_json::Value>,
    pub created_at: u64,
    pub expires_at: u64,
    pub cancelling_at: Option<u64>,
    pub completed_at: Option<u64>,
    pub in_progress_at: Option<u64>,
}

/// Groq Request Counts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqRequestCounts {
    pub total: u32,
    pub completed: u32,
    pub failed: u32,
}

/// Groq Batches Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqBatchesResponse {
    pub object: String,
    pub data: Vec<GroqBatch>,
}

/// Groq Error Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqErrorResponse {
    pub error: GroqError,
}

/// Groq Error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqError {
    pub message: String,
    pub r#type: Option<String>,
    pub param: Option<String>,
    pub code: Option<String>,
}

/// Groq-specific parameters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GroqParams {
    /// Frequency penalty
    pub frequency_penalty: Option<f32>,
    /// Presence penalty
    pub presence_penalty: Option<f32>,
    /// Logit bias
    pub logit_bias: Option<HashMap<String, f32>>,
    /// Whether to enable parallel function calling
    pub parallel_tool_calls: Option<bool>,
    /// Service tier
    pub service_tier: Option<String>,
    /// Reasoning effort (for qwen3 models)
    pub reasoning_effort: Option<String>,
    /// Reasoning format
    pub reasoning_format: Option<String>,
}

impl GroqParams {
    /// Create new Groq parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Set frequency penalty
    pub fn with_frequency_penalty(mut self, frequency_penalty: f32) -> Self {
        self.frequency_penalty = Some(frequency_penalty);
        self
    }

    /// Set presence penalty
    pub fn with_presence_penalty(mut self, presence_penalty: f32) -> Self {
        self.presence_penalty = Some(presence_penalty);
        self
    }

    /// Set parallel tool calls
    pub fn with_parallel_tool_calls(mut self, parallel_tool_calls: bool) -> Self {
        self.parallel_tool_calls = Some(parallel_tool_calls);
        self
    }

    /// Set service tier
    pub fn with_service_tier<S: Into<String>>(mut self, service_tier: S) -> Self {
        self.service_tier = Some(service_tier.into());
        self
    }

    /// Set reasoning effort (for qwen3 models)
    pub fn with_reasoning_effort<S: Into<String>>(mut self, reasoning_effort: S) -> Self {
        self.reasoning_effort = Some(reasoning_effort.into());
        self
    }

    /// Set reasoning format
    pub fn with_reasoning_format<S: Into<String>>(mut self, reasoning_format: S) -> Self {
        self.reasoning_format = Some(reasoning_format.into());
        self
    }
}
