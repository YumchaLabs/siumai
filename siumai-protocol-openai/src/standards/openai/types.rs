//! OpenAI-compatible protocol types
//!
//! These types represent the OpenAI(-compatible) wire format and are intended
//! for reuse across multiple providers that implement OpenAI-style APIs.

use serde::Deserialize;

// These core wire types are shared across OpenAI-like providers and live in `siumai-core`.
// Keep re-exports here to preserve the historical module path.
pub use siumai_core::standards::openai::types::{OpenAiFunction, OpenAiMessage, OpenAiToolCall};

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
