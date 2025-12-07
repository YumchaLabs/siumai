//! Text completion types (non-chat)

use crate::types::Usage;
use std::collections::HashMap;

/// Text completion request
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    /// Input prompt
    pub prompt: String,
    /// Model to use
    pub model: Option<String>,
    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,
    /// Temperature for randomness
    pub temperature: Option<f32>,
    /// Top-p sampling
    pub top_p: Option<f32>,
    /// Top-k sampling
    pub top_k: Option<u32>,
    /// Stop sequences
    pub stop: Option<Vec<String>>,
    /// Number of completions to generate
    pub n: Option<u32>,
    /// Whether to stream the response
    pub stream: Option<bool>,
    /// Additional parameters
    pub extra_params: HashMap<String, serde_json::Value>,
}

/// Text completion response
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    /// Generated text
    pub text: String,
    /// Finish reason
    pub finish_reason: Option<String>,
    /// Usage information
    pub usage: Option<Usage>,
    /// Model used
    pub model: Option<String>,
}

// Placeholder types for future implementation
pub type JsonSchema = ();
pub type StructuredResponse = ();
pub type BatchRequest = ();
pub type BatchResponse = ();
pub type CacheConfig = ();
pub type ThinkingResponse = ();
pub type SearchConfig = ();
pub type ExecutionResponse = ();
