//! Minimal chat transformers for standards

use crate::error::LlmError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessageInput {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatInput {
    pub messages: Vec<ChatMessageInput>,
    pub model: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub stop: Option<Vec<String>>,
    /// Provider-specific passthrough
    #[serde(default)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResult {
    pub content: String,
    pub finish_reason: Option<String>,
    pub usage: Option<ChatUsage>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

pub trait ChatRequestTransformer: Send + Sync {
    fn provider_id(&self) -> &str;
    fn transform_chat(&self, req: &ChatInput) -> Result<serde_json::Value, LlmError>;
}

pub trait ChatResponseTransformer: Send + Sync {
    fn provider_id(&self) -> &str;
    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResult, LlmError>;
}
