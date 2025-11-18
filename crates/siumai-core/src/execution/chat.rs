//! Minimal chat transformers for standards

use crate::error::LlmError;
use crate::types::FinishReasonCore;
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

/// Core representation of a single parsed tool call in chat content.
///
/// This is a provider-agnostic structure that standards can use to
/// surface tool calls alongside aggregated text and thinking content.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatParsedToolCallCore {
    /// Optional tool call id (if the provider exposes one).
    pub id: Option<String>,
    /// Tool/function name.
    pub name: String,
    /// JSON arguments for the tool call.
    pub arguments: serde_json::Value,
}

/// Core representation of parsed chat content (text + tool calls + thinking).
///
/// Standards like Anthropic and Gemini can populate this structure so
/// that higher layers can reconstruct richer `MessageContent` models
/// without re-parsing provider JSON.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatParsedContentCore {
    /// Aggregated assistant text (excluding thinking content).
    pub text: String,
    /// Parsed tool calls.
    pub tool_calls: Vec<ChatParsedToolCallCore>,
    /// Optional thinking / reasoning content.
    pub thinking: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResult {
    pub content: String,
    pub finish_reason: Option<FinishReasonCore>,
    pub usage: Option<ChatUsage>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    /// Optional parsed content model (text + tool calls + thinking).
    ///
    /// This field is filled by standards that expose a richer content
    /// view (e.g. Anthropic / Gemini). Providers that don't support it
    /// can leave this as `None`.
    #[serde(default)]
    pub parsed_content: Option<ChatParsedContentCore>,
}

pub trait ChatRequestTransformer: Send + Sync {
    fn provider_id(&self) -> &str;
    fn transform_chat(&self, req: &ChatInput) -> Result<serde_json::Value, LlmError>;
}

pub trait ChatResponseTransformer: Send + Sync {
    fn provider_id(&self) -> &str;
    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResult, LlmError>;
}
