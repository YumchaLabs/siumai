use serde::{Deserialize, Serialize};

/// OpenAI Compatible Chat Response with provider-specific fields
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiCompatibleChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAiCompatibleChoice>,
    pub usage: Option<OpenAiCompatibleUsage>,
}

/// OpenAI Compatible Choice with provider-specific fields
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiCompatibleChoice {
    pub index: u32,
    pub message: OpenAiCompatibleMessage,
    pub finish_reason: Option<String>,
}

/// OpenAI Compatible Message with provider-specific fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiCompatibleMessage {
    pub role: String,
    pub content: Option<serde_json::Value>,
    pub tool_calls: Option<Vec<OpenAiCompatibleToolCall>>,
    pub tool_call_id: Option<String>,

    // Provider-specific thinking/reasoning fields
    pub thinking: Option<String>,          // Standard thinking field
    pub reasoning_content: Option<String>, // DeepSeek reasoning field
    pub reasoning: Option<String>,         // Alternative reasoning field
}

/// OpenAI Compatible Tool Call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiCompatibleToolCall {
    pub id: String,
    pub r#type: String,
    pub function: Option<OpenAiCompatibleFunction>,
}

/// OpenAI Compatible Function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiCompatibleFunction {
    pub name: String,
    pub arguments: String,
}

/// OpenAI Compatible Usage
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiCompatibleUsage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}
