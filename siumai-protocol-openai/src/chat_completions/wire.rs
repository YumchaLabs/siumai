use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub(crate) struct ChatResponseWire {
    pub id: Option<String>,
    pub model: Option<String>,
    #[serde(default)]
    pub choices: Vec<ChoiceWire>,
    pub usage: Option<UsageWire>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChoiceWire {
    pub index: u32,
    pub message: AssistantMessageWire,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct AssistantMessageWire {
    pub content: Option<Value>,
    pub refusal: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCallWire>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ToolCallWire {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionWire,
}

#[derive(Debug, Deserialize)]
pub(crate) struct FunctionWire {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct UsageWire {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub prompt_tokens_details: Option<PromptTokenDetailsWire>,
    pub completion_tokens_details: Option<CompletionTokenDetailsWire>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct PromptTokenDetailsWire {
    pub cached_tokens: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct CompletionTokenDetailsWire {
    pub reasoning_tokens: Option<u64>,
    pub audio_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatStreamChunkWire {
    pub id: Option<String>,
    pub model: Option<String>,
    #[serde(default)]
    pub choices: Vec<StreamChoiceWire>,
    pub usage: Option<UsageWire>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct StreamChoiceWire {
    pub index: u32,
    #[serde(default)]
    pub delta: DeltaWire,
    pub finish_reason: Option<String>,
    pub usage: Option<UsageWire>,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct DeltaWire {
    pub content: Option<String>,
    pub refusal: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCallDeltaWire>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ToolCallDeltaWire {
    pub index: u32,
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub kind: Option<String>,
    pub function: Option<FunctionDeltaWire>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct FunctionDeltaWire {
    pub name: Option<String>,
    pub arguments: Option<String>,
}
