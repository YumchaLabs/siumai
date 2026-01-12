//! OpenAI-family non-streaming JSON response encoders (gateway/proxy).
//!
//! English-only comments in code as requested.

use crate::encoding::{JsonEncodeOptions, JsonResponseConverter};
use crate::error::LlmError;
use crate::types::{ChatResponse, ContentPart, FinishReason, Usage};
use serde::Serialize;

fn openai_finish_reason(reason: Option<&FinishReason>) -> Option<&'static str> {
    match reason? {
        FinishReason::Stop | FinishReason::StopSequence => Some("stop"),
        FinishReason::Length => Some("length"),
        FinishReason::ContentFilter => Some("content_filter"),
        FinishReason::ToolCalls => Some("tool_calls"),
        FinishReason::Error => Some("error"),
        FinishReason::Unknown => None,
        FinishReason::Other(_) => None,
    }
}

fn usage_json(u: &Usage) -> OpenAiUsage {
    OpenAiUsage {
        prompt_tokens: u.prompt_tokens,
        completion_tokens: u.completion_tokens,
        total_tokens: u.total_tokens,
        prompt_tokens_details: u.prompt_tokens_details.clone(),
        completion_tokens_details: u.completion_tokens_details.clone(),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_tokens_details: Option<crate::types::PromptTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    completion_tokens_details: Option<crate::types::CompletionTokensDetails>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiChatCompletionsJsonResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAiChatChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenAiUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiChatChoice {
    pub index: u32,
    pub message: OpenAiChatMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiChatMessage {
    pub role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: OpenAiToolCallFunction,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone)]
pub struct OpenAiChatCompletionsJsonResponseConverter;

impl OpenAiChatCompletionsJsonResponseConverter {
    pub fn new() -> Self {
        Self
    }
}

impl Default for OpenAiChatCompletionsJsonResponseConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonResponseConverter for OpenAiChatCompletionsJsonResponseConverter {
    fn serialize_response(
        &self,
        response: &ChatResponse,
        out: &mut Vec<u8>,
        opts: JsonEncodeOptions,
    ) -> Result<(), LlmError> {
        let id = response.id.clone().unwrap_or_else(|| "siumai".to_string());
        let model = response.model.clone().unwrap_or_default();

        let text = response.content_text().unwrap_or_default().to_string();

        let tool_calls: Vec<OpenAiToolCall> = response
            .tool_calls()
            .into_iter()
            .filter_map(|p| match p {
                ContentPart::ToolCall {
                    tool_call_id,
                    tool_name,
                    arguments,
                    ..
                } => Some(OpenAiToolCall {
                    id: tool_call_id.clone(),
                    kind: "function",
                    function: OpenAiToolCallFunction {
                        name: tool_name.clone(),
                        arguments: serde_json::to_string(arguments)
                            .unwrap_or_else(|_| arguments.to_string()),
                    },
                }),
                _ => None,
            })
            .collect();

        let content = if text.trim().is_empty() && !tool_calls.is_empty() {
            None
        } else {
            Some(text)
        };

        let message = OpenAiChatMessage {
            role: "assistant",
            content,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
        };

        let body = OpenAiChatCompletionsJsonResponse {
            id,
            object: "chat.completion",
            created: 0,
            model,
            choices: vec![OpenAiChatChoice {
                index: 0,
                message,
                finish_reason: openai_finish_reason(response.finish_reason.as_ref()),
            }],
            usage: response.usage.as_ref().map(usage_json),
            system_fingerprint: response.system_fingerprint.clone(),
            service_tier: response.service_tier.clone(),
        };

        if opts.pretty {
            serde_json::to_writer_pretty(out, &body).map_err(|e| {
                LlmError::JsonError(format!(
                    "Failed to serialize OpenAI chat response JSON: {e}"
                ))
            })?;
        } else {
            serde_json::to_writer(out, &body).map_err(|e| {
                LlmError::JsonError(format!(
                    "Failed to serialize OpenAI chat response JSON: {e}"
                ))
            })?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiResponsesJsonResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub status: &'static str,
    pub output: Vec<OpenAiResponseOutputItem>,
    pub output_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenAiUsage>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum OpenAiResponseOutputItem {
    #[serde(rename = "message")]
    Message {
        id: String,
        role: &'static str,
        content: Vec<OpenAiResponseMessageContent>,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum OpenAiResponseMessageContent {
    #[serde(rename = "output_text")]
    OutputText { text: String },
}

#[derive(Debug, Clone)]
pub struct OpenAiResponsesJsonResponseConverter;

impl OpenAiResponsesJsonResponseConverter {
    pub fn new() -> Self {
        Self
    }
}

impl Default for OpenAiResponsesJsonResponseConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonResponseConverter for OpenAiResponsesJsonResponseConverter {
    fn serialize_response(
        &self,
        response: &ChatResponse,
        out: &mut Vec<u8>,
        opts: JsonEncodeOptions,
    ) -> Result<(), LlmError> {
        let id = response.id.clone().unwrap_or_else(|| "siumai".to_string());
        let model = response.model.clone().unwrap_or_default();

        let text = response.content_text().unwrap_or_default().to_string();
        let mut output = Vec::new();

        if !text.trim().is_empty() {
            output.push(OpenAiResponseOutputItem::Message {
                id: format!("msg_{id}"),
                role: "assistant",
                content: vec![OpenAiResponseMessageContent::OutputText { text }],
            });
        }

        for p in response.tool_calls() {
            if let ContentPart::ToolCall {
                tool_call_id,
                tool_name,
                arguments,
                ..
            } = p
            {
                output.push(OpenAiResponseOutputItem::FunctionCall {
                    id: format!("fc_{tool_call_id}"),
                    call_id: tool_call_id.clone(),
                    name: tool_name.clone(),
                    arguments: serde_json::to_string(arguments)
                        .unwrap_or_else(|_| arguments.to_string()),
                });
            }
        }

        let body = OpenAiResponsesJsonResponse {
            id,
            object: "response",
            created: 0,
            model,
            status: "completed",
            output,
            output_text: response.content.all_text(),
            usage: response.usage.as_ref().map(usage_json),
        };

        if opts.pretty {
            serde_json::to_writer_pretty(out, &body).map_err(|e| {
                LlmError::JsonError(format!(
                    "Failed to serialize OpenAI Responses JSON response: {e}"
                ))
            })?;
        } else {
            serde_json::to_writer(out, &body).map_err(|e| {
                LlmError::JsonError(format!(
                    "Failed to serialize OpenAI Responses JSON response: {e}"
                ))
            })?;
        }
        Ok(())
    }
}
