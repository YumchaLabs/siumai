//! Anthropic Messages non-streaming JSON response encoder (gateway/proxy).
//!
//! English-only comments in code as requested.

use crate::encoding::{JsonEncodeOptions, JsonResponseConverter};
use crate::error::LlmError;
use crate::types::{ChatResponse, ContentPart, FinishReason, Usage};
use serde::Serialize;

fn anthropic_stop_reason(reason: Option<&FinishReason>) -> Option<&'static str> {
    match reason? {
        FinishReason::Stop | FinishReason::StopSequence => Some("end_turn"),
        FinishReason::Length => Some("max_tokens"),
        FinishReason::ToolCalls => Some("tool_use"),
        // Best-effort fallbacks.
        FinishReason::ContentFilter => Some("stop_sequence"),
        FinishReason::Error => Some("end_turn"),
        FinishReason::Unknown => None,
        FinishReason::Other(_) => None,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

fn usage_json(u: Option<&Usage>) -> AnthropicUsage {
    match u {
        Some(u) => AnthropicUsage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
        },
        None => AnthropicUsage {
            input_tokens: 0,
            output_tokens: 0,
        },
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AnthropicMessageResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub role: &'static str,
    pub model: String,
    pub content: Vec<AnthropicContentBlock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<&'static str>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Clone)]
pub struct AnthropicMessagesJsonResponseConverter;

impl AnthropicMessagesJsonResponseConverter {
    pub fn new() -> Self {
        Self
    }
}

impl Default for AnthropicMessagesJsonResponseConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonResponseConverter for AnthropicMessagesJsonResponseConverter {
    fn serialize_response(
        &self,
        response: &ChatResponse,
        out: &mut Vec<u8>,
        opts: JsonEncodeOptions,
    ) -> Result<(), LlmError> {
        let id = response.id.clone().unwrap_or_else(|| "siumai".to_string());
        let model = response.model.clone().unwrap_or_default();

        let mut content = Vec::new();

        let text = response.content_text().unwrap_or_default().to_string();
        if !text.trim().is_empty() {
            content.push(AnthropicContentBlock::Text { text });
        }

        for p in response.tool_calls() {
            if let ContentPart::ToolCall {
                tool_call_id,
                tool_name,
                arguments,
                ..
            } = p
            {
                content.push(AnthropicContentBlock::ToolUse {
                    id: tool_call_id.clone(),
                    name: tool_name.clone(),
                    input: arguments.clone(),
                });
            }
        }

        let body = AnthropicMessageResponse {
            id,
            kind: "message",
            role: "assistant",
            model,
            content,
            stop_reason: anthropic_stop_reason(response.finish_reason.as_ref()),
            stop_sequence: None,
            usage: usage_json(response.usage.as_ref()),
        };

        if opts.pretty {
            serde_json::to_writer_pretty(out, &body).map_err(|e| {
                LlmError::JsonError(format!(
                    "Failed to serialize Anthropic Messages JSON response: {e}"
                ))
            })?;
        } else {
            serde_json::to_writer(out, &body).map_err(|e| {
                LlmError::JsonError(format!(
                    "Failed to serialize Anthropic Messages JSON response: {e}"
                ))
            })?;
        }
        Ok(())
    }
}
