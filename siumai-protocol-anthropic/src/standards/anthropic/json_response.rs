//! Anthropic Messages non-streaming JSON response encoder (gateway/proxy).
//!
//! English-only comments in code as requested.

use crate::encoding::{JsonEncodeOptions, JsonResponseConverter};
use crate::error::LlmError;
use crate::provider_metadata::anthropic::AnthropicContentPartExt;
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
}

fn usage_json(u: Option<&Usage>, service_tier: Option<&str>) -> AnthropicUsage {
    match u {
        Some(u) => AnthropicUsage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            service_tier: service_tier.map(ToOwned::to_owned),
        },
        None => AnthropicUsage {
            input_tokens: 0,
            output_tokens: 0,
            service_tier: service_tier.map(ToOwned::to_owned),
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
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        caller: Option<serde_json::Value>,
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
        let raw_anthropic_meta = response
            .provider_metadata
            .as_ref()
            .and_then(|metadata| metadata.get("anthropic"));
        let thinking_signature = raw_anthropic_meta
            .and_then(|meta| meta.get("thinking_signature"))
            .and_then(|value| value.as_str())
            .map(ToOwned::to_owned);
        let redacted_thinking_data = raw_anthropic_meta
            .and_then(|meta| meta.get("redacted_thinking_data"))
            .and_then(|value| value.as_str())
            .map(ToOwned::to_owned);
        let stop_sequence = raw_anthropic_meta
            .and_then(|meta| meta.get("stopSequence"))
            .and_then(|value| value.as_str())
            .map(ToOwned::to_owned);

        match &response.content {
            crate::types::MessageContent::Text(text) => {
                if !text.trim().is_empty() {
                    content.push(AnthropicContentBlock::Text { text: text.clone() });
                }
            }
            crate::types::MessageContent::MultiModal(parts) => {
                let mut signature = thinking_signature.clone();

                for part in parts {
                    match part {
                        ContentPart::Text { text, .. } => {
                            content.push(AnthropicContentBlock::Text { text: text.clone() });
                        }
                        ContentPart::Reasoning { text, .. } => {
                            content.push(AnthropicContentBlock::Thinking {
                                thinking: text.clone(),
                                signature: signature.take(),
                            });
                        }
                        ContentPart::ToolCall {
                            tool_call_id,
                            tool_name,
                            arguments,
                            ..
                        } => {
                            let caller = part
                                .anthropic_tool_call_metadata()
                                .and_then(|meta| meta.caller)
                                .and_then(|caller| serde_json::to_value(caller).ok());

                            content.push(AnthropicContentBlock::ToolUse {
                                id: tool_call_id.clone(),
                                name: tool_name.clone(),
                                input: arguments.clone(),
                                caller,
                            });
                        }
                        _ => {}
                    }
                }
            }
            #[cfg(feature = "structured-messages")]
            crate::types::MessageContent::Json(value) => {
                content.push(AnthropicContentBlock::Text {
                    text: serde_json::to_string(value).unwrap_or_default(),
                });
            }
        }

        if let Some(data) = redacted_thinking_data {
            content.push(AnthropicContentBlock::RedactedThinking { data });
        }

        let body = AnthropicMessageResponse {
            id,
            kind: "message",
            role: "assistant",
            model,
            content,
            stop_reason: anthropic_stop_reason(response.finish_reason.as_ref()),
            stop_sequence,
            usage: usage_json(response.usage.as_ref(), response.service_tier.as_deref()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn anthropic_encoder_serializes_thinking_redacted_stop_sequence_and_caller() {
        let tool_call = ContentPart::ToolCall {
            tool_call_id: "toolu_1".to_string(),
            tool_name: "weather".to_string(),
            arguments: serde_json::json!({ "city": "Tokyo" }),
            provider_executed: None,
            provider_metadata: Some(HashMap::from([(
                "anthropic".to_string(),
                serde_json::json!({
                    "caller": {
                        "type": "code_execution_20250825",
                        "tool_id": "srvtoolu_1"
                    }
                }),
            )])),
        };

        let mut response = ChatResponse::new(crate::types::MessageContent::MultiModal(vec![
            ContentPart::text("Visible answer"),
            ContentPart::reasoning("Need to verify assumptions."),
            tool_call,
        ]));
        response.id = Some("msg_1".to_string());
        response.model = Some("claude-sonnet-4-5".to_string());
        response.finish_reason = Some(FinishReason::StopSequence);
        response.service_tier = Some("priority".to_string());
        response.provider_metadata = Some(HashMap::from([(
            "anthropic".to_string(),
            HashMap::from([
                ("thinking_signature".to_string(), serde_json::json!("sig_1")),
                (
                    "redacted_thinking_data".to_string(),
                    serde_json::json!("redacted_123"),
                ),
                ("stopSequence".to_string(), serde_json::json!("</tool>")),
            ]),
        )]));

        let mut out = Vec::new();
        AnthropicMessagesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(value["stop_reason"], serde_json::json!("end_turn"));
        assert_eq!(value["stop_sequence"], serde_json::json!("</tool>"));
        assert_eq!(
            value["usage"]["service_tier"],
            serde_json::json!("priority")
        );
        assert_eq!(value["content"][1]["type"], serde_json::json!("thinking"));
        assert_eq!(value["content"][1]["signature"], serde_json::json!("sig_1"));
        assert_eq!(
            value["content"][2]["caller"]["tool_id"],
            serde_json::json!("srvtoolu_1")
        );
        assert_eq!(
            value["content"][3]["type"],
            serde_json::json!("redacted_thinking")
        );
        assert_eq!(
            value["content"][3]["data"],
            serde_json::json!("redacted_123")
        );
    }
}
