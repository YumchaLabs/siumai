//! OpenAI-family non-streaming JSON response encoders (gateway/proxy).
//!
//! English-only comments in code as requested.

use crate::encoding::{JsonEncodeOptions, JsonResponseConverter};
use crate::error::LlmError;
use crate::provider_metadata::openai::{
    OpenAiChatResponseExt, OpenAiContentPartExt, OpenAiSourceExt,
};
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

fn openai_response_status(reason: Option<&FinishReason>) -> &'static str {
    match reason {
        Some(FinishReason::Error) => "failed",
        _ => "completed",
    }
}

fn openai_arguments_string(arguments: &serde_json::Value) -> String {
    if let Some(raw) = arguments.as_str() {
        raw.to_string()
    } else {
        serde_json::to_string(arguments).unwrap_or_else(|_| arguments.to_string())
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
    pub finish_reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenAiUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
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
    #[serde(rename = "reasoning")]
    Reasoning {
        id: String,
        summary: Vec<OpenAiResponseReasoningSummary>,
        #[serde(skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
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
    OutputText {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        annotations: Option<Vec<OpenAiResponseAnnotation>>,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum OpenAiResponseReasoningSummary {
    #[serde(rename = "summary_text")]
    SummaryText { text: String },
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum OpenAiResponseAnnotation {
    #[serde(rename = "url_citation")]
    UrlCitation {
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },
    #[serde(rename = "file_citation")]
    FileCitation {
        file_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        quote: Option<String>,
    },
    #[serde(rename = "container_file_citation")]
    ContainerFileCitation {
        file_id: String,
        container_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        index: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        quote: Option<String>,
    },
    #[serde(rename = "file_path")]
    FilePath {
        file_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        index: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
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
        let mut output = Vec::new();
        let raw_openai_meta = response
            .provider_metadata
            .as_ref()
            .and_then(|meta| meta.get("openai"));
        let message_item_id = raw_openai_meta
            .and_then(|meta| meta.get("itemId"))
            .and_then(|value| value.as_str())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| format!("msg_{id}"));

        let mut message_content: Vec<OpenAiResponseMessageContent> = match &response.content {
            crate::types::MessageContent::Text(text) => {
                vec![OpenAiResponseMessageContent::OutputText {
                    text: text.clone(),
                    annotations: None,
                }]
            }
            crate::types::MessageContent::MultiModal(parts) => parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text { text, .. } => {
                        Some(OpenAiResponseMessageContent::OutputText {
                            text: text.clone(),
                            annotations: None,
                        })
                    }
                    _ => None,
                })
                .collect(),
            #[cfg(feature = "structured-messages")]
            crate::types::MessageContent::Json(value) => {
                vec![OpenAiResponseMessageContent::OutputText {
                    text: serde_json::to_string(value).unwrap_or_default(),
                    annotations: None,
                }]
            }
        };

        let annotations = response
            .openai_metadata()
            .and_then(|meta| meta.sources)
            .map(|sources| {
                sources
                    .into_iter()
                    .filter_map(|source| {
                        let source_meta = source.openai_metadata();
                        match source.source_type.as_str() {
                            "url" => Some(OpenAiResponseAnnotation::UrlCitation {
                                url: source.url,
                                title: source.title,
                            }),
                            "document" => {
                                let file_id = source_meta
                                    .as_ref()
                                    .and_then(|meta| meta.file_id.clone())
                                    .unwrap_or_else(|| source.url.clone());
                                let quote = source.title.or(source.snippet);

                                if let Some(container_id) = source_meta
                                    .as_ref()
                                    .and_then(|meta| meta.container_id.clone())
                                {
                                    return Some(OpenAiResponseAnnotation::ContainerFileCitation {
                                        file_id,
                                        container_id,
                                        index: source_meta.as_ref().and_then(|meta| meta.index),
                                        filename: source.filename,
                                        quote,
                                    });
                                }

                                if source.media_type.as_deref() == Some("application/octet-stream")
                                {
                                    return Some(OpenAiResponseAnnotation::FilePath {
                                        file_id,
                                        index: source_meta.as_ref().and_then(|meta| meta.index),
                                        filename: source.filename,
                                    });
                                }

                                Some(OpenAiResponseAnnotation::FileCitation {
                                    file_id,
                                    filename: source.filename,
                                    quote,
                                })
                            }
                            _ => None,
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .filter(|annotations| !annotations.is_empty());

        if let Some(annotations) = annotations {
            if message_content.is_empty() {
                message_content.push(OpenAiResponseMessageContent::OutputText {
                    text: String::new(),
                    annotations: Some(annotations),
                });
            } else if let Some(OpenAiResponseMessageContent::OutputText {
                annotations: slot, ..
            }) = message_content.first_mut()
            {
                *slot = Some(annotations);
            }
        }

        if !message_content.is_empty()
            || raw_openai_meta
                .and_then(|meta| meta.get("itemId"))
                .and_then(|value| value.as_str())
                .is_some()
        {
            output.push(OpenAiResponseOutputItem::Message {
                id: message_item_id,
                role: "assistant",
                content: message_content,
            });
        }

        if let Some(parts) = response.content.as_multimodal() {
            for (index, part) in parts.iter().enumerate() {
                match part {
                    ContentPart::Reasoning { text, .. } => {
                        let meta = part.openai_metadata();
                        let item_id = meta
                            .as_ref()
                            .and_then(|meta| meta.item_id.clone())
                            .unwrap_or_else(|| format!("rs_{id}_{index}"));
                        let summary = vec![OpenAiResponseReasoningSummary::SummaryText {
                            text: text.clone(),
                        }];

                        output.push(OpenAiResponseOutputItem::Reasoning {
                            id: item_id,
                            summary,
                            encrypted_content: meta
                                .and_then(|meta| meta.reasoning_encrypted_content.clone()),
                        });
                    }
                    ContentPart::ToolCall {
                        tool_call_id,
                        tool_name,
                        arguments,
                        ..
                    } => {
                        let meta = part.openai_metadata();
                        let item_id = meta
                            .as_ref()
                            .and_then(|meta| meta.item_id.clone())
                            .unwrap_or_else(|| format!("fc_{tool_call_id}"));

                        output.push(OpenAiResponseOutputItem::FunctionCall {
                            id: item_id,
                            call_id: tool_call_id.clone(),
                            name: tool_name.clone(),
                            arguments: openai_arguments_string(arguments),
                        });
                    }
                    _ => {}
                }
            }
        }

        let body = OpenAiResponsesJsonResponse {
            id,
            object: "response",
            created: 0,
            model,
            status: openai_response_status(response.finish_reason.as_ref()),
            output,
            output_text: response.content.all_text(),
            finish_reason: openai_finish_reason(response.finish_reason.as_ref()),
            usage: response.usage.as_ref().map(usage_json),
            system_fingerprint: response.system_fingerprint.clone(),
            service_tier: response.service_tier.clone(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn responses_encoder_serializes_reasoning_ids_annotations_and_native_fields() {
        let reasoning = ContentPart::Reasoning {
            text: "Need to compare both options.".to_string(),
            provider_metadata: Some(HashMap::from([(
                "openai".to_string(),
                serde_json::json!({
                    "itemId": "rs_1",
                    "reasoningEncryptedContent": "enc_123"
                }),
            )])),
        };

        let tool_call = ContentPart::ToolCall {
            tool_call_id: "call_1".to_string(),
            tool_name: "get_weather".to_string(),
            arguments: serde_json::Value::String(r#"{"city":"Tokyo"}"#.to_string()),
            provider_executed: None,
            provider_metadata: Some(HashMap::from([(
                "openai".to_string(),
                serde_json::json!({
                    "itemId": "fc_1"
                }),
            )])),
        };

        let mut response = ChatResponse::new(crate::types::MessageContent::MultiModal(vec![
            reasoning,
            ContentPart::text("It is sunny."),
            tool_call,
        ]));
        response.id = Some("resp_1".to_string());
        response.model = Some("gpt-5-mini".to_string());
        response.finish_reason = Some(FinishReason::ToolCalls);
        response.system_fingerprint = Some("fp_123".to_string());
        response.service_tier = Some("priority".to_string());
        response.provider_metadata = Some(HashMap::from([(
            "openai".to_string(),
            HashMap::from([
                ("itemId".to_string(), serde_json::json!("msg_1")),
                (
                    "sources".to_string(),
                    serde_json::json!([
                        {
                            "id": "src_1",
                            "source_type": "url",
                            "url": "https://example.com",
                            "title": "Example"
                        },
                        {
                            "id": "src_2",
                            "source_type": "document",
                            "url": "file_123",
                            "title": "Design Doc",
                            "filename": "design.md",
                            "provider_metadata": {
                                "fileId": "file_123"
                            }
                        }
                    ]),
                ),
            ]),
        )]));

        let mut out = Vec::new();
        OpenAiResponsesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(value["finish_reason"], serde_json::json!("tool_calls"));
        assert_eq!(value["system_fingerprint"], serde_json::json!("fp_123"));
        assert_eq!(value["service_tier"], serde_json::json!("priority"));
        assert_eq!(value["output"][0]["id"], serde_json::json!("msg_1"));
        assert_eq!(
            value["output"][0]["content"][0]["annotations"][0]["type"],
            serde_json::json!("url_citation")
        );
        assert_eq!(
            value["output"][0]["content"][0]["annotations"][1]["type"],
            serde_json::json!("file_citation")
        );
        assert_eq!(value["output"][1]["type"], serde_json::json!("reasoning"));
        assert_eq!(value["output"][1]["id"], serde_json::json!("rs_1"));
        assert_eq!(
            value["output"][1]["encrypted_content"],
            serde_json::json!("enc_123")
        );
        assert_eq!(value["output"][2]["id"], serde_json::json!("fc_1"));
        assert_eq!(
            value["output"][2]["arguments"],
            serde_json::json!(r#"{"city":"Tokyo"}"#)
        );
    }
}
