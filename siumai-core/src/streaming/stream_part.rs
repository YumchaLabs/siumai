//! Vercel AI SDK aligned stream parts (typed).
//!
//! This module mirrors the `LanguageModelV3StreamPart` union from Vercel AI SDK
//! (packages/provider/src/language-model/v3/language-model-v3-stream-part.ts).
//!
//! It is intentionally kept as a lightweight, typed representation so users can:
//! - parse provider-specific `ChatStreamEvent::Custom` payloads into a stable schema
//! - implement custom bridging rules between providers without hand-writing JSON maps
//! - format stream parts into SSE `data: ...\n\n` frames when needed
//!
//! English-only comments in code as requested.

use crate::error::LlmError;
use crate::types::ChatStreamEvent;
use serde::{Deserialize, Serialize};

/// Provider metadata object keyed by provider name.
///
/// Vercel AI SDK uses `Record<string, JSONObject>`. We keep this type permissive
/// to preserve metadata for forward compatibility.
pub type SharedV3ProviderMetadata = serde_json::Map<String, serde_json::Value>;

/// Shared warning type (Vercel-aligned).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum SharedV3Warning {
    Unsupported {
        feature: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    Compatibility {
        feature: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    Other {
        message: String,
    },
}

/// Finish reason (Vercel-aligned).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LanguageModelV3FinishReason {
    pub unified: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
}

/// Response metadata (Vercel-aligned).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV3ResponseMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "modelId")]
    pub model_id: Option<String>,
}

/// Usage (Vercel-aligned).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV3Usage {
    #[serde(rename = "inputTokens")]
    pub input_tokens: LanguageModelV3InputTokens,
    #[serde(rename = "outputTokens")]
    pub output_tokens: LanguageModelV3OutputTokens,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV3InputTokens {
    pub total: Option<u64>,
    #[serde(rename = "noCache")]
    pub no_cache: Option<u64>,
    #[serde(rename = "cacheRead")]
    pub cache_read: Option<u64>,
    #[serde(rename = "cacheWrite")]
    pub cache_write: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV3OutputTokens {
    pub total: Option<u64>,
    pub text: Option<u64>,
    pub reasoning: Option<u64>,
}

/// Tool call (Vercel-aligned).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV3ToolCall {
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    #[serde(rename = "toolName")]
    pub tool_name: String,
    /// Stringified JSON tool arguments (Vercel expects a JSON string).
    pub input: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "providerExecuted"
    )]
    pub provider_executed: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "providerMetadata"
    )]
    pub provider_metadata: Option<SharedV3ProviderMetadata>,
}

/// Tool result (Vercel-aligned).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV3ToolResult {
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    #[serde(rename = "toolName")]
    pub tool_name: String,
    pub result: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "isError")]
    pub is_error: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preliminary: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "providerMetadata"
    )]
    pub provider_metadata: Option<SharedV3ProviderMetadata>,
}

/// Tool approval request (Vercel-aligned).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LanguageModelV3ToolApprovalRequest {
    #[serde(rename = "approvalId")]
    pub approval_id: String,
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "providerMetadata"
    )]
    pub provider_metadata: Option<SharedV3ProviderMetadata>,
}

/// File part (Vercel-aligned).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV3File {
    #[serde(rename = "mediaType")]
    pub media_type: String,
    pub data: LanguageModelV3FileData,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "providerMetadata"
    )]
    pub provider_metadata: Option<SharedV3ProviderMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum LanguageModelV3FileData {
    Base64(String),
    Bytes(Vec<u8>),
}

/// Source part (Vercel-aligned).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "sourceType", rename_all = "lowercase")]
pub enum LanguageModelV3Source {
    Url {
        id: String,
        url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<SharedV3ProviderMetadata>,
    },
    Document {
        id: String,
        #[serde(rename = "mediaType")]
        media_type: String,
        title: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<SharedV3ProviderMetadata>,
    },
}

/// Typed Vercel stream part union.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum LanguageModelV3StreamPart {
    TextStart {
        id: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<SharedV3ProviderMetadata>,
    },
    TextDelta {
        id: String,
        delta: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<SharedV3ProviderMetadata>,
    },
    TextEnd {
        id: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<SharedV3ProviderMetadata>,
    },

    ReasoningStart {
        id: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<SharedV3ProviderMetadata>,
    },
    ReasoningDelta {
        id: String,
        delta: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<SharedV3ProviderMetadata>,
    },
    ReasoningEnd {
        id: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<SharedV3ProviderMetadata>,
    },

    ToolInputStart {
        id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<SharedV3ProviderMetadata>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerExecuted"
        )]
        provider_executed: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        dynamic: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },
    ToolInputDelta {
        id: String,
        delta: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<SharedV3ProviderMetadata>,
    },
    ToolInputEnd {
        id: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<SharedV3ProviderMetadata>,
    },

    ToolApprovalRequest(LanguageModelV3ToolApprovalRequest),
    ToolCall(LanguageModelV3ToolCall),
    ToolResult(LanguageModelV3ToolResult),
    File(LanguageModelV3File),
    Source(LanguageModelV3Source),

    StreamStart {
        warnings: Vec<SharedV3Warning>,
    },

    ResponseMetadata(LanguageModelV3ResponseMetadata),

    Finish {
        usage: LanguageModelV3Usage,
        #[serde(rename = "finishReason")]
        finish_reason: LanguageModelV3FinishReason,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<SharedV3ProviderMetadata>,
    },

    Raw {
        #[serde(rename = "rawValue")]
        raw_value: serde_json::Value,
    },

    Error {
        error: serde_json::Value,
    },
}

/// Target namespace used when formatting a stream part into provider-specific custom events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamPartNamespace {
    OpenAi,
    Anthropic,
    Gemini,
}

/// Controls how v3 stream parts that cannot be represented in `ChatStreamEvent`
/// should be handled during protocol re-serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum V3UnsupportedPartBehavior {
    /// Drop the part (strictest behavior).
    #[default]
    Drop,
    /// Convert the part into a text delta (lossy, but preserves information).
    AsText,
}

impl LanguageModelV3StreamPart {
    /// Best-effort parse a v3 stream part from a JSON payload that is close to the
    /// Vercel schema but may contain minor shape differences.
    ///
    /// This is mainly used for cross-provider stream transcoding, where some
    /// providers emit `input` as a JSON object instead of a stringified JSON.
    pub fn parse_loose_json(value: &serde_json::Value) -> Option<Self> {
        let mut v = value.clone();

        fn normalize_in_place(v: &mut serde_json::Value) {
            let Some(obj) = v.as_object_mut() else {
                return;
            };
            let Some(tpe) = obj.get("type").and_then(|v| v.as_str()) else {
                return;
            };

            match tpe {
                "tool-call" => {
                    if let Some(input) = obj.get_mut("input")
                        && !matches!(input, serde_json::Value::String(_))
                        && let Ok(s) = serde_json::to_string(input)
                    {
                        *input = serde_json::Value::String(s);
                    }
                }
                "finish" => {
                    if let Some(fr) = obj.get_mut("finishReason")
                        && let serde_json::Value::String(s) = fr
                    {
                        *fr = serde_json::json!({ "unified": s, "raw": serde_json::Value::Null });
                    }
                }
                "source" => {
                    if obj.get("sourceType").is_none()
                        && let Some(st) = obj.remove("source_type")
                    {
                        obj.insert("sourceType".to_string(), st);
                    }
                }
                _ => {}
            }
        }

        normalize_in_place(&mut v);
        serde_json::from_value::<LanguageModelV3StreamPart>(v).ok()
    }

    /// Best-effort parse from a `ChatStreamEvent`.
    ///
    /// This is primarily intended for `ChatStreamEvent::Custom` where `data` follows
    /// the Vercel stream part JSON shape.
    pub fn try_from_chat_event(ev: &ChatStreamEvent) -> Option<Self> {
        match ev {
            ChatStreamEvent::Custom { data, .. } => Self::parse_loose_json(data),
            ChatStreamEvent::Error { error } => Some(LanguageModelV3StreamPart::Error {
                error: serde_json::json!(error),
            }),
            _ => None,
        }
    }

    /// Format as a provider-specific `ChatStreamEvent::Custom` (best-effort).
    ///
    /// This allows users to keep using the existing streaming pipeline while
    /// operating on typed stream parts in custom bridges.
    pub fn to_custom_event(&self, ns: StreamPartNamespace) -> Option<ChatStreamEvent> {
        let (event_type, data) = match ns {
            StreamPartNamespace::OpenAi => self.to_openai_custom_event_payload()?,
            StreamPartNamespace::Anthropic => self.to_anthropic_custom_event_payload()?,
            StreamPartNamespace::Gemini => self.to_gemini_custom_event_payload()?,
        };

        Some(ChatStreamEvent::Custom { event_type, data })
    }

    /// Encode this stream part as an SSE `data: ...\n\n` frame.
    pub fn to_data_sse_bytes(&self) -> Result<Vec<u8>, LlmError> {
        let json = serde_json::to_string(self).map_err(|e| {
            LlmError::ParseError(format!("Failed to serialize stream part JSON: {e}"))
        })?;
        Ok(format!("data: {json}\n\n").into_bytes())
    }

    /// Convert this typed stream part into best-effort `ChatStreamEvent`s.
    ///
    /// This is primarily intended for protocol re-serialization, where a target
    /// provider's encoder might not understand the source provider's custom events.
    ///
    /// Notes:
    /// - Not all Vercel stream parts have a lossless representation in `ChatStreamEvent`.
    /// - Unsupported parts are dropped (empty vec).
    pub fn to_best_effort_chat_events(&self) -> Vec<ChatStreamEvent> {
        match self {
            LanguageModelV3StreamPart::TextDelta { delta, .. } => {
                vec![ChatStreamEvent::ContentDelta {
                    delta: delta.clone(),
                    index: None,
                }]
            }
            LanguageModelV3StreamPart::ReasoningDelta { delta, .. } => {
                vec![ChatStreamEvent::ThinkingDelta {
                    delta: delta.clone(),
                }]
            }
            LanguageModelV3StreamPart::ToolInputStart { id, tool_name, .. } => {
                vec![ChatStreamEvent::ToolCallDelta {
                    id: id.clone(),
                    function_name: Some(tool_name.clone()),
                    arguments_delta: None,
                    index: None,
                }]
            }
            LanguageModelV3StreamPart::ToolInputDelta { id, delta, .. } => {
                vec![ChatStreamEvent::ToolCallDelta {
                    id: id.clone(),
                    function_name: None,
                    arguments_delta: Some(delta.clone()),
                    index: None,
                }]
            }
            LanguageModelV3StreamPart::ToolCall(call) => vec![ChatStreamEvent::ToolCallDelta {
                id: call.tool_call_id.clone(),
                function_name: Some(call.tool_name.clone()),
                arguments_delta: Some(call.input.clone()),
                index: None,
            }],
            _ => Vec::new(),
        }
    }

    /// Convert a v3 stream part into a lossy text representation.
    pub fn to_lossy_text(&self) -> Option<String> {
        match self {
            LanguageModelV3StreamPart::Source(src) => match src {
                LanguageModelV3Source::Url { url, title, .. } => Some(format!(
                    "[source] {} {}",
                    title.clone().unwrap_or_else(|| "url".to_string()),
                    url
                )),
                LanguageModelV3Source::Document {
                    title,
                    filename,
                    media_type,
                    ..
                } => Some(format!(
                    "[source] {} ({}){}",
                    title,
                    media_type,
                    filename
                        .as_ref()
                        .map(|f| format!(", {f}"))
                        .unwrap_or_default()
                )),
            },
            LanguageModelV3StreamPart::ToolResult(tr) => Some(format!(
                "[tool-result] {}: {}",
                tr.tool_name,
                serde_json::to_string(&tr.result).unwrap_or_else(|_| "{}".to_string())
            )),
            LanguageModelV3StreamPart::ToolApprovalRequest(req) => Some(format!(
                "[tool-approval-request] approvalId={} toolCallId={}",
                req.approval_id, req.tool_call_id
            )),
            LanguageModelV3StreamPart::Finish { finish_reason, .. } => {
                Some(format!("[finish] {}", finish_reason.unified))
            }
            LanguageModelV3StreamPart::Error { error } => Some(format!(
                "[error] {}",
                serde_json::to_string(error).unwrap_or_else(|_| "\"error\"".to_string())
            )),
            LanguageModelV3StreamPart::Raw { raw_value } => Some(format!(
                "[raw] {}",
                serde_json::to_string(raw_value).unwrap_or_else(|_| "\"raw\"".to_string())
            )),
            _ => None,
        }
    }

    fn to_openai_custom_event_payload(&self) -> Option<(String, serde_json::Value)> {
        let data = serde_json::to_value(self).ok()?;
        let event_type = match self {
            LanguageModelV3StreamPart::TextStart { .. } => "openai:text-start",
            LanguageModelV3StreamPart::TextDelta { .. } => "openai:text-delta",
            LanguageModelV3StreamPart::TextEnd { .. } => "openai:text-end",
            LanguageModelV3StreamPart::ReasoningStart { .. } => "openai:reasoning-start",
            LanguageModelV3StreamPart::ReasoningDelta { .. } => "openai:reasoning-delta",
            LanguageModelV3StreamPart::ReasoningEnd { .. } => "openai:reasoning-end",
            LanguageModelV3StreamPart::ToolInputStart { .. } => "openai:tool-input-start",
            LanguageModelV3StreamPart::ToolInputDelta { .. } => "openai:tool-input-delta",
            LanguageModelV3StreamPart::ToolInputEnd { .. } => "openai:tool-input-end",
            LanguageModelV3StreamPart::ToolApprovalRequest(_) => "openai:tool-approval-request",
            LanguageModelV3StreamPart::ToolCall(_) => "openai:tool-call",
            LanguageModelV3StreamPart::ToolResult(_) => "openai:tool-result",
            LanguageModelV3StreamPart::Source(_) => "openai:source",
            LanguageModelV3StreamPart::StreamStart { .. } => "openai:stream-start",
            LanguageModelV3StreamPart::ResponseMetadata(_) => "openai:response-metadata",
            LanguageModelV3StreamPart::Finish { .. } => "openai:finish",
            LanguageModelV3StreamPart::Error { .. } => "openai:error",
            LanguageModelV3StreamPart::Raw { .. } | LanguageModelV3StreamPart::File(_) => {
                return None;
            }
        };
        Some((event_type.to_string(), data))
    }

    fn to_anthropic_custom_event_payload(&self) -> Option<(String, serde_json::Value)> {
        let mut data = serde_json::to_value(self).ok()?;
        let event_type = match self {
            LanguageModelV3StreamPart::TextStart { .. } => "anthropic:text-start",
            LanguageModelV3StreamPart::TextDelta { .. } => "anthropic:text-delta",
            LanguageModelV3StreamPart::TextEnd { .. } => "anthropic:text-end",
            LanguageModelV3StreamPart::ToolCall(_) => "anthropic:tool-call",
            LanguageModelV3StreamPart::ToolResult(_) => "anthropic:tool-result",
            LanguageModelV3StreamPart::Source(_) => "anthropic:source",
            LanguageModelV3StreamPart::StreamStart { .. } => "anthropic:stream-start",
            LanguageModelV3StreamPart::ResponseMetadata(_) => "anthropic:response-metadata",
            LanguageModelV3StreamPart::Finish { .. } => "anthropic:finish",
            LanguageModelV3StreamPart::Error { .. } => "anthropic:error",

            // Anthropic does not currently emit Vercel-style tool-input-* parts in our pipeline.
            LanguageModelV3StreamPart::ToolInputStart { .. }
            | LanguageModelV3StreamPart::ToolInputDelta { .. }
            | LanguageModelV3StreamPart::ToolInputEnd { .. }
            | LanguageModelV3StreamPart::ToolApprovalRequest(_)
            | LanguageModelV3StreamPart::Raw { .. }
            | LanguageModelV3StreamPart::File(_) => return None,

            // Anthropic reasoning parts in our pipeline are keyed by contentBlockIndex.
            // Best-effort: map `id` into `contentBlockIndex` if it parses as u64.
            LanguageModelV3StreamPart::ReasoningStart { id, .. } => {
                let idx = id.parse::<u64>().ok()?;
                data = serde_json::json!({ "type": "reasoning-start", "contentBlockIndex": idx });
                "anthropic:reasoning-start"
            }
            LanguageModelV3StreamPart::ReasoningEnd { id, .. } => {
                let idx = id.parse::<u64>().ok()?;
                data = serde_json::json!({ "type": "reasoning-end", "contentBlockIndex": idx });
                "anthropic:reasoning-end"
            }
            LanguageModelV3StreamPart::ReasoningDelta { .. } => return None,
        };
        Some((event_type.to_string(), data))
    }

    fn to_gemini_custom_event_payload(&self) -> Option<(String, serde_json::Value)> {
        let data = serde_json::to_value(self).ok()?;
        let event_type = match self {
            LanguageModelV3StreamPart::ToolCall(_) | LanguageModelV3StreamPart::ToolResult(_) => {
                "gemini:tool"
            }
            LanguageModelV3StreamPart::Source(_) => "gemini:source",
            LanguageModelV3StreamPart::ReasoningStart { .. }
            | LanguageModelV3StreamPart::ReasoningDelta { .. }
            | LanguageModelV3StreamPart::ReasoningEnd { .. } => "gemini:reasoning",
            _ => return None,
        };
        Some((event_type.to_string(), data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_part_parses_from_custom_event_payload() {
        let ev = ChatStreamEvent::Custom {
            event_type: "openai:text-delta".to_string(),
            data: serde_json::json!({
                "type": "text-delta",
                "id": "0",
                "delta": "hello"
            }),
        };

        let part = LanguageModelV3StreamPart::try_from_chat_event(&ev).expect("parsed");
        match part {
            LanguageModelV3StreamPart::TextDelta { id, delta, .. } => {
                assert_eq!(id, "0");
                assert_eq!(delta, "hello");
            }
            other => panic!("unexpected part: {other:?}"),
        }
    }

    #[test]
    fn stream_part_formats_to_openai_custom_event() {
        let part = LanguageModelV3StreamPart::ToolCall(LanguageModelV3ToolCall {
            tool_call_id: "call_1".to_string(),
            tool_name: "web_search".to_string(),
            input: "{}".to_string(),
            provider_executed: Some(true),
            dynamic: None,
            provider_metadata: None,
        });

        let ev = part
            .to_custom_event(StreamPartNamespace::OpenAi)
            .expect("custom event");

        match ev {
            ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "openai:tool-call");
                assert_eq!(data.get("type").and_then(|v| v.as_str()), Some("tool-call"));
            }
            _ => panic!("expected custom event"),
        }
    }

    #[test]
    fn stream_part_formats_as_data_sse_frame() {
        let part = LanguageModelV3StreamPart::TextStart {
            id: "0".to_string(),
            provider_metadata: None,
        };
        let bytes = part.to_data_sse_bytes().expect("bytes");
        let text = String::from_utf8(bytes).expect("utf8");
        assert!(text.starts_with("data: "));
        assert!(text.ends_with("\n\n"));
    }

    #[test]
    fn stream_part_parse_loose_accepts_tool_call_input_object() {
        let v = serde_json::json!({
            "type": "tool-call",
            "toolCallId": "call_1",
            "toolName": "web_search",
            "input": { "query": "rust" }
        });

        let part = LanguageModelV3StreamPart::parse_loose_json(&v).expect("parsed");
        match part {
            LanguageModelV3StreamPart::ToolCall(call) => {
                assert_eq!(call.tool_call_id, "call_1");
                assert_eq!(call.tool_name, "web_search");
                assert!(call.input.contains("\"query\""));
            }
            other => panic!("unexpected part: {other:?}"),
        }
    }

    #[test]
    fn stream_part_parse_loose_accepts_finish_reason_string() {
        let v = serde_json::json!({
            "type": "finish",
            "usage": {
                "inputTokens": { "total": 1 },
                "outputTokens": { "total": 2 }
            },
            "finishReason": "stop"
        });

        let part = LanguageModelV3StreamPart::parse_loose_json(&v).expect("parsed");
        match part {
            LanguageModelV3StreamPart::Finish { finish_reason, .. } => {
                assert_eq!(finish_reason.unified, "stop");
            }
            other => panic!("unexpected part: {other:?}"),
        }
    }
}
