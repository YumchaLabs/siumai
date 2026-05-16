#![allow(clippy::large_enum_variant)]
//! Streaming event types for real-time responses

use super::chat::ChatResponse;
use super::chat::SourcePart;
use crate::types::{FinishReason, ProviderMetadataMap, ResponseMetadata, Usage, Warning};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;

/// Provider metadata object keyed by provider name.
pub type StreamProviderMetadata = ProviderMetadataMap;

fn serialize_stream_non_null_json_value<S>(
    value: &serde_json::Value,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if value.is_null() {
        return Err(serde::ser::Error::custom("expected non-null JSON value"));
    }

    value.serialize(serializer)
}

fn deserialize_stream_non_null_json_value<'de, D>(
    deserializer: D,
) -> Result<serde_json::Value, D::Error>
where
    D: Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    if value.is_null() {
        return Err(serde::de::Error::custom("expected non-null JSON value"));
    }

    Ok(value)
}

/// Binary-or-base64 file payload used by stream parts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ChatStreamFileData {
    Base64(String),
    Bytes(Vec<u8>),
}

/// Finish reason payload for stream parts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatStreamFinishInfo {
    pub unified: FinishReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
}

/// Tool approval request part carried during streaming.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatStreamToolApprovalRequest {
    #[serde(rename = "approvalId")]
    pub approval_id: String,
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "providerMetadata"
    )]
    pub provider_metadata: Option<StreamProviderMetadata>,
}

/// Tool call part carried during streaming.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatStreamToolCall {
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    #[serde(rename = "toolName")]
    pub tool_name: String,
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
    pub provider_metadata: Option<StreamProviderMetadata>,
}

/// Tool result part carried during streaming.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatStreamToolResult {
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    #[serde(rename = "toolName")]
    pub tool_name: String,
    #[serde(
        deserialize_with = "deserialize_stream_non_null_json_value",
        serialize_with = "serialize_stream_non_null_json_value"
    )]
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
    pub provider_metadata: Option<StreamProviderMetadata>,
}

/// Custom content part carried during streaming.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatStreamCustomContent {
    pub kind: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "providerMetadata"
    )]
    pub provider_metadata: Option<StreamProviderMetadata>,
}

/// Generated file part carried during streaming.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatStreamFilePart {
    #[serde(rename = "mediaType")]
    pub media_type: String,
    pub data: ChatStreamFileData,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "providerMetadata"
    )]
    pub provider_metadata: Option<StreamProviderMetadata>,
}

/// Runtime-only replay hints attached to structured stream parts.
///
/// These hints are intentionally kept outside `ChatStreamPart` so the stable
/// AI SDK-aligned part schema stays clean while protocol serializers can still
/// recover provider-specific wire details when lossless replay matters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ChatStreamReplay {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "openaiResponses"
    )]
    pub openai_responses: Option<ChatStreamOpenAiResponsesReplay>,
}

/// Replay hints used by the OpenAI Responses SSE serializer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ChatStreamOpenAiResponsesReplay {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "outputIndex"
    )]
    pub output_index: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "rawItem")]
    pub raw_item: Option<serde_json::Value>,
}

impl ChatStreamReplay {
    /// Build an OpenAI Responses replay envelope when at least one hint exists.
    pub fn openai_responses(
        output_index: Option<u64>,
        raw_item: Option<serde_json::Value>,
    ) -> Option<Self> {
        let replay = ChatStreamOpenAiResponsesReplay {
            output_index,
            raw_item,
        };

        if replay.output_index.is_none() && replay.raw_item.is_none() {
            None
        } else {
            Some(Self {
                openai_responses: Some(replay),
            })
        }
    }

    pub fn openai_responses_ref(&self) -> Option<&ChatStreamOpenAiResponsesReplay> {
        self.openai_responses.as_ref()
    }

    pub fn is_empty(&self) -> bool {
        self.openai_responses.is_none()
    }
}

/// Typed AI SDK-aligned stream-part contract available on the runtime event layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ChatStreamPart {
    TextStart {
        id: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<StreamProviderMetadata>,
    },
    TextDelta {
        id: String,
        delta: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<StreamProviderMetadata>,
    },
    TextEnd {
        id: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<StreamProviderMetadata>,
    },
    ReasoningStart {
        id: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<StreamProviderMetadata>,
    },
    ReasoningDelta {
        id: String,
        delta: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<StreamProviderMetadata>,
    },
    ReasoningEnd {
        id: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<StreamProviderMetadata>,
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
        provider_metadata: Option<StreamProviderMetadata>,
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
        provider_metadata: Option<StreamProviderMetadata>,
    },
    ToolInputEnd {
        id: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<StreamProviderMetadata>,
    },
    ToolApprovalRequest(ChatStreamToolApprovalRequest),
    ToolCall(ChatStreamToolCall),
    ToolResult(ChatStreamToolResult),
    Custom(ChatStreamCustomContent),
    File(ChatStreamFilePart),
    ReasoningFile(ChatStreamFilePart),
    Source {
        id: String,
        #[serde(flatten)]
        source: SourcePart,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<StreamProviderMetadata>,
    },
    StreamStart {
        warnings: Vec<Warning>,
    },
    ResponseMetadata(ResponseMetadata),
    Finish {
        usage: Usage,
        #[serde(rename = "finishReason")]
        finish_reason: ChatStreamFinishInfo,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<StreamProviderMetadata>,
    },
    Raw {
        #[serde(rename = "rawValue")]
        raw_value: serde_json::Value,
    },
    Error {
        error: serde_json::Value,
    },
}

/// Chat streaming event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChatStreamEvent {
    /// Stream start event with metadata
    StreamStart {
        /// Response metadata
        metadata: ResponseMetadata,
    },
    /// Stream end event with final response
    StreamEnd {
        /// Final response
        response: ChatResponse,
    },
    /// Typed AI SDK-style stream part.
    ///
    /// This is the structured semantic stream model. It represents text,
    /// reasoning, tools, sources, response metadata, warnings, usage, and
    /// custom content using AI SDK-aligned part variants.
    Part {
        /// Structured stream part.
        part: ChatStreamPart,
    },
    /// Structured stream part plus runtime replay hints.
    ///
    /// This is used when a provider parser can express stable semantics as a
    /// `ChatStreamPart` but still needs protocol-specific carrier data for
    /// lossless wire replay in a downstream serializer.
    PartWithReplay {
        /// Structured stream part.
        part: ChatStreamPart,
        /// Runtime-only replay metadata.
        replay: ChatStreamReplay,
    },
    /// Error occurred during streaming
    Error {
        /// Error message
        error: String,
    },
    /// Custom provider-specific event
    ///
    /// Allows providers to emit custom events without modifying the core enum.
    /// Users can pattern match on `event_type` to handle provider-specific features.
    ///
    /// # Example
    /// ```rust,ignore
    /// match event {
    ///     ChatStreamEvent::Custom { event_type, data } => {
    ///         match event_type.as_str() {
    ///             "openai:citation" => { /* Handle OpenAI citation */ }
    ///             "anthropic:thinking_progress" => { /* Handle thinking progress */ }
    ///             _ => { /* Ignore unknown custom events */ }
    ///         }
    ///     }
    ///     _ => { /* Handle standard events */ }
    /// }
    /// ```
    Custom {
        /// Event type identifier (e.g., "openai:function_call_progress", "anthropic:citation")
        event_type: String,
        /// Event data as JSON value
        data: serde_json::Value,
    },
}

/// Audio streaming event
#[derive(Debug, Clone)]
pub enum AudioStreamEvent {
    /// Audio data chunk
    AudioDelta {
        /// Audio data bytes
        data: Vec<u8>,
        /// Audio format
        format: String,
    },
    /// Metadata about the audio
    Metadata {
        /// Sample rate
        sample_rate: Option<u32>,
        /// Duration estimate
        duration: Option<f32>,
        /// Additional metadata
        metadata: HashMap<String, serde_json::Value>,
    },
    /// Stream finished
    Done {
        /// Total duration
        duration: Option<f32>,
        /// Final metadata
        metadata: HashMap<String, serde_json::Value>,
    },
    /// Error occurred during streaming
    Error {
        /// Error message
        error: String,
    },
}

impl ChatStreamEvent {
    /// Build a typed text delta event.
    pub fn text_delta_part(id: impl Into<String>, delta: impl Into<String>) -> Self {
        Self::Part {
            part: ChatStreamPart::TextDelta {
                id: id.into(),
                delta: delta.into(),
                provider_metadata: None,
            },
        }
    }

    /// Build a typed text delta event using a choice index as the stream part id.
    pub fn text_delta_for_index(index: Option<usize>, delta: impl Into<String>) -> Self {
        Self::text_delta_part(
            index
                .map(|index| index.to_string())
                .unwrap_or_else(|| "0".to_string()),
            delta,
        )
    }

    /// Build a typed reasoning delta event.
    pub fn reasoning_delta_part(id: impl Into<String>, delta: impl Into<String>) -> Self {
        Self::Part {
            part: ChatStreamPart::ReasoningDelta {
                id: id.into(),
                delta: delta.into(),
                provider_metadata: None,
            },
        }
    }

    /// Build a typed tool input start event.
    pub fn tool_input_start_part(id: impl Into<String>, tool_name: impl Into<String>) -> Self {
        Self::Part {
            part: ChatStreamPart::ToolInputStart {
                id: id.into(),
                tool_name: tool_name.into(),
                provider_metadata: None,
                provider_executed: None,
                dynamic: None,
                title: None,
            },
        }
    }

    /// Build a typed tool input delta event.
    pub fn tool_input_delta_part(id: impl Into<String>, delta: impl Into<String>) -> Self {
        Self::Part {
            part: ChatStreamPart::ToolInputDelta {
                id: id.into(),
                delta: delta.into(),
                provider_metadata: None,
            },
        }
    }

    /// Build a typed tool input end event.
    pub fn tool_input_end_part(id: impl Into<String>) -> Self {
        Self::Part {
            part: ChatStreamPart::ToolInputEnd {
                id: id.into(),
                provider_metadata: None,
            },
        }
    }

    /// Build a typed completed tool call event.
    pub fn tool_call_part(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        input: impl Into<String>,
    ) -> Self {
        Self::Part {
            part: ChatStreamPart::ToolCall(ChatStreamToolCall {
                tool_call_id: tool_call_id.into(),
                tool_name: tool_name.into(),
                input: input.into(),
                provider_executed: None,
                dynamic: None,
                provider_metadata: None,
            }),
        }
    }

    /// Build a typed finish event.
    pub fn finish_part(usage: Usage, finish_reason: FinishReason) -> Self {
        Self::Part {
            part: ChatStreamPart::Finish {
                usage,
                finish_reason: ChatStreamFinishInfo {
                    unified: finish_reason,
                    raw: None,
                },
                provider_metadata: None,
            },
        }
    }

    /// Borrow the structured stream part if this is a part-bearing event.
    pub fn part_ref(&self) -> Option<&ChatStreamPart> {
        match self {
            Self::Part { part } | Self::PartWithReplay { part, .. } => Some(part),
            _ => None,
        }
    }

    /// Borrow runtime replay hints if present.
    pub fn replay_ref(&self) -> Option<&ChatStreamReplay> {
        match self {
            Self::PartWithReplay { replay, .. } => Some(replay),
            _ => None,
        }
    }

    /// Borrow a typed text delta from this event.
    ///
    /// This intentionally reads only the typed stream-part lane. Legacy transport-style deltas are
    /// being removed from the public streaming model.
    pub fn text_delta(&self) -> Option<&str> {
        match self.part_ref() {
            Some(ChatStreamPart::TextDelta { delta, .. }) => Some(delta.as_str()),
            _ => None,
        }
    }

    /// Borrow a typed reasoning delta from this event.
    ///
    /// This intentionally reads only the typed stream-part lane. Legacy transport-style reasoning
    /// deltas are being removed from the public streaming model.
    pub fn reasoning_delta(&self) -> Option<&str> {
        match self.part_ref() {
            Some(ChatStreamPart::ReasoningDelta { delta, .. }) => Some(delta.as_str()),
            _ => None,
        }
    }

    /// Borrow typed finish usage from this event.
    pub fn finish_usage(&self) -> Option<&Usage> {
        match self.part_ref() {
            Some(ChatStreamPart::Finish { usage, .. }) => Some(usage),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_stream_event_is_send_sync_data() {
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<AudioStreamEvent>();
    }

    #[test]
    fn stream_part_serializes_finish_with_ai_sdk_shape() {
        let part = ChatStreamPart::Finish {
            usage: Usage::new(3, 5),
            finish_reason: ChatStreamFinishInfo {
                unified: FinishReason::Stop,
                raw: Some("stop".to_string()),
            },
            provider_metadata: Some(HashMap::from([(
                "openai".to_string(),
                serde_json::json!({ "responseId": "resp_1" }),
            )])),
        };

        let value = serde_json::to_value(&part).expect("serialize stream part");
        assert_eq!(value["type"], serde_json::json!("finish"));
        assert_eq!(value["finishReason"]["unified"], serde_json::json!("stop"));
        assert_eq!(value["finishReason"]["raw"], serde_json::json!("stop"));
        assert_eq!(
            value["providerMetadata"]["openai"]["responseId"],
            serde_json::json!("resp_1")
        );
    }

    #[test]
    fn stream_part_source_serializes_strict_union_shape() {
        let part = ChatStreamPart::Source {
            id: "src_1".to_string(),
            source: SourcePart::Document {
                media_type: "application/pdf".to_string(),
                title: "Guide".to_string(),
                filename: Some("guide.pdf".to_string()),
            },
            provider_metadata: Some(HashMap::from([(
                "anthropic".to_string(),
                serde_json::json!({ "startPageNumber": 1 }),
            )])),
        };

        let value = serde_json::to_value(&part).expect("serialize source part");
        assert_eq!(value["type"], serde_json::json!("source"));
        assert_eq!(value["sourceType"], serde_json::json!("document"));
        assert_eq!(value["mediaType"], serde_json::json!("application/pdf"));
        assert_eq!(value["title"], serde_json::json!("Guide"));
        assert_eq!(
            value["providerMetadata"]["anthropic"]["startPageNumber"],
            serde_json::json!(1)
        );
    }

    #[test]
    fn stream_tool_result_rejects_null_result_payload() {
        let invalid = serde_json::json!({
            "type": "tool-result",
            "toolCallId": "call_1",
            "toolName": "weather",
            "result": null
        });

        assert!(serde_json::from_value::<ChatStreamPart>(invalid).is_err());

        let part = ChatStreamPart::ToolResult(ChatStreamToolResult {
            tool_call_id: "call_1".to_string(),
            tool_name: "weather".to_string(),
            result: serde_json::Value::Null,
            is_error: None,
            preliminary: None,
            dynamic: None,
            provider_metadata: None,
        });
        assert!(serde_json::to_value(&part).is_err());
    }

    #[test]
    fn stream_event_supports_typed_part_variant() {
        let event = ChatStreamEvent::Part {
            part: ChatStreamPart::Custom(ChatStreamCustomContent {
                kind: "openai.compaction".to_string(),
                provider_metadata: Some(HashMap::from([(
                    "openai".to_string(),
                    serde_json::json!({ "itemId": "cmp_1" }),
                )])),
            }),
        };

        let value = serde_json::to_value(&event).expect("serialize event");
        assert!(value.get("Part").is_some());
    }

    #[test]
    fn stream_event_text_delta_reads_typed_part() {
        let typed = ChatStreamEvent::Part {
            part: ChatStreamPart::TextDelta {
                id: "0".to_string(),
                delta: "hello".to_string(),
                provider_metadata: None,
            },
        };

        assert_eq!(typed.text_delta(), Some("hello"));
    }

    #[test]
    fn stream_event_reasoning_delta_reads_typed_part() {
        let typed = ChatStreamEvent::Part {
            part: ChatStreamPart::ReasoningDelta {
                id: "0".to_string(),
                delta: "think".to_string(),
                provider_metadata: None,
            },
        };

        assert_eq!(typed.reasoning_delta(), Some("think"));
    }

    #[test]
    fn stream_event_exposes_part_replay_accessors() {
        let event = ChatStreamEvent::PartWithReplay {
            part: ChatStreamPart::ToolCall(ChatStreamToolCall {
                tool_call_id: "call_1".to_string(),
                tool_name: "web_search".to_string(),
                input: "{}".to_string(),
                provider_executed: Some(true),
                dynamic: Some(true),
                provider_metadata: None,
            }),
            replay: ChatStreamReplay::openai_responses(
                Some(2),
                Some(serde_json::json!({ "id": "call_1", "type": "custom_tool_call" })),
            )
            .expect("replay"),
        };

        assert!(matches!(
            event.part_ref(),
            Some(ChatStreamPart::ToolCall(_))
        ));
        assert_eq!(
            event
                .replay_ref()
                .and_then(ChatStreamReplay::openai_responses_ref)
                .and_then(|replay| replay.output_index),
            Some(2)
        );
    }
}
