#![allow(clippy::large_enum_variant)]
//! Streaming event types for real-time responses

use super::chat::ChatResponse;
use super::chat::SourcePart;
use crate::types::{
    FinishReason, ProviderMetadataMap, ResponseMetadata, Usage, Warning,
    provider_metadata_without_private_diagnostics,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;

/// Provider metadata object keyed by provider name.
///
/// Stream provider metadata is a public provider-scoped projection lane. Do not put raw HTTP
/// payloads, response headers, whole provider chunks, or unreviewed debug data here; use
/// `ResponseMetadata` transport diagnostics, `ChatStreamPart::Raw`, `ChatStreamReplay`, or
/// provider diagnostics handling instead.
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
    /// Tool execution owner.
    ///
    /// `Some(true)` means the provider/model service executed the tool. `None` and `Some(false)`
    /// mean the caller/runtime owns execution.
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
/// Replay hints may carry raw provider data and should not be treated as user-visible output.
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

    /// Return whether this replay envelope carries private provider wire data.
    pub fn contains_private_diagnostics(&self) -> bool {
        self.openai_responses
            .as_ref()
            .is_some_and(ChatStreamOpenAiResponsesReplay::contains_private_diagnostics)
    }
}

impl ChatStreamOpenAiResponsesReplay {
    /// Return whether this replay envelope carries raw OpenAI Responses item data.
    pub fn contains_private_diagnostics(&self) -> bool {
        self.raw_item.is_some()
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
    /// Tool input became available and may stream through following deltas.
    ///
    /// This variant carries only stable AI SDK stream fields: id, tool name, public provider
    /// metadata, execution ownership, dynamic flag, and title. Provider-specific replay hints such
    /// as OpenAI Responses `outputIndex` and raw items belong in `ChatStreamReplay`, not in this
    /// stable part.
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
    /// Raw provider chunk or event payload.
    ///
    /// This is a private diagnostics/replay carrier, not part of the stable public semantic
    /// projection. Consumers should only expose it through an explicit diagnostics/redaction
    /// policy.
    Raw {
        #[serde(rename = "rawValue")]
        raw_value: serde_json::Value,
    },
    Error {
        error: serde_json::Value,
    },
}

impl ChatStreamPart {
    fn provider_metadata_contains_private_diagnostics(
        provider_metadata: &Option<StreamProviderMetadata>,
    ) -> bool {
        provider_metadata.as_ref().is_some_and(|metadata| {
            provider_metadata_without_private_diagnostics(metadata) != *metadata
        })
    }

    fn public_provider_metadata(
        provider_metadata: &Option<StreamProviderMetadata>,
    ) -> Option<StreamProviderMetadata> {
        provider_metadata
            .as_ref()
            .map(provider_metadata_without_private_diagnostics)
            .filter(|metadata| !metadata.is_empty())
    }

    /// Return whether this stream part carries private diagnostics or raw provider data.
    ///
    /// AI SDK-style `providerMetadata` fields are public provider-scoped projections by contract,
    /// so this method only flags metadata keys reserved for raw/private diagnostics.
    pub fn contains_private_diagnostics(&self) -> bool {
        match self {
            Self::TextStart {
                provider_metadata, ..
            }
            | Self::TextDelta {
                provider_metadata, ..
            }
            | Self::TextEnd {
                provider_metadata, ..
            }
            | Self::ReasoningStart {
                provider_metadata, ..
            }
            | Self::ReasoningDelta {
                provider_metadata, ..
            }
            | Self::ReasoningEnd {
                provider_metadata, ..
            }
            | Self::ToolInputStart {
                provider_metadata, ..
            }
            | Self::ToolInputDelta {
                provider_metadata, ..
            }
            | Self::ToolInputEnd {
                provider_metadata, ..
            }
            | Self::Custom(ChatStreamCustomContent {
                provider_metadata, ..
            })
            | Self::File(ChatStreamFilePart {
                provider_metadata, ..
            })
            | Self::ReasoningFile(ChatStreamFilePart {
                provider_metadata, ..
            })
            | Self::Source {
                provider_metadata, ..
            }
            | Self::Finish {
                provider_metadata, ..
            } => Self::provider_metadata_contains_private_diagnostics(provider_metadata),
            Self::ToolApprovalRequest(request) => {
                Self::provider_metadata_contains_private_diagnostics(&request.provider_metadata)
            }
            Self::ToolCall(call) => {
                Self::provider_metadata_contains_private_diagnostics(&call.provider_metadata)
            }
            Self::ToolResult(result) => {
                Self::provider_metadata_contains_private_diagnostics(&result.provider_metadata)
            }
            Self::ResponseMetadata(metadata) => metadata.contains_private_diagnostics(),
            Self::Raw { .. } => true,
            _ => false,
        }
    }

    /// Return the public projection of this part, dropping raw-only diagnostics carriers.
    pub fn without_private_diagnostics(&self) -> Option<Self> {
        let mut part = self.clone();
        match &mut part {
            Self::TextStart {
                provider_metadata, ..
            }
            | Self::TextDelta {
                provider_metadata, ..
            }
            | Self::TextEnd {
                provider_metadata, ..
            }
            | Self::ReasoningStart {
                provider_metadata, ..
            }
            | Self::ReasoningDelta {
                provider_metadata, ..
            }
            | Self::ReasoningEnd {
                provider_metadata, ..
            }
            | Self::ToolInputStart {
                provider_metadata, ..
            }
            | Self::ToolInputDelta {
                provider_metadata, ..
            }
            | Self::ToolInputEnd {
                provider_metadata, ..
            }
            | Self::Custom(ChatStreamCustomContent {
                provider_metadata, ..
            })
            | Self::File(ChatStreamFilePart {
                provider_metadata, ..
            })
            | Self::ReasoningFile(ChatStreamFilePart {
                provider_metadata, ..
            })
            | Self::Source {
                provider_metadata, ..
            }
            | Self::Finish {
                provider_metadata, ..
            } => {
                *provider_metadata = Self::public_provider_metadata(provider_metadata);
                Some(part)
            }
            Self::ToolApprovalRequest(request) => {
                request.provider_metadata =
                    Self::public_provider_metadata(&request.provider_metadata);
                Some(part)
            }
            Self::ToolCall(call) => {
                call.provider_metadata = Self::public_provider_metadata(&call.provider_metadata);
                Some(part)
            }
            Self::ToolResult(result) => {
                result.provider_metadata =
                    Self::public_provider_metadata(&result.provider_metadata);
                Some(part)
            }
            Self::ResponseMetadata(metadata) => {
                *metadata = metadata.without_private_diagnostics();
                Some(part)
            }
            Self::Raw { .. } => None,
            _ => Some(part),
        }
    }
}

/// Chat streaming event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChatStreamEvent {
    /// Stream start event with metadata
    StreamStart {
        /// Response metadata
        metadata: ResponseMetadata,
    },
    /// Stream end event with the final response snapshot for this provider call.
    ///
    /// `response.content` is not a stream delta. Providers and protocol adapters may leave it
    /// empty, may replay the complete final content that was already emitted through typed stream
    /// parts, or may provide terminal-only content that was not available as live deltas. Consumers
    /// should use `Part` / `PartWithReplay` events for incremental UI updates and reconcile this
    /// terminal response as the final snapshot instead of appending it blindly.
    StreamEnd {
        /// Final response snapshot or replay/fallback response.
        response: ChatResponse,
    },
    /// Typed AI SDK-style stream part.
    ///
    /// This is the structured semantic stream model. It represents text,
    /// reasoning, tools, sources, response metadata, warnings, usage, and
    /// custom content using AI SDK-aligned part variants. Explicit raw parts remain private
    /// diagnostics/replay data and should not be projected into user-visible output by default.
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
    /// Custom provider-specific event.
    ///
    /// Allows providers to emit custom events without modifying the core enum.
    /// Users can pattern match on `event_type` to handle provider-specific features.
    /// Event types whose provider-local segment starts with `raw`, `private`, or `diagnostic`
    /// are reserved for private diagnostics and should be routed to diagnostics handling instead
    /// of being silently discarded or shown in user-visible output.
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

impl ChatStreamEvent {
    /// Return whether a custom event type is reserved for private diagnostics.
    ///
    /// The check is segment-based so names such as `raw_openai_event`, `openai:raw_response`,
    /// `openai.private_debug`, and `provider/diagnostic_headers` are all recognized.
    pub fn custom_event_type_is_private_diagnostics(event_type: &str) -> bool {
        let normalized = event_type.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            return false;
        }

        normalized
            .split([':', '.', '/'])
            .filter(|segment| !segment.is_empty())
            .any(|segment| {
                segment == "raw"
                    || segment == "private"
                    || segment == "diagnostic"
                    || segment == "diagnostics"
                    || segment.starts_with("raw_")
                    || segment.starts_with("private_")
                    || segment.starts_with("diagnostic_")
                    || segment.starts_with("diagnostics_")
            })
    }

    /// Return whether this event carries private diagnostics or raw provider data.
    ///
    /// This is a routing helper for bridges/adapters that need to keep public stream projection
    /// separate from raw diagnostics. It intentionally does not treat ordinary `providerMetadata`
    /// as private by itself; providers must only put reviewed public projection fields there.
    pub fn contains_private_diagnostics(&self) -> bool {
        match self {
            Self::StreamStart { metadata } => metadata.contains_private_diagnostics(),
            Self::Part { part } => part.contains_private_diagnostics(),
            Self::PartWithReplay { part, replay } => {
                part.contains_private_diagnostics() || replay.contains_private_diagnostics()
            }
            Self::Custom { event_type, .. } => {
                Self::custom_event_type_is_private_diagnostics(event_type)
            }
            Self::StreamEnd { response } => response.contains_private_diagnostics(),
            Self::Error { .. } => false,
        }
    }

    /// Return the public projection of this event, dropping raw-only diagnostics carriers.
    pub fn without_private_diagnostics(&self) -> Option<Self> {
        match self {
            Self::StreamStart { metadata } => Some(Self::StreamStart {
                metadata: metadata.without_private_diagnostics(),
            }),
            Self::StreamEnd { response } => Some(Self::StreamEnd {
                response: response.without_private_diagnostics(),
            }),
            Self::Part { part } => part
                .without_private_diagnostics()
                .map(|part| Self::Part { part }),
            Self::PartWithReplay { part, .. } => part
                .without_private_diagnostics()
                .map(|part| Self::Part { part }),
            Self::Custom { event_type, .. }
                if Self::custom_event_type_is_private_diagnostics(event_type) =>
            {
                None
            }
            Self::Custom { .. } | Self::Error { .. } => Some(self.clone()),
        }
    }
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

    #[test]
    fn tool_input_start_serializes_only_stable_ai_sdk_fields() {
        let part = ChatStreamPart::ToolInputStart {
            id: "call_1".to_string(),
            tool_name: "web_search".to_string(),
            provider_metadata: Some(HashMap::from([(
                "provider-a".to_string(),
                serde_json::json!({ "itemId": "item_1" }),
            )])),
            provider_executed: Some(true),
            dynamic: Some(true),
            title: Some("Web Search".to_string()),
        };

        let value = serde_json::to_value(&part).expect("serialize tool input start");
        assert_eq!(value["type"], serde_json::json!("tool-input-start"));
        assert_eq!(value["id"], serde_json::json!("call_1"));
        assert_eq!(value["toolName"], serde_json::json!("web_search"));
        assert_eq!(value["providerExecuted"], serde_json::json!(true));
        assert_eq!(value["dynamic"], serde_json::json!(true));
        assert_eq!(value["title"], serde_json::json!("Web Search"));
        assert_eq!(
            value["providerMetadata"]["provider-a"]["itemId"],
            serde_json::json!("item_1")
        );
        assert!(value.get("index").is_none());
        assert!(value.get("outputIndex").is_none());
        assert!(value.get("rawItem").is_none());
    }

    #[test]
    fn provider_metadata_is_public_projection_not_private_diagnostics() {
        let event = ChatStreamEvent::Part {
            part: ChatStreamPart::TextDelta {
                id: "0".to_string(),
                delta: "hello".to_string(),
                provider_metadata: Some(HashMap::from([(
                    "openai".to_string(),
                    serde_json::json!({ "responseId": "resp_1" }),
                )])),
            },
        };

        assert!(!event.contains_private_diagnostics());
    }

    #[test]
    fn provider_metadata_reserved_private_keys_are_projected_out() {
        let event = ChatStreamEvent::PartWithReplay {
            part: ChatStreamPart::ToolCall(ChatStreamToolCall {
                tool_call_id: "call_1".to_string(),
                tool_name: "search".to_string(),
                input: "{}".to_string(),
                provider_executed: Some(true),
                dynamic: None,
                provider_metadata: Some(HashMap::from([(
                    "openai".to_string(),
                    serde_json::json!({
                        "itemId": "item_1",
                        "rawItem": { "secret": true }
                    }),
                )])),
            }),
            replay: ChatStreamReplay::openai_responses(
                Some(0),
                Some(serde_json::json!({ "id": "raw_item_1" })),
            )
            .expect("replay"),
        };

        assert!(event.contains_private_diagnostics());

        let public = event
            .without_private_diagnostics()
            .expect("public projection");
        assert!(!public.contains_private_diagnostics());
        assert!(matches!(public, ChatStreamEvent::Part { .. }));
        let ChatStreamEvent::Part {
            part: ChatStreamPart::ToolCall(call),
        } = public
        else {
            panic!("expected projected tool call part");
        };
        let openai = call
            .provider_metadata
            .as_ref()
            .and_then(|metadata| metadata.get("openai"))
            .and_then(|metadata| metadata.as_object())
            .expect("openai metadata");
        assert_eq!(openai.get("itemId"), Some(&serde_json::json!("item_1")));
        assert!(openai.get("rawItem").is_none());
    }

    #[test]
    fn raw_stream_part_is_private_diagnostics() {
        let event = ChatStreamEvent::Part {
            part: ChatStreamPart::Raw {
                raw_value: serde_json::json!({ "provider": "chunk" }),
            },
        };

        assert!(event.contains_private_diagnostics());
        assert!(event.without_private_diagnostics().is_none());
    }

    #[test]
    fn response_metadata_headers_and_body_are_private_diagnostics() {
        let metadata = ResponseMetadata {
            id: Some("resp_1".to_string()),
            model: Some("gpt-4o".to_string()),
            created: None,
            provider: "openai".to_string(),
            request_id: Some("req_1".to_string()),
            headers: Some(HashMap::from([(
                "set-cookie".to_string(),
                "session=private".to_string(),
            )])),
            body: Some(serde_json::json!({ "raw": true })),
        };

        let event = ChatStreamEvent::Part {
            part: ChatStreamPart::ResponseMetadata(metadata.clone()),
        };

        assert!(event.contains_private_diagnostics());
        assert!(
            !metadata
                .without_private_diagnostics()
                .contains_private_diagnostics()
        );
    }

    #[test]
    fn stream_end_http_response_metadata_is_private_diagnostics() {
        let mut response =
            ChatResponse::new(crate::types::MessageContent::Text("done".to_string()));
        response.response = Some(crate::types::HttpResponseInfo {
            timestamp: chrono::Utc::now(),
            model_id: Some("gpt-4o".to_string()),
            headers: HashMap::from([("x-ratelimit-remaining".to_string(), "10".to_string())]),
            body: Some(serde_json::json!({ "raw": true })),
        });

        assert!(ChatStreamEvent::StreamEnd { response }.contains_private_diagnostics());
    }

    #[test]
    fn replay_raw_item_is_private_diagnostics() {
        let event = ChatStreamEvent::PartWithReplay {
            part: ChatStreamPart::TextDelta {
                id: "0".to_string(),
                delta: "hello".to_string(),
                provider_metadata: None,
            },
            replay: ChatStreamReplay::openai_responses(
                Some(0),
                Some(serde_json::json!({ "id": "item_1", "raw": true })),
            )
            .expect("replay"),
        };

        assert!(event.contains_private_diagnostics());
        assert!(matches!(
            event.without_private_diagnostics(),
            Some(ChatStreamEvent::Part { .. })
        ));
    }

    #[test]
    fn custom_raw_private_and_diagnostic_event_types_are_private_diagnostics() {
        for event_type in [
            "raw_openai_event",
            "openai:raw_response",
            "openai.private_debug",
            "provider/diagnostic_headers",
            "provider:diagnostics_body",
        ] {
            assert!(
                ChatStreamEvent::custom_event_type_is_private_diagnostics(event_type),
                "{event_type} should be private diagnostics"
            );

            assert!(
                ChatStreamEvent::Custom {
                    event_type: event_type.to_string(),
                    data: serde_json::json!({ "raw": true }),
                }
                .contains_private_diagnostics()
            );

            assert!(
                ChatStreamEvent::Custom {
                    event_type: event_type.to_string(),
                    data: serde_json::json!({ "raw": true }),
                }
                .without_private_diagnostics()
                .is_none()
            );
        }

        assert!(!ChatStreamEvent::custom_event_type_is_private_diagnostics(
            "openai:citation"
        ));
    }
}
