use crate::types::FinishReason;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{JSONValue, ProviderMetadata};

macro_rules! fixed_ui_message_chunk_type_marker {
    ($name:ident, $value:literal) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum $name {
            Marker,
        }

        impl Default for $name {
            fn default() -> Self {
                Self::Marker
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                serializer.serialize_str($value)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                if value == $value {
                    Ok(Self::Marker)
                } else {
                    Err(serde::de::Error::custom(format!(
                        "expected stream type marker `{}`, got `{value}`",
                        $value
                    )))
                }
            }
        }
    };
}

fixed_ui_message_chunk_type_marker!(UiMessageTextStartChunkMarker, "text-start");
fixed_ui_message_chunk_type_marker!(UiMessageTextDeltaChunkMarker, "text-delta");
fixed_ui_message_chunk_type_marker!(UiMessageTextEndChunkMarker, "text-end");
fixed_ui_message_chunk_type_marker!(UiMessageReasoningStartChunkMarker, "reasoning-start");
fixed_ui_message_chunk_type_marker!(UiMessageReasoningDeltaChunkMarker, "reasoning-delta");
fixed_ui_message_chunk_type_marker!(UiMessageReasoningEndChunkMarker, "reasoning-end");
fixed_ui_message_chunk_type_marker!(UiMessageCustomChunkMarker, "custom");
fixed_ui_message_chunk_type_marker!(UiMessageErrorChunkMarker, "error");
fixed_ui_message_chunk_type_marker!(UiMessageToolInputStartChunkMarker, "tool-input-start");
fixed_ui_message_chunk_type_marker!(UiMessageToolInputDeltaChunkMarker, "tool-input-delta");
fixed_ui_message_chunk_type_marker!(
    UiMessageToolInputAvailableChunkMarker,
    "tool-input-available"
);
fixed_ui_message_chunk_type_marker!(UiMessageToolInputErrorChunkMarker, "tool-input-error");
fixed_ui_message_chunk_type_marker!(
    UiMessageToolApprovalRequestChunkMarker,
    "tool-approval-request"
);
fixed_ui_message_chunk_type_marker!(
    UiMessageToolApprovalResponseChunkMarker,
    "tool-approval-response"
);
fixed_ui_message_chunk_type_marker!(
    UiMessageToolOutputAvailableChunkMarker,
    "tool-output-available"
);
fixed_ui_message_chunk_type_marker!(UiMessageToolOutputErrorChunkMarker, "tool-output-error");
fixed_ui_message_chunk_type_marker!(UiMessageToolOutputDeniedChunkMarker, "tool-output-denied");
fixed_ui_message_chunk_type_marker!(UiMessageSourceUrlChunkMarker, "source-url");
fixed_ui_message_chunk_type_marker!(UiMessageSourceDocumentChunkMarker, "source-document");
fixed_ui_message_chunk_type_marker!(UiMessageFileChunkMarker, "file");
fixed_ui_message_chunk_type_marker!(UiMessageReasoningFileChunkMarker, "reasoning-file");
fixed_ui_message_chunk_type_marker!(UiMessageStartStepChunkMarker, "start-step");
fixed_ui_message_chunk_type_marker!(UiMessageFinishStepChunkMarker, "finish-step");
fixed_ui_message_chunk_type_marker!(UiMessageStartChunkMarker, "start");
fixed_ui_message_chunk_type_marker!(UiMessageFinishChunkMarker, "finish");
fixed_ui_message_chunk_type_marker!(UiMessageAbortChunkMarker, "abort");
fixed_ui_message_chunk_type_marker!(UiMessageMetadataChunkMarker, "message-metadata");

fn deserialize_ui_message_data_chunk_type<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    if value.starts_with("data-") {
        Ok(value)
    } else {
        Err(serde::de::Error::custom(format!(
            "expected UI message data chunk type to start with `data-`, got `{value}`"
        )))
    }
}

macro_rules! ui_message_id_provider_metadata_chunk {
    ($name:ident, $marker:ident, $type:literal) => {
        #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
        #[serde(rename_all = "camelCase")]
        pub struct $name {
            #[serde(rename = "type")]
            marker: $marker,
            /// Chunk id.
            pub id: String,
            /// Provider-specific metadata.
            #[serde(skip_serializing_if = "Option::is_none")]
            pub provider_metadata: Option<ProviderMetadata>,
        }

        impl $name {
            /// Create a UI message stream chunk.
            pub fn new(id: impl Into<String>) -> Self {
                Self {
                    marker: $marker::Marker,
                    id: id.into(),
                    provider_metadata: None,
                }
            }

            /// Attach provider metadata.
            pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
                self.provider_metadata = Some(provider_metadata);
                self
            }

            /// Return the AI SDK UI message chunk discriminator.
            pub const fn r#type(&self) -> &'static str {
                $type
            }
        }
    };
}

ui_message_id_provider_metadata_chunk!(
    UiMessageTextStartChunk,
    UiMessageTextStartChunkMarker,
    "text-start"
);
ui_message_id_provider_metadata_chunk!(
    UiMessageTextEndChunk,
    UiMessageTextEndChunkMarker,
    "text-end"
);
ui_message_id_provider_metadata_chunk!(
    UiMessageReasoningStartChunk,
    UiMessageReasoningStartChunkMarker,
    "reasoning-start"
);
ui_message_id_provider_metadata_chunk!(
    UiMessageReasoningEndChunk,
    UiMessageReasoningEndChunkMarker,
    "reasoning-end"
);

/// Text delta chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageTextDeltaChunk {
    #[serde(rename = "type")]
    marker: UiMessageTextDeltaChunkMarker,
    /// Text block id.
    pub id: String,
    /// Text delta.
    pub delta: String,
    /// Provider-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl UiMessageTextDeltaChunk {
    /// Create a UI text delta chunk.
    pub fn new(id: impl Into<String>, delta: impl Into<String>) -> Self {
        Self {
            marker: UiMessageTextDeltaChunkMarker::Marker,
            id: id.into(),
            delta: delta.into(),
            provider_metadata: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "text-delta"
    }
}

/// Reasoning delta chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageReasoningDeltaChunk {
    #[serde(rename = "type")]
    marker: UiMessageReasoningDeltaChunkMarker,
    /// Reasoning block id.
    pub id: String,
    /// Reasoning delta.
    pub delta: String,
    /// Provider-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl UiMessageReasoningDeltaChunk {
    /// Create a UI reasoning delta chunk.
    pub fn new(id: impl Into<String>, delta: impl Into<String>) -> Self {
        Self {
            marker: UiMessageReasoningDeltaChunkMarker::Marker,
            id: id.into(),
            delta: delta.into(),
            provider_metadata: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "reasoning-delta"
    }
}

/// Custom chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageCustomChunk {
    #[serde(rename = "type")]
    marker: UiMessageCustomChunkMarker,
    /// Custom kind, normally `{provider}.{kind}`.
    pub kind: String,
    /// Provider-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl UiMessageCustomChunk {
    /// Create a custom UI chunk.
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            marker: UiMessageCustomChunkMarker::Marker,
            kind: kind.into(),
            provider_metadata: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "custom"
    }
}

/// Error chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageErrorChunk {
    #[serde(rename = "type")]
    marker: UiMessageErrorChunkMarker,
    /// Error text.
    pub error_text: String,
}

impl UiMessageErrorChunk {
    /// Create an error UI chunk.
    pub fn new(error_text: impl Into<String>) -> Self {
        Self {
            marker: UiMessageErrorChunkMarker::Marker,
            error_text: error_text.into(),
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "error"
    }
}

/// Tool input start chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageToolInputStartChunk {
    #[serde(rename = "type")]
    marker: UiMessageToolInputStartChunkMarker,
    pub tool_call_id: String,
    pub tool_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_executed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

impl UiMessageToolInputStartChunk {
    /// Create a tool input start chunk.
    pub fn new(tool_call_id: impl Into<String>, tool_name: impl Into<String>) -> Self {
        Self {
            marker: UiMessageToolInputStartChunkMarker::Marker,
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            provider_executed: None,
            provider_metadata: None,
            dynamic: None,
            title: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-input-start"
    }
}

/// Tool input delta chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageToolInputDeltaChunk {
    #[serde(rename = "type")]
    marker: UiMessageToolInputDeltaChunkMarker,
    pub tool_call_id: String,
    pub input_text_delta: String,
}

impl UiMessageToolInputDeltaChunk {
    /// Create a tool input delta chunk.
    pub fn new(tool_call_id: impl Into<String>, input_text_delta: impl Into<String>) -> Self {
        Self {
            marker: UiMessageToolInputDeltaChunkMarker::Marker,
            tool_call_id: tool_call_id.into(),
            input_text_delta: input_text_delta.into(),
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-input-delta"
    }
}

/// Tool input available chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageToolInputAvailableChunk<INPUT = JSONValue> {
    #[serde(rename = "type")]
    marker: UiMessageToolInputAvailableChunkMarker,
    pub tool_call_id: String,
    pub tool_name: String,
    pub input: INPUT,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_executed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

impl<INPUT> UiMessageToolInputAvailableChunk<INPUT> {
    /// Create a tool input available chunk.
    pub fn new(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        input: INPUT,
    ) -> Self {
        Self {
            marker: UiMessageToolInputAvailableChunkMarker::Marker,
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            input,
            provider_executed: None,
            provider_metadata: None,
            dynamic: None,
            title: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-input-available"
    }
}

/// Tool input error chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageToolInputErrorChunk<INPUT = JSONValue> {
    #[serde(rename = "type")]
    marker: UiMessageToolInputErrorChunkMarker,
    pub tool_call_id: String,
    pub tool_name: String,
    pub input: INPUT,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_executed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
    pub error_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

impl<INPUT> UiMessageToolInputErrorChunk<INPUT> {
    /// Create a tool input error chunk.
    pub fn new(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        input: INPUT,
        error_text: impl Into<String>,
    ) -> Self {
        Self {
            marker: UiMessageToolInputErrorChunkMarker::Marker,
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            input,
            provider_executed: None,
            provider_metadata: None,
            dynamic: None,
            error_text: error_text.into(),
            title: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-input-error"
    }
}

/// Tool approval request chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageToolApprovalRequestChunk {
    #[serde(rename = "type")]
    marker: UiMessageToolApprovalRequestChunkMarker,
    pub approval_id: String,
    pub tool_call_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_automatic: Option<bool>,
}

impl UiMessageToolApprovalRequestChunk {
    /// Create a tool approval request chunk.
    pub fn new(approval_id: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            marker: UiMessageToolApprovalRequestChunkMarker::Marker,
            approval_id: approval_id.into(),
            tool_call_id: tool_call_id.into(),
            is_automatic: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-approval-request"
    }
}

/// Tool approval response chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageToolApprovalResponseChunk {
    #[serde(rename = "type")]
    marker: UiMessageToolApprovalResponseChunkMarker,
    pub approval_id: String,
    pub approved: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_executed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl UiMessageToolApprovalResponseChunk {
    /// Create a tool approval response chunk.
    pub fn new(approval_id: impl Into<String>, approved: bool) -> Self {
        Self {
            marker: UiMessageToolApprovalResponseChunkMarker::Marker,
            approval_id: approval_id.into(),
            approved,
            reason: None,
            provider_executed: None,
            provider_metadata: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-approval-response"
    }
}

/// Tool output available chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageToolOutputAvailableChunk<OUTPUT = JSONValue> {
    #[serde(rename = "type")]
    marker: UiMessageToolOutputAvailableChunkMarker,
    pub tool_call_id: String,
    pub output: OUTPUT,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_executed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preliminary: Option<bool>,
}

impl<OUTPUT> UiMessageToolOutputAvailableChunk<OUTPUT> {
    /// Create a tool output available chunk.
    pub fn new(tool_call_id: impl Into<String>, output: OUTPUT) -> Self {
        Self {
            marker: UiMessageToolOutputAvailableChunkMarker::Marker,
            tool_call_id: tool_call_id.into(),
            output,
            provider_executed: None,
            provider_metadata: None,
            dynamic: None,
            preliminary: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-output-available"
    }
}

/// Tool output error chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageToolOutputErrorChunk {
    #[serde(rename = "type")]
    marker: UiMessageToolOutputErrorChunkMarker,
    pub tool_call_id: String,
    pub error_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_executed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
}

impl UiMessageToolOutputErrorChunk {
    /// Create a tool output error chunk.
    pub fn new(tool_call_id: impl Into<String>, error_text: impl Into<String>) -> Self {
        Self {
            marker: UiMessageToolOutputErrorChunkMarker::Marker,
            tool_call_id: tool_call_id.into(),
            error_text: error_text.into(),
            provider_executed: None,
            provider_metadata: None,
            dynamic: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-output-error"
    }
}

/// Tool output denied chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageToolOutputDeniedChunk {
    #[serde(rename = "type")]
    marker: UiMessageToolOutputDeniedChunkMarker,
    pub tool_call_id: String,
}

impl UiMessageToolOutputDeniedChunk {
    /// Create a tool output denied chunk.
    pub fn new(tool_call_id: impl Into<String>) -> Self {
        Self {
            marker: UiMessageToolOutputDeniedChunkMarker::Marker,
            tool_call_id: tool_call_id.into(),
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-output-denied"
    }
}

/// URL source chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageSourceUrlChunk {
    #[serde(rename = "type")]
    marker: UiMessageSourceUrlChunkMarker,
    pub source_id: String,
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl UiMessageSourceUrlChunk {
    /// Create a URL source chunk.
    pub fn new(source_id: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            marker: UiMessageSourceUrlChunkMarker::Marker,
            source_id: source_id.into(),
            url: url.into(),
            title: None,
            provider_metadata: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "source-url"
    }
}

/// Document source chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageSourceDocumentChunk {
    #[serde(rename = "type")]
    marker: UiMessageSourceDocumentChunkMarker,
    pub source_id: String,
    pub media_type: String,
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl UiMessageSourceDocumentChunk {
    /// Create a document source chunk.
    pub fn new(
        source_id: impl Into<String>,
        media_type: impl Into<String>,
        title: impl Into<String>,
    ) -> Self {
        Self {
            marker: UiMessageSourceDocumentChunkMarker::Marker,
            source_id: source_id.into(),
            media_type: media_type.into(),
            title: title.into(),
            filename: None,
            provider_metadata: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "source-document"
    }
}

/// File chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageFileChunk {
    #[serde(rename = "type")]
    marker: UiMessageFileChunkMarker,
    pub url: String,
    pub media_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl UiMessageFileChunk {
    /// Create a file chunk.
    pub fn new(url: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self {
            marker: UiMessageFileChunkMarker::Marker,
            url: url.into(),
            media_type: media_type.into(),
            provider_metadata: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "file"
    }
}

/// Reasoning file chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageReasoningFileChunk {
    #[serde(rename = "type")]
    marker: UiMessageReasoningFileChunkMarker,
    pub url: String,
    pub media_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl UiMessageReasoningFileChunk {
    /// Create a reasoning file chunk.
    pub fn new(url: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self {
            marker: UiMessageReasoningFileChunkMarker::Marker,
            url: url.into(),
            media_type: media_type.into(),
            provider_metadata: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "reasoning-file"
    }
}

/// Data chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageDataChunk<DATA = JSONValue> {
    /// Full data discriminator, e.g. `data-weather`.
    #[serde(
        rename = "type",
        deserialize_with = "deserialize_ui_message_data_chunk_type"
    )]
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub data: DATA,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transient: Option<bool>,
}

impl<DATA> UiMessageDataChunk<DATA> {
    /// Create a data UI chunk from the suffix after `data-`.
    pub fn new(data_type: impl AsRef<str>, data: DATA) -> Self {
        Self {
            kind: format!("data-{}", data_type.as_ref().trim_start_matches("data-")),
            id: None,
            data,
            transient: None,
        }
    }

    /// Return the full AI SDK UI message chunk discriminator.
    pub fn r#type(&self) -> &str {
        &self.kind
    }

    /// Return the suffix after `data-`, if the discriminator is valid.
    pub fn data_type(&self) -> Option<&str> {
        self.kind.strip_prefix("data-")
    }
}

/// Start-step chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UiMessageStartStepChunk {
    #[serde(rename = "type")]
    marker: UiMessageStartStepChunkMarker,
}

impl UiMessageStartStepChunk {
    /// Create a start-step chunk.
    pub fn new() -> Self {
        Self {
            marker: UiMessageStartStepChunkMarker::Marker,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "start-step"
    }
}

impl Default for UiMessageStartStepChunk {
    fn default() -> Self {
        Self::new()
    }
}

/// Finish-step chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UiMessageFinishStepChunk {
    #[serde(rename = "type")]
    marker: UiMessageFinishStepChunkMarker,
}

impl UiMessageFinishStepChunk {
    /// Create a finish-step chunk.
    pub fn new() -> Self {
        Self {
            marker: UiMessageFinishStepChunkMarker::Marker,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "finish-step"
    }
}

impl Default for UiMessageFinishStepChunk {
    fn default() -> Self {
        Self::new()
    }
}

/// Start chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageStartChunk<METADATA = JSONValue> {
    #[serde(rename = "type")]
    marker: UiMessageStartChunkMarker,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_metadata: Option<METADATA>,
}

impl<METADATA> UiMessageStartChunk<METADATA> {
    /// Create a start chunk.
    pub fn new() -> Self {
        Self {
            marker: UiMessageStartChunkMarker::Marker,
            message_id: None,
            message_metadata: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "start"
    }
}

impl<METADATA> Default for UiMessageStartChunk<METADATA> {
    fn default() -> Self {
        Self::new()
    }
}

/// Finish chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageFinishChunk<METADATA = JSONValue> {
    #[serde(rename = "type")]
    marker: UiMessageFinishChunkMarker,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_metadata: Option<METADATA>,
}

impl<METADATA> UiMessageFinishChunk<METADATA> {
    /// Create a finish chunk.
    pub fn new() -> Self {
        Self {
            marker: UiMessageFinishChunkMarker::Marker,
            finish_reason: None,
            message_metadata: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "finish"
    }
}

impl<METADATA> Default for UiMessageFinishChunk<METADATA> {
    fn default() -> Self {
        Self::new()
    }
}

/// Abort chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UiMessageAbortChunk {
    #[serde(rename = "type")]
    marker: UiMessageAbortChunkMarker,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl UiMessageAbortChunk {
    /// Create an abort chunk.
    pub fn new() -> Self {
        Self {
            marker: UiMessageAbortChunkMarker::Marker,
            reason: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "abort"
    }
}

impl Default for UiMessageAbortChunk {
    fn default() -> Self {
        Self::new()
    }
}

/// Message metadata chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageMetadataChunk<METADATA = JSONValue> {
    #[serde(rename = "type")]
    marker: UiMessageMetadataChunkMarker,
    pub message_metadata: METADATA,
}

impl<METADATA> UiMessageMetadataChunk<METADATA> {
    /// Create a message metadata chunk.
    pub fn new(message_metadata: METADATA) -> Self {
        Self {
            marker: UiMessageMetadataChunkMarker::Marker,
            message_metadata,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "message-metadata"
    }
}

/// AI SDK `UIMessageChunk` union from `ui-message-stream/ui-message-chunks.ts`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum UiMessageChunk<METADATA = JSONValue, DATA = JSONValue> {
    TextStart(UiMessageTextStartChunk),
    TextDelta(UiMessageTextDeltaChunk),
    TextEnd(UiMessageTextEndChunk),
    ReasoningStart(UiMessageReasoningStartChunk),
    ReasoningDelta(UiMessageReasoningDeltaChunk),
    ReasoningEnd(UiMessageReasoningEndChunk),
    Custom(UiMessageCustomChunk),
    Error(UiMessageErrorChunk),
    ToolInputStart(UiMessageToolInputStartChunk),
    ToolInputDelta(UiMessageToolInputDeltaChunk),
    ToolInputAvailable(UiMessageToolInputAvailableChunk<DATA>),
    ToolInputError(UiMessageToolInputErrorChunk<DATA>),
    ToolApprovalRequest(UiMessageToolApprovalRequestChunk),
    ToolApprovalResponse(UiMessageToolApprovalResponseChunk),
    ToolOutputAvailable(UiMessageToolOutputAvailableChunk<DATA>),
    ToolOutputError(UiMessageToolOutputErrorChunk),
    ToolOutputDenied(UiMessageToolOutputDeniedChunk),
    SourceUrl(UiMessageSourceUrlChunk),
    SourceDocument(UiMessageSourceDocumentChunk),
    File(UiMessageFileChunk),
    ReasoningFile(UiMessageReasoningFileChunk),
    Data(UiMessageDataChunk<DATA>),
    StartStep(UiMessageStartStepChunk),
    FinishStep(UiMessageFinishStepChunk),
    Start(UiMessageStartChunk<METADATA>),
    Finish(UiMessageFinishChunk<METADATA>),
    Abort(UiMessageAbortChunk),
    MessageMetadata(UiMessageMetadataChunk<METADATA>),
}

/// AI SDK export spelling for `UIMessageChunk`.
pub type UIMessageChunk<METADATA = JSONValue, DATA = JSONValue> = UiMessageChunk<METADATA, DATA>;

/// AI SDK export spelling for data UI message chunks.
pub type DataUIMessageChunk<DATA = JSONValue> = UiMessageDataChunk<DATA>;

/// AI SDK inferred UI message chunk. Rust exposes the resolved chunk union directly.
pub type InferUIMessageChunk<METADATA = JSONValue, DATA = JSONValue> =
    UiMessageChunk<METADATA, DATA>;

/// Check whether a UI message stream chunk is a `data-*` chunk.
pub fn is_data_ui_message_chunk<METADATA, DATA>(chunk: &UiMessageChunk<METADATA, DATA>) -> bool {
    matches!(chunk, UiMessageChunk::Data(_))
}

impl<METADATA, DATA> UiMessageChunk<METADATA, DATA> {
    /// Return the AI SDK UI message chunk discriminator.
    pub fn r#type(&self) -> &str {
        match self {
            Self::TextStart(part) => part.r#type(),
            Self::TextDelta(part) => part.r#type(),
            Self::TextEnd(part) => part.r#type(),
            Self::ReasoningStart(part) => part.r#type(),
            Self::ReasoningDelta(part) => part.r#type(),
            Self::ReasoningEnd(part) => part.r#type(),
            Self::Custom(part) => part.r#type(),
            Self::Error(part) => part.r#type(),
            Self::ToolInputStart(part) => part.r#type(),
            Self::ToolInputDelta(part) => part.r#type(),
            Self::ToolInputAvailable(part) => part.r#type(),
            Self::ToolInputError(part) => part.r#type(),
            Self::ToolApprovalRequest(part) => part.r#type(),
            Self::ToolApprovalResponse(part) => part.r#type(),
            Self::ToolOutputAvailable(part) => part.r#type(),
            Self::ToolOutputError(part) => part.r#type(),
            Self::ToolOutputDenied(part) => part.r#type(),
            Self::SourceUrl(part) => part.r#type(),
            Self::SourceDocument(part) => part.r#type(),
            Self::File(part) => part.r#type(),
            Self::ReasoningFile(part) => part.r#type(),
            Self::Data(part) => part.r#type(),
            Self::StartStep(part) => part.r#type(),
            Self::FinishStep(part) => part.r#type(),
            Self::Start(part) => part.r#type(),
            Self::Finish(part) => part.r#type(),
            Self::Abort(part) => part.r#type(),
            Self::MessageMetadata(part) => part.r#type(),
        }
    }
}

impl<METADATA, DATA> From<UiMessageTextStartChunk> for UiMessageChunk<METADATA, DATA> {
    fn from(value: UiMessageTextStartChunk) -> Self {
        Self::TextStart(value)
    }
}

impl<METADATA, DATA> From<UiMessageTextDeltaChunk> for UiMessageChunk<METADATA, DATA> {
    fn from(value: UiMessageTextDeltaChunk) -> Self {
        Self::TextDelta(value)
    }
}

impl<METADATA, DATA> From<UiMessageDataChunk<DATA>> for UiMessageChunk<METADATA, DATA> {
    fn from(value: UiMessageDataChunk<DATA>) -> Self {
        Self::Data(value)
    }
}

impl<METADATA, DATA> From<UiMessageFinishChunk<METADATA>> for UiMessageChunk<METADATA, DATA> {
    fn from(value: UiMessageFinishChunk<METADATA>) -> Self {
        Self::Finish(value)
    }
}
