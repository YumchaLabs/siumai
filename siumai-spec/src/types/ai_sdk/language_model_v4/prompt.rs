use crate::types::{
    AssistantContent, AssistantContentPart, AssistantModelMessage, CustomPart, FilePart, ImagePart,
    ModelMessage, ProviderOptionsMap, ProviderReference, ReasoningFilePart, ReasoningPart,
    SystemModelMessage, TextPart, ToolApprovalResponse, ToolCallPart, ToolContentPart,
    ToolModelMessage, ToolResultContentPart, ToolResultFileId, ToolResultOutput, ToolResultPart,
    UserContent, UserContentPart, UserModelMessage,
};
use serde::{Deserialize, Serialize};

use super::super::JSONValue;
use super::shared::*;

/// AI SDK V4 model prompt.
pub type LanguageModelV4Prompt = Vec<LanguageModelV4Message>;

/// AI SDK V4 prompt text part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LanguageModelV4TextPart {
    #[serde(rename = "type", default)]
    marker: LanguageModelV4TextMarker,
    /// Text content.
    pub text: String,
    /// Provider-specific request options.
    #[serde(
        rename = "providerOptions",
        alias = "provider_options",
        default,
        skip_serializing_if = "ProviderOptionsMap::is_empty",
        deserialize_with = "deserialize_language_model_v4_provider_options_map",
        serialize_with = "serialize_language_model_v4_provider_options_map"
    )]
    pub provider_options: ProviderOptionsMap,
}

impl LanguageModelV4TextPart {
    /// Create a V4 text prompt part.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            marker: LanguageModelV4TextMarker::Marker,
            text: text.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Project a stable text prompt part onto the V4 provider prompt shape.
    pub fn from_text_part(part: &TextPart) -> Self {
        Self {
            marker: LanguageModelV4TextMarker::Marker,
            text: part.text.clone(),
            provider_options: language_model_v4_provider_options_from_stable(
                &part.provider_options,
            ),
        }
    }

    /// Attach provider-specific request options.
    pub fn with_provider_options_map(mut self, provider_options: ProviderOptionsMap) -> Self {
        self.provider_options = provider_options;
        self
    }

    /// Return the AI SDK V4 prompt part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "text"
    }
}

/// AI SDK V4 prompt reasoning part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LanguageModelV4ReasoningPart {
    #[serde(rename = "type", default)]
    marker: LanguageModelV4ReasoningMarker,
    /// Reasoning text.
    pub text: String,
    /// Provider-specific request options.
    #[serde(
        rename = "providerOptions",
        alias = "provider_options",
        default,
        skip_serializing_if = "ProviderOptionsMap::is_empty",
        deserialize_with = "deserialize_language_model_v4_provider_options_map",
        serialize_with = "serialize_language_model_v4_provider_options_map"
    )]
    pub provider_options: ProviderOptionsMap,
}

impl LanguageModelV4ReasoningPart {
    /// Create a V4 reasoning prompt part.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            marker: LanguageModelV4ReasoningMarker::Marker,
            text: text.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Project a stable reasoning prompt part onto the V4 provider prompt shape.
    pub fn from_reasoning_part(part: &ReasoningPart) -> Self {
        Self {
            marker: LanguageModelV4ReasoningMarker::Marker,
            text: part.text.clone(),
            provider_options: language_model_v4_provider_options_from_stable(
                &part.provider_options,
            ),
        }
    }

    /// Attach provider-specific request options.
    pub fn with_provider_options_map(mut self, provider_options: ProviderOptionsMap) -> Self {
        self.provider_options = provider_options;
        self
    }

    /// Return the AI SDK V4 prompt part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "reasoning"
    }
}

/// AI SDK V4 prompt custom part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LanguageModelV4CustomPart {
    #[serde(rename = "type", default)]
    marker: LanguageModelV4CustomMarker,
    /// Provider-specific custom content kind, in `{provider}.{provider-type}` format.
    #[serde(
        deserialize_with = "deserialize_language_model_v4_custom_kind",
        serialize_with = "serialize_language_model_v4_custom_kind"
    )]
    pub kind: String,
    /// Provider-specific request options.
    #[serde(
        rename = "providerOptions",
        alias = "provider_options",
        default,
        skip_serializing_if = "ProviderOptionsMap::is_empty",
        deserialize_with = "deserialize_language_model_v4_provider_options_map",
        serialize_with = "serialize_language_model_v4_provider_options_map"
    )]
    pub provider_options: ProviderOptionsMap,
}

impl LanguageModelV4CustomPart {
    /// Create a V4 custom prompt part.
    ///
    /// Use `try_new` when the kind comes from untrusted input and should be checked before
    /// serialization.
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            marker: LanguageModelV4CustomMarker::Marker,
            kind: kind.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Create a V4 custom prompt part and validate the AI SDK kind format.
    pub fn try_new(kind: impl Into<String>) -> Option<Self> {
        let part = Self::new(kind);
        is_language_model_v4_custom_kind(&part.kind).then_some(part)
    }

    /// Project a stable custom prompt part onto the V4 provider prompt shape.
    pub fn try_from_custom_part(part: &CustomPart) -> Option<Self> {
        Some(Self {
            marker: LanguageModelV4CustomMarker::Marker,
            kind: is_language_model_v4_custom_kind(&part.kind).then(|| part.kind.clone())?,
            provider_options: language_model_v4_provider_options_from_stable(
                &part.provider_options,
            ),
        })
    }

    /// Attach provider-specific request options.
    pub fn with_provider_options_map(mut self, provider_options: ProviderOptionsMap) -> Self {
        self.provider_options = provider_options;
        self
    }

    /// Return the AI SDK V4 prompt part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "custom"
    }
}

/// AI SDK V4 prompt tool-call part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV4ToolCallPart {
    #[serde(rename = "type", default)]
    marker: LanguageModelV4ToolCallMarker,
    /// Tool-call id.
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    /// Tool name.
    #[serde(rename = "toolName")]
    pub tool_name: String,
    /// JSON-serializable tool input.
    pub input: JSONValue,
    /// Whether this tool call will be executed by the provider.
    #[serde(
        rename = "providerExecuted",
        alias = "provider_executed",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_executed: Option<bool>,
    /// Provider-specific request options.
    #[serde(
        rename = "providerOptions",
        alias = "provider_options",
        default,
        skip_serializing_if = "ProviderOptionsMap::is_empty",
        deserialize_with = "deserialize_language_model_v4_provider_options_map",
        serialize_with = "serialize_language_model_v4_provider_options_map"
    )]
    pub provider_options: ProviderOptionsMap,
}

impl LanguageModelV4ToolCallPart {
    /// Create a V4 prompt tool-call part.
    pub fn new(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        input: impl Into<JSONValue>,
    ) -> Self {
        Self {
            marker: LanguageModelV4ToolCallMarker::Marker,
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            input: input.into(),
            provider_executed: None,
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Project a stable tool-call prompt part onto the V4 provider prompt shape.
    pub fn from_tool_call_part(part: &ToolCallPart) -> Self {
        Self {
            marker: LanguageModelV4ToolCallMarker::Marker,
            tool_call_id: part.tool_call_id.clone(),
            tool_name: part.tool_name.clone(),
            input: part.input.clone(),
            provider_executed: part.provider_executed,
            provider_options: language_model_v4_provider_options_from_stable(
                &part.provider_options,
            ),
        }
    }

    /// Mark whether this tool call will be executed by the provider.
    pub const fn with_provider_executed(mut self, provider_executed: bool) -> Self {
        self.provider_executed = Some(provider_executed);
        self
    }

    /// Attach provider-specific request options.
    pub fn with_provider_options_map(mut self, provider_options: ProviderOptionsMap) -> Self {
        self.provider_options = provider_options;
        self
    }

    /// Return the AI SDK V4 prompt part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-call"
    }
}

/// AI SDK V4 prompt tool-result output.
///
/// This is narrower than the stable `ToolResultOutput` compatibility type. AI SDK V4 only accepts
/// canonical content parts inside `type: "content"` outputs, so legacy stable parts are projected
/// before they reach provider-facing prompts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum LanguageModelV4ToolResultOutput {
    /// Text tool output that should be sent directly to the API.
    Text {
        value: String,
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty",
            deserialize_with = "deserialize_language_model_v4_provider_options_map",
            serialize_with = "serialize_language_model_v4_provider_options_map"
        )]
        provider_options: ProviderOptionsMap,
    },
    /// JSON tool output.
    Json {
        value: JSONValue,
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty",
            deserialize_with = "deserialize_language_model_v4_provider_options_map",
            serialize_with = "serialize_language_model_v4_provider_options_map"
        )]
        provider_options: ProviderOptionsMap,
    },
    /// Output used when execution was denied.
    ExecutionDenied {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty",
            deserialize_with = "deserialize_language_model_v4_provider_options_map",
            serialize_with = "serialize_language_model_v4_provider_options_map"
        )]
        provider_options: ProviderOptionsMap,
    },
    /// Text error output.
    ErrorText {
        value: String,
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty",
            deserialize_with = "deserialize_language_model_v4_provider_options_map",
            serialize_with = "serialize_language_model_v4_provider_options_map"
        )]
        provider_options: ProviderOptionsMap,
    },
    /// JSON error output.
    ErrorJson {
        value: JSONValue,
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty",
            deserialize_with = "deserialize_language_model_v4_provider_options_map",
            serialize_with = "serialize_language_model_v4_provider_options_map"
        )]
        provider_options: ProviderOptionsMap,
    },
    /// Multimodal content output using only AI SDK V4 canonical content variants.
    Content {
        value: Vec<LanguageModelV4ToolResultContentPart>,
    },
}

impl LanguageModelV4ToolResultOutput {
    /// Create a V4 text tool output.
    pub fn text(value: impl Into<String>) -> Self {
        Self::Text {
            value: value.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Create a V4 JSON tool output.
    pub fn json(value: impl Into<JSONValue>) -> Self {
        Self::Json {
            value: value.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Create a V4 execution-denied tool output.
    pub fn execution_denied(reason: Option<String>) -> Self {
        Self::ExecutionDenied {
            reason,
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Create a V4 text error tool output.
    pub fn error_text(value: impl Into<String>) -> Self {
        Self::ErrorText {
            value: value.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Create a V4 JSON error tool output.
    pub fn error_json(value: impl Into<JSONValue>) -> Self {
        Self::ErrorJson {
            value: value.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Create a V4 content tool output.
    pub fn content(value: Vec<LanguageModelV4ToolResultContentPart>) -> Self {
        Self::Content { value }
    }

    /// Project a stable tool-result output onto the AI SDK V4 prompt shape.
    pub fn from_tool_result_output(output: &ToolResultOutput) -> Self {
        match output {
            ToolResultOutput::Text {
                value,
                provider_options,
            } => Self::Text {
                value: value.clone(),
                provider_options: language_model_v4_provider_options_from_stable(provider_options),
            },
            ToolResultOutput::Json {
                value,
                provider_options,
            } => Self::Json {
                value: value.clone(),
                provider_options: language_model_v4_provider_options_from_stable(provider_options),
            },
            ToolResultOutput::ExecutionDenied {
                reason,
                provider_options,
            } => Self::ExecutionDenied {
                reason: reason.clone(),
                provider_options: language_model_v4_provider_options_from_stable(provider_options),
            },
            ToolResultOutput::ErrorText {
                value,
                provider_options,
            } => Self::ErrorText {
                value: value.clone(),
                provider_options: language_model_v4_provider_options_from_stable(provider_options),
            },
            ToolResultOutput::ErrorJson {
                value,
                provider_options,
            } => Self::ErrorJson {
                value: value.clone(),
                provider_options: language_model_v4_provider_options_from_stable(provider_options),
            },
            ToolResultOutput::Content {
                value,
                provider_options,
            } => {
                let provider_options =
                    language_model_v4_provider_options_from_stable(provider_options);
                if !provider_options.is_empty() {
                    return Self::Json {
                        value: output.to_json_value(),
                        provider_options,
                    };
                }

                let mut projected = Vec::with_capacity(value.len());
                for part in value {
                    let Some(part) =
                        LanguageModelV4ToolResultContentPart::from_stable_content_part(part)
                    else {
                        return Self::Json {
                            value: output.to_json_value(),
                            provider_options: ProviderOptionsMap::default(),
                        };
                    };
                    projected.push(part);
                }

                Self::Content { value: projected }
            }
        }
    }

    /// Get provider options for variants that support output-level provider options.
    pub fn provider_options(&self) -> Option<&ProviderOptionsMap> {
        match self {
            Self::Text {
                provider_options, ..
            }
            | Self::Json {
                provider_options, ..
            }
            | Self::ExecutionDenied {
                provider_options, ..
            }
            | Self::ErrorText {
                provider_options, ..
            }
            | Self::ErrorJson {
                provider_options, ..
            } => Some(provider_options),
            Self::Content { .. } => None,
        }
    }
}

impl From<&ToolResultOutput> for LanguageModelV4ToolResultOutput {
    fn from(value: &ToolResultOutput) -> Self {
        Self::from_tool_result_output(value)
    }
}

impl From<ToolResultOutput> for LanguageModelV4ToolResultOutput {
    fn from(value: ToolResultOutput) -> Self {
        Self::from(&value)
    }
}

/// AI SDK V4 prompt tool-result content part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum LanguageModelV4ToolResultContentPart {
    /// Text content.
    Text {
        text: String,
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty",
            deserialize_with = "deserialize_language_model_v4_provider_options_map",
            serialize_with = "serialize_language_model_v4_provider_options_map"
        )]
        provider_options: ProviderOptionsMap,
    },
    /// File content encoded inline as base64 data.
    FileData {
        data: String,
        #[serde(rename = "mediaType", alias = "media_type")]
        media_type: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty",
            deserialize_with = "deserialize_language_model_v4_provider_options_map",
            serialize_with = "serialize_language_model_v4_provider_options_map"
        )]
        provider_options: ProviderOptionsMap,
    },
    /// File content referenced by URL.
    FileUrl {
        url: String,
        #[serde(rename = "mediaType", alias = "media_type")]
        media_type: String,
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty",
            deserialize_with = "deserialize_language_model_v4_provider_options_map",
            serialize_with = "serialize_language_model_v4_provider_options_map"
        )]
        provider_options: ProviderOptionsMap,
    },
    /// File content referenced by provider-owned file reference.
    FileReference {
        #[serde(rename = "providerReference", alias = "provider_reference")]
        provider_reference: ProviderReference,
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty",
            deserialize_with = "deserialize_language_model_v4_provider_options_map",
            serialize_with = "serialize_language_model_v4_provider_options_map"
        )]
        provider_options: ProviderOptionsMap,
    },
    /// Provider-specific custom content.
    Custom {
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty",
            deserialize_with = "deserialize_language_model_v4_provider_options_map",
            serialize_with = "serialize_language_model_v4_provider_options_map"
        )]
        provider_options: ProviderOptionsMap,
    },
}

impl LanguageModelV4ToolResultContentPart {
    /// Create a V4 text tool-result content part.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Create a V4 file-data tool-result content part.
    pub fn file_data(
        data: impl Into<String>,
        media_type: impl Into<String>,
        filename: Option<String>,
    ) -> Self {
        Self::FileData {
            data: data.into(),
            media_type: media_type.into(),
            filename,
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Create a V4 file-url tool-result content part.
    pub fn file_url(url: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self::FileUrl {
            url: url.into(),
            media_type: media_type.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Create a V4 file-reference tool-result content part.
    pub fn file_reference(provider_reference: impl Into<ProviderReference>) -> Self {
        Self::FileReference {
            provider_reference: provider_reference.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Create a V4 custom tool-result content part.
    pub fn custom() -> Self {
        Self::Custom {
            provider_options: ProviderOptionsMap::default(),
        }
    }

    fn from_stable_content_part(part: &ToolResultContentPart) -> Option<Self> {
        match part {
            ToolResultContentPart::Text {
                text,
                provider_options,
            } => Some(Self::Text {
                text: text.clone(),
                provider_options: language_model_v4_provider_options_from_stable(provider_options),
            }),
            ToolResultContentPart::FileData {
                data,
                media_type,
                filename,
                provider_options,
            } => Some(Self::FileData {
                data: data.clone(),
                media_type: media_type.clone(),
                filename: filename.clone(),
                provider_options: language_model_v4_provider_options_from_stable(provider_options),
            }),
            ToolResultContentPart::ImageData {
                data,
                media_type,
                provider_options,
            } => Some(Self::FileData {
                data: data.clone(),
                media_type: media_type.clone(),
                filename: None,
                provider_options: language_model_v4_provider_options_from_stable(provider_options),
            }),
            ToolResultContentPart::FileUrl {
                url,
                media_type,
                provider_options,
            } => media_type.as_ref().map(|media_type| Self::FileUrl {
                url: url.clone(),
                media_type: media_type.clone(),
                provider_options: language_model_v4_provider_options_from_stable(provider_options),
            }),
            ToolResultContentPart::ImageUrl {
                url,
                provider_options,
            } => Some(Self::FileUrl {
                url: url.clone(),
                media_type: "image/*".to_string(),
                provider_options: language_model_v4_provider_options_from_stable(provider_options),
            }),
            ToolResultContentPart::FileReference {
                provider_reference,
                provider_options,
            }
            | ToolResultContentPart::ImageFileReference {
                provider_reference,
                provider_options,
            } => Some(Self::FileReference {
                provider_reference: provider_reference.clone(),
                provider_options: language_model_v4_provider_options_from_stable(provider_options),
            }),
            ToolResultContentPart::FileId {
                file_id,
                provider_options,
            }
            | ToolResultContentPart::ImageFileId {
                file_id,
                provider_options,
            } => Some(Self::FileReference {
                provider_reference: language_model_v4_provider_reference_from_file_id(file_id)?,
                provider_options: language_model_v4_provider_options_from_stable(provider_options),
            }),
            ToolResultContentPart::Custom { provider_options } => Some(Self::Custom {
                provider_options: language_model_v4_provider_options_from_stable(provider_options),
            }),
        }
    }

    /// Get provider options for this content part.
    pub fn provider_options(&self) -> &ProviderOptionsMap {
        match self {
            Self::Text {
                provider_options, ..
            }
            | Self::FileData {
                provider_options, ..
            }
            | Self::FileUrl {
                provider_options, ..
            }
            | Self::FileReference {
                provider_options, ..
            }
            | Self::Custom {
                provider_options, ..
            } => provider_options,
        }
    }
}

fn language_model_v4_provider_reference_from_file_id(
    file_id: &ToolResultFileId,
) -> Option<ProviderReference> {
    match file_id {
        ToolResultFileId::Single(_) => None,
        ToolResultFileId::PerProvider(values) => Some(ProviderReference::from(values.clone())),
    }
}

/// AI SDK V4 prompt tool-result part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV4ToolResultPart {
    #[serde(rename = "type", default)]
    marker: LanguageModelV4ToolResultMarker,
    /// Tool-call id associated with this result.
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    /// Tool name.
    #[serde(rename = "toolName")]
    pub tool_name: String,
    /// Tool result output.
    pub output: LanguageModelV4ToolResultOutput,
    /// Provider-specific request options.
    #[serde(
        rename = "providerOptions",
        alias = "provider_options",
        default,
        skip_serializing_if = "ProviderOptionsMap::is_empty",
        deserialize_with = "deserialize_language_model_v4_provider_options_map",
        serialize_with = "serialize_language_model_v4_provider_options_map"
    )]
    pub provider_options: ProviderOptionsMap,
}

impl LanguageModelV4ToolResultPart {
    /// Create a V4 tool-result prompt part.
    pub fn new(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        output: impl Into<LanguageModelV4ToolResultOutput>,
    ) -> Self {
        Self {
            marker: LanguageModelV4ToolResultMarker::Marker,
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            output: output.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Project a stable tool-result prompt part onto the AI SDK V4 provider prompt shape.
    pub fn from_tool_result_part(part: &ToolResultPart) -> Self {
        Self {
            marker: LanguageModelV4ToolResultMarker::Marker,
            tool_call_id: part.tool_call_id.clone(),
            tool_name: part.tool_name.clone(),
            output: LanguageModelV4ToolResultOutput::from_tool_result_output(&part.output),
            provider_options: language_model_v4_provider_options_from_stable(
                &part.provider_options,
            ),
        }
    }

    /// Attach provider-specific request options.
    pub fn with_provider_options_map(mut self, provider_options: ProviderOptionsMap) -> Self {
        self.provider_options = provider_options;
        self
    }

    /// Return the AI SDK V4 prompt part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-result"
    }
}

impl From<&ToolResultPart> for LanguageModelV4ToolResultPart {
    fn from(value: &ToolResultPart) -> Self {
        Self::from_tool_result_part(value)
    }
}

impl From<ToolResultPart> for LanguageModelV4ToolResultPart {
    fn from(value: ToolResultPart) -> Self {
        Self::from(&value)
    }
}

/// AI SDK V4 prompt file part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LanguageModelV4FilePart {
    #[serde(rename = "type", default)]
    marker: LanguageModelV4FileMarker,
    /// Optional filename of the file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    /// File data or provider reference.
    pub data: LanguageModelV4FilePartData,
    /// IANA media type of the file.
    #[serde(rename = "mediaType", alias = "media_type")]
    pub media_type: String,
    /// Provider-specific request options.
    #[serde(
        rename = "providerOptions",
        alias = "provider_options",
        default,
        skip_serializing_if = "ProviderOptionsMap::is_empty",
        deserialize_with = "deserialize_language_model_v4_provider_options_map",
        serialize_with = "serialize_language_model_v4_provider_options_map"
    )]
    pub provider_options: ProviderOptionsMap,
}

impl LanguageModelV4FilePart {
    /// Create a V4 file prompt part.
    pub fn new(
        data: impl Into<LanguageModelV4FilePartData>,
        media_type: impl Into<String>,
    ) -> Self {
        Self {
            marker: LanguageModelV4FileMarker::Marker,
            filename: None,
            data: data.into(),
            media_type: media_type.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Project a stable file prompt part onto the V4 provider prompt shape.
    pub fn from_file_part(part: &FilePart) -> Self {
        Self {
            marker: LanguageModelV4FileMarker::Marker,
            filename: part.filename.clone(),
            data: LanguageModelV4FilePartData::from(&part.data),
            media_type: part.media_type.clone(),
            provider_options: language_model_v4_provider_options_from_stable(
                &part.provider_options,
            ),
        }
    }

    /// Project an image prompt part onto the V4 provider prompt file shape.
    pub fn from_image_part(part: &ImagePart) -> Self {
        Self {
            marker: LanguageModelV4FileMarker::Marker,
            filename: None,
            data: LanguageModelV4FilePartData::from(&part.image),
            media_type: part
                .media_type
                .clone()
                .unwrap_or_else(|| "image/*".to_string()),
            provider_options: language_model_v4_provider_options_from_stable(
                &part.provider_options,
            ),
        }
    }

    /// Attach an optional filename.
    pub fn with_filename(mut self, filename: impl Into<String>) -> Self {
        self.filename = Some(filename.into());
        self
    }

    /// Attach provider-specific request options.
    pub fn with_provider_options_map(mut self, provider_options: ProviderOptionsMap) -> Self {
        self.provider_options = provider_options;
        self
    }

    /// Return the AI SDK V4 prompt part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "file"
    }
}

/// AI SDK V4 prompt reasoning-file part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LanguageModelV4ReasoningFilePart {
    #[serde(rename = "type", default)]
    marker: LanguageModelV4ReasoningFileMarker,
    /// Reasoning-file data.
    pub data: LanguageModelV4DataContent,
    /// IANA media type of the file.
    #[serde(rename = "mediaType", alias = "media_type")]
    pub media_type: String,
    /// Provider-specific request options.
    #[serde(
        rename = "providerOptions",
        alias = "provider_options",
        default,
        skip_serializing_if = "ProviderOptionsMap::is_empty",
        deserialize_with = "deserialize_language_model_v4_provider_options_map",
        serialize_with = "serialize_language_model_v4_provider_options_map"
    )]
    pub provider_options: ProviderOptionsMap,
}

impl LanguageModelV4ReasoningFilePart {
    /// Create a V4 reasoning-file prompt part.
    pub fn new(data: impl Into<LanguageModelV4DataContent>, media_type: impl Into<String>) -> Self {
        Self {
            marker: LanguageModelV4ReasoningFileMarker::Marker,
            data: data.into(),
            media_type: media_type.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Project a stable reasoning-file prompt part onto the V4 provider prompt shape.
    pub fn from_reasoning_file_part(part: &ReasoningFilePart) -> Self {
        Self {
            marker: LanguageModelV4ReasoningFileMarker::Marker,
            data: language_model_v4_data_from_media(&part.data),
            media_type: part.media_type.clone(),
            provider_options: language_model_v4_provider_options_from_stable(
                &part.provider_options,
            ),
        }
    }

    /// Attach provider-specific request options.
    pub fn with_provider_options_map(mut self, provider_options: ProviderOptionsMap) -> Self {
        self.provider_options = provider_options;
        self
    }

    /// Return the AI SDK V4 prompt part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "reasoning-file"
    }
}

/// AI SDK V4 prompt tool-approval-response part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LanguageModelV4ToolApprovalResponsePart {
    #[serde(rename = "type", default)]
    marker: LanguageModelV4ToolApprovalResponseMarker,
    /// Approval request id.
    #[serde(rename = "approvalId")]
    pub approval_id: String,
    /// Whether the approval was granted.
    pub approved: bool,
    /// Optional reason for approval or denial.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Provider-specific request options.
    #[serde(
        rename = "providerOptions",
        alias = "provider_options",
        default,
        skip_serializing_if = "ProviderOptionsMap::is_empty",
        deserialize_with = "deserialize_language_model_v4_provider_options_map",
        serialize_with = "serialize_language_model_v4_provider_options_map"
    )]
    pub provider_options: ProviderOptionsMap,
}

impl LanguageModelV4ToolApprovalResponsePart {
    /// Create a V4 tool approval response prompt part.
    pub fn new(approval_id: impl Into<String>, approved: bool) -> Self {
        Self {
            marker: LanguageModelV4ToolApprovalResponseMarker::Marker,
            approval_id: approval_id.into(),
            approved,
            reason: None,
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Project a stable tool approval response onto the V4 provider prompt shape.
    pub fn from_tool_approval_response(part: &ToolApprovalResponse) -> Self {
        Self {
            marker: LanguageModelV4ToolApprovalResponseMarker::Marker,
            approval_id: part.approval_id.clone(),
            approved: part.approved,
            reason: part.reason.clone(),
            provider_options: language_model_v4_provider_options_from_stable(
                &part.provider_options,
            ),
        }
    }

    /// Attach an optional reason.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Attach provider-specific request options.
    pub fn with_provider_options_map(mut self, provider_options: ProviderOptionsMap) -> Self {
        self.provider_options = provider_options;
        self
    }

    /// Return the AI SDK V4 prompt part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-approval-response"
    }
}

/// AI SDK V4 user prompt content part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum LanguageModelV4UserContentPart {
    Text(LanguageModelV4TextPart),
    File(LanguageModelV4FilePart),
}

impl LanguageModelV4UserContentPart {
    /// Return the AI SDK V4 prompt part discriminator.
    pub fn r#type(&self) -> &'static str {
        match self {
            Self::Text(_) => "text",
            Self::File(part) => part.r#type(),
        }
    }
}

/// AI SDK V4 assistant prompt content part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum LanguageModelV4AssistantContentPart {
    Text(LanguageModelV4TextPart),
    File(LanguageModelV4FilePart),
    Custom(LanguageModelV4CustomPart),
    Reasoning(LanguageModelV4ReasoningPart),
    ReasoningFile(LanguageModelV4ReasoningFilePart),
    ToolCall(LanguageModelV4ToolCallPart),
    ToolResult(LanguageModelV4ToolResultPart),
}

impl LanguageModelV4AssistantContentPart {
    /// Return the AI SDK V4 prompt part discriminator.
    pub fn r#type(&self) -> &'static str {
        match self {
            Self::Text(_) => "text",
            Self::File(part) => part.r#type(),
            Self::Custom(_) => "custom",
            Self::Reasoning(_) => "reasoning",
            Self::ReasoningFile(part) => part.r#type(),
            Self::ToolCall(_) => "tool-call",
            Self::ToolResult(_) => "tool-result",
        }
    }
}

/// AI SDK V4 tool prompt content part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum LanguageModelV4ToolContentPart {
    ToolResult(LanguageModelV4ToolResultPart),
    ToolApprovalResponse(LanguageModelV4ToolApprovalResponsePart),
}

impl LanguageModelV4ToolContentPart {
    /// Return the AI SDK V4 prompt part discriminator.
    pub fn r#type(&self) -> &'static str {
        match self {
            Self::ToolResult(_) => "tool-result",
            Self::ToolApprovalResponse(part) => part.r#type(),
        }
    }
}

/// AI SDK V4 system prompt message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LanguageModelV4SystemMessage {
    #[serde(default)]
    role: LanguageModelV4SystemRoleMarker,
    /// System content.
    pub content: String,
    /// Provider-specific request options.
    #[serde(
        rename = "providerOptions",
        alias = "provider_options",
        default,
        skip_serializing_if = "ProviderOptionsMap::is_empty",
        deserialize_with = "deserialize_language_model_v4_provider_options_map",
        serialize_with = "serialize_language_model_v4_provider_options_map"
    )]
    pub provider_options: ProviderOptionsMap,
}

impl LanguageModelV4SystemMessage {
    /// Create a V4 system prompt message.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            role: LanguageModelV4SystemRoleMarker::Marker,
            content: content.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Project a stable system model message onto the V4 provider prompt shape.
    pub fn from_system_message(message: &SystemModelMessage) -> Self {
        Self {
            role: LanguageModelV4SystemRoleMarker::Marker,
            content: message.content.clone(),
            provider_options: language_model_v4_provider_options_from_stable(
                &message.provider_options,
            ),
        }
    }

    /// Attach provider-specific request options.
    pub fn with_provider_options_map(mut self, provider_options: ProviderOptionsMap) -> Self {
        self.provider_options = provider_options;
        self
    }
}

/// AI SDK V4 user prompt message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV4UserMessage {
    #[serde(default)]
    role: LanguageModelV4UserRoleMarker,
    /// User content parts.
    pub content: Vec<LanguageModelV4UserContentPart>,
    /// Provider-specific request options.
    #[serde(
        rename = "providerOptions",
        alias = "provider_options",
        default,
        skip_serializing_if = "ProviderOptionsMap::is_empty",
        deserialize_with = "deserialize_language_model_v4_provider_options_map",
        serialize_with = "serialize_language_model_v4_provider_options_map"
    )]
    pub provider_options: ProviderOptionsMap,
}

impl LanguageModelV4UserMessage {
    /// Create a V4 user prompt message.
    pub fn new(content: Vec<LanguageModelV4UserContentPart>) -> Self {
        Self {
            role: LanguageModelV4UserRoleMarker::Marker,
            content,
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Project a stable user model message onto the V4 provider prompt shape.
    pub fn from_user_message(message: &UserModelMessage) -> Self {
        let content = match &message.content {
            UserContent::Text(text) => {
                vec![LanguageModelV4UserContentPart::Text(
                    LanguageModelV4TextPart::new(text.clone()),
                )]
            }
            UserContent::Parts(parts) => parts
                .iter()
                .filter_map(|part| match part {
                    UserContentPart::Text(part) if part.text.is_empty() => None,
                    UserContentPart::Text(part) => Some(LanguageModelV4UserContentPart::Text(
                        LanguageModelV4TextPart::from_text_part(part),
                    )),
                    UserContentPart::Image(part) => Some(LanguageModelV4UserContentPart::File(
                        LanguageModelV4FilePart::from_image_part(part),
                    )),
                    UserContentPart::File(part) => Some(LanguageModelV4UserContentPart::File(
                        LanguageModelV4FilePart::from_file_part(part),
                    )),
                })
                .collect(),
        };

        Self {
            role: LanguageModelV4UserRoleMarker::Marker,
            content,
            provider_options: language_model_v4_provider_options_from_stable(
                &message.provider_options,
            ),
        }
    }

    /// Attach provider-specific request options.
    pub fn with_provider_options_map(mut self, provider_options: ProviderOptionsMap) -> Self {
        self.provider_options = provider_options;
        self
    }
}

/// AI SDK V4 assistant prompt message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV4AssistantMessage {
    #[serde(default)]
    role: LanguageModelV4AssistantRoleMarker,
    /// Assistant content parts.
    pub content: Vec<LanguageModelV4AssistantContentPart>,
    /// Provider-specific request options.
    #[serde(
        rename = "providerOptions",
        alias = "provider_options",
        default,
        skip_serializing_if = "ProviderOptionsMap::is_empty",
        deserialize_with = "deserialize_language_model_v4_provider_options_map",
        serialize_with = "serialize_language_model_v4_provider_options_map"
    )]
    pub provider_options: ProviderOptionsMap,
}

impl LanguageModelV4AssistantMessage {
    /// Create a V4 assistant prompt message.
    pub fn new(content: Vec<LanguageModelV4AssistantContentPart>) -> Self {
        Self {
            role: LanguageModelV4AssistantRoleMarker::Marker,
            content,
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Project a stable assistant model message onto the V4 provider prompt shape.
    pub fn from_assistant_message(message: &AssistantModelMessage) -> Self {
        let content = match &message.content {
            AssistantContent::Text(text) => {
                vec![LanguageModelV4AssistantContentPart::Text(
                    LanguageModelV4TextPart::new(text.clone()),
                )]
            }
            AssistantContent::Parts(parts) => parts
                .iter()
                .filter_map(|part| match part {
                    AssistantContentPart::Text(part)
                        if part.text.is_empty() && part.provider_options.is_empty() =>
                    {
                        None
                    }
                    AssistantContentPart::Text(part) => {
                        Some(LanguageModelV4AssistantContentPart::Text(
                            LanguageModelV4TextPart::from_text_part(part),
                        ))
                    }
                    AssistantContentPart::Custom(part) => {
                        LanguageModelV4CustomPart::try_from_custom_part(part)
                            .map(LanguageModelV4AssistantContentPart::Custom)
                    }
                    AssistantContentPart::File(part) => {
                        Some(LanguageModelV4AssistantContentPart::File(
                            LanguageModelV4FilePart::from_file_part(part),
                        ))
                    }
                    AssistantContentPart::Reasoning(part) => {
                        Some(LanguageModelV4AssistantContentPart::Reasoning(
                            LanguageModelV4ReasoningPart::from_reasoning_part(part),
                        ))
                    }
                    AssistantContentPart::ReasoningFile(part) => {
                        Some(LanguageModelV4AssistantContentPart::ReasoningFile(
                            LanguageModelV4ReasoningFilePart::from_reasoning_file_part(part),
                        ))
                    }
                    AssistantContentPart::ToolCall(part) => {
                        Some(LanguageModelV4AssistantContentPart::ToolCall(
                            LanguageModelV4ToolCallPart::from_tool_call_part(part),
                        ))
                    }
                    AssistantContentPart::ToolResult(part) => {
                        Some(LanguageModelV4AssistantContentPart::ToolResult(
                            LanguageModelV4ToolResultPart::from_tool_result_part(part),
                        ))
                    }
                    AssistantContentPart::ToolApprovalRequest(_) => None,
                })
                .collect(),
        };

        Self {
            role: LanguageModelV4AssistantRoleMarker::Marker,
            content,
            provider_options: language_model_v4_provider_options_from_stable(
                &message.provider_options,
            ),
        }
    }

    /// Attach provider-specific request options.
    pub fn with_provider_options_map(mut self, provider_options: ProviderOptionsMap) -> Self {
        self.provider_options = provider_options;
        self
    }
}

/// AI SDK V4 tool prompt message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV4ToolMessage {
    #[serde(default)]
    role: LanguageModelV4ToolRoleMarker,
    /// Tool content parts.
    pub content: Vec<LanguageModelV4ToolContentPart>,
    /// Provider-specific request options.
    #[serde(
        rename = "providerOptions",
        alias = "provider_options",
        default,
        skip_serializing_if = "ProviderOptionsMap::is_empty",
        deserialize_with = "deserialize_language_model_v4_provider_options_map",
        serialize_with = "serialize_language_model_v4_provider_options_map"
    )]
    pub provider_options: ProviderOptionsMap,
}

impl LanguageModelV4ToolMessage {
    /// Create a V4 tool prompt message.
    pub fn new(content: Vec<LanguageModelV4ToolContentPart>) -> Self {
        Self {
            role: LanguageModelV4ToolRoleMarker::Marker,
            content,
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Project a stable tool model message onto the V4 provider prompt shape.
    pub fn from_tool_message(message: &ToolModelMessage) -> Self {
        let content = message
            .content
            .iter()
            .filter_map(|part| match part {
                ToolContentPart::ToolResult(part) => {
                    Some(LanguageModelV4ToolContentPart::ToolResult(
                        LanguageModelV4ToolResultPart::from_tool_result_part(part),
                    ))
                }
                ToolContentPart::ToolApprovalResponse(part)
                    if part.provider_executed == Some(true) =>
                {
                    Some(LanguageModelV4ToolContentPart::ToolApprovalResponse(
                        LanguageModelV4ToolApprovalResponsePart::from_tool_approval_response(part),
                    ))
                }
                ToolContentPart::ToolApprovalResponse(_) => None,
            })
            .collect();

        Self {
            role: LanguageModelV4ToolRoleMarker::Marker,
            content,
            provider_options: language_model_v4_provider_options_from_stable(
                &message.provider_options,
            ),
        }
    }

    /// Attach provider-specific request options.
    pub fn with_provider_options_map(mut self, provider_options: ProviderOptionsMap) -> Self {
        self.provider_options = provider_options;
        self
    }
}

/// AI SDK V4 prompt message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum LanguageModelV4Message {
    System(LanguageModelV4SystemMessage),
    User(LanguageModelV4UserMessage),
    Assistant(LanguageModelV4AssistantMessage),
    Tool(LanguageModelV4ToolMessage),
}

impl LanguageModelV4Message {
    /// Return the message role.
    pub const fn role(&self) -> &'static str {
        match self {
            Self::System(_) => "system",
            Self::User(_) => "user",
            Self::Assistant(_) => "assistant",
            Self::Tool(_) => "tool",
        }
    }
}

impl From<&ModelMessage> for LanguageModelV4Message {
    fn from(value: &ModelMessage) -> Self {
        match value {
            ModelMessage::System(message) => {
                Self::System(LanguageModelV4SystemMessage::from_system_message(message))
            }
            ModelMessage::User(message) => {
                Self::User(LanguageModelV4UserMessage::from_user_message(message))
            }
            ModelMessage::Assistant(message) => Self::Assistant(
                LanguageModelV4AssistantMessage::from_assistant_message(message),
            ),
            ModelMessage::Tool(message) => {
                Self::Tool(LanguageModelV4ToolMessage::from_tool_message(message))
            }
        }
    }
}

impl From<ModelMessage> for LanguageModelV4Message {
    fn from(value: ModelMessage) -> Self {
        Self::from(&value)
    }
}

/// Project stable model messages onto the AI SDK V4 provider prompt shape.
pub fn prepare_language_model_v4_prompt(
    messages: impl IntoIterator<Item = ModelMessage>,
) -> LanguageModelV4Prompt {
    let mut prompt = Vec::new();

    for message in messages {
        match LanguageModelV4Message::from(message) {
            LanguageModelV4Message::Tool(mut current) => {
                if current.content.is_empty() {
                    continue;
                }

                if let Some(LanguageModelV4Message::Tool(previous)) = prompt.last_mut() {
                    previous.content.append(&mut current.content);
                } else {
                    prompt.push(LanguageModelV4Message::Tool(current));
                }
            }
            projected => prompt.push(projected),
        }
    }

    prompt
}
