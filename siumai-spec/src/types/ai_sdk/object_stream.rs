use crate::types::FinishReason;
use serde::{Deserialize, Serialize};

use super::{JSONValue, LanguageModelResponseMetadata, LanguageModelUsage, ProviderMetadata};

macro_rules! fixed_object_stream_type_marker {
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
                S: serde::Serializer,
            {
                serializer.serialize_str($value)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                if value == $value {
                    Ok(Self::Marker)
                } else {
                    Err(serde::de::Error::custom(format!(
                        "expected type `{}`, got `{value}`",
                        $value
                    )))
                }
            }
        }
    };
}

fixed_object_stream_type_marker!(ObjectStreamObjectPartMarker, "object");
fixed_object_stream_type_marker!(ObjectStreamTextDeltaPartMarker, "text-delta");
fixed_object_stream_type_marker!(ObjectStreamErrorPartMarker, "error");
fixed_object_stream_type_marker!(ObjectStreamFinishPartMarker, "finish");

/// Partial object event from AI SDK `ObjectStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ObjectStreamObjectPart<PARTIAL = JSONValue> {
    #[serde(rename = "type", default)]
    marker: ObjectStreamObjectPartMarker,
    /// Current partial object snapshot.
    pub object: PARTIAL,
}

impl<PARTIAL> ObjectStreamObjectPart<PARTIAL> {
    /// Create a partial-object stream part.
    pub fn new(object: PARTIAL) -> Self {
        Self {
            marker: ObjectStreamObjectPartMarker::Marker,
            object,
        }
    }

    /// Return the AI SDK `ObjectStreamPart` discriminator.
    pub const fn r#type(&self) -> &'static str {
        "object"
    }
}

/// Text-delta event from AI SDK `ObjectStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ObjectStreamTextDeltaPart {
    #[serde(rename = "type", default)]
    marker: ObjectStreamTextDeltaPartMarker,
    /// JSON text delta.
    pub text_delta: String,
}

impl ObjectStreamTextDeltaPart {
    /// Create an object-stream text delta.
    pub fn new(text_delta: impl Into<String>) -> Self {
        Self {
            marker: ObjectStreamTextDeltaPartMarker::Marker,
            text_delta: text_delta.into(),
        }
    }

    /// Return the AI SDK `ObjectStreamPart` discriminator.
    pub const fn r#type(&self) -> &'static str {
        "text-delta"
    }
}

/// Error event from AI SDK `ObjectStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ObjectStreamErrorPart {
    #[serde(rename = "type", default)]
    marker: ObjectStreamErrorPartMarker,
    /// Error payload.
    pub error: JSONValue,
}

impl ObjectStreamErrorPart {
    /// Create an object-stream error part.
    pub fn new(error: impl Into<JSONValue>) -> Self {
        Self {
            marker: ObjectStreamErrorPartMarker::Marker,
            error: error.into(),
        }
    }

    /// Return the AI SDK `ObjectStreamPart` discriminator.
    pub const fn r#type(&self) -> &'static str {
        "error"
    }
}

/// Finish event from AI SDK `ObjectStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ObjectStreamFinishPart {
    #[serde(rename = "type", default)]
    marker: ObjectStreamFinishPartMarker,
    /// Unified finish reason.
    pub finish_reason: FinishReason,
    /// Token usage.
    pub usage: LanguageModelUsage,
    /// Response metadata.
    pub response: LanguageModelResponseMetadata,
    /// Provider-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl ObjectStreamFinishPart {
    /// Create an object-stream finish part.
    pub fn new(
        finish_reason: FinishReason,
        usage: LanguageModelUsage,
        response: LanguageModelResponseMetadata,
    ) -> Self {
        Self {
            marker: ObjectStreamFinishPartMarker::Marker,
            finish_reason,
            usage,
            response,
            provider_metadata: None,
        }
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }

    /// Return the AI SDK `ObjectStreamPart` discriminator.
    pub const fn r#type(&self) -> &'static str {
        "finish"
    }
}

/// AI SDK `ObjectStreamPart` union from `generate-object/stream-object-result.ts`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
#[allow(clippy::large_enum_variant)]
pub enum ObjectStreamPart<PARTIAL = JSONValue> {
    /// Partial object snapshot.
    Object(ObjectStreamObjectPart<PARTIAL>),
    /// JSON text delta.
    TextDelta(ObjectStreamTextDeltaPart),
    /// Stream error payload.
    Error(ObjectStreamErrorPart),
    /// Terminal stream metadata.
    Finish(ObjectStreamFinishPart),
}

impl<PARTIAL> ObjectStreamPart<PARTIAL> {
    /// Return the AI SDK `ObjectStreamPart` discriminator.
    pub fn r#type(&self) -> &'static str {
        match self {
            Self::Object(part) => part.r#type(),
            Self::TextDelta(part) => part.r#type(),
            Self::Error(part) => part.r#type(),
            Self::Finish(part) => part.r#type(),
        }
    }
}

impl<PARTIAL> From<ObjectStreamObjectPart<PARTIAL>> for ObjectStreamPart<PARTIAL> {
    fn from(value: ObjectStreamObjectPart<PARTIAL>) -> Self {
        Self::Object(value)
    }
}

impl<PARTIAL> From<ObjectStreamTextDeltaPart> for ObjectStreamPart<PARTIAL> {
    fn from(value: ObjectStreamTextDeltaPart) -> Self {
        Self::TextDelta(value)
    }
}

impl<PARTIAL> From<ObjectStreamErrorPart> for ObjectStreamPart<PARTIAL> {
    fn from(value: ObjectStreamErrorPart) -> Self {
        Self::Error(value)
    }
}

impl<PARTIAL> From<ObjectStreamFinishPart> for ObjectStreamPart<PARTIAL> {
    fn from(value: ObjectStreamFinishPart) -> Self {
        Self::Finish(value)
    }
}
