use crate::types::{ProviderMetadataMap, ProviderOptionsMap, Warning};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;

/// AI SDK-style JSON value alias.
pub type JSONValue = serde_json::Value;

/// AI SDK-style JSON object alias.
pub type JSONObject = serde_json::Map<String, JSONValue>;

pub(super) fn serialize_ai_sdk_non_null_json_value<S>(
    value: &JSONValue,
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

pub(super) fn deserialize_ai_sdk_non_null_json_value<'de, D>(
    deserializer: D,
) -> Result<JSONValue, D::Error>
where
    D: Deserializer<'de>,
{
    let value = JSONValue::deserialize(deserializer)?;
    if value.is_null() {
        return Err(serde::de::Error::custom("expected non-null JSON value"));
    }

    Ok(value)
}

/// AI SDK-style JSON Schema draft-07 value alias.
pub type JSONSchema7 = serde_json::Value;

/// AI SDK-style shared call warning.
///
/// This mirrors `SharedV4Warning`. The wider stable `Warning` type still accepts legacy
/// compatibility inputs, but AI SDK-facing result and callback payloads normalize those legacy
/// variants to the canonical `unsupported { feature, details }` shape.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum CallWarning {
    /// A feature is not supported by the model/provider.
    Unsupported {
        /// Unsupported feature name.
        feature: String,
        /// Optional details about why it is unsupported.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    /// A compatibility feature is used.
    Compatibility {
        /// Feature being used in compatibility mode.
        feature: String,
        /// Optional compatibility details.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    /// A deprecated setting or feature is being used.
    Deprecated {
        /// Deprecated setting or feature name.
        setting: String,
        /// Human-readable replacement guidance.
        message: String,
    },
    /// Other warning.
    Other {
        /// Human-readable warning message.
        message: String,
    },
}

impl CallWarning {
    /// Create an unsupported warning.
    pub fn unsupported(feature: impl Into<String>, details: Option<impl Into<String>>) -> Self {
        Self::Unsupported {
            feature: feature.into(),
            details: details.map(Into::into),
        }
    }

    /// Create a compatibility warning.
    pub fn compatibility(feature: impl Into<String>, details: Option<impl Into<String>>) -> Self {
        Self::Compatibility {
            feature: feature.into(),
            details: details.map(Into::into),
        }
    }

    /// Create a deprecated warning.
    pub fn deprecated(setting: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Deprecated {
            setting: setting.into(),
            message: message.into(),
        }
    }

    /// Create an other warning.
    pub fn other(message: impl Into<String>) -> Self {
        Self::Other {
            message: message.into(),
        }
    }
}

impl From<Warning> for CallWarning {
    fn from(value: Warning) -> Self {
        match value {
            Warning::Unsupported { feature, details } => Self::Unsupported { feature, details },
            Warning::UnsupportedSetting { setting, details } => Self::Unsupported {
                feature: setting,
                details,
            },
            Warning::UnsupportedTool { tool_name, details } => Self::Unsupported {
                feature: tool_name,
                details,
            },
            Warning::Compatibility { feature, details } => Self::Compatibility { feature, details },
            Warning::Deprecated { setting, message } => Self::Deprecated { setting, message },
            Warning::Other { message } => Self::Other { message },
        }
    }
}

impl From<&Warning> for CallWarning {
    fn from(value: &Warning) -> Self {
        Self::from(value.clone())
    }
}

impl From<CallWarning> for Warning {
    fn from(value: CallWarning) -> Self {
        match value {
            CallWarning::Unsupported { feature, details } => {
                Warning::Unsupported { feature, details }
            }
            CallWarning::Compatibility { feature, details } => {
                Warning::Compatibility { feature, details }
            }
            CallWarning::Deprecated { setting, message } => {
                Warning::Deprecated { setting, message }
            }
            CallWarning::Other { message } => Warning::Other { message },
        }
    }
}

/// AI SDK-style shared provider-metadata root.
pub type ProviderMetadata = ProviderMetadataMap;

/// AI SDK-style shared provider-options root.
pub type ProviderOptions = ProviderOptionsMap;

/// AI SDK-style shared execution context object.
pub type Context = HashMap<String, JSONValue>;

/// Passive AI SDK `TelemetryOptions` configuration.
///
/// Function-valued telemetry integrations remain runtime-only and are intentionally not modeled here.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TelemetryOptions {
    /// Enable or disable telemetry for this call.
    #[serde(alias = "is_enabled", skip_serializing_if = "Option::is_none")]
    pub is_enabled: Option<bool>,
    /// Enable or disable input recording.
    #[serde(alias = "record_inputs", skip_serializing_if = "Option::is_none")]
    pub record_inputs: Option<bool>,
    /// Enable or disable output recording.
    #[serde(alias = "record_outputs", skip_serializing_if = "Option::is_none")]
    pub record_outputs: Option<bool>,
    /// Identifier used to group telemetry data by function.
    #[serde(alias = "function_id", skip_serializing_if = "Option::is_none")]
    pub function_id: Option<String>,
}

impl TelemetryOptions {
    /// Create empty telemetry options.
    pub const fn new() -> Self {
        Self {
            is_enabled: None,
            record_inputs: None,
            record_outputs: None,
            function_id: None,
        }
    }

    /// Set whether telemetry is enabled.
    pub const fn with_is_enabled(mut self, is_enabled: bool) -> Self {
        self.is_enabled = Some(is_enabled);
        self
    }

    /// Set whether inputs should be recorded.
    pub const fn with_record_inputs(mut self, record_inputs: bool) -> Self {
        self.record_inputs = Some(record_inputs);
        self
    }

    /// Set whether outputs should be recorded.
    pub const fn with_record_outputs(mut self, record_outputs: bool) -> Self {
        self.record_outputs = Some(record_outputs);
        self
    }

    /// Set the telemetry function id.
    pub fn with_function_id(mut self, function_id: impl Into<String>) -> Self {
        self.function_id = Some(function_id.into());
        self
    }
}

/// AI SDK-style single embedding vector.
pub type Embedding = Vec<f32>;
