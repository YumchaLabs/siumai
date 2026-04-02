//! Common enums and metadata types used across the library.
//!
//! This module intentionally excludes parameter/HTTP/usage types, which live in
//! `types::params`, `types::http`, and `types::usage`.

use serde::{Deserialize, Serialize};

/// Warning from the model provider
///
/// Warnings indicate non-fatal issues during generation, such as unsupported settings
/// or deprecated features. The generation continues despite warnings.
///
/// # Examples
///
/// ```rust
/// use siumai::types::Warning;
///
/// let warning = Warning::unsupported("topK", Some("This provider doesn't support topK"));
/// let tool_warning = Warning::unsupported_tool("calculator", Some("This model doesn't support custom tools"));
/// let compatibility_warning = Warning::compatibility(
///     "systemMessageMode=remove",
///     Some("System messages are removed for this model"),
/// );
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum Warning {
    /// An unsupported feature was provided.
    Unsupported {
        /// The unsupported feature name.
        feature: String,
        /// Optional details about why it's unsupported.
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    /// An unsupported setting was provided
    ///
    /// Legacy compatibility variant. Prefer `Unsupported { feature }` for new
    /// warnings.
    UnsupportedSetting {
        /// The name of the unsupported setting
        setting: String,
        /// Optional details about why it's unsupported
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    /// An unsupported tool was provided
    ///
    /// Legacy compatibility variant. Prefer `Unsupported { feature }` for new
    /// warnings.
    UnsupportedTool {
        /// The name of the unsupported tool
        tool_name: String,
        /// Optional details about why it's unsupported
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    /// A compatibility warning indicating behavior differs from the ideal contract
    Compatibility {
        /// The feature or behavior with compatibility caveats
        feature: String,
        /// Optional details about the compatibility adjustment
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    /// Other warning types
    Other {
        /// Warning message
        message: String,
    },
}

impl Warning {
    /// Create an unsupported warning using the AI SDK-style shared shape.
    pub fn unsupported(feature: impl Into<String>, details: Option<impl Into<String>>) -> Self {
        Self::Unsupported {
            feature: feature.into(),
            details: details.map(|d| d.into()),
        }
    }

    /// Create an unsupported setting warning
    pub fn unsupported_setting(
        setting: impl Into<String>,
        details: Option<impl Into<String>>,
    ) -> Self {
        Self::unsupported(setting, details)
    }

    /// Create an unsupported tool warning
    pub fn unsupported_tool(
        tool_name: impl Into<String>,
        details: Option<impl Into<String>>,
    ) -> Self {
        Self::unsupported(tool_name, details)
    }

    /// Create a compatibility warning
    pub fn compatibility(feature: impl Into<String>, details: Option<impl Into<String>>) -> Self {
        Self::Compatibility {
            feature: feature.into(),
            details: details.map(|d| d.into()),
        }
    }

    /// Create a generic warning
    pub fn other(message: impl Into<String>) -> Self {
        Self::Other {
            message: message.into(),
        }
    }
}

/// Provider type enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderType {
    OpenAi,
    Anthropic,
    Gemini,
    Ollama,
    DeepSeek,
    XAI,
    Groq,
    MiniMaxi,
    Custom(String),
}

impl std::fmt::Display for ProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenAi => write!(f, "openai"),
            Self::Anthropic => write!(f, "anthropic"),
            Self::Gemini => write!(f, "gemini"),
            Self::Ollama => write!(f, "ollama"),
            Self::DeepSeek => write!(f, "deepseek"),
            Self::XAI => write!(f, "xai"),
            Self::Groq => write!(f, "groq"),
            Self::MiniMaxi => write!(f, "minimaxi"),
            Self::Custom(name) => write!(f, "{name}"),
        }
    }
}

impl ProviderType {
    /// Construct a ProviderType from a provider name string.
    /// Known names map to concrete variants; others map to Custom(name).
    pub fn from_name(name: &str) -> Self {
        match name {
            "openai" => Self::OpenAi,
            "anthropic" => Self::Anthropic,
            "gemini" => Self::Gemini,
            "ollama" => Self::Ollama,
            "deepseek" => Self::DeepSeek,
            "xai" => Self::XAI,
            "groq" => Self::Groq,
            "minimaxi" => Self::MiniMaxi,
            other => Self::Custom(other.to_string()),
        }
    }
}

/// Reason why the model stopped generating tokens.
///
/// This enum follows industry standards (OpenAI, Anthropic, Gemini, etc.) and is compatible
/// with the AI SDK (Vercel) finish reason types.
///
/// # Examples
///
/// ```rust
/// use siumai::types::FinishReason;
///
/// // Check if the response completed normally
/// let finish_reason = Some(FinishReason::Stop);
/// match finish_reason {
///     Some(FinishReason::Stop) => println!("Completed successfully"),
///     Some(FinishReason::Length) => println!("Reached max tokens"),
///     Some(FinishReason::ContentFilter) => println!("Content filtered"),
///     Some(FinishReason::Unknown) => println!("Unknown finish reason"),
///     _ => println!("Other reason"),
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Model generated stop sequence or completed naturally.
    ///
    /// Maps to:
    /// - OpenAI: `stop`
    /// - Anthropic: `end_turn`
    /// - Gemini: `STOP`
    /// - Groq/xAI: `stop`
    Stop,

    /// Model reached the maximum number of tokens (`max_tokens` parameter).
    ///
    /// Maps to:
    /// - OpenAI: `length`
    /// - Anthropic: `max_tokens`
    /// - Gemini: `MAX_TOKENS`
    /// - Groq/xAI: `length`
    Length,

    /// Model triggered tool/function calls.
    ///
    /// Maps to:
    /// - OpenAI: `tool_calls`
    /// - Anthropic: `tool_use`
    /// - Groq/xAI: `tool_calls`
    ToolCalls,

    /// Content was filtered due to safety/policy violations.
    ///
    /// Maps to:
    /// - OpenAI: `content_filter`
    /// - Anthropic: `refusal`
    /// - Gemini: `SAFETY`, `RECITATION`, `PROHIBITED_CONTENT`
    /// - Groq/xAI: `content_filter`
    ContentFilter,

    /// Model stopped due to a custom stop sequence.
    ///
    /// Maps to Anthropic's `stop_sequence`.
    StopSequence,

    /// An error occurred during generation.
    ///
    /// Maps to AI SDK's `error` reason.
    Error,

    /// Other provider-specific finish reason.
    ///
    /// Maps to AI SDK's `other` reason.
    Other(String),

    /// Unknown finish reason.
    ///
    /// Used when the provider did not provide a finish reason or it was not recognized.
    Unknown,
}

/// Response metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMetadata {
    /// Response ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// AI SDK-style model identifier.
    #[serde(
        rename = "modelId",
        alias = "model",
        skip_serializing_if = "Option::is_none"
    )]
    pub model: Option<String>,
    /// AI SDK-style response start timestamp.
    #[serde(
        rename = "timestamp",
        alias = "created",
        skip_serializing_if = "Option::is_none"
    )]
    pub created: Option<chrono::DateTime<chrono::Utc>>,
    /// Provider name
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub provider: String,
    /// Request ID
    #[serde(
        rename = "requestId",
        alias = "request_id",
        skip_serializing_if = "Option::is_none"
    )]
    pub request_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::{ProviderType, Warning};

    #[test]
    fn provider_type_maps_deepseek_name() {
        assert_eq!(ProviderType::from_name("deepseek"), ProviderType::DeepSeek);
        assert_eq!(ProviderType::DeepSeek.to_string(), "deepseek");
    }

    #[test]
    fn compatibility_warning_serializes_with_vercel_shape() {
        let warning = Warning::compatibility(
            "systemMessageMode=remove",
            Some("system messages are removed for this model"),
        );

        let value = serde_json::to_value(&warning).expect("serialize warning");
        assert_eq!(value["type"], serde_json::json!("compatibility"));
        assert_eq!(
            value["feature"],
            serde_json::json!("systemMessageMode=remove")
        );
        assert_eq!(
            value["details"],
            serde_json::json!("system messages are removed for this model")
        );
    }

    #[test]
    fn unsupported_warning_serializes_with_vercel_shape() {
        let warning = Warning::unsupported_setting(
            "size",
            Some("This model does not support the `size` option."),
        );

        let value = serde_json::to_value(&warning).expect("serialize warning");
        assert_eq!(value["type"], serde_json::json!("unsupported"));
        assert_eq!(value["feature"], serde_json::json!("size"));
        assert_eq!(
            value["details"],
            serde_json::json!("This model does not support the `size` option.")
        );
    }

    #[test]
    fn legacy_unsupported_setting_shape_still_deserializes() {
        let value = serde_json::json!({
            "type": "unsupported-setting",
            "setting": "topK",
            "details": "provider does not support topK"
        });

        let warning = serde_json::from_value::<Warning>(value).expect("deserialize warning");
        assert_eq!(
            warning,
            Warning::UnsupportedSetting {
                setting: "topK".to_string(),
                details: Some("provider does not support topK".to_string()),
            }
        );
    }
}
