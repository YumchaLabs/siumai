//! Common enums and metadata types used across the library.
//!
//! This module intentionally excludes parameter/HTTP/usage types, which live in
//! `types::params`, `types::http`, and `types::usage`.

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use std::fmt;

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
    /// A deprecated setting or feature is being used.
    Deprecated {
        /// The deprecated setting or feature name.
        setting: String,
        /// A human-readable message explaining what to use instead.
        message: String,
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

    /// Create a deprecation warning.
    pub fn deprecated(setting: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Deprecated {
            setting: setting.into(),
            message: message.into(),
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
    Azure,
    Anthropic,
    Gemini,
    Vertex,
    AnthropicVertex,
    VertexMaas,
    Ollama,
    DeepSeek,
    DeepInfra,
    Cohere,
    TogetherAi,
    Bedrock,
    Mistral,
    Fireworks,
    Perplexity,
    XAI,
    Groq,
    MiniMaxi,
    Custom(String),
}

impl std::fmt::Display for ProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenAi => write!(f, "openai"),
            Self::Azure => write!(f, "azure"),
            Self::Anthropic => write!(f, "anthropic"),
            Self::Gemini => write!(f, "gemini"),
            Self::Vertex => write!(f, "vertex"),
            Self::AnthropicVertex => write!(f, "anthropic-vertex"),
            Self::VertexMaas => write!(f, "vertex-maas"),
            Self::Ollama => write!(f, "ollama"),
            Self::DeepSeek => write!(f, "deepseek"),
            Self::DeepInfra => write!(f, "deepinfra"),
            Self::Cohere => write!(f, "cohere"),
            Self::TogetherAi => write!(f, "togetherai"),
            Self::Bedrock => write!(f, "bedrock"),
            Self::Mistral => write!(f, "mistral"),
            Self::Fireworks => write!(f, "fireworks"),
            Self::Perplexity => write!(f, "perplexity"),
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
            "openai" | "openai-chat" | "openai-responses" => Self::OpenAi,
            "azure" | "azure-chat" => Self::Azure,
            "anthropic" => Self::Anthropic,
            "gemini" => Self::Gemini,
            "vertex" | "google-vertex" => Self::Vertex,
            "anthropic-vertex" | "google-vertex-anthropic" => Self::AnthropicVertex,
            "vertex-maas" | "google-vertex-maas" | "vertex.maas" | "vertexMaas" => Self::VertexMaas,
            "ollama" => Self::Ollama,
            "deepseek" => Self::DeepSeek,
            "deepinfra" => Self::DeepInfra,
            "cohere" => Self::Cohere,
            "togetherai" => Self::TogetherAi,
            "bedrock" => Self::Bedrock,
            "mistral" => Self::Mistral,
            "fireworks" => Self::Fireworks,
            "perplexity" => Self::Perplexity,
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
#[derive(Debug, Clone, PartialEq, Eq)]
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

impl FinishReason {
    fn from_wire_value(value: &str) -> Self {
        match value {
            "stop" => Self::Stop,
            "length" => Self::Length,
            "tool_calls" => Self::ToolCalls,
            "content_filter" => Self::ContentFilter,
            "stop_sequence" => Self::StopSequence,
            "error" => Self::Error,
            "unknown" => Self::Unknown,
            other => Self::Other(other.to_string()),
        }
    }

    fn as_wire_str(&self) -> &str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ToolCalls => "tool_calls",
            Self::ContentFilter => "content_filter",
            Self::StopSequence => "stop_sequence",
            Self::Error => "error",
            Self::Other(value) => value.as_str(),
            Self::Unknown => "unknown",
        }
    }
}

impl Serialize for FinishReason {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_wire_str())
    }
}

impl<'de> Deserialize<'de> for FinishReason {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct FinishReasonVisitor;

        impl<'de> Visitor<'de> for FinishReasonVisitor {
            type Value = FinishReason;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter
                    .write_str("a finish reason string or a legacy single-key finish reason object")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(FinishReason::from_wire_value(value))
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(FinishReason::from_wire_value(&value))
            }

            fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let Some(key) = map.next_key::<String>()? else {
                    return Err(de::Error::custom(
                        "expected a non-empty finish reason object",
                    ));
                };

                let reason = match key.as_str() {
                    "other" => FinishReason::Other(map.next_value::<String>()?),
                    "stop" | "length" | "tool_calls" | "content_filter" | "stop_sequence"
                    | "error" | "unknown" => {
                        let _: Option<de::IgnoredAny> = map.next_value()?;
                        FinishReason::from_wire_value(&key)
                    }
                    _ => {
                        return Err(de::Error::custom(format!(
                            "unsupported legacy finish reason variant: {key}"
                        )));
                    }
                };

                if map.next_key::<de::IgnoredAny>()?.is_some() {
                    return Err(de::Error::custom(
                        "expected a single-key finish reason object",
                    ));
                }

                Ok(reason)
            }
        }

        deserializer.deserialize_any(FinishReasonVisitor)
    }
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
    /// Response headers when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
}

#[cfg(test)]
mod tests {
    use super::{FinishReason, ProviderType, ResponseMetadata, Warning};
    use chrono::{DateTime, Utc};
    use std::collections::HashMap;

    #[test]
    fn provider_type_maps_deepseek_name() {
        assert_eq!(ProviderType::from_name("deepseek"), ProviderType::DeepSeek);
        assert_eq!(ProviderType::DeepSeek.to_string(), "deepseek");
    }

    #[test]
    fn provider_type_maps_deepinfra_name() {
        assert_eq!(
            ProviderType::from_name("deepinfra"),
            ProviderType::DeepInfra
        );
        assert_eq!(ProviderType::DeepInfra.to_string(), "deepinfra");
    }

    #[test]
    fn provider_type_maps_azure_name() {
        assert_eq!(ProviderType::from_name("azure"), ProviderType::Azure);
        assert_eq!(ProviderType::from_name("azure-chat"), ProviderType::Azure);
        assert_eq!(ProviderType::Azure.to_string(), "azure");
    }

    #[test]
    fn provider_type_maps_openai_family_variants() {
        assert_eq!(ProviderType::from_name("openai"), ProviderType::OpenAi);
        assert_eq!(ProviderType::from_name("openai-chat"), ProviderType::OpenAi);
        assert_eq!(
            ProviderType::from_name("openai-responses"),
            ProviderType::OpenAi
        );
        assert_eq!(ProviderType::OpenAi.to_string(), "openai");
    }

    #[test]
    fn provider_type_maps_cohere_name() {
        assert_eq!(ProviderType::from_name("cohere"), ProviderType::Cohere);
        assert_eq!(ProviderType::Cohere.to_string(), "cohere");
    }

    #[test]
    fn provider_type_maps_togetherai_name() {
        assert_eq!(
            ProviderType::from_name("togetherai"),
            ProviderType::TogetherAi
        );
        assert_eq!(ProviderType::TogetherAi.to_string(), "togetherai");
    }

    #[test]
    fn provider_type_maps_bedrock_name() {
        assert_eq!(ProviderType::from_name("bedrock"), ProviderType::Bedrock);
        assert_eq!(ProviderType::Bedrock.to_string(), "bedrock");
    }

    #[test]
    fn provider_type_maps_mistral_name() {
        assert_eq!(ProviderType::from_name("mistral"), ProviderType::Mistral);
        assert_eq!(ProviderType::Mistral.to_string(), "mistral");
    }

    #[test]
    fn provider_type_maps_fireworks_name() {
        assert_eq!(
            ProviderType::from_name("fireworks"),
            ProviderType::Fireworks
        );
        assert_eq!(ProviderType::Fireworks.to_string(), "fireworks");
    }

    #[test]
    fn provider_type_maps_perplexity_name() {
        assert_eq!(
            ProviderType::from_name("perplexity"),
            ProviderType::Perplexity
        );
        assert_eq!(ProviderType::Perplexity.to_string(), "perplexity");
    }

    #[test]
    fn provider_type_maps_vertex_maas_name() {
        assert_eq!(
            ProviderType::from_name("vertex-maas"),
            ProviderType::VertexMaas
        );
        assert_eq!(
            ProviderType::from_name("google-vertex-maas"),
            ProviderType::VertexMaas
        );
        assert_eq!(
            ProviderType::from_name("vertex.maas"),
            ProviderType::VertexMaas
        );
        assert_eq!(ProviderType::VertexMaas.to_string(), "vertex-maas");
    }

    #[test]
    fn provider_type_maps_vertex_name() {
        assert_eq!(ProviderType::from_name("vertex"), ProviderType::Vertex);
        assert_eq!(
            ProviderType::from_name("google-vertex"),
            ProviderType::Vertex
        );
        assert_eq!(ProviderType::Vertex.to_string(), "vertex");
    }

    #[test]
    fn provider_type_maps_anthropic_vertex_name() {
        assert_eq!(
            ProviderType::from_name("anthropic-vertex"),
            ProviderType::AnthropicVertex
        );
        assert_eq!(
            ProviderType::from_name("google-vertex-anthropic"),
            ProviderType::AnthropicVertex
        );
        assert_eq!(
            ProviderType::AnthropicVertex.to_string(),
            "anthropic-vertex"
        );
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

    #[test]
    fn deprecated_warning_serializes_with_shared_v4_shape() {
        let warning = Warning::deprecated("legacyOption", "Use `replacementOption` instead.");

        let value = serde_json::to_value(&warning).expect("serialize warning");
        assert_eq!(value["type"], serde_json::json!("deprecated"));
        assert_eq!(value["setting"], serde_json::json!("legacyOption"));
        assert_eq!(
            value["message"],
            serde_json::json!("Use `replacementOption` instead.")
        );
    }

    #[test]
    fn response_metadata_serializes_headers_when_available() {
        let metadata = ResponseMetadata {
            id: Some("resp_1".to_string()),
            model: Some("gpt-4o".to_string()),
            created: Some(
                DateTime::parse_from_rfc3339("2026-04-21T09:00:00Z")
                    .expect("valid timestamp")
                    .with_timezone(&Utc),
            ),
            provider: "openai".to_string(),
            request_id: Some("req_1".to_string()),
            headers: Some(HashMap::from([(
                "x-request-id".to_string(),
                "req_1".to_string(),
            )])),
        };

        let value = serde_json::to_value(&metadata).expect("serialize metadata");
        assert_eq!(value["headers"]["x-request-id"], serde_json::json!("req_1"));
    }

    #[test]
    fn finish_reason_other_serializes_as_plain_string() {
        let value =
            serde_json::to_value(FinishReason::Other("other".to_string())).expect("serialize");
        assert_eq!(value, serde_json::json!("other"));
    }

    #[test]
    fn finish_reason_deserializes_unknown_string_as_other() {
        let reason: FinishReason =
            serde_json::from_value(serde_json::json!("provider_custom")).expect("deserialize");
        assert_eq!(reason, FinishReason::Other("provider_custom".to_string()));
    }

    #[test]
    fn finish_reason_deserializes_legacy_other_object_shape() {
        let reason: FinishReason =
            serde_json::from_value(serde_json::json!({ "other": "provider_custom" }))
                .expect("deserialize legacy other");
        assert_eq!(reason, FinishReason::Other("provider_custom".to_string()));
    }
}
