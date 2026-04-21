use base64::{Engine, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::ProviderOptionsMap;

/// Tool result output - supports multiple formats
///
/// This enum represents different types of tool execution results,
/// aligned with Vercel AI SDK's ToolResultOutput design.
///
/// # Examples
///
/// ```rust
/// use siumai::types::ToolResultOutput;
///
/// // Simple text result
/// let result = ToolResultOutput::text("Success");
///
/// // JSON result
/// let result = ToolResultOutput::json(serde_json::json!({
///     "temperature": 18,
///     "condition": "sunny"
/// }));
///
/// // Error result
/// let result = ToolResultOutput::error_text("API timeout");
///
/// // Execution denied
/// let result = ToolResultOutput::execution_denied(Some("User rejected".to_string()));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ToolResultOutput {
    /// Plain text output
    Text {
        value: String,
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options: ProviderOptionsMap,
    },

    /// JSON output
    Json {
        value: serde_json::Value,
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options: ProviderOptionsMap,
    },

    /// Execution was denied
    ExecutionDenied {
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options: ProviderOptionsMap,
    },

    /// Text error
    ErrorText {
        value: String,
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options: ProviderOptionsMap,
    },

    /// JSON error
    ErrorJson {
        value: serde_json::Value,
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options: ProviderOptionsMap,
    },

    /// Multimodal content (text/files/images/custom parts)
    Content {
        value: Vec<ToolResultContentPart>,
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options: ProviderOptionsMap,
    },
}

/// Provider file id reference used by AI SDK tool-result content parts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum ToolResultFileId {
    Single(String),
    PerProvider(HashMap<String, String>),
}

impl ToolResultFileId {
    /// Resolve the best available file id for one of the given providers.
    pub fn preferred_value<'a>(&'a self, providers: &[&str]) -> Option<&'a str> {
        match self {
            Self::Single(value) => Some(value.as_str()),
            Self::PerProvider(values) => providers
                .iter()
                .find_map(|provider| values.get(*provider).map(String::as_str))
                .or_else(|| {
                    values
                        .iter()
                        .min_by(|(left, _), (right, _)| left.cmp(right))
                        .map(|(_, value)| value.as_str())
                }),
        }
    }
}

impl From<String> for ToolResultFileId {
    fn from(value: String) -> Self {
        Self::Single(value)
    }
}

impl From<&str> for ToolResultFileId {
    fn from(value: &str) -> Self {
        Self::Single(value.to_string())
    }
}

impl From<HashMap<String, String>> for ToolResultFileId {
    fn from(value: HashMap<String, String>) -> Self {
        Self::PerProvider(value)
    }
}

/// Content part for tool results
///
/// This is a Vercel-aligned subset of content parts that can appear inside
/// `ToolResultOutput::Content`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ToolResultContentPart {
    /// Text content
    Text {
        text: String,
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options: ProviderOptionsMap,
    },

    /// File content encoded inline as base64 data.
    FileData {
        data: String,
        #[serde(rename = "mediaType", alias = "media_type")]
        media_type: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options: ProviderOptionsMap,
    },

    /// File content referenced by URL.
    FileUrl {
        url: String,
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options: ProviderOptionsMap,
    },

    /// File content referenced by provider file id or provider-reference map.
    #[serde(rename = "file-reference", alias = "file-id")]
    FileId {
        #[serde(
            alias = "providerReference",
            alias = "provider_reference",
            alias = "fileId",
            alias = "file_id",
            rename = "providerReference"
        )]
        file_id: ToolResultFileId,
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options: ProviderOptionsMap,
    },

    /// Image content encoded inline as base64 data.
    ImageData {
        data: String,
        #[serde(rename = "mediaType", alias = "media_type")]
        media_type: String,
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options: ProviderOptionsMap,
    },

    /// Image content referenced by URL.
    ImageUrl {
        url: String,
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options: ProviderOptionsMap,
    },

    /// Image content referenced by provider file id or provider-reference map.
    #[serde(rename = "image-file-reference", alias = "image-file-id")]
    ImageFileId {
        #[serde(
            alias = "providerReference",
            alias = "provider_reference",
            alias = "fileId",
            alias = "file_id",
            rename = "providerReference"
        )]
        file_id: ToolResultFileId,
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options: ProviderOptionsMap,
    },

    /// Provider-specific custom content.
    Custom {
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options: ProviderOptionsMap,
    },
}

impl ToolResultOutput {
    /// Create a text output
    pub fn text(value: impl Into<String>) -> Self {
        Self::Text {
            value: value.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Create a JSON output
    pub fn json(value: serde_json::Value) -> Self {
        Self::Json {
            value,
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Create an execution denied output
    pub fn execution_denied(reason: Option<String>) -> Self {
        Self::ExecutionDenied {
            reason,
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Create a text error output
    pub fn error_text(value: impl Into<String>) -> Self {
        Self::ErrorText {
            value: value.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Create a JSON error output
    pub fn error_json(value: serde_json::Value) -> Self {
        Self::ErrorJson {
            value,
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Create a multimodal content output
    pub fn content(value: Vec<ToolResultContentPart>) -> Self {
        Self::Content {
            value,
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Check if this is an error result
    pub fn is_error(&self) -> bool {
        matches!(self, Self::ErrorText { .. } | Self::ErrorJson { .. })
    }

    /// Check if execution was denied
    pub fn is_execution_denied(&self) -> bool {
        matches!(self, Self::ExecutionDenied { .. })
    }

    /// Convert to a simple string representation (for backward compatibility)
    pub fn to_string_lossy(&self) -> String {
        match self {
            Self::Text { value, .. } => value.clone(),
            Self::Json { value, .. } => serde_json::to_string(value).unwrap_or_default(),
            Self::ExecutionDenied { reason, .. } => reason
                .clone()
                .unwrap_or_else(|| "Execution denied".to_string()),
            Self::ErrorText { value, .. } => value.clone(),
            Self::ErrorJson { value, .. } => serde_json::to_string(value).unwrap_or_default(),
            Self::Content { value, .. } => serde_json::to_string(value)
                .unwrap_or_else(|_| format!("Tool content with {} parts", value.len())),
        }
    }

    /// Convert to JSON value for provider APIs
    pub fn to_json_value(&self) -> serde_json::Value {
        match self {
            Self::Text { value, .. } => serde_json::Value::String(value.clone()),
            Self::Json { value, .. } => value.clone(),
            Self::ExecutionDenied { reason, .. } => {
                serde_json::json!({
                    "type": "execution-denied",
                    "reason": reason
                })
            }
            Self::ErrorText { value, .. } => serde_json::Value::String(value.clone()),
            Self::ErrorJson { value, .. } => value.clone(),
            Self::Content { value, .. } => serde_json::Value::Array(
                value
                    .iter()
                    .map(|part| serde_json::to_value(part).unwrap_or(serde_json::Value::Null))
                    .collect(),
            ),
        }
    }
}

impl ToolResultContentPart {
    /// Create a text content part
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Create an image-data content part.
    pub fn image_data(data: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self::ImageData {
            data: data.into(),
            media_type: media_type.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Create an image content part from URL.
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::ImageUrl {
            url: url.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Create an image-data content part from base64 data.
    ///
    /// This legacy convenience constructor defaults to `image/jpeg`.
    pub fn image_base64(data: impl Into<String>) -> Self {
        Self::image_data(data, "image/jpeg")
    }

    /// Create an image-data content part from binary bytes.
    pub fn image_binary(data: Vec<u8>, media_type: impl Into<String>) -> Self {
        Self::image_data(STANDARD.encode(data), media_type)
    }

    /// Create an image-file-id content part.
    pub fn image_file_id(file_id: impl Into<ToolResultFileId>) -> Self {
        Self::ImageFileId {
            file_id: file_id.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Create an image-file-reference content part.
    pub fn image_file_reference(provider_reference: impl Into<ToolResultFileId>) -> Self {
        Self::image_file_id(provider_reference)
    }

    /// Create a file-data content part.
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

    /// Create a file-data content part from base64 data.
    pub fn file_base64(
        data: impl Into<String>,
        media_type: impl Into<String>,
        filename: Option<String>,
    ) -> Self {
        Self::file_data(data, media_type, filename)
    }

    /// Create a file-data content part from binary bytes.
    pub fn file_binary(
        data: Vec<u8>,
        media_type: impl Into<String>,
        filename: Option<String>,
    ) -> Self {
        Self::file_data(STANDARD.encode(data), media_type, filename)
    }

    /// Create a file-url content part.
    pub fn file_url(url: impl Into<String>) -> Self {
        Self::FileUrl {
            url: url.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Create a file-id content part.
    pub fn file_id(file_id: impl Into<ToolResultFileId>) -> Self {
        Self::FileId {
            file_id: file_id.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Create a file-reference content part.
    pub fn file_reference(provider_reference: impl Into<ToolResultFileId>) -> Self {
        Self::file_id(provider_reference)
    }

    /// Create a custom tool-result content part.
    pub fn custom() -> Self {
        Self::Custom {
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Get provider options if the part supports request-side provider configuration.
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
            | Self::FileId {
                provider_options, ..
            }
            | Self::ImageData {
                provider_options, ..
            }
            | Self::ImageUrl {
                provider_options, ..
            }
            | Self::ImageFileId {
                provider_options, ..
            }
            | Self::Custom {
                provider_options, ..
            } => provider_options,
        }
    }

    /// Alias of `provider_options()` that matches the wider shared-type naming convention.
    pub fn provider_options_map(&self) -> &ProviderOptionsMap {
        self.provider_options()
    }

    /// Get mutable provider options for the part.
    pub fn provider_options_mut(&mut self) -> &mut ProviderOptionsMap {
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
            | Self::FileId {
                provider_options, ..
            }
            | Self::ImageData {
                provider_options, ..
            }
            | Self::ImageUrl {
                provider_options, ..
            }
            | Self::ImageFileId {
                provider_options, ..
            }
            | Self::Custom {
                provider_options, ..
            } => provider_options,
        }
    }

    /// Alias of `provider_options_mut()` that matches the wider shared-type naming convention.
    pub fn provider_options_map_mut(&mut self) -> &mut ProviderOptionsMap {
        self.provider_options_mut()
    }

    /// Replace provider options for this tool result content part.
    pub fn with_provider_options_map(mut self, provider_options_map: ProviderOptionsMap) -> Self {
        *self.provider_options_map_mut() = provider_options_map;
        self
    }

    /// Attach provider-specific request options to this tool result content part.
    pub fn with_provider_option(
        mut self,
        provider_id: impl AsRef<str>,
        value: serde_json::Value,
    ) -> Self {
        self.provider_options_mut().insert(provider_id, value);
        self
    }

    /// Get provider-specific request options for a provider id on this tool result content part.
    pub fn provider_option(&self, provider_id: impl AsRef<str>) -> Option<&serde_json::Value> {
        self.provider_options().get(provider_id)
    }
}

#[cfg(test)]
mod tests {
    use super::{ToolResultContentPart, ToolResultFileId, ToolResultOutput};
    use std::collections::HashMap;

    #[test]
    fn custom_tool_result_content_serializes_provider_options() {
        let part = ToolResultContentPart::custom()
            .with_provider_option("anthropic", serde_json::json!({ "type": "tool-reference" }));

        let value = serde_json::to_value(&part).expect("serialize custom tool result part");
        assert_eq!(value["type"], serde_json::json!("custom"));
        assert_eq!(
            value["providerOptions"]["anthropic"]["type"],
            serde_json::json!("tool-reference")
        );
    }

    #[test]
    fn tool_result_content_serializes_vercel_aligned_variants() {
        let file_id = ToolResultContentPart::file_id(HashMap::from([
            ("openai".to_string(), "file_openai".to_string()),
            ("anthropic".to_string(), "file_anthropic".to_string()),
        ]));
        let image_data = ToolResultContentPart::image_data("aGVsbG8=", "image/png");
        let file_data = ToolResultContentPart::file_data(
            "JVBERi0x",
            "application/pdf",
            Some("report.pdf".to_string()),
        );

        let file_id_json = serde_json::to_value(&file_id).expect("serialize file-reference part");
        assert_eq!(file_id_json["type"], serde_json::json!("file-reference"));
        assert_eq!(
            file_id_json["providerReference"]["openai"],
            serde_json::json!("file_openai")
        );

        let image_data_json = serde_json::to_value(&image_data).expect("serialize image-data part");
        assert_eq!(image_data_json["type"], serde_json::json!("image-data"));
        assert_eq!(image_data_json["mediaType"], serde_json::json!("image/png"));

        let file_data_json = serde_json::to_value(&file_data).expect("serialize file-data part");
        assert_eq!(file_data_json["type"], serde_json::json!("file-data"));
        assert_eq!(file_data_json["filename"], serde_json::json!("report.pdf"));
        assert_eq!(
            file_data_json["mediaType"],
            serde_json::json!("application/pdf")
        );
    }

    #[test]
    fn tool_result_file_id_prefers_requested_provider() {
        let file_id = ToolResultFileId::PerProvider(HashMap::from([
            ("anthropic".to_string(), "file_ant".to_string()),
            ("openai".to_string(), "file_oa".to_string()),
        ]));

        assert_eq!(file_id.preferred_value(&["openai"]), Some("file_oa"));
        assert_eq!(file_id.preferred_value(&["google"]), Some("file_ant"));
    }

    #[test]
    fn tool_result_content_accepts_provider_reference_aliases() {
        let file_reference: ToolResultContentPart = serde_json::from_value(serde_json::json!({
            "type": "file-reference",
            "providerReference": {
                "openai": "file_openai",
                "anthropic": "file_anthropic"
            }
        }))
        .expect("deserialize file-reference alias");
        let image_reference: ToolResultContentPart = serde_json::from_value(serde_json::json!({
            "type": "image-file-reference",
            "providerReference": {
                "openai": "image_openai"
            }
        }))
        .expect("deserialize image-file-reference alias");

        assert_eq!(
            file_reference,
            ToolResultContentPart::file_reference(HashMap::from([
                ("openai".to_string(), "file_openai".to_string()),
                ("anthropic".to_string(), "file_anthropic".to_string()),
            ]))
        );
        assert_eq!(
            image_reference,
            ToolResultContentPart::image_file_reference(HashMap::from([(
                "openai".to_string(),
                "image_openai".to_string(),
            )]))
        );
    }

    #[test]
    fn content_output_json_value_preserves_explicit_tool_result_parts() {
        let output = ToolResultOutput::content(vec![
            ToolResultContentPart::image_data("aGVsbG8=", "image/png"),
            ToolResultContentPart::file_url("https://example.com/report.pdf"),
            ToolResultContentPart::custom()
                .with_provider_option("anthropic", serde_json::json!({ "type": "tool-reference" })),
        ]);

        let value = output.to_json_value();
        let arr = value.as_array().expect("content array");
        assert_eq!(arr[0]["type"], serde_json::json!("image-data"));
        assert_eq!(arr[0]["mediaType"], serde_json::json!("image/png"));
        assert_eq!(arr[1]["type"], serde_json::json!("file-url"));
        assert_eq!(
            arr[1]["url"],
            serde_json::json!("https://example.com/report.pdf")
        );
        assert_eq!(arr[2]["type"], serde_json::json!("custom"));
        assert_eq!(
            arr[2]["providerOptions"]["anthropic"]["type"],
            serde_json::json!("tool-reference")
        );
    }

    #[test]
    fn tool_result_output_and_parts_expose_shared_provider_option_helpers() {
        let mut provider_options = crate::types::ProviderOptionsMap::default();
        provider_options.insert("openai", serde_json::json!({ "store": false }));

        let part =
            ToolResultContentPart::custom().with_provider_options_map(provider_options.clone());
        assert_eq!(part.provider_options_map(), &provider_options);
        assert_eq!(
            part.provider_option("openai"),
            Some(&serde_json::json!({ "store": false }))
        );

        let output = ToolResultOutput::json(serde_json::json!({ "ok": true }))
            .with_provider_options_map(provider_options.clone());
        assert_eq!(output.provider_options_map(), &provider_options);
        assert_eq!(
            output.provider_option("openai"),
            Some(&serde_json::json!({ "store": false }))
        );
    }
}

impl ToolResultOutput {
    /// Get provider options for this output variant.
    pub fn provider_options(&self) -> &ProviderOptionsMap {
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
            }
            | Self::Content {
                provider_options, ..
            } => provider_options,
        }
    }

    /// Alias of `provider_options()` that matches the wider shared-type naming convention.
    pub fn provider_options_map(&self) -> &ProviderOptionsMap {
        self.provider_options()
    }

    /// Get mutable provider options for this output variant.
    pub fn provider_options_mut(&mut self) -> &mut ProviderOptionsMap {
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
            }
            | Self::Content {
                provider_options, ..
            } => provider_options,
        }
    }

    /// Alias of `provider_options_mut()` that matches the wider shared-type naming convention.
    pub fn provider_options_map_mut(&mut self) -> &mut ProviderOptionsMap {
        self.provider_options_mut()
    }

    /// Replace provider options for this tool result output.
    pub fn with_provider_options_map(mut self, provider_options_map: ProviderOptionsMap) -> Self {
        *self.provider_options_map_mut() = provider_options_map;
        self
    }

    /// Attach provider-specific request options to this tool result output.
    pub fn with_provider_option(
        mut self,
        provider_id: impl AsRef<str>,
        value: serde_json::Value,
    ) -> Self {
        self.provider_options_mut().insert(provider_id, value);
        self
    }

    /// Get provider-specific request options for a provider id on this tool result output.
    pub fn provider_option(&self, provider_id: impl AsRef<str>) -> Option<&serde_json::Value> {
        self.provider_options().get(provider_id)
    }
}
