use serde::{Deserialize, Serialize};

use super::{ImageDetail, MediaSource};

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
    ///
    /// Use this for simple text results that should be sent directly to the API.
    Text { value: String },

    /// JSON output
    ///
    /// Use this for structured data results.
    Json { value: serde_json::Value },

    /// Execution was denied
    ///
    /// Use this when the user or system denies the execution of a tool call.
    /// This is useful for tool approval workflows.
    ExecutionDenied {
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },

    /// Text error
    ///
    /// Use this for error messages in plain text format.
    ErrorText { value: String },

    /// JSON error
    ///
    /// Use this for structured error information.
    ErrorJson { value: serde_json::Value },

    /// Multimodal content (text, images, files)
    ///
    /// Use this when the tool result contains multiple content parts,
    /// such as text combined with images or files.
    ///
    /// Note: This creates a recursive structure. The inner ContentParts
    /// should not contain ToolCall or ToolResult to avoid infinite recursion.
    Content { value: Vec<ToolResultContentPart> },
}

/// Content part for tool results
///
/// This is a subset of ContentPart that can be used in tool results.
/// It excludes ToolCall and ToolResult to avoid infinite recursion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ToolResultContentPart {
    /// Text content
    Text { text: String },

    /// Image content
    Image {
        #[serde(flatten)]
        source: MediaSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
    },

    /// File content
    File {
        #[serde(flatten)]
        source: MediaSource,
        media_type: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
}

impl ToolResultOutput {
    /// Create a text output
    pub fn text(value: impl Into<String>) -> Self {
        Self::Text {
            value: value.into(),
        }
    }

    /// Create a JSON output
    pub fn json(value: serde_json::Value) -> Self {
        Self::Json { value }
    }

    /// Create an execution denied output
    pub fn execution_denied(reason: Option<String>) -> Self {
        Self::ExecutionDenied { reason }
    }

    /// Create a text error output
    pub fn error_text(value: impl Into<String>) -> Self {
        Self::ErrorText {
            value: value.into(),
        }
    }

    /// Create a JSON error output
    pub fn error_json(value: serde_json::Value) -> Self {
        Self::ErrorJson { value }
    }

    /// Create a multimodal content output
    pub fn content(value: Vec<ToolResultContentPart>) -> Self {
        Self::Content { value }
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
            Self::Text { value } => value.clone(),
            Self::Json { value } => serde_json::to_string(value).unwrap_or_default(),
            Self::ExecutionDenied { reason } => reason
                .clone()
                .unwrap_or_else(|| "Execution denied".to_string()),
            Self::ErrorText { value } => value.clone(),
            Self::ErrorJson { value } => serde_json::to_string(value).unwrap_or_default(),
            Self::Content { value } => {
                // Simplified representation
                format!("Multimodal content with {} parts", value.len())
            }
        }
    }

    /// Convert to JSON value for provider APIs
    pub fn to_json_value(&self) -> serde_json::Value {
        match self {
            Self::Text { value } => serde_json::Value::String(value.clone()),
            Self::Json { value } => value.clone(),
            Self::ExecutionDenied { reason } => {
                serde_json::json!({
                    "error": "execution_denied",
                    "reason": reason.clone().unwrap_or_else(|| "Execution denied".to_string())
                })
            }
            Self::ErrorText { value } => serde_json::Value::String(value.clone()),
            Self::ErrorJson { value } => value.clone(),
            Self::Content { value } => {
                // Convert multimodal content to JSON array
                let content_array: Vec<serde_json::Value> = value
                    .iter()
                    .map(|part| match part {
                        ToolResultContentPart::Text { text } => {
                            serde_json::json!({"type": "text", "text": text})
                        }
                        ToolResultContentPart::Image { source, .. } => {
                            use MediaSource;
                            match source {
                                MediaSource::Url { url } => {
                                    serde_json::json!({"type": "image", "url": url})
                                }
                                MediaSource::Base64 { data } => {
                                    serde_json::json!({"type": "image", "data": data})
                                }
                                MediaSource::Binary { .. } => {
                                    serde_json::json!({"type": "text", "text": "[Binary image data]"})
                                }
                            }
                        }
                        ToolResultContentPart::File { .. } => {
                            serde_json::json!({"type": "text", "text": "[File attachment]"})
                        }
                    })
                    .collect();
                serde_json::Value::Array(content_array)
            }
        }
    }
}

impl ToolResultContentPart {
    /// Create a text content part
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Create an image content part from URL
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::Image {
            source: MediaSource::url(url),
            detail: None,
        }
    }

    /// Create an image content part from base64 data
    pub fn image_base64(data: impl Into<String>) -> Self {
        Self::Image {
            source: MediaSource::base64(data),
            detail: None,
        }
    }

    /// Create a file content part from URL
    pub fn file_url(url: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self::File {
            source: MediaSource::url(url),
            media_type: media_type.into(),
            filename: None,
        }
    }

    /// Create a file content part from base64 data
    pub fn file_base64(
        data: impl Into<String>,
        media_type: impl Into<String>,
        filename: Option<String>,
    ) -> Self {
        Self::File {
            source: MediaSource::base64(data),
            media_type: media_type.into(),
            filename,
        }
    }
}
