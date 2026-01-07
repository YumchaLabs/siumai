//! Content types for chat messages

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::metadata::{ToolCallInfo, ToolResultInfo};

/// Message content - supports multimodality
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MessageContent {
    /// Plain text
    Text(String),
    /// Multimodal content
    MultiModal(Vec<ContentPart>),
    /// Structured JSON content (optional feature)
    #[cfg(feature = "structured-messages")]
    Json(serde_json::Value),
}

impl MessageContent {
    /// Extract text content if available
    pub fn text(&self) -> Option<&str> {
        match self {
            MessageContent::Text(text) => Some(text),
            MessageContent::MultiModal(parts) => {
                // Return the first text part found
                for part in parts {
                    if let ContentPart::Text { text } = part {
                        return Some(text);
                    }
                }
                None
            }
            #[cfg(feature = "structured-messages")]
            MessageContent::Json(_) => None,
        }
    }

    /// Extract all text content
    pub fn all_text(&self) -> String {
        match self {
            MessageContent::Text(text) => text.clone(),
            MessageContent::MultiModal(parts) => {
                let mut result = String::new();
                for part in parts {
                    if let ContentPart::Text { text } = part {
                        if !result.is_empty() {
                            result.push(' ');
                        }
                        result.push_str(text);
                    }
                }
                result
            }
            #[cfg(feature = "structured-messages")]
            MessageContent::Json(v) => serde_json::to_string(v).unwrap_or_default(),
        }
    }

    /// Get multimodal content parts if this is multimodal content
    pub fn as_multimodal(&self) -> Option<&Vec<ContentPart>> {
        match self {
            MessageContent::MultiModal(parts) => Some(parts),
            _ => None,
        }
    }
}

/// Media source - unified way to represent media data across providers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum MediaSource {
    /// URL (http, https, gs, data URLs, etc.)
    Url { url: String },
    /// Base64-encoded data
    Base64 { data: String },
    /// Binary data (will be base64-encoded when needed)
    #[serde(skip)]
    Binary { data: Vec<u8> },
}

impl MediaSource {
    /// Create from URL string
    pub fn url(url: impl Into<String>) -> Self {
        Self::Url { url: url.into() }
    }

    /// Create from base64 string
    pub fn base64(data: impl Into<String>) -> Self {
        Self::Base64 { data: data.into() }
    }

    /// Create from binary data
    pub fn binary(data: Vec<u8>) -> Self {
        Self::Binary { data }
    }

    /// Get as URL if available
    pub fn as_url(&self) -> Option<&str> {
        match self {
            Self::Url { url } => Some(url),
            _ => None,
        }
    }

    /// Get as base64 if available, or convert binary to base64
    pub fn as_base64(&self) -> Option<String> {
        match self {
            Self::Base64 { data } => Some(data.clone()),
            Self::Binary { data } => Some(base64_encode(data)),
            _ => None,
        }
    }

    /// Check if this is a URL
    pub fn is_url(&self) -> bool {
        matches!(self, Self::Url { .. })
    }

    /// Check if this is base64 data
    pub fn is_base64(&self) -> bool {
        matches!(self, Self::Base64 { .. })
    }

    /// Check if this is binary data
    pub fn is_binary(&self) -> bool {
        matches!(self, Self::Binary { .. })
    }
}

/// Image detail level (for providers that support it)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    Auto,
    Low,
    High,
}

impl From<&str> for ImageDetail {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "low" => ImageDetail::Low,
            "high" => ImageDetail::High,
            _ => ImageDetail::Auto,
        }
    }
}

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

/// Content part - provider-agnostic multimodal content
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ContentPart {
    /// Text content
    Text { text: String },

    /// Image content - supports URL, base64, or binary data
    Image {
        /// Image data source
        #[serde(flatten)]
        source: MediaSource,
        /// Optional detail level (for providers that support it, e.g., OpenAI)
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
    },

    /// Audio content - supports URL, base64, or binary data
    Audio {
        /// Audio data source
        #[serde(flatten)]
        source: MediaSource,
        /// Media type (e.g., "audio/wav", "audio/mp3", "audio/mpeg")
        #[serde(skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,
    },

    /// File content (PDF, documents, etc.)
    File {
        /// File data source
        #[serde(flatten)]
        source: MediaSource,
        /// Media type (e.g., "application/pdf", "text/plain")
        media_type: String,
        /// Optional filename (for providers that support it)
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },

    /// Tool call (function call request from AI)
    ///
    /// This represents a request from the AI model to call a tool/function.
    /// The tool can be either user-defined (executed by your code) or
    /// provider-defined (executed by the provider, e.g., web search).
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ContentPart;
    /// use serde_json::json;
    ///
    /// let tool_call = ContentPart::tool_call(
    ///     "call_123",
    ///     "search",
    ///     json!({"query": "rust programming"}),
    ///     None, // user-defined function
    /// );
    /// ```
    #[serde(rename = "tool-call")]
    ToolCall {
        /// Tool call ID (used to match with tool result)
        #[serde(rename = "toolCallId")]
        tool_call_id: String,

        /// Tool/function name
        #[serde(rename = "toolName")]
        tool_name: String,

        /// Arguments as JSON value
        #[serde(rename = "input")]
        arguments: serde_json::Value,

        /// Whether this tool will be executed by the provider
        ///
        /// - `Some(true)`: Provider-defined tool (e.g., web search, code execution)
        /// - `None` or `Some(false)`: User-defined function
        #[serde(
            rename = "providerExecuted",
            alias = "provider_executed",
            skip_serializing_if = "Option::is_none"
        )]
        provider_executed: Option<bool>,

        /// Provider-specific metadata (Vercel-aligned).
        #[serde(
            rename = "providerMetadata",
            alias = "provider_metadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<HashMap<String, serde_json::Value>>,
    },

    /// Tool approval response (for MCP approval workflows)
    ///
    /// This represents a user/system approval decision for a pending tool call,
    /// aligned with Vercel AI SDK's `tool-approval-response` part.
    #[serde(rename = "tool-approval-response")]
    ToolApprovalResponse {
        /// The approval request ID (MCP approval request item id)
        #[serde(rename = "approvalId")]
        approval_id: String,
        /// Whether the tool call was approved
        approved: bool,
    },

    /// Tool result (function execution result)
    ///
    /// This represents the result of executing a tool/function.
    /// It matches a previous tool call by ID.
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::{ContentPart, ToolResultOutput};
    /// use serde_json::json;
    ///
    /// // Success result
    /// let result = ContentPart::tool_result_json(
    ///     "call_123",
    ///     "search",
    ///     json!({"results": ["..."]}),
    /// );
    ///
    /// // Error result
    /// let error = ContentPart::tool_error(
    ///     "call_123",
    ///     "search",
    ///     "API timeout",
    /// );
    /// ```
    #[serde(rename = "tool-result")]
    ToolResult {
        /// Tool call ID (matches the tool call)
        #[serde(rename = "toolCallId")]
        tool_call_id: String,

        /// Tool/function name
        #[serde(rename = "toolName")]
        tool_name: String,

        /// Structured output
        #[serde(rename = "output")]
        output: ToolResultOutput,

        /// Whether this result was generated by the provider
        #[serde(
            rename = "providerExecuted",
            alias = "provider_executed",
            skip_serializing_if = "Option::is_none"
        )]
        provider_executed: Option<bool>,

        /// Provider-specific metadata (Vercel-aligned).
        #[serde(
            rename = "providerMetadata",
            alias = "provider_metadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<HashMap<String, serde_json::Value>>,
    },

    /// Reasoning/thinking content
    ///
    /// This represents the model's reasoning or thinking process.
    /// Supported by models like:
    /// - OpenAI o1, o3
    /// - DeepSeek-R1
    /// - Gemini (thought summary)
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ContentPart;
    ///
    /// let reasoning = ContentPart::reasoning(
    ///     "Let me think about this step by step..."
    /// );
    /// ```
    Reasoning {
        text: String,

        /// Provider-specific metadata (Vercel-aligned).
        #[serde(
            rename = "providerMetadata",
            alias = "provider_metadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<HashMap<String, serde_json::Value>>,
    },
}

// Helper function for base64 encoding
fn base64_encode(data: &[u8]) -> String {
    use base64::{Engine, engine::general_purpose::STANDARD};
    STANDARD.encode(data)
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

impl ContentPart {
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

    /// Create an image content part from URL with detail level
    pub fn image_url_with_detail(url: impl Into<String>, detail: ImageDetail) -> Self {
        Self::Image {
            source: MediaSource::url(url),
            detail: Some(detail),
        }
    }

    /// Create an image content part from base64 data
    pub fn image_base64(data: impl Into<String>) -> Self {
        Self::Image {
            source: MediaSource::base64(data),
            detail: None,
        }
    }

    /// Create an image content part from binary data
    pub fn image_binary(data: Vec<u8>) -> Self {
        Self::Image {
            source: MediaSource::binary(data),
            detail: None,
        }
    }

    /// Create an audio content part from URL
    pub fn audio_url(url: impl Into<String>, media_type: Option<String>) -> Self {
        Self::Audio {
            source: MediaSource::url(url),
            media_type,
        }
    }

    /// Create an audio content part from base64 data
    pub fn audio_base64(data: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self::Audio {
            source: MediaSource::base64(data),
            media_type: Some(media_type.into()),
        }
    }

    /// Create an audio content part from binary data
    pub fn audio_binary(data: Vec<u8>, media_type: impl Into<String>) -> Self {
        Self::Audio {
            source: MediaSource::binary(data),
            media_type: Some(media_type.into()),
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

    /// Create a file content part from binary data
    pub fn file_binary(
        data: Vec<u8>,
        media_type: impl Into<String>,
        filename: Option<String>,
    ) -> Self {
        Self::File {
            source: MediaSource::binary(data),
            media_type: media_type.into(),
            filename,
        }
    }

    /// Create a tool call content part
    ///
    /// # Arguments
    ///
    /// * `tool_call_id` - Unique ID for this tool call
    /// * `tool_name` - Name of the tool/function to call
    /// * `arguments` - JSON value of arguments
    /// * `provider_executed` - Whether this tool is executed by the provider
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ContentPart;
    /// use serde_json::json;
    ///
    /// // User-defined function
    /// let call = ContentPart::tool_call(
    ///     "call_123",
    ///     "get_weather",
    ///     json!({"location": "Tokyo"}),
    ///     None,
    /// );
    ///
    /// // Provider-defined tool
    /// let search = ContentPart::tool_call(
    ///     "call_456",
    ///     "web_search",
    ///     json!({"query": "rust"}),
    ///     Some(true),
    /// );
    /// ```
    pub fn tool_call(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        arguments: serde_json::Value,
        provider_executed: Option<bool>,
    ) -> Self {
        Self::ToolCall {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            arguments,
            provider_executed,
            provider_metadata: None,
        }
    }

    /// Create a tool approval response content part.
    pub fn tool_approval_response(approval_id: impl Into<String>, approved: bool) -> Self {
        Self::ToolApprovalResponse {
            approval_id: approval_id.into(),
            approved,
        }
    }

    /// Create a tool result content part with text output
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ContentPart;
    ///
    /// let result = ContentPart::tool_result_text(
    ///     "call_123",
    ///     "get_weather",
    ///     "Temperature is 18°C, sunny",
    /// );
    /// ```
    pub fn tool_result_text(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        result: impl Into<String>,
    ) -> Self {
        Self::ToolResult {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            output: ToolResultOutput::text(result),
            provider_executed: None,
            provider_metadata: None,
        }
    }

    /// Create a tool result content part with JSON output
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ContentPart;
    /// use serde_json::json;
    ///
    /// let result = ContentPart::tool_result_json(
    ///     "call_123",
    ///     "get_weather",
    ///     json!({"temperature": 18, "condition": "sunny"}),
    /// );
    /// ```
    pub fn tool_result_json(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        result: serde_json::Value,
    ) -> Self {
        Self::ToolResult {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            output: ToolResultOutput::json(result),
            provider_executed: None,
            provider_metadata: None,
        }
    }

    /// Create a tool error content part with text error
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ContentPart;
    ///
    /// let error = ContentPart::tool_error(
    ///     "call_123",
    ///     "get_weather",
    ///     "API timeout",
    /// );
    /// ```
    pub fn tool_error(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        error: impl Into<String>,
    ) -> Self {
        Self::ToolResult {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            output: ToolResultOutput::error_text(error),
            provider_executed: None,
            provider_metadata: None,
        }
    }

    /// Create a tool error content part with JSON error
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ContentPart;
    /// use serde_json::json;
    ///
    /// let error = ContentPart::tool_error_json(
    ///     "call_123",
    ///     "get_weather",
    ///     json!({"error": "API timeout", "code": 504}),
    /// );
    /// ```
    pub fn tool_error_json(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        error: serde_json::Value,
    ) -> Self {
        Self::ToolResult {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            output: ToolResultOutput::error_json(error),
            provider_executed: None,
            provider_metadata: None,
        }
    }

    /// Create a tool result with execution denied
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ContentPart;
    ///
    /// let denied = ContentPart::tool_execution_denied(
    ///     "call_123",
    ///     "delete_file",
    ///     Some("User rejected the operation".to_string()),
    /// );
    /// ```
    pub fn tool_execution_denied(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        reason: Option<String>,
    ) -> Self {
        Self::ToolResult {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            output: ToolResultOutput::execution_denied(reason),
            provider_executed: None,
            provider_metadata: None,
        }
    }

    /// Create a tool result with multimodal content
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::{ContentPart, ToolResultContentPart};
    ///
    /// let result = ContentPart::tool_result_content(
    ///     "call_123",
    ///     "generate_image",
    ///     vec![
    ///         ToolResultContentPart::text("Generated image:"),
    ///         ToolResultContentPart::image_url("https://example.com/image.png"),
    ///     ],
    /// );
    /// ```
    pub fn tool_result_content(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        content: Vec<ToolResultContentPart>,
    ) -> Self {
        Self::ToolResult {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            output: ToolResultOutput::content(content),
            provider_executed: None,
            provider_metadata: None,
        }
    }

    /// Create a reasoning content part
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ContentPart;
    ///
    /// let reasoning = ContentPart::reasoning(
    ///     "Let me think about this step by step..."
    /// );
    /// ```
    pub fn reasoning(text: impl Into<String>) -> Self {
        Self::Reasoning {
            text: text.into(),
            provider_metadata: None,
        }
    }

    /// Check if this is a text part
    pub fn is_text(&self) -> bool {
        matches!(self, Self::Text { .. })
    }

    /// Check if this is an image part
    pub fn is_image(&self) -> bool {
        matches!(self, Self::Image { .. })
    }

    /// Check if this is an audio part
    pub fn is_audio(&self) -> bool {
        matches!(self, Self::Audio { .. })
    }

    /// Check if this is a file part
    pub fn is_file(&self) -> bool {
        matches!(self, Self::File { .. })
    }

    /// Check if this is a tool call
    pub fn is_tool_call(&self) -> bool {
        matches!(self, Self::ToolCall { .. })
    }

    /// Check if this is a tool result
    pub fn is_tool_result(&self) -> bool {
        matches!(self, Self::ToolResult { .. })
    }

    /// Check if this is reasoning
    pub fn is_reasoning(&self) -> bool {
        matches!(self, Self::Reasoning { .. })
    }

    /// Get the text content if this is a text part
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text } => Some(text),
            _ => None,
        }
    }

    /// Get the tool call ID if this is a tool call
    pub fn as_tool_call_id(&self) -> Option<&str> {
        match self {
            Self::ToolCall { tool_call_id, .. } => Some(tool_call_id),
            _ => None,
        }
    }

    /// Get the tool name if this is a tool call or tool result
    pub fn as_tool_name(&self) -> Option<&str> {
        match self {
            Self::ToolCall { tool_name, .. } | Self::ToolResult { tool_name, .. } => {
                Some(tool_name)
            }
            _ => None,
        }
    }

    /// Get tool call information if this is a tool call
    ///
    /// Returns a structured view of the tool call fields, making it easier to
    /// work with tool calls without manual pattern matching.
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::{ChatResponse, ContentPart};
    ///
    /// # let response = ChatResponse::empty();
    /// for tool_call in response.tool_calls() {
    ///     if let Some(info) = tool_call.as_tool_call() {
    ///         println!("Tool: {}", info.tool_name);
    ///         println!("ID: {}", info.tool_call_id);
    ///         println!("Args: {}", info.arguments);
    ///     }
    /// }
    /// ```
    pub fn as_tool_call(&self) -> Option<ToolCallInfo<'_>> {
        match self {
            Self::ToolCall {
                tool_call_id,
                tool_name,
                arguments,
                provider_executed,
                ..
            } => Some(ToolCallInfo {
                tool_call_id,
                tool_name,
                arguments,
                provider_executed: provider_executed.as_ref(),
            }),
            _ => None,
        }
    }

    /// Get tool result information if this is a tool result
    ///
    /// Returns a structured view of the tool result fields.
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::{ChatMessage, ContentPart};
    ///
    /// # let msg = ChatMessage::user("test").build();
    /// for part in msg.content.as_multimodal().unwrap_or(&vec![]) {
    ///     if let Some(info) = part.as_tool_result() {
    ///         println!("Tool: {}", info.tool_name);
    ///         println!("ID: {}", info.tool_call_id);
    ///     }
    /// }
    /// ```
    pub fn as_tool_result(&self) -> Option<ToolResultInfo<'_>> {
        match self {
            Self::ToolResult {
                tool_call_id,
                tool_name,
                output,
                provider_executed,
                ..
            } => Some(ToolResultInfo {
                tool_call_id,
                tool_name,
                output,
                provider_executed: provider_executed.as_ref(),
            }),
            _ => None,
        }
    }
}
