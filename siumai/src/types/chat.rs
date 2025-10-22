//! Chat-related types and message handling

use super::common::{CommonParams, FinishReason, HttpConfig, Usage, Warning};
use super::tools::{Tool, ToolCall};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Message role
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Developer, // Developer role for system-level instructions
    Tool,
}

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
/// let result = ToolResultOutput::execution_denied(Some("User rejected"));
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
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_executed: Option<bool>,
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
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_executed: Option<bool>,
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
    Reasoning { text: String },
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
    ///     Some("User rejected the operation"),
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
        Self::Reasoning { text: text.into() }
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

/// Tool call information (borrowed view)
///
/// Provides convenient access to tool call fields without pattern matching.
#[derive(Debug, Clone, Copy)]
pub struct ToolCallInfo<'a> {
    /// The tool call ID
    pub tool_call_id: &'a str,
    /// The tool name
    pub tool_name: &'a str,
    /// The tool arguments (JSON)
    pub arguments: &'a serde_json::Value,
    /// Whether the tool was executed by the provider
    pub provider_executed: Option<&'a bool>,
}

/// Tool result information (borrowed view)
///
/// Provides convenient access to tool result fields without pattern matching.
#[derive(Debug, Clone, Copy)]
pub struct ToolResultInfo<'a> {
    /// The tool call ID
    pub tool_call_id: &'a str,
    /// The tool name
    pub tool_name: &'a str,
    /// The tool output
    pub output: &'a ToolResultOutput,
    /// Whether the tool was executed by the provider
    pub provider_executed: Option<&'a bool>,
}

/// Cache control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheControl {
    /// Ephemeral cache
    Ephemeral,
    /// Persistent cache
    Persistent { ttl: Option<std::time::Duration> },
}

/// Message metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MessageMetadata {
    /// Message ID
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Timestamp
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
    /// Cache control (Anthropic-specific)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
    /// Custom metadata
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub custom: HashMap<String, serde_json::Value>,
}

/// Chat message
///
/// A message in a conversation. Content can include text, images, audio, files,
/// tool calls, tool results, and reasoning.
///
/// # Examples
///
/// ```rust
/// use siumai::types::{ChatMessage, ContentPart};
///
/// // Simple text message
/// let msg = ChatMessage::user("Hello!").build();
///
/// // Message with tool call
/// let msg = ChatMessage::assistant_with_content(vec![
///     ContentPart::text("Let me search for that..."),
///     ContentPart::tool_call("call_123", "search", r#"{"query":"rust"}"#, None),
/// ]).build();
///
/// // Tool result message
/// let msg = ChatMessage::tool_result(
///     "call_123",
///     "search",
///     r#"{"results":["..."]}"#,
///     false,
/// ).build();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role
    pub role: MessageRole,
    /// Content - can be text, multimodal (images, audio, files), tool calls, tool results, or reasoning
    pub content: MessageContent,
    /// Message metadata
    #[serde(default)]
    pub metadata: MessageMetadata,
}

impl ChatMessage {
    /// Creates a user message
    pub fn user<S: Into<String>>(content: S) -> ChatMessageBuilder {
        ChatMessageBuilder::user(content)
    }

    /// Creates a system message
    pub fn system<S: Into<String>>(content: S) -> ChatMessageBuilder {
        ChatMessageBuilder::system(content)
    }

    /// Creates an assistant message
    pub fn assistant<S: Into<String>>(content: S) -> ChatMessageBuilder {
        ChatMessageBuilder::assistant(content)
    }

    /// Creates a developer message
    pub fn developer<S: Into<String>>(content: S) -> ChatMessageBuilder {
        ChatMessageBuilder::developer(content)
    }

    /// Creates a tool message (deprecated - use tool_result instead)
    #[deprecated(since = "0.12.0", note = "Use `tool_result` instead")]
    pub fn tool<S: Into<String>>(content: S, tool_call_id: S) -> ChatMessageBuilder {
        ChatMessageBuilder::tool(content, tool_call_id)
    }

    /// Creates an assistant message with multimodal content
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::{ChatMessage, ContentPart};
    ///
    /// let msg = ChatMessage::assistant_with_content(vec![
    ///     ContentPart::text("Let me search for that..."),
    ///     ContentPart::tool_call("call_123", "search", r#"{"query":"rust"}"#, None),
    /// ]).build();
    /// ```
    pub fn assistant_with_content(content: Vec<ContentPart>) -> ChatMessageBuilder {
        ChatMessageBuilder {
            role: MessageRole::Assistant,
            content: Some(MessageContent::MultiModal(content)),
            metadata: MessageMetadata::default(),
        }
    }

    /// Creates a tool result message with text output
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ChatMessage;
    ///
    /// let msg = ChatMessage::tool_result_text(
    ///     "call_123",
    ///     "get_weather",
    ///     "Temperature is 18°C",
    /// ).build();
    /// ```
    pub fn tool_result_text(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        result: impl Into<String>,
    ) -> ChatMessageBuilder {
        ChatMessageBuilder {
            role: MessageRole::Tool,
            content: Some(MessageContent::MultiModal(vec![
                ContentPart::tool_result_text(tool_call_id, tool_name, result),
            ])),
            metadata: MessageMetadata::default(),
        }
    }

    /// Creates a tool result message with JSON output
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ChatMessage;
    /// use serde_json::json;
    ///
    /// let msg = ChatMessage::tool_result_json(
    ///     "call_123",
    ///     "get_weather",
    ///     json!({"temperature": 18}),
    /// ).build();
    /// ```
    pub fn tool_result_json(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        result: serde_json::Value,
    ) -> ChatMessageBuilder {
        ChatMessageBuilder {
            role: MessageRole::Tool,
            content: Some(MessageContent::MultiModal(vec![
                ContentPart::tool_result_json(tool_call_id, tool_name, result),
            ])),
            metadata: MessageMetadata::default(),
        }
    }

    /// Creates a tool error message
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ChatMessage;
    ///
    /// let msg = ChatMessage::tool_error(
    ///     "call_123",
    ///     "get_weather",
    ///     "API timeout",
    /// ).build();
    /// ```
    pub fn tool_error(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        error: impl Into<String>,
    ) -> ChatMessageBuilder {
        ChatMessageBuilder {
            role: MessageRole::Tool,
            content: Some(MessageContent::MultiModal(vec![ContentPart::tool_error(
                tool_call_id,
                tool_name,
                error,
            )])),
            metadata: MessageMetadata::default(),
        }
    }

    /// Creates a tool error message with JSON error
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ChatMessage;
    /// use serde_json::json;
    ///
    /// let msg = ChatMessage::tool_error_json(
    ///     "call_123",
    ///     "get_weather",
    ///     json!({"error": "API timeout", "code": 504}),
    /// ).build();
    /// ```
    pub fn tool_error_json(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        error: serde_json::Value,
    ) -> ChatMessageBuilder {
        ChatMessageBuilder {
            role: MessageRole::Tool,
            content: Some(MessageContent::MultiModal(vec![
                ContentPart::tool_error_json(tool_call_id, tool_name, error),
            ])),
            metadata: MessageMetadata::default(),
        }
    }

    /// Gets the text content of the message
    pub fn content_text(&self) -> Option<&str> {
        match &self.content {
            MessageContent::Text(text) => Some(text),
            MessageContent::MultiModal(parts) => parts.iter().find_map(|part| {
                if let ContentPart::Text { text } = part {
                    Some(text.as_str())
                } else {
                    None
                }
            }),
            #[cfg(feature = "structured-messages")]
            MessageContent::Json(_) => None,
        }
    }

    /// Create a user message from static string (zero-copy for string literals)
    pub fn user_static(content: &'static str) -> ChatMessageBuilder {
        ChatMessageBuilder::user(content)
    }

    /// Create an assistant message from static string (zero-copy for string literals)
    pub fn assistant_static(content: &'static str) -> ChatMessageBuilder {
        ChatMessageBuilder::assistant(content)
    }

    /// Create a system message from static string (zero-copy for string literals)
    pub fn system_static(content: &'static str) -> ChatMessageBuilder {
        ChatMessageBuilder::system(content)
    }

    /// Create a user message with pre-allocated capacity for content
    pub fn user_with_capacity(content: String, _capacity_hint: usize) -> ChatMessageBuilder {
        // Note: In a real implementation, you might use the capacity hint
        // to pre-allocate string buffers for multimodal content
        ChatMessageBuilder::user(content)
    }

    /// Check if message is empty (optimization for filtering)
    pub const fn is_empty(&self) -> bool {
        match &self.content {
            MessageContent::Text(text) => text.is_empty(),
            MessageContent::MultiModal(parts) => parts.is_empty(),
            #[cfg(feature = "structured-messages")]
            MessageContent::Json(_) => false,
        }
    }

    /// Get content length for memory estimation
    pub fn content_length(&self) -> usize {
        match &self.content {
            MessageContent::Text(text) => text.len(),
            MessageContent::MultiModal(parts) => parts
                .iter()
                .map(|part| match part {
                    ContentPart::Text { text } => text.len(),
                    ContentPart::Image { source, .. }
                    | ContentPart::Audio { source, .. }
                    | ContentPart::File { source, .. } => match source {
                        MediaSource::Url { url } => url.len(),
                        MediaSource::Base64 { data } => data.len(),
                        MediaSource::Binary { data } => data.len(),
                    },
                    ContentPart::ToolCall { arguments, .. } => serde_json::to_string(arguments)
                        .map(|s| s.len())
                        .unwrap_or(0),
                    ContentPart::ToolResult { output, .. } => output.to_string_lossy().len(),
                    ContentPart::Reasoning { text } => text.len(),
                })
                .sum(),
            #[cfg(feature = "structured-messages")]
            MessageContent::Json(v) => serde_json::to_string(v).map(|s| s.len()).unwrap_or(0),
        }
    }

    /// Extract all tool calls from content
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::{ChatMessage, ContentPart};
    ///
    /// let msg = ChatMessage::assistant_with_content(vec![
    ///     ContentPart::text("Let me search..."),
    ///     ContentPart::tool_call("call_123", "search", r#"{}"#, None),
    /// ]).build();
    ///
    /// let tool_calls = msg.tool_calls();
    /// assert_eq!(tool_calls.len(), 1);
    /// ```
    pub fn tool_calls(&self) -> Vec<&ContentPart> {
        match &self.content {
            MessageContent::MultiModal(parts) => {
                parts.iter().filter(|p| p.is_tool_call()).collect()
            }
            _ => vec![],
        }
    }

    /// Extract all tool results from content
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ChatMessage;
    ///
    /// let msg = ChatMessage::tool_result(
    ///     "call_123",
    ///     "search",
    ///     r#"{"results":[]}"#,
    ///     false,
    /// ).build();
    ///
    /// let results = msg.tool_results();
    /// assert_eq!(results.len(), 1);
    /// ```
    pub fn tool_results(&self) -> Vec<&ContentPart> {
        match &self.content {
            MessageContent::MultiModal(parts) => {
                parts.iter().filter(|p| p.is_tool_result()).collect()
            }
            _ => vec![],
        }
    }

    /// Check if message contains tool calls
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls().is_empty()
    }

    /// Check if message contains tool results
    pub fn has_tool_results(&self) -> bool {
        !self.tool_results().is_empty()
    }

    /// Extract all reasoning content from message
    pub fn reasoning(&self) -> Vec<&str> {
        match &self.content {
            MessageContent::MultiModal(parts) => parts
                .iter()
                .filter_map(|p| {
                    if let ContentPart::Reasoning { text } = p {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
                .collect(),
            _ => vec![],
        }
    }

    /// Check if message contains reasoning
    pub fn has_reasoning(&self) -> bool {
        !self.reasoning().is_empty()
    }

    /// Get message metadata (always available due to default)
    pub fn metadata(&self) -> &MessageMetadata {
        &self.metadata
    }

    /// Get mutable reference to metadata
    pub fn metadata_mut(&mut self) -> &mut MessageMetadata {
        &mut self.metadata
    }

    /// Check if message has any metadata set
    pub fn has_metadata(&self) -> bool {
        self.metadata.id.is_some()
            || self.metadata.timestamp.is_some()
            || self.metadata.cache_control.is_some()
            || !self.metadata.custom.is_empty()
    }
}

/// Chat message builder
#[derive(Debug, Clone)]
pub struct ChatMessageBuilder {
    role: MessageRole,
    content: Option<MessageContent>,
    metadata: MessageMetadata,
}

impl ChatMessageBuilder {
    /// Creates a user message builder
    pub fn user<S: Into<String>>(content: S) -> Self {
        Self {
            role: MessageRole::User,
            content: Some(MessageContent::Text(content.into())),
            metadata: MessageMetadata::default(),
        }
    }

    /// Creates a user message builder with pre-allocated capacity
    pub fn user_with_capacity(content: String, _capacity_hint: usize) -> Self {
        // Note: In a real implementation, you might use the capacity hint
        // to pre-allocate vectors for multimodal content
        Self {
            role: MessageRole::User,
            content: Some(MessageContent::Text(content)),
            metadata: MessageMetadata::default(),
        }
    }

    /// Creates a system message builder
    pub fn system<S: Into<String>>(content: S) -> Self {
        Self {
            role: MessageRole::System,
            content: Some(MessageContent::Text(content.into())),
            metadata: MessageMetadata::default(),
        }
    }

    /// Creates an assistant message builder
    pub fn assistant<S: Into<String>>(content: S) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: Some(MessageContent::Text(content.into())),
            metadata: MessageMetadata::default(),
        }
    }

    /// Creates a developer message builder
    pub fn developer<S: Into<String>>(content: S) -> Self {
        Self {
            role: MessageRole::Developer,
            content: Some(MessageContent::Text(content.into())),
            metadata: MessageMetadata::default(),
        }
    }

    /// Creates a tool message builder (deprecated - use ChatMessage::tool_result_text instead)
    #[deprecated(since = "0.12.0", note = "Use `ChatMessage::tool_result_text` instead")]
    pub fn tool<S: Into<String>>(content: S, tool_call_id: S) -> Self {
        // Convert to new format: create a tool result content part
        Self {
            role: MessageRole::Tool,
            content: Some(MessageContent::MultiModal(vec![ContentPart::ToolResult {
                tool_call_id: tool_call_id.into(),
                tool_name: String::new(), // Unknown in old API
                output: ToolResultOutput::text(content),
                provider_executed: None,
            }])),
            metadata: MessageMetadata::default(),
        }
    }

    /// Sets cache control
    pub const fn cache_control(mut self, cache: CacheControl) -> Self {
        self.metadata.cache_control = Some(cache);
        self
    }

    /// Sets cache control for a specific multimodal content part (Anthropic only)
    /// The index refers to the position in the final content array after transformation.
    pub fn cache_control_for_part(mut self, index: usize, _cache: CacheControl) -> Self {
        use serde_json::Value;
        // Collect existing indices from metadata.custom
        let key = "anthropic_content_cache_indices".to_string();
        let mut indices: Vec<usize> = self
            .metadata
            .custom
            .get(&key)
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_u64().map(|u| u as usize))
                    .collect()
            })
            .unwrap_or_default();
        if !indices.contains(&index) {
            indices.push(index);
        }
        indices.sort_unstable();
        self.metadata.custom.insert(
            key,
            Value::Array(indices.into_iter().map(|i| Value::from(i as u64)).collect()),
        );
        self
    }

    /// Sets cache control for multiple multimodal content parts (Anthropic only)
    pub fn cache_control_for_parts<I: IntoIterator<Item = usize>>(
        self,
        idx: I,
        cache: CacheControl,
    ) -> Self {
        let mut me = self;
        for i in idx {
            me = me.cache_control_for_part(i, cache.clone());
        }
        me
    }

    /// Adds image content
    pub fn with_image(mut self, image_url: String, detail: Option<String>) -> Self {
        let image_part = ContentPart::Image {
            source: MediaSource::Url { url: image_url },
            detail: detail.map(|d| ImageDetail::from(d.as_str())),
        };

        match self.content {
            Some(MessageContent::Text(text)) => {
                self.content = Some(MessageContent::MultiModal(vec![
                    ContentPart::Text { text },
                    image_part,
                ]));
            }
            Some(MessageContent::MultiModal(ref mut parts)) => {
                parts.push(image_part);
            }
            #[cfg(feature = "structured-messages")]
            Some(MessageContent::Json(v)) => {
                let text = serde_json::to_string(&v).unwrap_or_default();
                self.content = Some(MessageContent::MultiModal(vec![
                    ContentPart::Text { text },
                    image_part,
                ]));
            }
            None => {
                self.content = Some(MessageContent::MultiModal(vec![image_part]));
            }
        }

        self
    }

    /// Adds tool calls (deprecated - use with_content_parts instead)
    #[deprecated(
        since = "0.12.0",
        note = "Tool calls are now part of content. Use `with_content_parts` or create message with `ChatMessage::assistant_with_content`"
    )]
    pub fn with_tool_calls(mut self, tool_calls: Vec<crate::types::ToolCall>) -> Self {
        // Convert old ToolCall to new ContentPart::ToolCall
        let mut parts = match self.content {
            Some(MessageContent::Text(text)) if !text.is_empty() => {
                vec![ContentPart::Text { text }]
            }
            Some(MessageContent::MultiModal(parts)) => parts,
            _ => vec![],
        };

        for tc in tool_calls {
            if let Some(function) = tc.function {
                // Parse arguments string to JSON Value
                let arguments = serde_json::from_str(&function.arguments)
                    .unwrap_or_else(|_| serde_json::Value::String(function.arguments.clone()));

                parts.push(ContentPart::ToolCall {
                    tool_call_id: tc.id,
                    tool_name: function.name,
                    arguments,
                    provider_executed: None,
                });
            }
        }

        self.content = Some(MessageContent::MultiModal(parts));
        self
    }

    /// Adds content parts to the message
    pub fn with_content_parts(mut self, new_parts: Vec<ContentPart>) -> Self {
        let mut parts = match self.content {
            Some(MessageContent::Text(text)) if !text.is_empty() => {
                vec![ContentPart::Text { text }]
            }
            Some(MessageContent::MultiModal(parts)) => parts,
            _ => vec![],
        };

        parts.extend(new_parts);
        self.content = Some(MessageContent::MultiModal(parts));
        self
    }

    /// Builds the message
    pub fn build(self) -> ChatMessage {
        ChatMessage {
            role: self.role,
            content: self.content.unwrap_or(MessageContent::Text(String::new())),
            metadata: self.metadata,
        }
    }
}

/// Chat request configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatRequest {
    /// The conversation messages
    pub messages: Vec<ChatMessage>,
    /// Optional tools to use in the chat
    pub tools: Option<Vec<Tool>>,
    /// Tool choice strategy
    ///
    /// Controls how the model should use the provided tools:
    /// - `Auto` (default): Model decides whether to call tools
    /// - `Required`: Model must call at least one tool
    /// - `None`: Model cannot call any tools
    /// - `Tool { name }`: Model must call the specified tool
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::{ChatRequest, ChatMessage, ToolChoice};
    ///
    /// let request = ChatRequest::new(vec![
    ///     ChatMessage::user("What's the weather?").build()
    /// ])
    /// .with_tool_choice(ToolChoice::tool("weather"));
    /// ```
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<crate::types::ToolChoice>,
    /// Common parameters (for backward compatibility)
    pub common_params: CommonParams,

    /// Provider-specific options (type-safe!)
    #[serde(default)]
    pub provider_options: crate::types::ProviderOptions,

    /// HTTP configuration
    pub http_config: Option<HttpConfig>,

    /// Stream the response
    pub stream: bool,
    /// Optional telemetry configuration
    #[serde(skip)]
    pub telemetry: Option<crate::telemetry::TelemetryConfig>,
}

impl ChatRequest {
    /// Create a new chat request with messages
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self {
            messages,
            tools: None,
            tool_choice: None,
            common_params: CommonParams::default(),
            provider_options: crate::types::ProviderOptions::None,
            http_config: None,
            stream: false,
            telemetry: None,
        }
    }

    /// Create a builder for the chat request
    pub fn builder() -> ChatRequestBuilder {
        ChatRequestBuilder::new()
    }

    /// Add a message to the request
    pub fn with_message(mut self, message: ChatMessage) -> Self {
        self.messages.push(message);
        self
    }

    /// Add multiple messages to the request
    pub fn with_messages(mut self, messages: Vec<ChatMessage>) -> Self {
        self.messages.extend(messages);
        self
    }

    /// Add tools to the request
    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set tool choice strategy
    ///
    /// Controls how the model should use the provided tools.
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::{ChatRequest, ChatMessage, ToolChoice};
    ///
    /// // Force the model to call a specific tool
    /// let request = ChatRequest::new(vec![
    ///     ChatMessage::user("What's the weather?").build()
    /// ])
    /// .with_tool_choice(ToolChoice::tool("weather"));
    ///
    /// // Require the model to call at least one tool
    /// let request = ChatRequest::new(vec![
    ///     ChatMessage::user("Help me").build()
    /// ])
    /// .with_tool_choice(ToolChoice::Required);
    /// ```
    pub fn with_tool_choice(mut self, choice: crate::types::ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Enable streaming
    pub const fn with_streaming(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Set common parameters
    pub fn with_common_params(mut self, params: CommonParams) -> Self {
        self.common_params = params;
        self
    }

    /// Set model parameters (alias for `common_params`)
    pub fn with_model_params(mut self, params: CommonParams) -> Self {
        self.common_params = params;
        self
    }

    // ============================================================================
    // 🎯 NEW: Type-safe provider options (v0.12+)
    // ============================================================================

    /// Set provider-specific options (type-safe!)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use siumai::types::{ChatRequest, ProviderOptions, XaiOptions, XaiSearchParameters};
    ///
    /// let req = ChatRequest::new(messages)
    ///     .with_provider_options(ProviderOptions::Xai(
    ///         XaiOptions::new().with_default_search()
    ///     ));
    /// ```
    pub fn with_provider_options(mut self, options: crate::types::ProviderOptions) -> Self {
        self.provider_options = options;
        self
    }

    /// Convenience: Set OpenAI-specific options
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use siumai::types::{ChatRequest, OpenAiOptions};
    ///
    /// let req = ChatRequest::new(messages)
    ///     .with_openai_options(
    ///         OpenAiOptions::new().with_web_search()
    ///     );
    /// ```
    pub fn with_openai_options(mut self, options: crate::types::OpenAiOptions) -> Self {
        self.provider_options = crate::types::ProviderOptions::OpenAi(options);
        self
    }

    /// Convenience: Set xAI-specific options
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use siumai::types::{ChatRequest, XaiOptions};
    ///
    /// let req = ChatRequest::new(messages)
    ///     .with_xai_options(
    ///         XaiOptions::new().with_default_search()
    ///     );
    /// ```
    pub fn with_xai_options(mut self, options: crate::types::XaiOptions) -> Self {
        self.provider_options = crate::types::ProviderOptions::Xai(options);
        self
    }

    /// Convenience: Set Anthropic-specific options
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use siumai::types::{ChatRequest, AnthropicOptions, PromptCachingConfig};
    ///
    /// let req = ChatRequest::new(messages)
    ///     .with_anthropic_options(
    ///         AnthropicOptions::new()
    ///             .with_prompt_caching(PromptCachingConfig::default())
    ///     );
    /// ```
    pub fn with_anthropic_options(mut self, options: crate::types::AnthropicOptions) -> Self {
        self.provider_options = crate::types::ProviderOptions::Anthropic(options);
        self
    }

    /// Convenience: Set Gemini-specific options
    pub fn with_gemini_options(mut self, options: crate::types::GeminiOptions) -> Self {
        self.provider_options = crate::types::ProviderOptions::Gemini(options);
        self
    }

    /// Convenience: Set Groq-specific options
    pub fn with_groq_options(mut self, options: crate::types::GroqOptions) -> Self {
        self.provider_options = crate::types::ProviderOptions::Groq(options);
        self
    }

    /// Convenience: Set Ollama-specific options
    pub fn with_ollama_options(mut self, options: crate::types::OllamaOptions) -> Self {
        self.provider_options = crate::types::ProviderOptions::Ollama(options);
        self
    }

    /// Set HTTP configuration
    pub fn with_http_config(mut self, config: HttpConfig) -> Self {
        self.http_config = Some(config);
        self
    }
}

/// Chat request builder
#[derive(Debug, Clone)]
pub struct ChatRequestBuilder {
    messages: Vec<ChatMessage>,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<crate::types::ToolChoice>,
    common_params: CommonParams,
    provider_options: crate::types::ProviderOptions,
    http_config: Option<HttpConfig>,
    stream: bool,
}

impl ChatRequestBuilder {
    /// Create a new chat request builder
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            tools: None,
            tool_choice: None,
            common_params: CommonParams::default(),
            provider_options: crate::types::ProviderOptions::None,
            http_config: None,
            stream: false,
        }
    }

    /// Add a message to the request
    pub fn message(mut self, message: ChatMessage) -> Self {
        self.messages.push(message);
        self
    }

    /// Add multiple messages to the request
    pub fn messages(mut self, messages: Vec<ChatMessage>) -> Self {
        self.messages.extend(messages);
        self
    }

    /// Add tools to the request
    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set tool choice strategy
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::{ChatRequestBuilder, ToolChoice};
    ///
    /// let request = ChatRequestBuilder::new()
    ///     .tool_choice(ToolChoice::tool("weather"))
    ///     .build();
    /// ```
    pub fn tool_choice(mut self, choice: crate::types::ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Enable streaming
    pub const fn stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Set common parameters
    pub fn common_params(mut self, params: CommonParams) -> Self {
        self.common_params = params;
        self
    }

    /// Set model parameters (alias for `common_params`)
    pub fn model_params(mut self, params: CommonParams) -> Self {
        self.common_params = params;
        self
    }

    // Convenience methods for common parameters

    /// Set the model name
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.common_params.model = model.into();
        self
    }

    /// Set the temperature (0.0 to 2.0)
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.common_params.temperature = Some(temperature);
        self
    }

    /// Set the maximum number of tokens to generate
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.common_params.max_tokens = Some(max_tokens);
        self
    }

    /// Set the top_p sampling parameter
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.common_params.top_p = Some(top_p);
        self
    }

    /// Set stop sequences
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.common_params.stop_sequences = Some(sequences);
        self
    }

    /// Set the random seed for reproducibility
    pub fn seed(mut self, seed: u64) -> Self {
        self.common_params.seed = Some(seed);
        self
    }

    // ============================================================================
    // 🎯 NEW: Type-safe provider options (v0.12+)
    // ============================================================================

    /// Set provider-specific options (type-safe!)
    pub fn provider_options(mut self, options: crate::types::ProviderOptions) -> Self {
        self.provider_options = options;
        self
    }

    /// Convenience: Set OpenAI-specific options
    pub fn openai_options(mut self, options: crate::types::OpenAiOptions) -> Self {
        self.provider_options = crate::types::ProviderOptions::OpenAi(options);
        self
    }

    /// Convenience: Set xAI-specific options
    pub fn xai_options(mut self, options: crate::types::XaiOptions) -> Self {
        self.provider_options = crate::types::ProviderOptions::Xai(options);
        self
    }

    /// Convenience: Set Anthropic-specific options
    pub fn anthropic_options(mut self, options: crate::types::AnthropicOptions) -> Self {
        self.provider_options = crate::types::ProviderOptions::Anthropic(options);
        self
    }

    /// Convenience: Set Gemini-specific options
    pub fn gemini_options(mut self, options: crate::types::GeminiOptions) -> Self {
        self.provider_options = crate::types::ProviderOptions::Gemini(options);
        self
    }

    /// Convenience: Set Groq-specific options
    pub fn groq_options(mut self, options: crate::types::GroqOptions) -> Self {
        self.provider_options = crate::types::ProviderOptions::Groq(options);
        self
    }

    /// Convenience: Set Ollama-specific options
    pub fn ollama_options(mut self, options: crate::types::OllamaOptions) -> Self {
        self.provider_options = crate::types::ProviderOptions::Ollama(options);
        self
    }

    /// Set HTTP configuration
    pub fn http_config(mut self, config: HttpConfig) -> Self {
        self.http_config = Some(config);
        self
    }

    /// Build the chat request
    pub fn build(self) -> ChatRequest {
        ChatRequest {
            messages: self.messages,
            tools: self.tools,
            tool_choice: self.tool_choice,
            common_params: self.common_params,
            provider_options: self.provider_options,
            http_config: self.http_config,
            stream: self.stream,
            telemetry: None,
        }
    }
}

impl Default for ChatRequestBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Audio output from the model
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioOutput {
    /// Unique identifier for this audio response
    pub id: String,
    /// Unix timestamp (in seconds) when this audio expires
    pub expires_at: i64,
    /// Base64-encoded audio data
    pub data: String,
    /// Transcript of the audio
    pub transcript: String,
}

/// Chat response from the provider
///
/// The response content can include text, tool calls, reasoning, and other content types.
/// Tool calls and reasoning are now part of the content, not separate fields.
///
/// # Examples
///
/// ```rust
/// use siumai::types::{ChatResponse, MessageContent, ContentPart};
///
/// // Response with tool calls
/// let response = ChatResponse::new(MessageContent::MultiModal(vec![
///     ContentPart::text("Let me search for that..."),
///     ContentPart::tool_call("call_123", "search", r#"{"query":"rust"}"#, None),
/// ]));
///
/// // Check for tool calls
/// if response.has_tool_calls() {
///     let tool_calls = response.tool_calls();
///     println!("Found {} tool calls", tool_calls.len());
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// Response ID
    pub id: Option<String>,
    /// The response content (can include text, tool calls, reasoning, etc.)
    pub content: MessageContent,
    /// Model used for the response
    pub model: Option<String>,
    /// Usage statistics
    pub usage: Option<Usage>,
    /// Finish reason
    pub finish_reason: Option<FinishReason>,
    /// Audio output (if audio modality was requested)
    pub audio: Option<AudioOutput>,
    /// System fingerprint (backend configuration identifier)
    ///
    /// This fingerprint represents the backend configuration that the model runs with.
    /// Can be used in conjunction with the `seed` request parameter to understand when
    /// backend changes have been made that might impact determinism.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    /// Service tier used for processing the request
    ///
    /// Indicates the actual service tier used (e.g., "default", "scale", "flex", "priority").
    /// May differ from the requested service tier based on availability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    /// Warnings from the model provider (e.g., unsupported settings)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<Vec<Warning>>,
    /// Provider-specific metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ChatResponse {
    /// Create a new chat response
    pub fn new(content: MessageContent) -> Self {
        Self {
            id: None,
            content,
            model: None,
            usage: None,
            finish_reason: None,
            audio: None,
            system_fingerprint: None,
            service_tier: None,
            warnings: None,
            metadata: HashMap::new(),
        }
    }

    /// Create an empty response (typically used for stream end events)
    pub fn empty() -> Self {
        Self {
            id: None,
            content: MessageContent::Text(String::new()),
            model: None,
            usage: None,
            finish_reason: None,
            audio: None,
            system_fingerprint: None,
            service_tier: None,
            warnings: None,
            metadata: HashMap::new(),
        }
    }

    /// Create an empty response with a specific finish reason
    pub fn empty_with_finish_reason(reason: FinishReason) -> Self {
        Self {
            id: None,
            content: MessageContent::Text(String::new()),
            model: None,
            usage: None,
            finish_reason: Some(reason),
            audio: None,
            system_fingerprint: None,
            service_tier: None,
            warnings: None,
            metadata: HashMap::new(),
        }
    }

    /// Get the text content of the response
    pub fn content_text(&self) -> Option<&str> {
        self.content.text()
    }

    /// Get all text content of the response
    pub fn text(&self) -> Option<String> {
        Some(self.content.all_text())
    }

    /// Extract all tool calls from response content
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::{ChatResponse, MessageContent, ContentPart};
    ///
    /// let response = ChatResponse::new(MessageContent::MultiModal(vec![
    ///     ContentPart::text("Let me search..."),
    ///     ContentPart::tool_call("call_123", "search", r#"{}"#, None),
    /// ]));
    ///
    /// let tool_calls = response.tool_calls();
    /// assert_eq!(tool_calls.len(), 1);
    /// ```
    pub fn tool_calls(&self) -> Vec<&ContentPart> {
        match &self.content {
            MessageContent::MultiModal(parts) => {
                parts.iter().filter(|p| p.is_tool_call()).collect()
            }
            _ => vec![],
        }
    }

    /// Check if the response has tool calls
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls().is_empty()
    }

    /// Get tool calls (deprecated - use tool_calls() instead)
    #[deprecated(since = "0.12.0", note = "Use `tool_calls()` instead")]
    pub fn get_tool_calls(&self) -> Option<Vec<&ContentPart>> {
        let calls = self.tool_calls();
        if calls.is_empty() { None } else { Some(calls) }
    }

    /// Extract all reasoning/thinking content from response
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::{ChatResponse, MessageContent, ContentPart};
    ///
    /// let response = ChatResponse::new(MessageContent::MultiModal(vec![
    ///     ContentPart::reasoning("Let me think..."),
    ///     ContentPart::text("The answer is 42"),
    /// ]));
    ///
    /// let reasoning = response.reasoning();
    /// assert_eq!(reasoning.len(), 1);
    /// ```
    pub fn reasoning(&self) -> Vec<&str> {
        match &self.content {
            MessageContent::MultiModal(parts) => parts
                .iter()
                .filter_map(|p| {
                    if let ContentPart::Reasoning { text } = p {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
                .collect(),
            _ => vec![],
        }
    }

    /// Check if the response has reasoning/thinking content
    pub fn has_reasoning(&self) -> bool {
        !self.reasoning().is_empty()
    }

    /// Convert response to messages for conversation history
    ///
    /// Returns an assistant message containing the response content
    /// (including tool calls, reasoning, and other content if present).
    ///
    /// This is useful for building conversation history in multi-step tool calling scenarios,
    /// similar to Vercel AI SDK's `response.messages` property.
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::{ChatResponse, ChatMessage};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = siumai::providers::openai::OpenAiClient::new("key");
    /// # let request = siumai::types::ChatRequest::builder().build();
    /// let mut messages = vec![ChatMessage::user("What's the weather?").build()];
    ///
    /// let response = client.chat_request(request).await?;
    ///
    /// // Add response messages to conversation history
    /// messages.extend(response.to_messages());
    ///
    /// // Now messages contains both user message and assistant response
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_messages(&self) -> Vec<ChatMessage> {
        vec![ChatMessage {
            role: MessageRole::Assistant,
            content: self.content.clone(),
            metadata: MessageMetadata::default(),
        }]
    }

    /// Convert response to a single assistant message
    ///
    /// This is equivalent to `to_messages()[0]` but more explicit.
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ChatResponse;
    ///
    /// # let response = ChatResponse::empty();
    /// let assistant_msg = response.to_assistant_message();
    /// ```
    pub fn to_assistant_message(&self) -> ChatMessage {
        ChatMessage {
            role: MessageRole::Assistant,
            content: self.content.clone(),
            metadata: MessageMetadata::default(),
        }
    }

    /// Get thinking content if available (deprecated - use reasoning() instead)
    #[deprecated(since = "0.12.0", note = "Use `reasoning()` instead")]
    pub fn has_thinking(&self) -> bool {
        self.has_reasoning()
    }

    /// Get thinking content if available (deprecated - use reasoning() instead)
    #[deprecated(since = "0.12.0", note = "Use `reasoning()` instead")]
    pub fn get_thinking(&self) -> Option<String> {
        let reasoning = self.reasoning();
        if reasoning.is_empty() {
            None
        } else {
            Some(reasoning.join("\n"))
        }
    }

    /// Get thinking content with fallback to empty string
    pub fn thinking_or_empty(&self) -> String {
        let reasoning_parts = self.reasoning();
        reasoning_parts
            .first()
            .map(|s| s.to_string())
            .unwrap_or_default()
    }

    /// Get the response ID for use in multi-turn conversations (OpenAI Responses API)
    ///
    /// This is particularly useful for OpenAI's Responses API, where you can chain
    /// conversations by passing the previous response ID to the next request.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use siumai::prelude::*;
    /// # use siumai::types::{ChatRequest, OpenAiOptions, ResponsesApiConfig};
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = Siumai::builder().openai().api_key("key").model("gpt-4o-mini").build().await?;
    /// // Turn 1
    /// let response1 = client.chat(vec![user!("What is Rust?")]).await?;
    /// let response_id = response1.response_id().expect("Response ID not found");
    ///
    /// // Turn 2 - automatically loads context from Turn 1
    /// let request2 = ChatRequest::new(vec![user!("Can you give me a code example?")])
    ///     .with_openai_options(
    ///         OpenAiOptions::new().with_responses_api(
    ///             ResponsesApiConfig::new()
    ///                 .with_previous_response(response_id.to_string())
    ///         )
    ///     );
    /// let response2 = client.chat_request(request2).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn response_id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// Get provider-specific metadata value by key
    ///
    /// # Example
    /// ```rust,no_run
    /// # use siumai::prelude::*;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = Siumai::builder().openai().api_key("key").model("gpt-4o-mini").build().await?;
    /// let response = client.chat(vec![user!("Hello")]).await?;
    /// if let Some(value) = response.get_metadata("openai.response_id") {
    ///     println!("OpenAI Response ID: {:?}", value);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.metadata.get(key)
    }

    /// Check if response has any metadata
    pub fn has_metadata(&self) -> bool {
        !self.metadata.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_response_helper_methods() {
        // Test response_id()
        let mut response = ChatResponse::new(MessageContent::Text("Hello".to_string()));
        assert_eq!(response.response_id(), None);

        response.id = Some("resp_123".to_string());
        assert_eq!(response.response_id(), Some("resp_123"));

        // Test get_metadata() and has_metadata()
        assert!(!response.has_metadata());
        assert_eq!(response.get_metadata("key1"), None);

        response
            .metadata
            .insert("key1".to_string(), serde_json::json!("value1"));
        response
            .metadata
            .insert("key2".to_string(), serde_json::json!(42));

        assert!(response.has_metadata());
        assert_eq!(
            response.get_metadata("key1"),
            Some(&serde_json::json!("value1"))
        );
        assert_eq!(response.get_metadata("key2"), Some(&serde_json::json!(42)));
        assert_eq!(response.get_metadata("nonexistent"), None);
    }
}
