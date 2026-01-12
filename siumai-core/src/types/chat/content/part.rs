use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::super::metadata::{ToolCallInfo, ToolResultInfo};
use super::{ImageDetail, MediaSource, ToolResultContentPart, ToolResultOutput};

/// Content part - provider-agnostic multimodal content
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ContentPart {
    /// Text content
    Text {
        text: String,

        /// Provider-specific metadata (Vercel-aligned).
        #[serde(
            rename = "providerMetadata",
            alias = "provider_metadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<HashMap<String, serde_json::Value>>,
    },

    /// Image content - supports URL, base64, or binary data
    Image {
        /// Image data source
        #[serde(flatten)]
        source: MediaSource,
        /// Optional detail level (for providers that support it, e.g., OpenAI)
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,

        /// Provider-specific metadata (Vercel-aligned).
        #[serde(
            rename = "providerMetadata",
            alias = "provider_metadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<HashMap<String, serde_json::Value>>,
    },

    /// Audio content - supports URL, base64, or binary data
    Audio {
        /// Audio data source
        #[serde(flatten)]
        source: MediaSource,
        /// Media type (e.g., "audio/wav", "audio/mp3", "audio/mpeg")
        #[serde(skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,

        /// Provider-specific metadata (Vercel-aligned).
        #[serde(
            rename = "providerMetadata",
            alias = "provider_metadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<HashMap<String, serde_json::Value>>,
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

        /// Provider-specific metadata (Vercel-aligned).
        #[serde(
            rename = "providerMetadata",
            alias = "provider_metadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<HashMap<String, serde_json::Value>>,
    },

    /// Source citation (Vercel-aligned).
    ///
    /// Providers may surface citations/attributions as a list of sources.
    /// This content part represents a single source entry in the unified content stream.
    #[serde(rename = "source")]
    Source {
        /// Source id (Vercel-aligned, e.g. "id-0").
        id: String,

        /// Source type (Vercel-aligned, e.g. "url", "document").
        #[serde(rename = "sourceType")]
        source_type: String,

        /// Source URL (or provider file id for document sources).
        url: String,

        /// Human-readable title (often the URL itself for web citations).
        title: String,
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

    /// Tool approval request (for MCP approval workflows)
    ///
    /// This represents a model request to obtain user/system approval for a pending tool call,
    /// aligned with Vercel AI SDK's `tool-approval-request` part.
    #[serde(rename = "tool-approval-request")]
    ToolApprovalRequest {
        /// The approval request ID (MCP approval request item id)
        #[serde(rename = "approvalId")]
        approval_id: String,

        /// The tool call ID associated with this approval request.
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
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

impl ContentPart {
    /// Create a text content part
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            provider_metadata: None,
        }
    }

    /// Create an image content part from URL
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::Image {
            source: MediaSource::url(url),
            detail: None,
            provider_metadata: None,
        }
    }

    /// Create an image content part from URL with detail level
    pub fn image_url_with_detail(url: impl Into<String>, detail: ImageDetail) -> Self {
        Self::Image {
            source: MediaSource::url(url),
            detail: Some(detail),
            provider_metadata: None,
        }
    }

    /// Create an image content part from base64 data
    pub fn image_base64(data: impl Into<String>) -> Self {
        Self::Image {
            source: MediaSource::base64(data),
            detail: None,
            provider_metadata: None,
        }
    }

    /// Create an image content part from binary data
    pub fn image_binary(data: Vec<u8>) -> Self {
        Self::Image {
            source: MediaSource::binary(data),
            detail: None,
            provider_metadata: None,
        }
    }

    /// Create an audio content part from URL
    pub fn audio_url(url: impl Into<String>, media_type: Option<String>) -> Self {
        Self::Audio {
            source: MediaSource::url(url),
            media_type,
            provider_metadata: None,
        }
    }

    /// Create an audio content part from base64 data
    pub fn audio_base64(data: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self::Audio {
            source: MediaSource::base64(data),
            media_type: Some(media_type.into()),
            provider_metadata: None,
        }
    }

    /// Create an audio content part from binary data
    pub fn audio_binary(data: Vec<u8>, media_type: impl Into<String>) -> Self {
        Self::Audio {
            source: MediaSource::binary(data),
            media_type: Some(media_type.into()),
            provider_metadata: None,
        }
    }

    /// Create a file content part from URL
    pub fn file_url(url: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self::File {
            source: MediaSource::url(url),
            media_type: media_type.into(),
            filename: None,
            provider_metadata: None,
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
            provider_metadata: None,
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
            provider_metadata: None,
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

    /// Create a source content part (Vercel-aligned).
    pub fn source(
        id: impl Into<String>,
        source_type: impl Into<String>,
        url: impl Into<String>,
        title: impl Into<String>,
    ) -> Self {
        Self::Source {
            id: id.into(),
            source_type: source_type.into(),
            url: url.into(),
            title: title.into(),
        }
    }

    /// Create a tool approval response content part.
    pub fn tool_approval_response(approval_id: impl Into<String>, approved: bool) -> Self {
        Self::ToolApprovalResponse {
            approval_id: approval_id.into(),
            approved,
        }
    }

    /// Create a tool approval request content part.
    pub fn tool_approval_request(
        approval_id: impl Into<String>,
        tool_call_id: impl Into<String>,
    ) -> Self {
        Self::ToolApprovalRequest {
            approval_id: approval_id.into(),
            tool_call_id: tool_call_id.into(),
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
    ///     "Temperature is 18掳C, sunny",
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

    /// Check if this is a source
    pub fn is_source(&self) -> bool {
        matches!(self, Self::Source { .. })
    }

    /// Check if this is reasoning
    pub fn is_reasoning(&self) -> bool {
        matches!(self, Self::Reasoning { .. })
    }

    /// Get the text content if this is a text part
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text, .. } => Some(text),
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
