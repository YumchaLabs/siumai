//! Chat message types

use serde::{Deserialize, Serialize};

use super::content::{ContentPart, ImageDetail, MediaSource, MessageContent, ToolResultOutput};
use super::metadata::{CacheControl, MessageMetadata};

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
    #[allow(deprecated)]
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
