//! Chat-related types and message handling

use super::common::{CommonParams, FinishReason, HttpConfig, Usage};
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
}

/// Content part
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ContentPart {
    Text {
        text: String,
    },
    Image {
        image_url: String,
        detail: Option<String>,
    },
    Audio {
        audio_url: String,
        format: String,
    },
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role
    pub role: MessageRole,
    /// Content
    pub content: MessageContent,
    /// Message metadata
    #[serde(default)]
    pub metadata: MessageMetadata,
    /// Tool calls
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Tool call ID
    pub tool_call_id: Option<String>,
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

    /// Creates a tool message
    pub fn tool<S: Into<String>>(content: S, tool_call_id: S) -> ChatMessageBuilder {
        ChatMessageBuilder::tool(content, tool_call_id)
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
                    ContentPart::Image { image_url, .. } => image_url.len(),
                    ContentPart::Audio { audio_url, .. } => audio_url.len(),
                })
                .sum(),
            #[cfg(feature = "structured-messages")]
            MessageContent::Json(v) => serde_json::to_string(v).map(|s| s.len()).unwrap_or(0),
        }
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
    tool_calls: Option<Vec<ToolCall>>,
    tool_call_id: Option<String>,
}

impl ChatMessageBuilder {
    /// Creates a user message builder
    pub fn user<S: Into<String>>(content: S) -> Self {
        Self {
            role: MessageRole::User,
            content: Some(MessageContent::Text(content.into())),
            metadata: MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: None,
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
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Creates a system message builder
    pub fn system<S: Into<String>>(content: S) -> Self {
        Self {
            role: MessageRole::System,
            content: Some(MessageContent::Text(content.into())),
            metadata: MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Creates an assistant message builder
    pub fn assistant<S: Into<String>>(content: S) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: Some(MessageContent::Text(content.into())),
            metadata: MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Creates a developer message builder
    pub fn developer<S: Into<String>>(content: S) -> Self {
        Self {
            role: MessageRole::Developer,
            content: Some(MessageContent::Text(content.into())),
            metadata: MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Creates a tool message builder
    pub fn tool<S: Into<String>>(content: S, tool_call_id: S) -> Self {
        Self {
            role: MessageRole::Tool,
            content: Some(MessageContent::Text(content.into())),
            metadata: MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
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
        let image_part = ContentPart::Image { image_url, detail };

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

    /// Adds tool calls
    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = Some(tool_calls);
        self
    }

    /// Builds the message
    pub fn build(self) -> ChatMessage {
        ChatMessage {
            role: self.role,
            content: self.content.unwrap_or(MessageContent::Text(String::new())),
            metadata: self.metadata,
            tool_calls: self.tool_calls,
            tool_call_id: self.tool_call_id,
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

/// Chat response from the provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// Response ID
    pub id: Option<String>,
    /// The response content
    pub content: MessageContent,
    /// Model used for the response
    pub model: Option<String>,
    /// Usage statistics
    pub usage: Option<Usage>,
    /// Finish reason
    pub finish_reason: Option<FinishReason>,
    /// Tool calls in the response
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Thinking content (if available)
    pub thinking: Option<String>,
    /// Provider-specific metadata
    pub metadata: HashMap<String, serde_json::Value>,
    // Reserved for future: warnings/provider_metadata (kept in metadata for now)
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
            tool_calls: None,
            thinking: None,
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

    /// Check if the response has tool calls
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty())
    }

    /// Get tool calls
    pub const fn get_tool_calls(&self) -> Option<&Vec<ToolCall>> {
        self.tool_calls.as_ref()
    }

    /// Check if the response has thinking content
    pub fn has_thinking(&self) -> bool {
        self.thinking.as_ref().is_some_and(|t| !t.is_empty())
    }

    /// Get thinking content if available
    pub fn get_thinking(&self) -> Option<&str> {
        self.thinking.as_deref()
    }

    /// Get thinking content with fallback to empty string
    pub fn thinking_or_empty(&self) -> &str {
        self.thinking.as_deref().unwrap_or("")
    }
}
