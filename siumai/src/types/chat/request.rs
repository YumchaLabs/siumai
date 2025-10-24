//! Chat request types

use serde::{Deserialize, Serialize};

use super::message::ChatMessage;
use crate::types::common::{CommonParams, HttpConfig};
use crate::types::tools::Tool;

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
    ///         OpenAiOptions::new()
    ///             .with_service_tier(siumai::types::provider_options::openai::ServiceTier::Standard)
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
