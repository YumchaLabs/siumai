//! Chat request types

use serde::{Deserialize, Serialize};

use super::message::ChatMessage;
use crate::types::tools::Tool;
use crate::types::{CommonParams, HttpConfig, ProviderOptionsMap};

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

    /// Open provider options map (Vercel-aligned).
    ///
    /// Provider implementations should prefer this open map over the closed enum
    /// during the fearless refactor.
    #[serde(
        default,
        rename = "providerOptions",
        skip_serializing_if = "ProviderOptionsMap::is_empty"
    )]
    pub provider_options_map: ProviderOptionsMap,

    /// HTTP configuration
    pub http_config: Option<HttpConfig>,

    /// Stream the response
    pub stream: bool,
    /// Optional telemetry configuration
    #[serde(skip)]
    pub telemetry: Option<crate::observability::telemetry::TelemetryConfig>,
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
            provider_options_map: ProviderOptionsMap::default(),
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

    /// Replace the full provider options map (open JSON map).
    pub fn with_provider_options_map(mut self, map: ProviderOptionsMap) -> Self {
        self.provider_options_map = map;
        self
    }

    /// Set provider options for a provider id (open JSON map).
    pub fn with_provider_option(
        mut self,
        provider_id: impl AsRef<str>,
        options: serde_json::Value,
    ) -> Self {
        self.provider_options_map.insert(provider_id, options);
        self
    }

    /// Get provider options for a provider id (open JSON map).
    pub fn provider_option(&self, provider_id: impl AsRef<str>) -> Option<&serde_json::Value> {
        self.provider_options_map.get(provider_id)
    }

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
        if let Some((provider_id, value)) = options.to_provider_options_map_entry() {
            self.provider_options_map.insert(provider_id, value);
        }
        self.provider_options = options;
        self
    }

    /// Convenience: Set OpenAI-specific options
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use siumai::types::ChatRequest;
    /// use siumai::provider_ext::openai::{OpenAiOptions, ServiceTier};
    ///
    /// let req = ChatRequest::new(messages)
    ///     .with_openai_options(
    ///         OpenAiOptions::new()
    ///             .with_service_tier(ServiceTier::Standard)
    ///     );
    /// ```
    #[cfg(feature = "openai")]
    pub fn with_openai_options<T: serde::Serialize>(self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        self.with_provider_options(crate::types::ProviderOptions::OpenAi(value))
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
    #[cfg(feature = "xai")]
    pub fn with_xai_options(self, options: crate::types::XaiOptions) -> Self {
        self.with_provider_options(crate::types::ProviderOptions::Xai(options))
    }

    /// Convenience: Set Anthropic-specific options
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use siumai::types::ChatRequest;
    /// use siumai::provider_ext::anthropic::{AnthropicOptions, PromptCachingConfig};
    ///
    /// let req = ChatRequest::new(messages)
    ///     .with_anthropic_options(
    ///         AnthropicOptions::new()
    ///             .with_prompt_caching(PromptCachingConfig::default())
    ///     );
    /// ```
    #[cfg(feature = "anthropic")]
    pub fn with_anthropic_options<T: serde::Serialize>(self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        self.with_provider_options(crate::types::ProviderOptions::Anthropic(value))
    }

    /// Convenience: Set Gemini-specific options
    #[cfg(feature = "google")]
    pub fn with_gemini_options<T: serde::Serialize>(self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        self.with_provider_options(crate::types::ProviderOptions::Gemini(value))
    }

    /// Convenience: Set Groq-specific options
    #[cfg(feature = "groq")]
    pub fn with_groq_options(self, options: crate::types::GroqOptions) -> Self {
        self.with_provider_options(crate::types::ProviderOptions::Groq(options))
    }

    /// Convenience: Set Ollama-specific options
    #[cfg(feature = "ollama")]
    pub fn with_ollama_options(self, options: crate::types::OllamaOptions) -> Self {
        self.with_provider_options(crate::types::ProviderOptions::Ollama(options))
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
    provider_options_map: ProviderOptionsMap,
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
            provider_options_map: ProviderOptionsMap::default(),
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

    /// Replace the full provider options map (open JSON map).
    pub fn provider_options_map(mut self, map: ProviderOptionsMap) -> Self {
        self.provider_options_map = map;
        self
    }

    /// Set provider options for a provider id (open JSON map).
    pub fn provider_option(
        mut self,
        provider_id: impl AsRef<str>,
        options: serde_json::Value,
    ) -> Self {
        self.provider_options_map.insert(provider_id, options);
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
        if let Some((provider_id, value)) = options.to_provider_options_map_entry() {
            self.provider_options_map.insert(provider_id, value);
        }
        self.provider_options = options;
        self
    }

    /// Convenience: Set OpenAI-specific options
    #[cfg(feature = "openai")]
    pub fn openai_options<T: serde::Serialize>(self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        self.provider_options(crate::types::ProviderOptions::OpenAi(value))
    }

    /// Convenience: Set xAI-specific options
    #[cfg(feature = "xai")]
    pub fn xai_options(self, options: crate::types::XaiOptions) -> Self {
        self.provider_options(crate::types::ProviderOptions::Xai(options))
    }

    /// Convenience: Set Anthropic-specific options
    #[cfg(feature = "anthropic")]
    pub fn anthropic_options<T: serde::Serialize>(self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        self.provider_options(crate::types::ProviderOptions::Anthropic(value))
    }

    /// Convenience: Set Gemini-specific options
    #[cfg(feature = "google")]
    pub fn gemini_options<T: serde::Serialize>(self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        self.provider_options(crate::types::ProviderOptions::Gemini(value))
    }

    /// Convenience: Set Groq-specific options
    #[cfg(feature = "groq")]
    pub fn groq_options(self, options: crate::types::GroqOptions) -> Self {
        self.provider_options(crate::types::ProviderOptions::Groq(options))
    }

    /// Convenience: Set Ollama-specific options
    #[cfg(feature = "ollama")]
    pub fn ollama_options(self, options: crate::types::OllamaOptions) -> Self {
        self.provider_options(crate::types::ProviderOptions::Ollama(options))
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
            provider_options_map: self.provider_options_map,
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
