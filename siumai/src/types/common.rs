//! Common types and enums used across the library

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Provider type enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderType {
    OpenAi,
    Anthropic,
    Gemini,
    Ollama,
    XAI,
    Groq,
    Custom(String),
}

impl std::fmt::Display for ProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenAi => write!(f, "openai"),
            Self::Anthropic => write!(f, "anthropic"),
            Self::Gemini => write!(f, "gemini"),
            Self::Ollama => write!(f, "ollama"),
            Self::XAI => write!(f, "xai"),
            Self::Groq => write!(f, "groq"),
            Self::Custom(name) => write!(f, "{name}"),
        }
    }
}

impl ProviderType {
    /// Construct a ProviderType from a provider name string.
    /// Known names map to concrete variants; others map to Custom(name).
    pub fn from_name(name: &str) -> Self {
        match name {
            "openai" => Self::OpenAi,
            "anthropic" => Self::Anthropic,
            "gemini" => Self::Gemini,
            "ollama" => Self::Ollama,
            "xai" => Self::XAI,
            "groq" => Self::Groq,
            other => Self::Custom(other.to_string()),
        }
    }
}

/// Common AI parameters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CommonParams {
    /// Model name
    pub model: String,

    /// Temperature parameter (must be non-negative)
    pub temperature: Option<f32>,

    /// Maximum output tokens (deprecated for o1/o3 models, use max_completion_tokens instead)
    pub max_tokens: Option<u32>,

    /// Maximum completion tokens (for o1/o3 reasoning models)
    /// This is an upper bound for the number of tokens that can be generated for a completion,
    /// including visible output tokens and reasoning tokens.
    pub max_completion_tokens: Option<u32>,

    /// `top_p` parameter
    pub top_p: Option<f32>,

    /// Stop sequences
    pub stop_sequences: Option<Vec<String>>,

    /// Random seed
    pub seed: Option<u64>,
}

impl CommonParams {
    /// Create `CommonParams` with pre-allocated model string capacity
    pub const fn with_model_capacity(model: String, _capacity_hint: usize) -> Self {
        Self {
            model,
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            top_p: None,
            stop_sequences: None,
            seed: None,
        }
    }

    /// Check if parameters are effectively empty (for optimization)
    pub const fn is_minimal(&self) -> bool {
        self.model.is_empty()
            && self.temperature.is_none()
            && self.max_tokens.is_none()
            && self.max_completion_tokens.is_none()
            && self.top_p.is_none()
            && self.stop_sequences.is_none()
            && self.seed.is_none()
    }

    /// Estimate memory usage for caching decisions
    pub fn memory_footprint(&self) -> usize {
        let mut size = std::mem::size_of::<Self>();
        size += self.model.capacity();
        if let Some(ref stop_seqs) = self.stop_sequences {
            size += stop_seqs
                .iter()
                .map(std::string::String::capacity)
                .sum::<usize>();
        }
        size
    }

    /// Create a hash for caching (performance optimized)
    pub fn cache_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.model.hash(&mut hasher);
        self.temperature
            .map(|t| (t * 1000.0) as u32)
            .hash(&mut hasher);
        self.max_tokens.hash(&mut hasher);
        self.top_p.map(|t| (t * 1000.0) as u32).hash(&mut hasher);
        hasher.finish()
    }

    /// Validate common parameters
    pub fn validate_params(&self) -> Result<(), crate::error::LlmError> {
        // Validate model name
        if self.model.is_empty() {
            return Err(crate::error::LlmError::InvalidParameter(
                "Model name cannot be empty".to_string(),
            ));
        }

        // Validate temperature (must be non-negative)
        if let Some(temp) = self.temperature {
            if temp < 0.0 {
                return Err(crate::error::LlmError::InvalidParameter(
                    "Temperature must be non-negative".to_string(),
                ));
            }
        }

        // Validate top_p (must be between 0.0 and 1.0)
        if let Some(top_p) = self.top_p {
            if !(0.0..=1.0).contains(&top_p) {
                return Err(crate::error::LlmError::InvalidParameter(
                    "top_p must be between 0.0 and 1.0".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Create a builder for common parameters
    pub fn builder() -> CommonParamsBuilder {
        CommonParamsBuilder::new()
    }
}

/// Builder for CommonParams with validation
#[derive(Debug, Clone, Default)]
pub struct CommonParamsBuilder {
    model: String,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    top_p: Option<f32>,
    stop_sequences: Option<Vec<String>>,
    seed: Option<u64>,
}

impl CommonParamsBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model name
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = model.into();
        self
    }

    /// Set the temperature with validation
    pub fn temperature(mut self, temperature: f32) -> Result<Self, crate::error::LlmError> {
        if !(0.0..=2.0).contains(&temperature) {
            return Err(crate::error::LlmError::InvalidParameter(
                "Temperature must be between 0.0 and 2.0".to_string(),
            ));
        }
        self.temperature = Some(temperature);
        Ok(self)
    }

    /// Set the max tokens
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set the top_p with validation
    pub fn top_p(mut self, top_p: f32) -> Result<Self, crate::error::LlmError> {
        if !(0.0..=1.0).contains(&top_p) {
            return Err(crate::error::LlmError::InvalidParameter(
                "top_p must be between 0.0 and 1.0".to_string(),
            ));
        }
        self.top_p = Some(top_p);
        Ok(self)
    }

    /// Set stop sequences
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(sequences);
        self
    }

    /// Set the random seed
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Build the CommonParams
    pub fn build(self) -> Result<CommonParams, crate::error::LlmError> {
        let params = CommonParams {
            model: self.model,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            max_completion_tokens: None,
            top_p: self.top_p,
            stop_sequences: self.stop_sequences,
            seed: self.seed,
        };

        params.validate_params()?;
        Ok(params)
    }
}
// ProviderParams has been removed in v0.12.0
// Use provider-specific options instead:
// - OpenAiOptions for OpenAI-specific features
// - AnthropicOptions for Anthropic-specific features
// - XaiOptions for xAI-specific features
// - GeminiOptions for Gemini-specific features
// - GroqOptions for Groq-specific features
// - OllamaOptions for Ollama-specific features
// See the migration guide in docs/architecture/provider-spec-refactor.md

/// HTTP configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpConfig {
    /// Request timeout
    #[serde(with = "duration_option_serde")]
    pub timeout: Option<Duration>,
    /// Connection timeout
    #[serde(with = "duration_option_serde")]
    pub connect_timeout: Option<Duration>,
    /// Custom headers
    pub headers: HashMap<String, String>,
    /// Proxy settings
    pub proxy: Option<String>,
    /// User agent
    pub user_agent: Option<String>,
    /// Whether to disable compression for streaming (SSE) requests.
    ///
    /// When `true`, streaming requests explicitly set `Accept-Encoding: identity`
    /// to avoid intermediary/proxy compression which can break long-lived SSE
    /// connections. Default is `true` for stability.
    pub stream_disable_compression: bool,
}

// Helper module for Duration serialization
mod duration_option_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Option<Duration>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match duration {
            Some(d) => d.as_secs().serialize(serializer),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Duration>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs: Option<u64> = Option::deserialize(deserializer)?;
        Ok(secs.map(Duration::from_secs))
    }
}

impl Default for HttpConfig {
    fn default() -> Self {
        // Determine default for stream_disable_compression from env var (default: true)
        let sdc = match std::env::var("SIUMAI_STREAM_DISABLE_COMPRESSION") {
            Ok(val) => {
                let v = val.trim().to_lowercase();
                !(v == "false" || v == "0" || v == "off" || v == "no")
            }
            Err(_) => true,
        };
        Self {
            timeout: Some(crate::defaults::http::REQUEST_TIMEOUT),
            connect_timeout: Some(crate::defaults::http::CONNECT_TIMEOUT),
            headers: HashMap::new(),
            proxy: None,
            user_agent: Some(crate::defaults::http::USER_AGENT.to_string()),
            stream_disable_compression: sdc,
        }
    }
}

/// Usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Input tokens used
    pub prompt_tokens: u32,
    /// Output tokens generated
    pub completion_tokens: u32,
    /// Total tokens used
    pub total_tokens: u32,
    /// Cached tokens (if applicable)
    #[deprecated(
        since = "0.11.0",
        note = "Use prompt_tokens_details.cached_tokens instead"
    )]
    pub cached_tokens: Option<u32>,
    /// Reasoning tokens (for models like o1)
    #[deprecated(
        since = "0.11.0",
        note = "Use completion_tokens_details.reasoning_tokens instead"
    )]
    pub reasoning_tokens: Option<u32>,
    /// Detailed breakdown of prompt tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    /// Detailed breakdown of completion tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

/// Breakdown of tokens used in the prompt
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PromptTokensDetails {
    /// Audio input tokens present in the prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u32>,
    /// Cached tokens present in the prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
}

/// Breakdown of tokens used in the completion
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompletionTokensDetails {
    /// Tokens generated by the model for reasoning
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
    /// Audio output tokens generated by the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u32>,
    /// Accepted prediction tokens (when using Predicted Outputs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accepted_prediction_tokens: Option<u32>,
    /// Rejected prediction tokens (when using Predicted Outputs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rejected_prediction_tokens: Option<u32>,
}

impl Usage {
    /// Create new usage statistics
    pub const fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            #[allow(deprecated)]
            cached_tokens: None,
            #[allow(deprecated)]
            reasoning_tokens: None,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        }
    }

    /// Create a builder for constructing Usage with detailed token information
    pub fn builder() -> UsageBuilder {
        UsageBuilder::default()
    }

    /// Create usage with all fields (for backward compatibility during migration)
    #[allow(deprecated)]
    pub fn with_legacy_fields(
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: u32,
        cached_tokens: Option<u32>,
        reasoning_tokens: Option<u32>,
    ) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cached_tokens,
            reasoning_tokens,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        }
    }

    /// Merge usage statistics
    pub fn merge(&mut self, other: &Usage) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.total_tokens += other.total_tokens;

        // Merge deprecated fields for backward compatibility
        #[allow(deprecated)]
        {
            if let Some(other_cached) = other.cached_tokens {
                self.cached_tokens = Some(self.cached_tokens.unwrap_or(0) + other_cached);
            }
            if let Some(other_reasoning) = other.reasoning_tokens {
                self.reasoning_tokens = Some(self.reasoning_tokens.unwrap_or(0) + other_reasoning);
            }
        }

        // Merge prompt tokens details
        if let Some(ref other_prompt_details) = other.prompt_tokens_details {
            let self_details = self
                .prompt_tokens_details
                .get_or_insert_with(Default::default);
            if let Some(audio) = other_prompt_details.audio_tokens {
                self_details.audio_tokens = Some(self_details.audio_tokens.unwrap_or(0) + audio);
            }
            if let Some(cached) = other_prompt_details.cached_tokens {
                self_details.cached_tokens = Some(self_details.cached_tokens.unwrap_or(0) + cached);
            }
        }

        // Merge completion tokens details
        if let Some(ref other_completion_details) = other.completion_tokens_details {
            let self_details = self
                .completion_tokens_details
                .get_or_insert_with(Default::default);
            if let Some(reasoning) = other_completion_details.reasoning_tokens {
                self_details.reasoning_tokens =
                    Some(self_details.reasoning_tokens.unwrap_or(0) + reasoning);
            }
            if let Some(audio) = other_completion_details.audio_tokens {
                self_details.audio_tokens = Some(self_details.audio_tokens.unwrap_or(0) + audio);
            }
            if let Some(accepted) = other_completion_details.accepted_prediction_tokens {
                self_details.accepted_prediction_tokens =
                    Some(self_details.accepted_prediction_tokens.unwrap_or(0) + accepted);
            }
            if let Some(rejected) = other_completion_details.rejected_prediction_tokens {
                self_details.rejected_prediction_tokens =
                    Some(self_details.rejected_prediction_tokens.unwrap_or(0) + rejected);
            }
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
/// match finish_reason {
///     Some(FinishReason::Stop) => println!("✅ Completed successfully"),
///     Some(FinishReason::Length) => println!("⚠️ Reached max tokens"),
///     Some(FinishReason::ContentFilter) => println!("❌ Content filtered"),
///     Some(FinishReason::Incomplete) => println!("❌ Stream interrupted"),
///     _ => println!("ℹ️ Other reason"),
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Model generated stop sequence or completed naturally.
    ///
    /// This is the most common finish reason, indicating the model completed its response
    /// successfully. Maps to:
    /// - OpenAI: `stop`
    /// - Anthropic: `end_turn`
    /// - Gemini: `STOP`
    /// - Groq/xAI: `stop`
    Stop,

    /// Model reached the maximum number of tokens (`max_tokens` parameter).
    ///
    /// The response was truncated because it hit the token limit. Consider increasing
    /// `max_tokens` if you need longer responses. Maps to:
    /// - OpenAI: `length`
    /// - Anthropic: `max_tokens`
    /// - Gemini: `MAX_TOKENS`
    /// - Groq/xAI: `length`
    Length,

    /// Model triggered tool/function calls.
    ///
    /// The model wants to call one or more tools. You should execute the tools and
    /// continue the conversation with the results. Maps to:
    /// - OpenAI: `tool_calls`
    /// - Anthropic: `tool_use`
    /// - Groq/xAI: `tool_calls`
    ToolCalls,

    /// Content was filtered due to safety/policy violations.
    ///
    /// The model's output was blocked by content filters. Maps to:
    /// - OpenAI: `content_filter`
    /// - Anthropic: `refusal`
    /// - Gemini: `SAFETY`, `RECITATION`, `PROHIBITED_CONTENT`
    /// - Groq/xAI: `content_filter`
    ContentFilter,

    /// Model stopped due to a custom stop sequence.
    ///
    /// The model encountered one of the stop sequences specified in the request.
    /// Maps to Anthropic's `stop_sequence`.
    StopSequence,

    /// An error occurred during generation.
    ///
    /// The model stopped due to an internal error. Check the error details for more
    /// information. Maps to AI SDK's `error` reason.
    Error,

    /// Other provider-specific finish reason.
    ///
    /// The provider returned a finish reason that doesn't map to standard categories.
    /// The string contains the original provider-specific reason.
    ///
    /// Maps to AI SDK's `other` reason.
    Other(String),

    /// Unknown finish reason.
    ///
    /// The model has not transmitted a finish reason. This can occur when:
    /// - The provider doesn't send a finish_reason field
    /// - The stream ended without a proper completion event (connection lost, server error, client cancelled)
    /// - The finish_reason value is not recognized
    ///
    /// **Important**: Always check for this reason to detect potentially incomplete responses.
    ///
    /// Maps to AI SDK's `unknown` reason. This follows the industry standard where
    /// `unknown` indicates the absence of a finish reason, not necessarily an error.
    Unknown,
}

/// Builder for constructing Usage with detailed token information
#[derive(Default)]
pub struct UsageBuilder {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: Option<u32>,
    prompt_details: Option<PromptTokensDetails>,
    completion_details: Option<CompletionTokensDetails>,
}

impl UsageBuilder {
    /// Set prompt tokens
    pub fn prompt_tokens(mut self, tokens: u32) -> Self {
        self.prompt_tokens = tokens;
        self
    }

    /// Set completion tokens
    pub fn completion_tokens(mut self, tokens: u32) -> Self {
        self.completion_tokens = tokens;
        self
    }

    /// Set total tokens (if not set, will be calculated as prompt + completion)
    pub fn total_tokens(mut self, tokens: u32) -> Self {
        self.total_tokens = Some(tokens);
        self
    }

    /// Add cached tokens to prompt details
    pub fn with_cached_tokens(mut self, cached: u32) -> Self {
        let details = self.prompt_details.get_or_insert_with(Default::default);
        details.cached_tokens = Some(cached);
        self
    }

    /// Add audio input tokens to prompt details
    pub fn with_prompt_audio_tokens(mut self, audio: u32) -> Self {
        let details = self.prompt_details.get_or_insert_with(Default::default);
        details.audio_tokens = Some(audio);
        self
    }

    /// Add reasoning tokens to completion details
    pub fn with_reasoning_tokens(mut self, reasoning: u32) -> Self {
        let details = self.completion_details.get_or_insert_with(Default::default);
        details.reasoning_tokens = Some(reasoning);
        self
    }

    /// Add audio output tokens to completion details
    pub fn with_completion_audio_tokens(mut self, audio: u32) -> Self {
        let details = self.completion_details.get_or_insert_with(Default::default);
        details.audio_tokens = Some(audio);
        self
    }

    /// Add accepted prediction tokens to completion details
    pub fn with_accepted_prediction_tokens(mut self, accepted: u32) -> Self {
        let details = self.completion_details.get_or_insert_with(Default::default);
        details.accepted_prediction_tokens = Some(accepted);
        self
    }

    /// Add rejected prediction tokens to completion details
    pub fn with_rejected_prediction_tokens(mut self, rejected: u32) -> Self {
        let details = self.completion_details.get_or_insert_with(Default::default);
        details.rejected_prediction_tokens = Some(rejected);
        self
    }

    /// Set prompt token details directly
    pub fn with_prompt_details(mut self, details: PromptTokensDetails) -> Self {
        self.prompt_details = Some(details);
        self
    }

    /// Set completion token details directly
    pub fn with_completion_details(mut self, details: CompletionTokensDetails) -> Self {
        self.completion_details = Some(details);
        self
    }

    /// Build the Usage struct
    #[allow(deprecated)]
    pub fn build(self) -> Usage {
        let total = self
            .total_tokens
            .unwrap_or(self.prompt_tokens + self.completion_tokens);

        Usage {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            total_tokens: total,
            cached_tokens: self.prompt_details.as_ref().and_then(|d| d.cached_tokens),
            reasoning_tokens: self
                .completion_details
                .as_ref()
                .and_then(|d| d.reasoning_tokens),
            prompt_tokens_details: self.prompt_details,
            completion_tokens_details: self.completion_details,
        }
    }
}

impl PromptTokensDetails {
    /// Create with only cached tokens
    pub fn with_cached(cached: u32) -> Self {
        Self {
            audio_tokens: None,
            cached_tokens: Some(cached),
        }
    }

    /// Create with only audio tokens
    pub fn with_audio(audio: u32) -> Self {
        Self {
            audio_tokens: Some(audio),
            cached_tokens: None,
        }
    }
}

impl CompletionTokensDetails {
    /// Create with only reasoning tokens
    pub fn with_reasoning(reasoning: u32) -> Self {
        Self {
            reasoning_tokens: Some(reasoning),
            audio_tokens: None,
            accepted_prediction_tokens: None,
            rejected_prediction_tokens: None,
        }
    }

    /// Create with only audio tokens
    pub fn with_audio(audio: u32) -> Self {
        Self {
            reasoning_tokens: None,
            audio_tokens: Some(audio),
            accepted_prediction_tokens: None,
            rejected_prediction_tokens: None,
        }
    }
}

/// Response metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMetadata {
    /// Response ID
    pub id: Option<String>,
    /// Model name
    pub model: Option<String>,
    /// Creation time
    pub created: Option<chrono::DateTime<chrono::Utc>>,
    /// Provider name
    pub provider: String,
    /// Request ID
    pub request_id: Option<String>,
}
