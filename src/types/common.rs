//! Common types and enums used across the library

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use validator::Validate;

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

/// Common AI parameters
#[derive(Debug, Clone, Serialize, Deserialize, Default, Validate)]
pub struct CommonParams {
    /// Model name
    #[validate(length(min = 1, message = "Model name cannot be empty"))]
    pub model: String,

    /// Temperature parameter (must be non-negative)
    #[validate(range(min = 0.0, message = "Temperature must be non-negative"))]
    pub temperature: Option<f32>,

    /// Maximum output tokens
    pub max_tokens: Option<u32>,

    /// `top_p` parameter
    #[validate(range(min = 0.0, max = 1.0, message = "top_p must be between 0.0 and 1.0"))]
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
        use validator::Validate;
        self.validate()
            .map_err(|e| crate::error::LlmError::InvalidParameter(e.to_string()))?;
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
            top_p: self.top_p,
            stop_sequences: self.stop_sequences,
            seed: self.seed,
        };

        params.validate_params()?;
        Ok(params)
    }
}

/// Provider-specific parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderParams {
    /// A map for provider-specific parameters
    pub params: HashMap<String, serde_json::Value>,
}

impl ProviderParams {
    /// Creates new provider parameters
    pub fn new() -> Self {
        Self {
            params: HashMap::new(),
        }
    }

    /// Adds a parameter
    pub fn with_param<T: Serialize>(mut self, key: &str, value: T) -> Self {
        self.params
            .insert(key.to_string(), serde_json::to_value(value).unwrap());
        self
    }

    /// Gets a parameter
    pub fn get<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Option<T> {
        self.params
            .get(key)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// Creates provider parameters from `OpenAI` parameters
    pub fn from_openai(openai_params: crate::params::OpenAiParams) -> Self {
        let mut params = HashMap::new();

        // Serialize the OpenAI params to a JSON value and then convert to HashMap
        if let Ok(json_value) = serde_json::to_value(&openai_params)
            && let Ok(map) =
                serde_json::from_value::<HashMap<String, serde_json::Value>>(json_value)
        {
            params = map;
        }

        Self { params }
    }

    /// Creates provider parameters from `Anthropic` parameters
    pub fn from_anthropic(anthropic_params: crate::params::AnthropicParams) -> Self {
        let mut params = HashMap::new();

        // Serialize the Anthropic params to a JSON value and then convert to HashMap
        if let Ok(json_value) = serde_json::to_value(&anthropic_params)
            && let Ok(map) =
                serde_json::from_value::<HashMap<String, serde_json::Value>>(json_value)
        {
            params = map;
        }

        Self { params }
    }

    /// Creates provider parameters from `Gemini` parameters
    pub fn from_gemini(gemini_params: crate::params::GeminiParams) -> Self {
        let mut params = HashMap::new();

        // Serialize the Gemini params to a JSON value and then convert to HashMap
        if let Ok(json_value) = serde_json::to_value(&gemini_params)
            && let Ok(map) =
                serde_json::from_value::<HashMap<String, serde_json::Value>>(json_value)
        {
            params = map;
        }

        Self { params }
    }

    /// Quick setup for OpenAI-specific parameters.
    pub fn openai() -> Self {
        Self::new()
            .with_param("frequency_penalty", 0.0)
            .with_param("presence_penalty", 0.0)
            .with_param("parallel_tool_calls", true)
    }

    /// Quick setup for Anthropic-specific parameters.
    pub fn anthropic() -> Self {
        Self::new().with_param("max_tokens", 4096)
    }

    /// Quick setup for Gemini-specific parameters.
    pub fn gemini() -> Self {
        Self::new()
            .with_param("candidate_count", 1)
            .with_param("top_k", 40)
    }
}

impl Default for ProviderParams {
    fn default() -> Self {
        Self::new()
    }
}

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
        Self {
            timeout: Some(crate::defaults::http::REQUEST_TIMEOUT),
            connect_timeout: Some(crate::defaults::http::CONNECT_TIMEOUT),
            headers: HashMap::new(),
            proxy: None,
            user_agent: Some(crate::defaults::http::USER_AGENT.to_string()),
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
    pub cached_tokens: Option<u32>,
    /// Reasoning tokens (for models like o1)
    pub reasoning_tokens: Option<u32>,
}

impl Usage {
    /// Create new usage statistics
    pub const fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            cached_tokens: None,
            reasoning_tokens: None,
        }
    }

    /// Merge usage statistics
    pub fn merge(&mut self, other: &Usage) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.total_tokens += other.total_tokens;
        if let Some(other_cached) = other.cached_tokens {
            self.cached_tokens = Some(self.cached_tokens.unwrap_or(0) + other_cached);
        }
        if let Some(other_reasoning) = other.reasoning_tokens {
            self.reasoning_tokens = Some(self.reasoning_tokens.unwrap_or(0) + other_reasoning);
        }
    }
}

/// Finish reason for the response
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Natural completion
    Stop,
    /// Maximum tokens reached
    Length,
    /// Tool call requested
    ToolCalls,
    /// Content filtered
    ContentFilter,
    /// Model stopped due to stop sequence
    StopSequence,
    /// Error occurred
    Error,
    /// Other reason
    Other(String),
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
