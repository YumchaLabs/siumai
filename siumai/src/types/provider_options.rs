//! Type-safe provider-specific options
//!
//! This module provides strongly-typed options for provider-specific features,
//! replacing the weakly-typed `HashMap<String, Value>` approach.
//!
//! # User Extensibility
//!
//! Users can extend the system with custom provider features by implementing
//! the `CustomProviderOptions` trait:
//!
//! ```rust,ignore
//! use siumai::types::{CustomProviderOptions, ChatRequest, ProviderOptions};
//!
//! #[derive(Debug, Clone)]
//! struct MyCustomFeature {
//!     pub custom_param: String,
//! }
//!
//! impl CustomProviderOptions for MyCustomFeature {
//!     fn provider_id(&self) -> &str {
//!         "my-provider"
//!     }
//!
//!     fn to_json(&self) -> Result<serde_json::Value, crate::error::LlmError> {
//!         Ok(serde_json::json!({
//!             "custom_param": self.custom_param
//!         }))
//!     }
//! }
//!
//! // Usage
//! let feature = MyCustomFeature { custom_param: "value".to_string() };
//! let req = ChatRequest::new(messages)
//!     .with_provider_options(ProviderOptions::Custom {
//!         provider_id: "my-provider".to_string(),
//!         options: feature.to_json()?.as_object().unwrap().clone().into_iter().collect(),
//!     });
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Re-export OpenAiBuiltInTool from tools module to avoid duplication
pub use super::tools::OpenAiBuiltInTool;

// ============================================================================
// Custom Provider Options Trait
// ============================================================================

/// Trait for user-defined custom provider options
///
/// This trait allows users to extend the system with new provider features
/// without waiting for library updates. Implement this trait to add support
/// for provider-specific features that aren't yet built into the library.
///
/// # Example
///
/// ```rust,ignore
/// use siumai::types::CustomProviderOptions;
/// use siumai::error::LlmError;
///
/// #[derive(Debug, Clone)]
/// pub struct CustomXaiFeature {
///     pub deferred: Option<bool>,
///     pub parallel_function_calling: Option<bool>,
/// }
///
/// impl CustomProviderOptions for CustomXaiFeature {
///     fn provider_id(&self) -> &str {
///         "xai"
///     }
///
///     fn to_json(&self) -> Result<serde_json::Value, LlmError> {
///         let mut map = serde_json::Map::new();
///         if let Some(deferred) = self.deferred {
///             map.insert("deferred".to_string(), serde_json::Value::Bool(deferred));
///         }
///         if let Some(parallel) = self.parallel_function_calling {
///             map.insert("parallel_function_calling".to_string(),
///                       serde_json::Value::Bool(parallel));
///         }
///         Ok(serde_json::Value::Object(map))
///     }
/// }
/// ```
pub trait CustomProviderOptions: Send + Sync + std::fmt::Debug {
    /// Get the provider ID this options is for
    fn provider_id(&self) -> &str;

    /// Convert to JSON for request body injection
    ///
    /// This method should return a JSON object containing the custom parameters
    /// that will be merged into the request body.
    fn to_json(&self) -> Result<serde_json::Value, crate::error::LlmError>;

    /// Apply to request body (called by ProviderSpec::chat_before_send)
    ///
    /// Default implementation merges the JSON from `to_json()` into the request body.
    /// Override this method if you need custom merging logic.
    fn apply_to_request(&self, body: &mut serde_json::Value) -> Result<(), crate::error::LlmError> {
        let json = self.to_json()?;
        if let serde_json::Value::Object(map) = json {
            if let Some(body_obj) = body.as_object_mut() {
                for (k, v) in map {
                    body_obj.insert(k, v);
                }
            }
        }
        Ok(())
    }
}

/// Type-safe provider-specific options
///
/// This enum provides compile-time type safety for provider-specific features,
/// replacing the previous `ProviderParams` HashMap approach.
///
/// # Example
///
/// ```rust,ignore
/// use siumai::types::{ChatRequest, ProviderOptions, XaiOptions, XaiSearchParameters, SearchMode};
///
/// let req = ChatRequest::new(messages)
///     .with_xai_options(
///         XaiOptions::new()
///             .with_search(XaiSearchParameters {
///                 mode: SearchMode::On,
///                 return_citations: Some(true),
///                 ..Default::default()
///             })
///     );
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "provider", content = "options")]
pub enum ProviderOptions {
    /// No provider-specific options
    #[default]
    None,
    /// OpenAI-specific options
    #[serde(rename = "openai")]
    OpenAi(OpenAiOptions),
    /// Anthropic-specific options
    #[serde(rename = "anthropic")]
    Anthropic(AnthropicOptions),
    /// xAI (Grok) specific options
    #[serde(rename = "xai")]
    Xai(XaiOptions),
    /// Google Gemini specific options
    #[serde(rename = "gemini")]
    Gemini(GeminiOptions),
    /// Groq-specific options
    #[serde(rename = "groq")]
    Groq(GroqOptions),
    /// Ollama-specific options
    #[serde(rename = "ollama")]
    Ollama(OllamaOptions),
    /// Custom provider options (for user extensions)
    #[serde(rename = "custom")]
    Custom {
        provider_id: String,
        options: HashMap<String, serde_json::Value>,
    },
}

impl ProviderOptions {
    /// Get the provider ID this options is for
    pub fn provider_id(&self) -> Option<&str> {
        match self {
            Self::None => None,
            Self::OpenAi(_) => Some("openai"),
            Self::Anthropic(_) => Some("anthropic"),
            Self::Xai(_) => Some("xai"),
            Self::Gemini(_) => Some("gemini"),
            Self::Groq(_) => Some("groq"),
            Self::Ollama(_) => Some("ollama"),
            Self::Custom { provider_id, .. } => Some(provider_id),
        }
    }

    /// Check if options match the given provider
    pub fn is_for_provider(&self, provider_id: &str) -> bool {
        self.provider_id().map_or(false, |id| id == provider_id)
    }

    /// Check if this is None
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Create Custom variant from a CustomProviderOptions implementation
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let custom_feature = MyCustomFeature { ... };
    /// let options = ProviderOptions::from_custom(custom_feature)?;
    /// ```
    pub fn from_custom<T: CustomProviderOptions>(
        custom: T,
    ) -> Result<Self, crate::error::LlmError> {
        let provider_id = custom.provider_id().to_string();
        let json = custom.to_json()?;

        // Convert JSON object to HashMap
        let options = if let serde_json::Value::Object(map) = json {
            map.into_iter().collect()
        } else {
            HashMap::new()
        };

        Ok(Self::Custom {
            provider_id,
            options,
        })
    }
}

// ============================================================================
// OpenAI Options
// ============================================================================

/// OpenAI-specific options
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpenAiOptions {
    /// Responses API configuration
    pub responses_api: Option<ResponsesApiConfig>,
    /// Built-in tools (web search, file search, computer use)
    pub built_in_tools: Vec<OpenAiBuiltInTool>,
    /// Reasoning effort (for o1/o3 models)
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Service tier preference
    pub service_tier: Option<ServiceTier>,
}

impl OpenAiOptions {
    /// Create new OpenAI options
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable Responses API with configuration
    pub fn with_responses_api(mut self, config: ResponsesApiConfig) -> Self {
        self.responses_api = Some(config);
        self
    }

    /// Add a built-in tool
    pub fn with_built_in_tool(mut self, tool: OpenAiBuiltInTool) -> Self {
        self.built_in_tools.push(tool);
        self
    }

    /// Enable web search (shorthand for built-in tool)
    pub fn with_web_search(mut self) -> Self {
        self.built_in_tools.push(OpenAiBuiltInTool::WebSearch);
        self
    }

    /// Enable file search with vector store IDs
    pub fn with_file_search(mut self, vector_store_ids: Vec<String>) -> Self {
        self.built_in_tools.push(OpenAiBuiltInTool::FileSearch {
            vector_store_ids: Some(vector_store_ids),
        });
        self
    }

    /// Enable computer use with display settings
    pub fn with_computer_use(
        mut self,
        display_width: u32,
        display_height: u32,
        environment: String,
    ) -> Self {
        self.built_in_tools.push(OpenAiBuiltInTool::ComputerUse {
            display_width,
            display_height,
            environment,
        });
        self
    }

    /// Set reasoning effort
    pub fn with_reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    /// Set service tier
    pub fn with_service_tier(mut self, tier: ServiceTier) -> Self {
        self.service_tier = Some(tier);
        self
    }
}

/// Responses API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesApiConfig {
    /// Whether to use Responses API endpoint
    pub enabled: bool,
    /// Previous response ID for continuation
    pub previous_response_id: Option<String>,
    /// Response format schema
    pub response_format: Option<serde_json::Value>,
}

impl Default for ResponsesApiConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            previous_response_id: None,
            response_format: None,
        }
    }
}

impl ResponsesApiConfig {
    /// Create new Responses API config (enabled by default)
    pub fn new() -> Self {
        Self::default()
    }

    /// Set previous response ID for continuation
    pub fn with_previous_response(mut self, response_id: String) -> Self {
        self.previous_response_id = Some(response_id);
        self
    }

    /// Set response format schema
    pub fn with_response_format(mut self, format: serde_json::Value) -> Self {
        self.response_format = Some(format);
        self
    }
}

/// Reasoning effort for o1/o3 models
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

/// Service tier preference
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ServiceTier {
    Auto,
    Default,
}

// ============================================================================
// xAI Options
// ============================================================================

/// xAI (Grok) specific options
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct XaiOptions {
    /// Reasoning effort for Grok models
    pub reasoning_effort: Option<String>,
    /// Web search parameters
    pub search_parameters: Option<XaiSearchParameters>,
}

impl XaiOptions {
    /// Create new xAI options
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable web search with configuration
    pub fn with_search(mut self, params: XaiSearchParameters) -> Self {
        self.search_parameters = Some(params);
        self
    }

    /// Enable web search with default settings
    pub fn with_default_search(mut self) -> Self {
        self.search_parameters = Some(XaiSearchParameters::default());
        self
    }

    /// Set reasoning effort
    pub fn with_reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }
}

/// xAI web search parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XaiSearchParameters {
    /// Search mode
    pub mode: SearchMode,
    /// Whether to return citations
    pub return_citations: Option<bool>,
    /// Maximum number of search results
    pub max_search_results: Option<u32>,
    /// Start date for search (YYYY-MM-DD)
    pub from_date: Option<String>,
    /// End date for search (YYYY-MM-DD)
    pub to_date: Option<String>,
    /// Search sources configuration
    pub sources: Option<Vec<SearchSource>>,
}

impl Default for XaiSearchParameters {
    fn default() -> Self {
        Self {
            mode: SearchMode::Auto,
            return_citations: Some(true),
            max_search_results: Some(5),
            from_date: None,
            to_date: None,
            sources: None,
        }
    }
}

/// Search mode
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    /// Automatically decide whether to search
    Auto,
    /// Always search
    On,
    /// Never search
    Off,
}

/// Search source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSource {
    /// Source type
    #[serde(rename = "type")]
    pub source_type: SearchSourceType,
    /// Country code for localized search
    pub country: Option<String>,
    /// Allowed websites
    pub allowed_websites: Option<Vec<String>>,
    /// Excluded websites
    pub excluded_websites: Option<Vec<String>>,
    /// Enable safe search
    pub safe_search: Option<bool>,
}

/// Search source type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchSourceType {
    /// Web search
    Web,
    /// News search
    News,
    /// X (Twitter) search
    X,
}

// ============================================================================
// Anthropic Options
// ============================================================================

/// Anthropic-specific options
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnthropicOptions {
    /// Prompt caching configuration
    pub prompt_caching: Option<PromptCachingConfig>,
    /// Thinking mode (extended thinking)
    pub thinking_mode: Option<ThinkingModeConfig>,
}

impl AnthropicOptions {
    /// Create new Anthropic options
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable prompt caching
    pub fn with_prompt_caching(mut self, config: PromptCachingConfig) -> Self {
        self.prompt_caching = Some(config);
        self
    }

    /// Enable thinking mode
    pub fn with_thinking_mode(mut self, config: ThinkingModeConfig) -> Self {
        self.thinking_mode = Some(config);
        self
    }
}

/// Prompt caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptCachingConfig {
    /// Whether prompt caching is enabled
    pub enabled: bool,
    /// Cache control markers
    pub cache_control: Vec<AnthropicCacheControl>,
}

impl Default for PromptCachingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cache_control: vec![],
        }
    }
}

/// Anthropic cache control marker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicCacheControl {
    /// Cache type
    pub cache_type: AnthropicCacheType,
    /// Message index to apply cache control to
    pub message_index: usize,
}

/// Anthropic cache type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicCacheType {
    /// Ephemeral cache (5 minutes TTL)
    Ephemeral,
}

/// Thinking mode configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingModeConfig {
    /// Whether thinking mode is enabled
    pub enabled: bool,
    /// Thinking budget (tokens)
    pub thinking_budget: Option<u32>,
}

impl Default for ThinkingModeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            thinking_budget: None,
        }
    }
}

// ============================================================================
// Gemini Options
// ============================================================================

/// Google Gemini specific options
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GeminiOptions {
    /// Code execution configuration
    pub code_execution: Option<CodeExecutionConfig>,
    /// Search grounding (web search)
    pub search_grounding: Option<SearchGroundingConfig>,
}

impl GeminiOptions {
    /// Create new Gemini options
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable code execution
    pub fn with_code_execution(mut self, config: CodeExecutionConfig) -> Self {
        self.code_execution = Some(config);
        self
    }

    /// Enable search grounding
    pub fn with_search_grounding(mut self, config: SearchGroundingConfig) -> Self {
        self.search_grounding = Some(config);
        self
    }
}

/// Code execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExecutionConfig {
    /// Whether code execution is enabled
    pub enabled: bool,
}

impl Default for CodeExecutionConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Search grounding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchGroundingConfig {
    /// Whether search grounding is enabled
    pub enabled: bool,
    /// Dynamic retrieval configuration
    pub dynamic_retrieval_config: Option<DynamicRetrievalConfig>,
}

impl Default for SearchGroundingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            dynamic_retrieval_config: None,
        }
    }
}

/// Dynamic retrieval configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicRetrievalConfig {
    /// Retrieval mode
    pub mode: DynamicRetrievalMode,
    /// Dynamic threshold
    pub dynamic_threshold: Option<f32>,
}

/// Dynamic retrieval mode
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DynamicRetrievalMode {
    /// Unspecified mode
    ModeUnspecified,
    /// Dynamic mode
    ModeDynamic,
}

// ============================================================================
// Groq Options
// ============================================================================

/// Groq-specific options
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GroqOptions {
    /// Additional Groq-specific parameters
    pub extra_params: HashMap<String, serde_json::Value>,
}

impl GroqOptions {
    /// Create new Groq options
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a custom parameter
    pub fn with_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_params.insert(key.into(), value);
        self
    }
}

// ============================================================================
// Ollama Options
// ============================================================================

/// Ollama-specific options
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OllamaOptions {
    /// Keep model loaded in memory for this duration
    pub keep_alive: Option<String>,
    /// Use raw mode (bypass templating)
    pub raw: Option<bool>,
    /// Format for structured outputs
    pub format: Option<String>,
    /// Additional Ollama-specific parameters
    pub extra_params: HashMap<String, serde_json::Value>,
}

impl OllamaOptions {
    /// Create new Ollama options
    pub fn new() -> Self {
        Self::default()
    }

    /// Set keep alive duration
    pub fn with_keep_alive(mut self, duration: impl Into<String>) -> Self {
        self.keep_alive = Some(duration.into());
        self
    }

    /// Enable raw mode
    pub fn with_raw_mode(mut self, raw: bool) -> Self {
        self.raw = Some(raw);
        self
    }

    /// Set output format
    pub fn with_format(mut self, format: impl Into<String>) -> Self {
        self.format = Some(format.into());
        self
    }

    /// Add a custom parameter
    pub fn with_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_params.insert(key.into(), value);
        self
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_options_provider_id() {
        assert_eq!(ProviderOptions::None.provider_id(), None);
        assert_eq!(
            ProviderOptions::OpenAi(OpenAiOptions::default()).provider_id(),
            Some("openai")
        );
        assert_eq!(
            ProviderOptions::Xai(XaiOptions::default()).provider_id(),
            Some("xai")
        );
        assert_eq!(
            ProviderOptions::Anthropic(AnthropicOptions::default()).provider_id(),
            Some("anthropic")
        );
    }

    #[test]
    fn test_provider_options_is_for_provider() {
        let options = ProviderOptions::Xai(XaiOptions::default());
        assert!(options.is_for_provider("xai"));
        assert!(!options.is_for_provider("openai"));
    }

    #[test]
    fn test_openai_options_builder() {
        let options = OpenAiOptions::new()
            .with_web_search()
            .with_file_search(vec!["vs_123".to_string()])
            .with_reasoning_effort(ReasoningEffort::High);

        assert_eq!(options.built_in_tools.len(), 2);
        assert!(matches!(
            options.built_in_tools[0],
            OpenAiBuiltInTool::WebSearch
        ));
        assert!(matches!(
            options.reasoning_effort,
            Some(ReasoningEffort::High)
        ));
    }

    #[test]
    fn test_responses_api_config() {
        let config = ResponsesApiConfig::new()
            .with_previous_response("resp_123".to_string())
            .with_response_format(serde_json::json!({"type": "json_object"}));

        assert!(config.enabled);
        assert_eq!(config.previous_response_id, Some("resp_123".to_string()));
        assert!(config.response_format.is_some());
    }

    #[test]
    fn test_xai_options_builder() {
        let options = XaiOptions::new()
            .with_default_search()
            .with_reasoning_effort("high");

        assert!(options.search_parameters.is_some());
        assert_eq!(options.reasoning_effort, Some("high".to_string()));
    }

    #[test]
    fn test_xai_search_parameters_default() {
        let params = XaiSearchParameters::default();
        assert!(matches!(params.mode, SearchMode::Auto));
        assert_eq!(params.return_citations, Some(true));
        assert_eq!(params.max_search_results, Some(5));
    }

    #[test]
    fn test_xai_search_parameters_custom() {
        let params = XaiSearchParameters {
            mode: SearchMode::On,
            return_citations: Some(true),
            max_search_results: Some(10),
            from_date: Some("2024-01-01".to_string()),
            to_date: Some("2024-12-31".to_string()),
            sources: Some(vec![SearchSource {
                source_type: SearchSourceType::Web,
                country: Some("US".to_string()),
                allowed_websites: Some(vec!["arxiv.org".to_string()]),
                excluded_websites: None,
                safe_search: Some(true),
            }]),
        };

        assert!(matches!(params.mode, SearchMode::On));
        assert_eq!(params.max_search_results, Some(10));
        assert!(params.sources.is_some());
    }

    #[test]
    fn test_anthropic_options_builder() {
        let options = AnthropicOptions::new()
            .with_prompt_caching(PromptCachingConfig::default())
            .with_thinking_mode(ThinkingModeConfig::default());

        assert!(options.prompt_caching.is_some());
        assert!(options.thinking_mode.is_some());
    }

    #[test]
    fn test_gemini_options_builder() {
        let options = GeminiOptions::new()
            .with_code_execution(CodeExecutionConfig::default())
            .with_search_grounding(SearchGroundingConfig::default());

        assert!(options.code_execution.is_some());
        assert!(options.search_grounding.is_some());
    }

    #[test]
    fn test_ollama_options_builder() {
        let options = OllamaOptions::new()
            .with_keep_alive("5m")
            .with_raw_mode(true)
            .with_format("json");

        assert_eq!(options.keep_alive, Some("5m".to_string()));
        assert_eq!(options.raw, Some(true));
        assert_eq!(options.format, Some("json".to_string()));
    }

    #[test]
    fn test_serialization_openai() {
        let options = ProviderOptions::OpenAi(OpenAiOptions::new().with_web_search());

        let json = serde_json::to_string(&options).unwrap();
        let deserialized: ProviderOptions = serde_json::from_str(&json).unwrap();

        assert!(matches!(deserialized, ProviderOptions::OpenAi(_)));
    }

    #[test]
    fn test_serialization_xai() {
        let options = ProviderOptions::Xai(XaiOptions::new().with_default_search());

        let json = serde_json::to_string(&options).unwrap();
        let deserialized: ProviderOptions = serde_json::from_str(&json).unwrap();

        assert!(matches!(deserialized, ProviderOptions::Xai(_)));
    }

    #[test]
    fn test_custom_provider_options() {
        let mut custom_opts = HashMap::new();
        custom_opts.insert("custom_param".to_string(), serde_json::json!("value"));

        let options = ProviderOptions::Custom {
            provider_id: "my_provider".to_string(),
            options: custom_opts,
        };

        assert_eq!(options.provider_id(), Some("my_provider"));
        assert!(options.is_for_provider("my_provider"));
    }
}
