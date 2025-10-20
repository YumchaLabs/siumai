//! `OpenAI` Parameter Mapping
//!
//! Contains OpenAI-specific parameter mapping and validation logic.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Truncation strategy for Responses API
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TruncationStrategy {
    /// Automatically truncate to fit context window
    Auto,
    /// Fail if context window is exceeded (default)
    Disabled,
}

impl Default for TruncationStrategy {
    fn default() -> Self {
        Self::Disabled
    }
}

/// Includable items for Responses API
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IncludableItem {
    /// Include outputs of python code execution in code interpreter tool calls
    #[serde(rename = "code_interpreter_call.outputs")]
    CodeInterpreterCallOutputs,
    /// Include image urls from computer call output
    #[serde(rename = "computer_call_output.output.image_url")]
    ComputerCallOutputImageUrl,
    /// Include search results of file search tool calls
    #[serde(rename = "file_search_call.results")]
    FileSearchCallResults,
    /// Include image urls from input messages
    #[serde(rename = "message.input_image.image_url")]
    MessageInputImageUrl,
    /// Include logprobs with assistant messages
    #[serde(rename = "message.output_text.logprobs")]
    MessageOutputTextLogprobs,
    /// Include encrypted reasoning content
    #[serde(rename = "reasoning.encrypted_content")]
    ReasoningEncryptedContent,
}

/// Sort order for list operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SortOrder {
    /// Ascending order
    Asc,
    /// Descending order
    Desc,
}

use crate::error::LlmError;
use crate::types::ProviderType;

// OpenAI ParameterMapper removed; use Transformers for mapping/validation.

/// OpenAI-specific parameter extensions
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenAiParams {
    /// Response format
    pub response_format: Option<ResponseFormat>,

    /// Tool choice strategy
    pub tool_choice: Option<ToolChoice>,

    /// Parallel tool calls
    pub parallel_tool_calls: Option<bool>,

    /// Persist response in server-side store (Responses API)
    pub store: Option<bool>,

    /// Custom metadata for Responses API
    pub metadata: Option<HashMap<String, String>>,

    /// User ID
    pub user: Option<String>,

    /// Frequency penalty (-2.0 to 2.0) - OpenAI standard range
    pub frequency_penalty: Option<f32>,

    /// Presence penalty (-2.0 to 2.0) - OpenAI standard range
    pub presence_penalty: Option<f32>,

    /// Logit bias
    pub logit_bias: Option<HashMap<String, f32>>,

    /// Number of choices to return
    pub n: Option<u32>,

    /// Whether to stream the response
    pub stream: Option<bool>,

    /// Logprobs configuration
    pub logprobs: Option<bool>,

    /// Top logprobs to return
    pub top_logprobs: Option<u32>,

    /// Response modalities (text, audio)
    pub modalities: Option<Vec<String>>,

    /// Reasoning effort level for reasoning models
    pub reasoning_effort: Option<ReasoningEffort>,

    /// Maximum completion tokens (replaces `max_tokens` for some models)
    pub max_completion_tokens: Option<u32>,

    /// Service tier for prioritized access
    pub service_tier: Option<ServiceTier>,

    // Responses API specific parameters
    /// System instructions for Responses API
    pub instructions: Option<String>,

    /// Additional output data to include in the response
    pub include: Option<Vec<IncludableItem>>,

    /// Truncation strategy for the model response
    pub truncation: Option<TruncationStrategy>,

    /// Reasoning configuration for o-series models
    pub reasoning: Option<serde_json::Value>,

    /// Maximum output tokens (Responses API)
    pub max_output_tokens: Option<u32>,

    /// Maximum number of tool calls
    pub max_tool_calls: Option<u32>,

    /// Text response configuration
    pub text: Option<serde_json::Value>,

    /// Prompt configuration
    pub prompt: Option<serde_json::Value>,

    /// Whether to run in background
    pub background: Option<bool>,

    /// Stream options configuration
    pub stream_options: Option<serde_json::Value>,

    /// Safety identifier for abuse detection
    pub safety_identifier: Option<String>,

    /// Prompt cache key for optimization
    pub prompt_cache_key: Option<String>,

    // Chat Completions API specific parameters
    /// Audio output configuration
    pub audio: Option<serde_json::Value>,

    /// Web search options
    pub web_search_options: Option<serde_json::Value>,

    /// Prediction configuration for Predicted Outputs
    pub prediction: Option<serde_json::Value>,

    /// Verbosity level
    pub verbosity: Option<Verbosity>,

    /// Function call (deprecated, use tool_choice)
    #[deprecated(note = "Use tool_choice instead")]
    pub function_call: Option<serde_json::Value>,

    /// Functions (deprecated, use tools)
    #[deprecated(note = "Use tools instead")]
    pub functions: Option<Vec<serde_json::Value>>,
}

/// Verbosity level for responses
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Verbosity {
    /// Low verbosity
    Low,
    /// Medium verbosity (default)
    Medium,
    /// High verbosity
    High,
}

impl Default for Verbosity {
    fn default() -> Self {
        Self::Medium
    }
}

/// Service tier for request processing
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ServiceTier {
    /// Auto-select based on project settings (default)
    Auto,
    /// Standard pricing and performance
    Default,
    /// Flex processing
    Flex,
    /// Scale processing
    Scale,
    /// Priority processing
    Priority,
}

impl Default for ServiceTier {
    fn default() -> Self {
        Self::Auto
    }
}

impl super::common::ProviderParamsExt for OpenAiParams {
    fn provider_type(&self) -> ProviderType {
        ProviderType::OpenAi
    }
}

impl OpenAiParams {
    /// Validate OpenAI-specific parameters
    pub fn validate_params(&self) -> Result<(), LlmError> {
        // Validate frequency_penalty (-2.0 to 2.0)
        if let Some(penalty) = self.frequency_penalty {
            if !(-2.0..=2.0).contains(&penalty) {
                return Err(LlmError::InvalidParameter(
                    "Frequency penalty must be between -2.0 and 2.0".to_string(),
                ));
            }
        }

        // Validate presence_penalty (-2.0 to 2.0)
        if let Some(penalty) = self.presence_penalty {
            if !(-2.0..=2.0).contains(&penalty) {
                return Err(LlmError::InvalidParameter(
                    "Presence penalty must be between -2.0 and 2.0".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Create a builder for OpenAI parameters
    pub fn builder() -> OpenAiParamsBuilder {
        OpenAiParamsBuilder::new()
    }
}

/// Builder for OpenAI parameters with validation
#[derive(Debug, Clone, Default)]
pub struct OpenAiParamsBuilder {
    response_format: Option<ResponseFormat>,
    tool_choice: Option<ToolChoice>,
    parallel_tool_calls: Option<bool>,
    store: Option<bool>,
    metadata: Option<HashMap<String, String>>,

    user: Option<String>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    logit_bias: Option<HashMap<String, f32>>,
    n: Option<u32>,
    stream: Option<bool>,
    logprobs: Option<bool>,
    top_logprobs: Option<u32>,
    modalities: Option<Vec<String>>,
    reasoning_effort: Option<ReasoningEffort>,
    max_completion_tokens: Option<u32>,
    service_tier: Option<ServiceTier>,

    // Responses API specific parameters
    instructions: Option<String>,
    include: Option<Vec<IncludableItem>>,
    truncation: Option<TruncationStrategy>,
    reasoning: Option<serde_json::Value>,
    max_output_tokens: Option<u32>,
    max_tool_calls: Option<u32>,
    text: Option<serde_json::Value>,
    prompt: Option<serde_json::Value>,
    background: Option<bool>,
    stream_options: Option<serde_json::Value>,
    safety_identifier: Option<String>,
    prompt_cache_key: Option<String>,

    // Chat Completions API specific parameters
    audio: Option<serde_json::Value>,
    web_search_options: Option<serde_json::Value>,
    prediction: Option<serde_json::Value>,
    verbosity: Option<Verbosity>,
    function_call: Option<serde_json::Value>,
    functions: Option<Vec<serde_json::Value>>,
}

impl OpenAiParamsBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set response format
    pub fn response_format(mut self, format: ResponseFormat) -> Self {
        self.response_format = Some(format);
        self
    }

    /// Set tool choice
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Set parallel tool calls
    /// Set store flag
    pub fn store(mut self, store: bool) -> Self {
        self.store = Some(store);
        self
    }

    /// Set metadata
    pub fn metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn parallel_tool_calls(mut self, parallel: bool) -> Self {
        self.parallel_tool_calls = Some(parallel);
        self
    }

    /// Set user ID
    pub fn user<S: Into<String>>(mut self, user: S) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Set frequency penalty with validation
    pub fn frequency_penalty(mut self, penalty: f32) -> Result<Self, LlmError> {
        if !(-2.0..=2.0).contains(&penalty) {
            return Err(LlmError::InvalidParameter(
                "Frequency penalty must be between -2.0 and 2.0".to_string(),
            ));
        }
        self.frequency_penalty = Some(penalty);
        Ok(self)
    }

    /// Set presence penalty with validation
    pub fn presence_penalty(mut self, penalty: f32) -> Result<Self, LlmError> {
        if !(-2.0..=2.0).contains(&penalty) {
            return Err(LlmError::InvalidParameter(
                "Presence penalty must be between -2.0 and 2.0".to_string(),
            ));
        }
        self.presence_penalty = Some(penalty);
        Ok(self)
    }

    /// Set logit bias
    pub fn logit_bias(mut self, bias: HashMap<String, f32>) -> Self {
        self.logit_bias = Some(bias);
        self
    }

    /// Set number of choices
    pub fn n(mut self, n: u32) -> Self {
        self.n = Some(n);
        self
    }

    /// Set streaming
    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }

    /// Set logprobs
    pub fn logprobs(mut self, logprobs: bool) -> Self {
        self.logprobs = Some(logprobs);
        self
    }

    /// Set top logprobs
    pub fn top_logprobs(mut self, top_logprobs: u32) -> Self {
        self.top_logprobs = Some(top_logprobs);
        self
    }

    /// Set modalities
    pub fn modalities(mut self, modalities: Vec<String>) -> Self {
        self.modalities = Some(modalities);
        self
    }

    /// Set reasoning effort
    pub fn reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    /// Set max completion tokens
    pub fn max_completion_tokens(mut self, tokens: u32) -> Self {
        self.max_completion_tokens = Some(tokens);
        self
    }

    /// Set service tier
    pub fn service_tier(mut self, tier: ServiceTier) -> Self {
        self.service_tier = Some(tier);
        self
    }

    /// Set instructions for Responses API
    pub fn instructions<S: Into<String>>(mut self, instructions: S) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Set include array for Responses API
    pub fn include(mut self, include: Vec<IncludableItem>) -> Self {
        self.include = Some(include);
        self
    }

    /// Set truncation strategy for Responses API
    pub fn truncation(mut self, truncation: TruncationStrategy) -> Self {
        self.truncation = Some(truncation);
        self
    }

    /// Set reasoning configuration for Responses API
    pub fn reasoning(mut self, reasoning: serde_json::Value) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    /// Set max output tokens for Responses API
    pub fn max_output_tokens(mut self, tokens: u32) -> Self {
        self.max_output_tokens = Some(tokens);
        self
    }

    /// Set max tool calls for Responses API
    pub fn max_tool_calls(mut self, calls: u32) -> Self {
        self.max_tool_calls = Some(calls);
        self
    }

    /// Set text configuration for Responses API
    pub fn text(mut self, text: serde_json::Value) -> Self {
        self.text = Some(text);
        self
    }

    /// Set prompt configuration for Responses API
    pub fn prompt(mut self, prompt: serde_json::Value) -> Self {
        self.prompt = Some(prompt);
        self
    }

    /// Set background mode for Responses API
    pub fn background(mut self, background: bool) -> Self {
        self.background = Some(background);
        self
    }

    /// Set stream options for Responses API
    pub fn stream_options(mut self, options: serde_json::Value) -> Self {
        self.stream_options = Some(options);
        self
    }

    /// Set safety identifier for abuse detection
    pub fn safety_identifier<S: Into<String>>(mut self, identifier: S) -> Self {
        self.safety_identifier = Some(identifier.into());
        self
    }

    /// Set prompt cache key for optimization
    pub fn prompt_cache_key<S: Into<String>>(mut self, key: S) -> Self {
        self.prompt_cache_key = Some(key.into());
        self
    }

    /// Set audio output configuration
    pub fn audio(mut self, audio: serde_json::Value) -> Self {
        self.audio = Some(audio);
        self
    }

    /// Set web search options
    pub fn web_search_options(mut self, options: serde_json::Value) -> Self {
        self.web_search_options = Some(options);
        self
    }

    /// Set prediction configuration for Predicted Outputs
    pub fn prediction(mut self, prediction: serde_json::Value) -> Self {
        self.prediction = Some(prediction);
        self
    }

    /// Set verbosity level
    pub fn verbosity(mut self, verbosity: Verbosity) -> Self {
        self.verbosity = Some(verbosity);
        self
    }

    /// Set function call (deprecated, use tool_choice)
    #[deprecated(note = "Use tool_choice instead")]
    pub fn function_call(mut self, function_call: serde_json::Value) -> Self {
        self.function_call = Some(function_call);
        self
    }

    /// Set functions (deprecated, use tools)
    #[deprecated(note = "Use tools instead")]
    pub fn functions(mut self, functions: Vec<serde_json::Value>) -> Self {
        self.functions = Some(functions);
        self
    }

    /// Build the OpenAI parameters
    #[allow(deprecated)]
    pub fn build(self) -> Result<OpenAiParams, LlmError> {
        let params = OpenAiParams {
            response_format: self.response_format,
            tool_choice: self.tool_choice,
            parallel_tool_calls: self.parallel_tool_calls,
            store: self.store,
            metadata: self.metadata,
            user: self.user,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            logit_bias: self.logit_bias,
            n: self.n,
            stream: self.stream,
            logprobs: self.logprobs,
            top_logprobs: self.top_logprobs,
            modalities: self.modalities,
            reasoning_effort: self.reasoning_effort,
            max_completion_tokens: self.max_completion_tokens,
            service_tier: self.service_tier,
            instructions: self.instructions,
            include: self.include,
            truncation: self.truncation,
            reasoning: self.reasoning,
            max_output_tokens: self.max_output_tokens,
            max_tool_calls: self.max_tool_calls,
            text: self.text,
            prompt: self.prompt,
            background: self.background,
            stream_options: self.stream_options,
            safety_identifier: self.safety_identifier,
            prompt_cache_key: self.prompt_cache_key,
            audio: self.audio,
            web_search_options: self.web_search_options,
            prediction: self.prediction,
            verbosity: self.verbosity,
            function_call: self.function_call,
            functions: self.functions,
        };

        params.validate_params()?;
        Ok(params)
    }
}

/// `OpenAI` Response Format
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseFormat {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
    #[serde(rename = "json_schema")]
    JsonSchema { schema: serde_json::Value },
}

/// `OpenAI` Tool Choice
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    String(String), // "none", "auto", "required"
    Function {
        #[serde(rename = "type")]
        choice_type: String, // "function"
        function: FunctionChoice,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionChoice {
    pub name: String,
}

/// Reasoning effort level for reasoning models (o1 series)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// Minimal reasoning effort - fastest responses
    Minimal,
    /// Low reasoning effort - faster responses
    Low,
    /// Medium reasoning effort - balanced performance (default)
    Medium,
    /// High reasoning effort - more thorough reasoning
    High,
}

impl Default for ReasoningEffort {
    fn default() -> Self {
        Self::Medium
    }
}

// tests removed; covered by Transformers
