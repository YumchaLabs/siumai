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
use validator::Validate;

use super::common::{ParameterMapper as CommonMapper, ParameterValidator};
use super::mapper::{ParameterConstraints, ParameterMapper};
use crate::error::LlmError;
use crate::types::{CommonParams, ProviderParams, ProviderType};

/// `OpenAI` Parameter Mapper
#[derive(Debug, Clone)]
pub struct OpenAiParameterMapper;

impl ParameterMapper for OpenAiParameterMapper {
    fn map_common_params(&self, params: &CommonParams) -> serde_json::Value {
        let mut json = CommonMapper::map_common_to_json(params);

        // Handle OpenAI-specific stop sequences format
        if let Some(stop) = &params.stop_sequences {
            json["stop"] = stop.clone().into();
        }

        json
    }

    fn merge_provider_params(
        &self,
        base: serde_json::Value,
        provider: &ProviderParams,
    ) -> serde_json::Value {
        CommonMapper::merge_provider_params(base, provider)
    }

    fn validate_params(&self, params: &serde_json::Value) -> Result<(), LlmError> {
        // Validate OpenAI-specific parameter constraints
        if let Some(temp) = params.get("temperature")
            && let Some(temp_val) = temp.as_f64()
        {
            ParameterValidator::validate_temperature(temp_val, 0.0, 2.0, "OpenAI")?;
        }

        if let Some(top_p) = params.get("top_p")
            && let Some(top_p_val) = top_p.as_f64()
        {
            ParameterValidator::validate_top_p(top_p_val)?;
        }

        if let Some(max_tokens) = params.get("max_tokens")
            && let Some(max_tokens_val) = max_tokens.as_u64()
        {
            ParameterValidator::validate_max_tokens(max_tokens_val, 1, 128_000, "OpenAI")?;
        }

        // Validate OpenAI-specific parameters
        if let Some(frequency_penalty) = params.get("frequency_penalty")
            && let Some(penalty_val) = frequency_penalty.as_f64()
        {
            ParameterValidator::validate_numeric_range(
                penalty_val,
                -2.0,
                2.0,
                "frequency_penalty",
                "OpenAI",
            )?;
        }

        if let Some(presence_penalty) = params.get("presence_penalty")
            && let Some(penalty_val) = presence_penalty.as_f64()
        {
            ParameterValidator::validate_numeric_range(
                penalty_val,
                -2.0,
                2.0,
                "presence_penalty",
                "OpenAI",
            )?;
        }

        if let Some(n) = params.get("n")
            && let Some(n_val) = n.as_u64()
            && (n_val == 0 || n_val > 128)
        {
            return Err(LlmError::InvalidParameter(
                "n must be between 1 and 128 for OpenAI".to_string(),
            ));
        }

        // Validate max_completion_tokens
        if let Some(max_completion_tokens) = params.get("max_completion_tokens")
            && let Some(tokens_val) = max_completion_tokens.as_u64()
        {
            ParameterValidator::validate_max_tokens(
                tokens_val,
                1,
                128_000,
                "OpenAI max_completion_tokens",
            )?;
        }

        // Validate top_logprobs
        if let Some(top_logprobs) = params.get("top_logprobs")
            && let Some(logprobs_val) = top_logprobs.as_u64()
            && logprobs_val > 20
        {
            return Err(LlmError::InvalidParameter(
                "top_logprobs must be between 0 and 20 for OpenAI".to_string(),
            ));
        }

        // Validate modalities
        if let Some(modalities) = params.get("modalities")
            && let Some(modalities_array) = modalities.as_array()
        {
            for modality in modalities_array {
                if let Some(modality_str) = modality.as_str()
                    && !["text", "audio"].contains(&modality_str)
                {
                    return Err(LlmError::InvalidParameter(format!(
                        "Invalid modality '{modality_str}'. Supported modalities: text, audio"
                    )));
                }
            }
        }

        // Validate service_tier
        if let Some(service_tier) = params.get("service_tier")
            && let Some(tier_str) = service_tier.as_str()
            && !["auto", "default"].contains(&tier_str)
        {
            return Err(LlmError::InvalidParameter(format!(
                "Invalid service_tier '{tier_str}'. Supported tiers: auto, default"
            )));
        }

        Ok(())
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::OpenAi
    }

    fn supported_params(&self) -> Vec<&'static str> {
        vec![
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "stop",
            "seed",
            "frequency_penalty",
            "presence_penalty",
            "logit_bias",
            "user",
            "n",
            "stream",
            "response_format",
            "tool_choice",
            "tools",
            "parallel_tool_calls",
            "modalities",
            "reasoning_effort",
            "max_completion_tokens",
            "service_tier",
            "logprobs",
            "top_logprobs",
            "store",
            "metadata",
            "audio",
            "web_search_options",
            "prediction",
            "verbosity",
            "function_call",
            "functions",
        ]
    }

    fn get_param_constraints(&self) -> ParameterConstraints {
        ParameterConstraints {
            temperature_min: 0.0,
            temperature_max: 2.0,
            max_tokens_min: 1,
            max_tokens_max: 128_000,
            top_p_min: 0.0,
            top_p_max: 1.0,
        }
    }
}

/// OpenAI-specific parameter extensions
#[derive(Debug, Clone, Serialize, Deserialize, Default, Validate)]
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
    #[validate(range(min = -2.0, max = 2.0, message = "Frequency penalty must be between -2.0 and 2.0"))]
    pub frequency_penalty: Option<f32>,

    /// Presence penalty (-2.0 to 2.0) - OpenAI standard range
    #[validate(range(min = -2.0, max = 2.0, message = "Presence penalty must be between -2.0 and 2.0"))]
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
        use validator::Validate;
        self.validate()
            .map_err(|e| LlmError::InvalidParameter(e.to_string()))?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_parameter_mapping() {
        let mapper = OpenAiParameterMapper;
        let params = CommonParams {
            model: "gpt-4".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(1000),
            top_p: Some(0.9),
            stop_sequences: Some(vec!["STOP".to_string()]),
            seed: Some(42),
        };

        let mapped_params = mapper.map_common_params(&params);
        assert_eq!(mapped_params["model"], "gpt-4");
        assert_eq!(mapped_params["max_tokens"], 1000);
        assert_eq!(mapped_params["seed"], 42);
        assert_eq!(mapped_params["stop"], serde_json::json!(["STOP"]));
    }

    #[test]
    fn test_openai_parameter_validation() {
        let mapper = OpenAiParameterMapper;

        // Valid parameters
        let valid_params = serde_json::json!({
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 1000,
            "frequency_penalty": 0.5,
            "presence_penalty": -0.5,
            "n": 1
        });
        assert!(mapper.validate_params(&valid_params).is_ok());

        // High temperature (now allowed with relaxed validation)
        let high_temp = serde_json::json!({
            "temperature": 3.0
        });
        assert!(mapper.validate_params(&high_temp).is_ok()); // Now allowed

        // Negative temperature (still invalid)
        let invalid_temp = serde_json::json!({
            "temperature": -1.0
        });
        assert!(mapper.validate_params(&invalid_temp).is_err());

        // Invalid frequency penalty
        let invalid_penalty = serde_json::json!({
            "frequency_penalty": 3.0
        });
        assert!(mapper.validate_params(&invalid_penalty).is_err());
    }

    #[test]
    fn test_openai_params_builder() {
        let params = OpenAiParamsBuilder::new()
            .response_format(ResponseFormat::JsonObject)
            .tool_choice(ToolChoice::String("auto".to_string()))
            .parallel_tool_calls(true)
            .user("test-user".to_string())
            .frequency_penalty(0.5)
            .unwrap()
            .presence_penalty(-0.2)
            .unwrap()
            .n(2)
            .stream(false)
            .logprobs(true)
            .top_logprobs(5)
            .build();

        let params = params.unwrap();
        assert!(params.response_format.is_some());
        assert!(params.tool_choice.is_some());
        assert_eq!(params.parallel_tool_calls, Some(true));
        assert_eq!(params.user, Some("test-user".to_string()));
        assert_eq!(params.frequency_penalty, Some(0.5));
        assert_eq!(params.presence_penalty, Some(-0.2));
        assert_eq!(params.n, Some(2));
        assert_eq!(params.stream, Some(false));
        assert_eq!(params.logprobs, Some(true));
        assert_eq!(params.top_logprobs, Some(5));
    }

    #[test]
    fn test_enum_types() {
        // Test TruncationStrategy
        let truncation = TruncationStrategy::Auto;
        let json = serde_json::to_string(&truncation).unwrap();
        assert_eq!(json, "\"auto\"");

        // Test IncludableItem
        let item = IncludableItem::MessageOutputTextLogprobs;
        let json = serde_json::to_string(&item).unwrap();
        assert_eq!(json, "\"message.output_text.logprobs\"");

        // Test SortOrder
        let order = SortOrder::Desc;
        let json = serde_json::to_string(&order).unwrap();
        assert_eq!(json, "\"desc\"");

        // Test ServiceTier
        let tier = ServiceTier::Priority;
        let json = serde_json::to_string(&tier).unwrap();
        assert_eq!(json, "\"priority\"");

        // Test ReasoningEffort
        let effort = ReasoningEffort::Minimal;
        let json = serde_json::to_string(&effort).unwrap();
        assert_eq!(json, "\"minimal\"");

        // Test Verbosity
        let verbosity = Verbosity::High;
        let json = serde_json::to_string(&verbosity).unwrap();
        assert_eq!(json, "\"high\"");
    }

    #[test]
    fn test_enum_defaults() {
        assert_eq!(TruncationStrategy::default(), TruncationStrategy::Disabled);
        assert_eq!(ServiceTier::default(), ServiceTier::Auto);
        assert_eq!(ReasoningEffort::default(), ReasoningEffort::Medium);
        assert_eq!(Verbosity::default(), Verbosity::Medium);
    }
}
