//! Google Gemini Types
//!
//! This module contains type definitions for Google Gemini API requests and responses.
//! Based on the Gemini `OpenAPI` specification.

use serde::{Deserialize, Serialize};

/// Gemini Generate Content Request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateContentRequest {
    /// Required. The name of the Model to use for generating the completion.
    pub model: String,
    /// Required. The content of the current conversation with the model.
    pub contents: Vec<Content>,
    /// Optional. Developer set system instructions.
    #[serde(skip_serializing_if = "Option::is_none", rename = "systemInstruction")]
    pub system_instruction: Option<Content>,
    /// Optional. A list of Tools the Model may use to generate the next response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<GeminiTool>>,
    /// Optional. Tool configuration for any Tool specified in the request.
    #[serde(skip_serializing_if = "Option::is_none", rename = "toolConfig")]
    pub tool_config: Option<ToolConfig>,
    /// Optional. A list of unique `SafetySetting` instances for blocking unsafe content.
    #[serde(skip_serializing_if = "Option::is_none", rename = "safetySettings")]
    pub safety_settings: Option<Vec<SafetySetting>>,
    /// Optional. Configuration options for model generation and outputs.
    #[serde(skip_serializing_if = "Option::is_none", rename = "generationConfig")]
    pub generation_config: Option<GenerationConfig>,
    /// Optional. The name of the content cached to use as context.
    #[serde(skip_serializing_if = "Option::is_none", rename = "cachedContent")]
    pub cached_content: Option<String>,
}

/// Gemini Generate Content Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateContentResponse {
    /// Candidate responses from the model.
    #[serde(default)]
    pub candidates: Vec<Candidate>,
    /// Returns the prompt's feedback related to the content filters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_feedback: Option<PromptFeedback>,
    /// Output only. Metadata on the generation requests' token usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_metadata: Option<UsageMetadata>,
    /// Output only. The model version used to generate the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
    /// Output only. `response_id` is used to identify each response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
}

/// The base structured datatype containing multi-part content of a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Content {
    /// Optional. The producer of the content. Must be either 'user' or 'model'.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Ordered Parts that constitute a single message.
    pub parts: Vec<Part>,
}

/// A datatype containing media that is part of a multi-part Content message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Part {
    /// Text content
    Text {
        text: String,
        /// Optional. Whether this is a thought summary (for thinking models)
        #[serde(skip_serializing_if = "Option::is_none")]
        thought: Option<bool>,
    },
    /// Inline data (images, audio, etc.)
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: Blob,
    },
    /// File data
    FileData {
        #[serde(rename = "fileData")]
        file_data: FileData,
    },
    /// Function call
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: FunctionCall,
    },
    /// Function response
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: FunctionResponse,
    },
    /// Executable code
    ExecutableCode {
        #[serde(rename = "executableCode")]
        executable_code: ExecutableCode,
    },
    /// Code execution result
    CodeExecutionResult {
        #[serde(rename = "codeExecutionResult")]
        code_execution_result: CodeExecutionResult,
    },
}

/// Raw media bytes with MIME type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Blob {
    /// The IANA standard MIME type of the source data.
    pub mime_type: String,
    /// Raw bytes for media formats.
    pub data: String, // Base64 encoded
}

/// URI based data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileData {
    /// Required. URI.
    pub file_uri: String,
    /// Optional. The IANA standard MIME type of the source data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// A predicted `FunctionCall` returned from the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Required. The name of the function to call.
    pub name: String,
    /// Optional. The function parameters and values in JSON object format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<serde_json::Value>,
}

/// The result output from a `FunctionCall`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionResponse {
    /// Required. The name of the function to call.
    pub name: String,
    /// Required. The function response in JSON object format.
    pub response: serde_json::Value,
}

/// Code generated by the model that is meant to be executed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutableCode {
    /// Required. Programming language of the code.
    pub language: CodeLanguage,
    /// Required. The code to be executed.
    pub code: String,
}

/// Programming language enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CodeLanguage {
    #[serde(rename = "LANGUAGE_UNSPECIFIED")]
    Unspecified,
    #[serde(rename = "PYTHON")]
    Python,
}

/// Result of executing the `ExecutableCode`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExecutionResult {
    /// Required. Outcome of the code execution.
    pub outcome: CodeExecutionOutcome,
    /// Optional. Contains stdout when code execution is successful, stderr or other description otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
}

/// Code execution outcome enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CodeExecutionOutcome {
    #[serde(rename = "OUTCOME_UNSPECIFIED")]
    Unspecified,
    #[serde(rename = "OUTCOME_OK")]
    Ok,
    #[serde(rename = "OUTCOME_FAILED")]
    Failed,
    #[serde(rename = "OUTCOME_DEADLINE_EXCEEDED")]
    DeadlineExceeded,
}

/// A candidate response generated by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    /// Output only. Generated content returned from the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Content>,
    /// Optional. Output only. The reason why the model stopped generating tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    /// List of ratings for the safety of a response candidate.
    #[serde(default)]
    pub safety_ratings: Vec<SafetyRating>,
    /// Output only. Citation information for model-generated candidate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citation_metadata: Option<CitationMetadata>,
    /// Output only. Token count for this candidate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_count: Option<i32>,
    /// Output only. Index of the candidate in the list of candidates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<i32>,
}

/// Defines the reason why the model stopped generating tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FinishReason {
    #[serde(rename = "FINISH_REASON_UNSPECIFIED")]
    Unspecified,
    #[serde(rename = "STOP")]
    Stop,
    #[serde(rename = "MAX_TOKENS")]
    MaxTokens,
    #[serde(rename = "SAFETY")]
    Safety,
    #[serde(rename = "RECITATION")]
    Recitation,
    #[serde(rename = "LANGUAGE")]
    Language,
    #[serde(rename = "OTHER")]
    Other,
    #[serde(rename = "BLOCKLIST")]
    Blocklist,
    #[serde(rename = "PROHIBITED_CONTENT")]
    ProhibitedContent,
    #[serde(rename = "SPII")]
    Spii,
    #[serde(rename = "MALFORMED_FUNCTION_CALL")]
    MalformedFunctionCall,
}

/// A collection of source attributions for a piece of content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationMetadata {
    /// Citations to sources for a specific response.
    #[serde(default)]
    pub citation_sources: Vec<CitationSource>,
}

/// A citation to a source for a portion of a specific response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationSource {
    /// Optional. Start of segment of the response that is attributed to this source.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_index: Option<i32>,
    /// Optional. End of the attributed segment, exclusive.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_index: Option<i32>,
    /// Optional. URI that is attributed as a source for a portion of the text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    /// Optional. License for the GitHub project that is attributed as a source for segment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
}

/// Safety rating for a piece of content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyRating {
    /// Required. The category for this rating.
    pub category: HarmCategory,
    /// Required. The probability of harm for this content.
    pub probability: HarmProbability,
    /// Was this content blocked because of this rating?
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocked: Option<bool>,
}

/// The category of a rating.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HarmCategory {
    #[serde(rename = "HARM_CATEGORY_UNSPECIFIED")]
    Unspecified,
    #[serde(rename = "HARM_CATEGORY_DEROGATORY")]
    Derogatory,
    #[serde(rename = "HARM_CATEGORY_TOXICITY")]
    Toxicity,
    #[serde(rename = "HARM_CATEGORY_VIOLENCE")]
    Violence,
    #[serde(rename = "HARM_CATEGORY_SEXUAL")]
    Sexual,
    #[serde(rename = "HARM_CATEGORY_MEDICAL")]
    Medical,
    #[serde(rename = "HARM_CATEGORY_DANGEROUS")]
    Dangerous,
    #[serde(rename = "HARM_CATEGORY_HARASSMENT")]
    Harassment,
    #[serde(rename = "HARM_CATEGORY_HATE_SPEECH")]
    HateSpeech,
    #[serde(rename = "HARM_CATEGORY_SEXUALLY_EXPLICIT")]
    SexuallyExplicit,
    #[serde(rename = "HARM_CATEGORY_DANGEROUS_CONTENT")]
    DangerousContent,
    #[serde(rename = "HARM_CATEGORY_CIVIC_INTEGRITY")]
    CivicIntegrity,
}

/// The probability that a piece of content is harmful.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HarmProbability {
    #[serde(rename = "HARM_PROBABILITY_UNSPECIFIED")]
    Unspecified,
    #[serde(rename = "NEGLIGIBLE")]
    Negligible,
    #[serde(rename = "LOW")]
    Low,
    #[serde(rename = "MEDIUM")]
    Medium,
    #[serde(rename = "HIGH")]
    High,
}

/// Safety setting, affecting the safety-blocking behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetySetting {
    /// Required. The category for this setting.
    pub category: HarmCategory,
    /// Required. Controls the probability threshold at which harm is blocked.
    pub threshold: HarmBlockThreshold,
}

/// Block at and beyond a specified harm probability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HarmBlockThreshold {
    #[serde(rename = "HARM_BLOCK_THRESHOLD_UNSPECIFIED")]
    Unspecified,
    #[serde(rename = "BLOCK_LOW_AND_ABOVE")]
    BlockLowAndAbove,
    #[serde(rename = "BLOCK_MEDIUM_AND_ABOVE")]
    BlockMediumAndAbove,
    #[serde(rename = "BLOCK_ONLY_HIGH")]
    BlockOnlyHigh,
    #[serde(rename = "BLOCK_NONE")]
    BlockNone,
}

/// Configuration options for model generation and outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Optional. Number of generated responses to return.
    #[serde(skip_serializing_if = "Option::is_none", rename = "candidateCount")]
    pub candidate_count: Option<i32>,
    /// Optional. The set of character sequences that will stop output generation.
    #[serde(skip_serializing_if = "Option::is_none", rename = "stopSequences")]
    pub stop_sequences: Option<Vec<String>>,
    /// Optional. The maximum number of tokens to include in a candidate.
    #[serde(skip_serializing_if = "Option::is_none", rename = "maxOutputTokens")]
    pub max_output_tokens: Option<i32>,
    /// Optional. Controls the randomness of the output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Optional. The maximum cumulative probability of tokens to consider when sampling.
    #[serde(skip_serializing_if = "Option::is_none", rename = "topP")]
    pub top_p: Option<f32>,
    /// Optional. The maximum number of tokens to consider when sampling.
    #[serde(skip_serializing_if = "Option::is_none", rename = "topK")]
    pub top_k: Option<i32>,
    /// Optional. Output response mimetype of the generated candidate text.
    #[serde(skip_serializing_if = "Option::is_none", rename = "responseMimeType")]
    pub response_mime_type: Option<String>,
    /// Optional. Output response schema of the generated candidate text.
    #[serde(skip_serializing_if = "Option::is_none", rename = "responseSchema")]
    pub response_schema: Option<serde_json::Value>,
    /// Optional. Configuration for thinking behavior.
    #[serde(skip_serializing_if = "Option::is_none", rename = "thinkingConfig")]
    pub thinking_config: Option<ThinkingConfig>,
    /// Optional. Output response modalities (e.g., ["TEXT", "IMAGE"]).
    #[serde(skip_serializing_if = "Option::is_none", rename = "responseModalities")]
    pub response_modalities: Option<Vec<String>>,
}

/// Configuration for thinking behavior in Gemini models.
///
/// Note: Different models have different thinking capabilities. The API will
/// return appropriate errors if unsupported configurations are used.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingConfig {
    /// Thinking budget in tokens.
    /// - Set to -1 for dynamic thinking (model decides when and how much to think)
    /// - Set to 0 to attempt to disable thinking (may not work on all models)
    /// - Set to specific value to limit thinking tokens
    ///
    /// The actual supported range depends on the specific model being used.
    #[serde(skip_serializing_if = "Option::is_none", rename = "thinkingBudget")]
    pub thinking_budget: Option<i32>,

    /// Whether to include thought summaries in the response.
    /// This controls the visibility of thinking summaries, not the thinking process itself.
    #[serde(skip_serializing_if = "Option::is_none", rename = "includeThoughts")]
    pub include_thoughts: Option<bool>,
}

/// A set of the feedback metadata the prompt specified in GenerateContentRequest.content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptFeedback {
    /// Optional. If set, the prompt was blocked and no candidates are returned.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_reason: Option<BlockReason>,
    /// Ratings for safety of the prompt.
    #[serde(default)]
    pub safety_ratings: Vec<SafetyRating>,
}

/// Specifies what was the reason why prompt was blocked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BlockReason {
    #[serde(rename = "BLOCK_REASON_UNSPECIFIED")]
    Unspecified,
    #[serde(rename = "SAFETY")]
    Safety,
    #[serde(rename = "OTHER")]
    Other,
    #[serde(rename = "BLOCKLIST")]
    Blocklist,
    #[serde(rename = "PROHIBITED_CONTENT")]
    ProhibitedContent,
    #[serde(rename = "IMAGE_SAFETY")]
    ImageSafety,
}

/// Metadata on the generation requests' token usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageMetadata {
    /// Number of tokens in the prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_token_count: Option<i32>,
    /// Total token count for the generation request (prompt + response candidates).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_token_count: Option<i32>,
    /// Number of tokens in the cached part of the prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_content_token_count: Option<i32>,
    /// Number of tokens in the response candidate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidates_token_count: Option<i32>,
    /// Number of tokens used for thinking (only for thinking models).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thoughts_token_count: Option<i32>,
}

/// Tool details that the model may use to generate response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GeminiTool {
    /// Function calling tool
    FunctionDeclarations {
        function_declarations: Vec<FunctionDeclaration>,
    },
    /// Code execution tool
    CodeExecution { code_execution: CodeExecution },
}

/// Structured representation of a function declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDeclaration {
    /// Required. The name of the function.
    pub name: String,
    /// Required. A brief description of the function.
    pub description: String,
    /// Optional. Describes the parameters to this function.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
    /// Optional. Describes the output from this function in JSON Schema format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<serde_json::Value>,
}

/// Tool that executes code generated by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExecution {
    // This is an empty object in the API spec
}

/// Tool configuration for any Tool specified in the request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolConfig {
    /// Optional. Function calling config.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_calling_config: Option<FunctionCallingConfig>,
}

/// Configuration for specifying function calling behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallingConfig {
    /// Optional. Specifies the mode in which function calling should execute.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<FunctionCallingMode>,
    /// Optional. A set of function names that, when provided, limits the functions the model will call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_function_names: Option<Vec<String>>,
}

/// Defines the execution behavior for function calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FunctionCallingMode {
    #[serde(rename = "MODE_UNSPECIFIED")]
    Unspecified,
    #[serde(rename = "AUTO")]
    Auto,
    #[serde(rename = "ANY")]
    Any,
    #[serde(rename = "NONE")]
    None,
}

/// Gemini-specific configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiConfig {
    /// API key for authentication
    pub api_key: String,
    /// Base URL for the Gemini API
    pub base_url: String,
    /// Default model to use
    pub model: String,
    /// Default generation configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
    /// Default safety settings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_settings: Option<Vec<SafetySetting>>,
    /// HTTP timeout in seconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u64>,
}

impl Default for GeminiConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            model: "gemini-1.5-flash".to_string(),
            generation_config: None,
            safety_settings: None,
            timeout: Some(30),
        }
    }
}

impl GeminiConfig {
    /// Create a new Gemini configuration with the given API key
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            ..Default::default()
        }
    }

    /// Set the model to use
    pub fn with_model(mut self, model: String) -> Self {
        self.model = model;
        self
    }

    /// Set the base URL
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    /// Set generation configuration
    pub fn with_generation_config(mut self, config: GenerationConfig) -> Self {
        self.generation_config = Some(config);
        self
    }

    /// Set safety settings
    pub fn with_safety_settings(mut self, settings: Vec<SafetySetting>) -> Self {
        self.safety_settings = Some(settings);
        self
    }

    /// Set HTTP timeout
    pub const fn with_timeout(mut self, timeout: u64) -> Self {
        self.timeout = Some(timeout);
        self
    }
}

impl GenerationConfig {
    /// Create a new generation configuration
    pub const fn new() -> Self {
        Self {
            candidate_count: None,
            stop_sequences: None,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            response_mime_type: None,
            response_schema: None,
            thinking_config: None,
            response_modalities: None,
        }
    }

    /// Set the number of candidates to generate
    pub const fn with_candidate_count(mut self, count: i32) -> Self {
        self.candidate_count = Some(count);
        self
    }

    /// Set stop sequences
    pub fn with_stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(sequences);
        self
    }

    /// Set maximum output tokens
    pub const fn with_max_output_tokens(mut self, tokens: i32) -> Self {
        self.max_output_tokens = Some(tokens);
        self
    }

    /// Set temperature
    pub const fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set top-p
    pub const fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set top-k
    pub const fn with_top_k(mut self, top_k: i32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Set response MIME type
    pub fn with_response_mime_type(mut self, mime_type: String) -> Self {
        self.response_mime_type = Some(mime_type);
        self
    }

    /// Set response schema
    pub fn with_response_schema(mut self, schema: serde_json::Value) -> Self {
        self.response_schema = Some(schema);
        self
    }

    /// Set thinking configuration
    pub const fn with_thinking_config(mut self, config: ThinkingConfig) -> Self {
        self.thinking_config = Some(config);
        self
    }

    /// Set response modalities
    pub fn with_response_modalities(mut self, modalities: Vec<String>) -> Self {
        self.response_modalities = Some(modalities);
        self
    }
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl ThinkingConfig {
    /// Create a new thinking configuration
    pub const fn new() -> Self {
        Self {
            thinking_budget: None,
            include_thoughts: None,
        }
    }

    /// Create thinking configuration with specific budget
    pub const fn with_budget(budget: i32) -> Self {
        Self {
            thinking_budget: Some(budget),
            include_thoughts: None,
        }
    }

    /// Create thinking configuration with thought summaries enabled
    pub const fn with_thoughts() -> Self {
        Self {
            thinking_budget: None,
            include_thoughts: Some(true),
        }
    }

    /// Create dynamic thinking configuration (model decides budget)
    pub const fn dynamic() -> Self {
        Self {
            thinking_budget: Some(-1),
            include_thoughts: Some(true),
        }
    }

    /// Create configuration that attempts to disable thinking
    /// Note: Not all models support disabling thinking
    pub const fn disabled() -> Self {
        Self {
            thinking_budget: Some(0),
            include_thoughts: Some(false),
        }
    }

    /// Basic validation (only check for obviously invalid values)
    pub fn validate(&self) -> Result<(), String> {
        if let Some(budget) = self.thinking_budget
            && budget < -1
        {
            return Err("Thinking budget cannot be less than -1".to_string());
        }
        Ok(())
    }
}

impl Default for ThinkingConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thinking_config_json_serialization() {
        let thinking_config = ThinkingConfig {
            thinking_budget: Some(-1),
            include_thoughts: Some(true),
        };

        let json = serde_json::to_string(&thinking_config).unwrap();
        println!("ThinkingConfig JSON: {}", json);

        // Verify the JSON contains the correct field names
        assert!(json.contains("thinkingBudget"));
        assert!(json.contains("includeThoughts"));
        assert!(!json.contains("thinking_budget"));
        assert!(!json.contains("include_thoughts"));
    }

    #[test]
    fn test_generation_config_json_serialization() {
        let thinking_config = ThinkingConfig {
            thinking_budget: Some(-1),
            include_thoughts: Some(true),
        };

        let generation_config = GenerationConfig {
            candidate_count: Some(1),
            stop_sequences: None,
            max_output_tokens: Some(1000),
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(40),
            response_mime_type: None,
            response_schema: None,
            thinking_config: Some(thinking_config),
            response_modalities: None,
        };

        let json = serde_json::to_string(&generation_config).unwrap();
        println!("GenerationConfig JSON: {}", json);

        // Verify the JSON contains the correct field names
        assert!(json.contains("candidateCount"));
        assert!(json.contains("maxOutputTokens"));
        assert!(json.contains("topP"));
        assert!(json.contains("topK"));
        assert!(json.contains("thinkingConfig"));
        assert!(json.contains("thinkingBudget"));
        assert!(json.contains("includeThoughts"));
    }
}

impl SafetySetting {
    /// Create a new safety setting
    pub const fn new(category: HarmCategory, threshold: HarmBlockThreshold) -> Self {
        Self {
            category,
            threshold,
        }
    }

    /// Create a safety setting that blocks low and above
    pub const fn block_low_and_above(category: HarmCategory) -> Self {
        Self::new(category, HarmBlockThreshold::BlockLowAndAbove)
    }

    /// Create a safety setting that blocks medium and above
    pub const fn block_medium_and_above(category: HarmCategory) -> Self {
        Self::new(category, HarmBlockThreshold::BlockMediumAndAbove)
    }

    /// Create a safety setting that blocks only high
    pub const fn block_only_high(category: HarmCategory) -> Self {
        Self::new(category, HarmBlockThreshold::BlockOnlyHigh)
    }

    /// Create a safety setting that blocks none
    pub const fn block_none(category: HarmCategory) -> Self {
        Self::new(category, HarmBlockThreshold::BlockNone)
    }
}

impl Content {
    /// Create new content with the given role and parts
    pub const fn new(role: Option<String>, parts: Vec<Part>) -> Self {
        Self { role, parts }
    }

    /// Create user content with text
    pub fn user_text(text: String) -> Self {
        Self {
            role: Some("user".to_string()),
            parts: vec![Part::Text {
                text,
                thought: None,
            }],
        }
    }

    /// Create model content with text
    pub fn model_text(text: String) -> Self {
        Self {
            role: Some("model".to_string()),
            parts: vec![Part::Text {
                text,
                thought: None,
            }],
        }
    }

    /// Create system content with text
    pub fn system_text(text: String) -> Self {
        Self {
            role: None, // System instructions don't have a role
            parts: vec![Part::Text {
                text,
                thought: None,
            }],
        }
    }
}

impl Part {
    /// Create a text part
    pub const fn text(text: String) -> Self {
        Self::Text {
            text,
            thought: None,
        }
    }

    /// Create a thought summary part
    pub const fn thought_summary(text: String) -> Self {
        Self::Text {
            text,
            thought: Some(true),
        }
    }

    /// Create an inline data part
    pub const fn inline_data(mime_type: String, data: String) -> Self {
        Self::InlineData {
            inline_data: Blob { mime_type, data },
        }
    }

    /// Create a file data part
    pub const fn file_data(file_uri: String, mime_type: Option<String>) -> Self {
        Self::FileData {
            file_data: FileData {
                file_uri,
                mime_type,
            },
        }
    }

    /// Create a function call part
    pub const fn function_call(name: String, args: Option<serde_json::Value>) -> Self {
        Self::FunctionCall {
            function_call: FunctionCall { name, args },
        }
    }

    /// Create a function response part
    pub const fn function_response(name: String, response: serde_json::Value) -> Self {
        Self::FunctionResponse {
            function_response: FunctionResponse { name, response },
        }
    }
}

// File management types

/// Gemini File object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiFile {
    /// Immutable. Identifier. The File resource name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Optional. The human-readable display name for the File.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    /// Output only. MIME type of the file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Output only. Size of the file in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<String>,
    /// Output only. The timestamp of when the File was created.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub create_time: Option<String>,
    /// Output only. The timestamp of when the File was last updated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub update_time: Option<String>,
    /// Output only. The timestamp of when the File will be deleted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expiration_time: Option<String>,
    /// Output only. SHA-256 hash of the uploaded bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256_hash: Option<String>,
    /// Output only. The uri of the File.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    /// Output only. Processing state of the File.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state: Option<GeminiFileState>,
    /// Output only. Error status if File processing failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<GeminiStatus>,
    /// Output only. Metadata for a video.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video_metadata: Option<VideoFileMetadata>,
}

/// Processing state of the File
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeminiFileState {
    #[serde(rename = "STATE_UNSPECIFIED")]
    Unspecified,
    #[serde(rename = "PROCESSING")]
    Processing,
    #[serde(rename = "ACTIVE")]
    Active,
    #[serde(rename = "FAILED")]
    Failed,
}

/// Error status for file processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiStatus {
    /// The status code
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<i32>,
    /// A developer-facing error message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// A list of messages that carry the error details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Vec<serde_json::Value>>,
}

/// Metadata for a video file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoFileMetadata {
    /// Duration of the video
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video_duration: Option<String>,
}

/// Request for `CreateFile`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateFileRequest {
    /// Optional. Metadata for the file to create.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<GeminiFile>,
}

/// Response for `CreateFile`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateFileResponse {
    /// Metadata for the created file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<GeminiFile>,
}

/// Response for `ListFiles`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListFilesResponse {
    /// The list of Files.
    #[serde(default)]
    pub files: Vec<GeminiFile>,
    /// A token that can be sent as `page_token` into a subsequent `ListFiles` call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_page_token: Option<String>,
}

/// Response for `DownloadFile`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadFileResponse {
    // This is typically just raw bytes, but we'll handle it in the implementation
}

// ================================================================================================
// Embedding Types
// ================================================================================================

/// Gemini-specific embedding configuration options
///
/// This struct provides type-safe configuration for Gemini embedding requests,
/// including task type optimization, context titles, and custom dimensions.
///
/// # Example
/// ```rust,no_run
/// use siumai::providers::gemini::GeminiEmbeddingOptions;
/// use siumai::types::EmbeddingTaskType;
///
/// let options = GeminiEmbeddingOptions::new()
///     .with_task_type(EmbeddingTaskType::RetrievalQuery)
///     .with_title("Search Context")
///     .with_output_dimensionality(768);
/// ```
#[derive(Debug, Clone, Default)]
pub struct GeminiEmbeddingOptions {
    /// Task type for optimization (Gemini-specific feature)
    pub task_type: Option<crate::types::EmbeddingTaskType>,
    /// Title for additional context (helps with embedding quality)
    pub title: Option<String>,
    /// Custom output dimensions (128-3072, must be supported by model)
    pub output_dimensionality: Option<u32>,
}

impl GeminiEmbeddingOptions {
    /// Create new Gemini embedding options with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set task type for optimization
    ///
    /// Different task types optimize the embedding for specific use cases:
    /// - `RetrievalQuery`: For search queries
    /// - `RetrievalDocument`: For documents to be searched
    /// - `SemanticSimilarity`: For similarity comparison
    /// - `Classification`: For text classification
    /// - `Clustering`: For grouping similar texts
    /// - `QuestionAnswering`: For Q&A systems
    /// - `FactVerification`: For fact checking
    pub fn with_task_type(mut self, task_type: crate::types::EmbeddingTaskType) -> Self {
        self.task_type = Some(task_type);
        self
    }

    /// Set title for additional context
    ///
    /// The title provides additional context that can improve embedding quality.
    /// This is particularly useful for documents or when the text needs context.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set custom output dimensions
    ///
    /// Allows customizing the embedding vector size. Must be between 128-3072
    /// and supported by the specific model being used.
    pub fn with_output_dimensionality(mut self, dimensions: u32) -> Self {
        self.output_dimensionality = Some(dimensions);
        self
    }

    /// Apply these options to an EmbeddingRequest
    ///
    /// This method modifies the provided EmbeddingRequest to include
    /// Gemini-specific parameters.
    pub fn apply_to_request(
        self,
        mut request: crate::types::EmbeddingRequest,
    ) -> crate::types::EmbeddingRequest {
        if let Some(task_type) = self.task_type {
            request = request.with_task_type(task_type);
        }
        if let Some(title) = self.title {
            request = request.with_provider_param("title", serde_json::Value::String(title));
        }
        if let Some(dims) = self.output_dimensionality {
            request.dimensions = Some(dims);
        }
        request
    }
}

/// Extension trait for EmbeddingRequest to add Gemini-specific configuration
pub trait GeminiEmbeddingRequestExt {
    /// Configure this request with Gemini-specific options
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::types::EmbeddingRequest;
    /// use siumai::providers::gemini::{GeminiEmbeddingOptions, GeminiEmbeddingRequestExt};
    /// use siumai::types::EmbeddingTaskType;
    ///
    /// let request = EmbeddingRequest::new(vec!["text".to_string()])
    ///     .with_gemini_config(
    ///         GeminiEmbeddingOptions::new()
    ///             .with_task_type(EmbeddingTaskType::RetrievalQuery)
    ///             .with_title("Search Context")
    ///     );
    /// ```
    fn with_gemini_config(self, config: GeminiEmbeddingOptions) -> Self;

    /// Quick method to set Gemini task type
    fn with_gemini_task_type(self, task_type: crate::types::EmbeddingTaskType) -> Self;

    /// Quick method to set Gemini title
    fn with_gemini_title(self, title: impl Into<String>) -> Self;

    /// Quick method to set Gemini output dimensions
    fn with_gemini_dimensions(self, dimensions: u32) -> Self;
}

impl GeminiEmbeddingRequestExt for crate::types::EmbeddingRequest {
    fn with_gemini_config(self, config: GeminiEmbeddingOptions) -> Self {
        config.apply_to_request(self)
    }

    fn with_gemini_task_type(self, task_type: crate::types::EmbeddingTaskType) -> Self {
        self.with_task_type(task_type)
    }

    fn with_gemini_title(self, title: impl Into<String>) -> Self {
        self.with_provider_param("title", serde_json::Value::String(title.into()))
    }

    fn with_gemini_dimensions(self, dimensions: u32) -> Self {
        let mut request = self;
        request.dimensions = Some(dimensions);
        request
    }
}
