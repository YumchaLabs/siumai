use serde::{Deserialize, Serialize};

use super::{Candidate, Content, GeminiTool, SafetySetting, ToolConfig};

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

/// Configuration options for model generation and outputs.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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

impl GenerationConfig {
    /// Create a new generation configuration
    pub fn new() -> Self {
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
    pub fn with_candidate_count(mut self, count: i32) -> Self {
        self.candidate_count = Some(count);
        self
    }
    /// Set stop sequences
    pub fn with_stop_sequences(mut self, stop: Vec<String>) -> Self {
        self.stop_sequences = Some(stop);
        self
    }
    /// Set max output tokens
    pub fn with_max_output_tokens(mut self, max: i32) -> Self {
        self.max_output_tokens = Some(max);
        self
    }
    /// Set temperature
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }
    /// Set top_p
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }
    /// Set top_k
    pub fn with_top_k(mut self, top_k: i32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Set response schema for structured output
    pub fn with_response_schema(mut self, schema: serde_json::Value) -> Self {
        self.response_schema = Some(schema);
        self
    }

    /// Set response mime type
    pub fn with_response_mime_type(mut self, mime: String) -> Self {
        self.response_mime_type = Some(mime);
        self
    }

    /// Attach thinking configuration
    pub fn with_thinking_config(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking_config = Some(thinking);
        self
    }
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

impl ThinkingConfig {
    /// Create a new empty thinking config
    pub const fn new() -> Self {
        Self {
            thinking_budget: None,
            include_thoughts: None,
        }
    }
    /// Dynamic thinking: model decides when/how much to think
    pub const fn dynamic() -> Self {
        Self {
            thinking_budget: Some(-1),
            include_thoughts: Some(true),
        }
    }
    /// Attempt to disable thinking
    pub const fn disabled() -> Self {
        Self {
            thinking_budget: Some(0),
            include_thoughts: Some(false),
        }
    }
}

impl Default for ThinkingConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// A set of the feedback metadata the prompt specified in GenerateContentRequest.content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptFeedback {
    /// Optional. If set, the prompt was blocked and no candidates are returned.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_reason: Option<BlockReason>,
    /// Ratings for safety of the prompt.
    #[serde(default)]
    pub safety_ratings: Vec<super::SafetyRating>,
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
