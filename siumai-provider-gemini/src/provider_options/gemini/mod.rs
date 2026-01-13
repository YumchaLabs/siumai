//! Google Gemini provider options.
//!
//! These types are carried via the open `providerOptions` JSON map (`provider_id = "gemini"`),
//! and should be carried via `providerOptions["gemini"]`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Gemini response modalities for `generationConfig.responseModalities`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeminiResponseModality {
    #[serde(rename = "TEXT")]
    Text,
    #[serde(rename = "IMAGE")]
    Image,
    #[serde(rename = "AUDIO")]
    Audio,
}

/// Gemini thinking configuration for `generationConfig.thinkingConfig`.
///
/// Vercel AI SDK alignment: `providerOptions.google.thinkingConfig`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GeminiThinkingConfig {
    /// Thinking budget in tokens.
    /// - Use `-1` for dynamic thinking (model decides).
    /// - Use `0` to attempt to disable thinking (may not work for all models).
    #[serde(skip_serializing_if = "Option::is_none", rename = "thinkingBudget")]
    pub thinking_budget: Option<i32>,
    /// Whether to include thought summaries in the response.
    #[serde(skip_serializing_if = "Option::is_none", rename = "includeThoughts")]
    pub include_thoughts: Option<bool>,
    /// Thinking level hint (Gemini 3+).
    #[serde(skip_serializing_if = "Option::is_none", rename = "thinkingLevel")]
    pub thinking_level: Option<GeminiThinkingLevel>,
}

impl GeminiThinkingConfig {
    /// Create a new empty thinking config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set thinking budget.
    pub const fn with_thinking_budget(mut self, budget: i32) -> Self {
        self.thinking_budget = Some(budget);
        self
    }

    /// Set whether to include thought summaries.
    pub const fn with_include_thoughts(mut self, include: bool) -> Self {
        self.include_thoughts = Some(include);
        self
    }

    /// Set thinking level hint.
    pub const fn with_thinking_level(mut self, level: GeminiThinkingLevel) -> Self {
        self.thinking_level = Some(level);
        self
    }
}

/// Thinking level hint (Gemini 3+).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeminiThinkingLevel {
    #[serde(rename = "minimal")]
    Minimal,
    #[serde(rename = "low")]
    Low,
    #[serde(rename = "medium")]
    Medium,
    #[serde(rename = "high")]
    High,
}

/// Harm category for Gemini safety settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeminiHarmCategory {
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

/// Harm block threshold for Gemini safety settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeminiHarmBlockThreshold {
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
    #[serde(rename = "OFF")]
    Off,
}

/// Safety setting for Gemini requests (`safetySettings`).
///
/// Vercel AI SDK alignment: `providerOptions.google.safetySettings`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiSafetySetting {
    pub category: GeminiHarmCategory,
    pub threshold: GeminiHarmBlockThreshold,
}

/// Google Gemini specific options
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GeminiOptions {
    /// Code execution configuration
    #[deprecated(
        note = "Use provider-defined tools instead: siumai::hosted_tools::google::code_execution()"
    )]
    pub code_execution: Option<CodeExecutionConfig>,
    /// Search grounding (web search)
    #[deprecated(
        note = "Use provider-defined tools instead: siumai::hosted_tools::google::google_search()"
    )]
    pub search_grounding: Option<SearchGroundingConfig>,
    /// File Search configuration (Gemini File Search tool)
    #[deprecated(
        note = "Use provider-defined tools instead: siumai::hosted_tools::google::file_search()"
    )]
    pub file_search: Option<FileSearchConfig>,
    /// Preferred MIME type for responses (e.g., "application/json")
    pub response_mime_type: Option<String>,
    /// Optional JSON Schema output schema (Gemini `responseJsonSchema`).
    pub response_json_schema: Option<serde_json::Value>,
    /// Optional cached content reference used as extra context.
    ///
    /// Format: `cachedContents/{cachedContent}`.
    ///
    /// Vercel AI SDK alignment: `providerOptions.google.cachedContent`.
    pub cached_content: Option<String>,

    /// Optional response modalities for `generationConfig.responseModalities`.
    ///
    /// Examples: `[TEXT]`, `[TEXT, IMAGE]`.
    pub response_modalities: Option<Vec<GeminiResponseModality>>,

    /// Optional thinking config for `generationConfig.thinkingConfig`.
    pub thinking_config: Option<GeminiThinkingConfig>,

    /// Optional safety settings (top-level `safetySettings`).
    pub safety_settings: Option<Vec<GeminiSafetySetting>>,

    /// Optional labels (top-level `labels`), mainly used with Vertex.
    pub labels: Option<HashMap<String, String>>,

    /// Optional audio timestamp understanding for audio-only files.
    ///
    /// Vercel AI SDK alignment: `providerOptions.google.audioTimestamp`.
    pub audio_timestamp: Option<bool>,

    /// Optional. If specified, the media resolution specified will be used.
    ///
    /// Vercel AI SDK alignment: `providerOptions.google.mediaResolution`.
    pub media_resolution: Option<String>,

    /// Optional. Configures the image generation aspect ratio for Gemini image models.
    ///
    /// Vercel AI SDK alignment: `providerOptions.google.imageConfig`.
    pub image_config: Option<GeminiImageConfig>,

    /// Optional. Configuration for grounding retrieval (location context for Maps/Search grounding).
    ///
    /// Vercel AI SDK alignment: `providerOptions.google.retrievalConfig`.
    pub retrieval_config: Option<GeminiRetrievalConfig>,

    /// Optional. Enable structured outputs (responseSchema) for `responseFormat`.
    ///
    /// Vercel AI SDK alignment: `providerOptions.google.structuredOutputs`.
    pub structured_outputs: Option<bool>,

    /// Optional. If true, export logprobs results in response.
    pub response_logprobs: Option<bool>,

    /// Optional. Number of top logprobs to return at each decoding step.
    pub logprobs: Option<i32>,
}

/// Image config for Gemini image generation models.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GeminiImageConfig {
    pub aspect_ratio: Option<String>,
    pub image_size: Option<String>,
}

/// Grounding retrieval config (location context).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GeminiRetrievalConfig {
    pub lat_lng: Option<GeminiLatLng>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiLatLng {
    pub latitude: f64,
    pub longitude: f64,
}

impl GeminiOptions {
    /// Create new Gemini options
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable code execution
    #[deprecated(
        note = "Use provider-defined tools instead: siumai::hosted_tools::google::code_execution()"
    )]
    #[allow(deprecated)]
    pub fn with_code_execution(mut self, config: CodeExecutionConfig) -> Self {
        self.code_execution = Some(config);
        self
    }

    /// Enable search grounding
    #[deprecated(
        note = "Use provider-defined tools instead: siumai::hosted_tools::google::google_search()"
    )]
    #[allow(deprecated)]
    pub fn with_search_grounding(mut self, config: SearchGroundingConfig) -> Self {
        self.search_grounding = Some(config);
        self
    }

    /// Enable File Search with given store names
    #[deprecated(
        note = "Use provider-defined tools instead: siumai::hosted_tools::google::file_search().with_file_search_store_names(...).build()"
    )]
    #[allow(deprecated)]
    pub fn with_file_search_store_names<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let stores: Vec<String> = names.into_iter().map(Into::into).collect();
        self.file_search = Some(FileSearchConfig {
            file_search_store_names: stores,
        });
        self
    }

    /// Set preferred response MIME type (e.g., application/json)
    pub fn with_response_mime_type(mut self, mime: impl Into<String>) -> Self {
        self.response_mime_type = Some(mime.into());
        self
    }

    /// Set JSON Schema for structured output (Gemini `responseJsonSchema`).
    pub fn with_response_json_schema(mut self, schema: serde_json::Value) -> Self {
        self.response_json_schema = Some(schema);
        self
    }

    /// Attach a cached content reference as extra context.
    ///
    /// Format: `cachedContents/{cachedContent}`.
    pub fn with_cached_content(mut self, cached_content: impl Into<String>) -> Self {
        self.cached_content = Some(cached_content.into());
        self
    }

    /// Set response modalities for this request (Gemini-specific).
    pub fn with_response_modalities(mut self, modalities: Vec<GeminiResponseModality>) -> Self {
        self.response_modalities = Some(modalities);
        self
    }

    /// Set thinking config for this request (Gemini-specific).
    pub fn with_thinking_config(mut self, thinking: GeminiThinkingConfig) -> Self {
        self.thinking_config = Some(thinking);
        self
    }

    /// Set safety settings for this request (Gemini-specific).
    pub fn with_safety_settings(mut self, settings: Vec<GeminiSafetySetting>) -> Self {
        self.safety_settings = Some(settings);
        self
    }

    /// Set labels for this request (Gemini-specific).
    pub fn with_labels(mut self, labels: HashMap<String, String>) -> Self {
        self.labels = Some(labels);
        self
    }

    /// Enable or disable audio timestamp understanding for audio-only files.
    pub const fn with_audio_timestamp(mut self, enabled: bool) -> Self {
        self.audio_timestamp = Some(enabled);
        self
    }

    /// Set media resolution for this request.
    pub fn with_media_resolution(mut self, resolution: impl Into<String>) -> Self {
        self.media_resolution = Some(resolution.into());
        self
    }

    /// Set image config for this request.
    pub fn with_image_config(mut self, config: GeminiImageConfig) -> Self {
        self.image_config = Some(config);
        self
    }

    /// Set retrieval config for this request.
    pub fn with_retrieval_config(mut self, config: GeminiRetrievalConfig) -> Self {
        self.retrieval_config = Some(config);
        self
    }

    /// Enable or disable structured outputs for responseFormat schema mapping.
    pub const fn with_structured_outputs(mut self, enabled: bool) -> Self {
        self.structured_outputs = Some(enabled);
        self
    }

    /// Enable or disable logprobs in response.
    pub const fn with_response_logprobs(mut self, enabled: bool) -> Self {
        self.response_logprobs = Some(enabled);
        self
    }

    /// Set number of top logprobs to return.
    pub const fn with_logprobs(mut self, k: i32) -> Self {
        self.logprobs = Some(k);
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

/// File Search configuration (Gemini File Search tool)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSearchConfig {
    /// Names of File Search stores to use for retrieval
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub file_search_store_names: Vec<String>,
}
