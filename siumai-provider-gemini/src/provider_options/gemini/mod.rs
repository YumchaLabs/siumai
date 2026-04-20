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

    /// Optional top-level threshold escape hatch kept for AI SDK surface parity.
    ///
    /// Note: the upstream AI SDK type exposes this field, but current Google runtime mapping
    /// primarily uses `safetySettings`. Prefer `with_safety_settings(...)` for deterministic
    /// request behavior.
    pub threshold: Option<GeminiHarmBlockThreshold>,

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

    /// Optional. Stream tool arguments incrementally in Vertex streaming responses only.
    ///
    /// Vercel AI SDK alignment: `providerOptions.google.streamFunctionCallArguments`.
    pub stream_function_call_arguments: Option<bool>,

    /// Optional. Enable structured outputs (responseSchema) for `responseFormat`.
    ///
    /// Vercel AI SDK alignment: `providerOptions.google.structuredOutputs`.
    pub structured_outputs: Option<bool>,

    /// Optional. If true, export logprobs results in response.
    pub response_logprobs: Option<bool>,

    /// Optional. Number of top logprobs to return at each decoding step.
    pub logprobs: Option<i32>,

    /// Optional service tier hint.
    ///
    /// Accepted values follow the AI SDK surface (`standard`, `flex`, `priority`).
    pub service_tier: Option<String>,
}

/// AI SDK-style alias for Google language-model options.
pub type GoogleLanguageModelOptions = GeminiOptions;

/// Deprecated AI SDK compatibility alias for Google language-model options.
#[deprecated(note = "Use `GoogleLanguageModelOptions` instead.")]
pub type GoogleGenerativeAIProviderOptions = GoogleLanguageModelOptions;

/// Image config for Gemini image generation models.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GeminiImageConfig {
    pub aspect_ratio: Option<String>,
    pub image_size: Option<String>,
}

/// Provider-owned image-model options for Gemini/Google image requests.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GeminiImageOptions {
    /// Output aspect ratio (for example `1:1`, `16:9`, `9:16`).
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "aspectRatio",
        alias = "aspect_ratio"
    )]
    pub aspect_ratio: Option<String>,
    /// Person-generation policy (`dont_allow`, `allow_adult`, `allow_all`).
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "personGeneration",
        alias = "person_generation"
    )]
    pub person_generation: Option<String>,
    /// Forward-compatible provider-owned escape hatch for newly introduced fields.
    #[serde(flatten, default, skip_serializing_if = "HashMap::is_empty")]
    pub extra_fields: HashMap<String, serde_json::Value>,
}

impl GeminiImageOptions {
    /// Create empty Gemini image options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the image aspect ratio.
    pub fn with_aspect_ratio(mut self, aspect_ratio: impl Into<String>) -> Self {
        self.aspect_ratio = Some(aspect_ratio.into());
        self
    }

    /// Set the person-generation policy.
    pub fn with_person_generation(mut self, person_generation: impl Into<String>) -> Self {
        self.person_generation = Some(person_generation.into());
        self
    }

    /// Add an extra provider-owned field.
    pub fn with_extra_field(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_fields.insert(key.into(), value);
        self
    }
}

/// AI SDK-style alias for Google image-model options.
pub type GoogleImageModelOptions = GeminiImageOptions;

/// Deprecated AI SDK compatibility alias for Google image-model options.
#[deprecated(note = "Use `GoogleImageModelOptions` instead.")]
pub type GoogleGenerativeAIImageProviderOptions = GoogleImageModelOptions;

/// Inline multimodal data for Google embedding requests.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GoogleEmbeddingInlineData {
    #[serde(rename = "mimeType", alias = "mime_type")]
    pub mime_type: String,
    pub data: String,
}

/// Multimodal part for Google embedding requests.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum GoogleEmbeddingContentPart {
    Text {
        text: String,
    },
    InlineData {
        #[serde(rename = "inlineData", alias = "inline_data")]
        inline_data: GoogleEmbeddingInlineData,
    },
}

/// Provider-owned embedding-model options for Google embedding requests.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct GoogleEmbeddingModelOptions {
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "outputDimensionality",
        alias = "output_dimensionality"
    )]
    pub output_dimensionality: Option<u32>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "taskType",
        alias = "task_type"
    )]
    pub task_type: Option<crate::types::EmbeddingTaskType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<Option<Vec<GoogleEmbeddingContentPart>>>>,
}

impl GoogleEmbeddingModelOptions {
    /// Create empty Google embedding options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set output dimensionality.
    pub const fn with_output_dimensionality(mut self, dimensions: u32) -> Self {
        self.output_dimensionality = Some(dimensions);
        self
    }

    /// Set embedding task type.
    pub fn with_task_type(mut self, task_type: crate::types::EmbeddingTaskType) -> Self {
        self.task_type = Some(task_type);
        self
    }

    /// Set per-input multimodal content overrides.
    pub fn with_content(mut self, content: Vec<Option<Vec<GoogleEmbeddingContentPart>>>) -> Self {
        self.content = Some(content);
        self
    }
}

/// Deprecated AI SDK compatibility alias for Google embedding-model options.
#[deprecated(note = "Use `GoogleEmbeddingModelOptions` instead.")]
pub type GoogleGenerativeAIEmbeddingProviderOptions = GoogleEmbeddingModelOptions;

/// Reference image entry for Google video generation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GoogleReferenceImage {
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "bytesBase64Encoded",
        alias = "bytes_base64_encoded"
    )]
    pub bytes_base64_encoded: Option<String>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "gcsUri",
        alias = "gcs_uri"
    )]
    pub gcs_uri: Option<String>,
}

/// Provider-owned video-model options for Google video requests.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct GoogleVideoModelOptions {
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "pollIntervalMs",
        alias = "poll_interval_ms"
    )]
    pub poll_interval_ms: Option<u64>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "pollTimeoutMs",
        alias = "poll_timeout_ms"
    )]
    pub poll_timeout_ms: Option<u64>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "personGeneration",
        alias = "person_generation"
    )]
    pub person_generation: Option<String>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "negativePrompt",
        alias = "negative_prompt"
    )]
    pub negative_prompt: Option<String>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "referenceImages",
        alias = "reference_images"
    )]
    pub reference_images: Option<Vec<GoogleReferenceImage>>,
    #[serde(flatten, default, skip_serializing_if = "HashMap::is_empty")]
    pub extra_fields: HashMap<String, serde_json::Value>,
}

impl GoogleVideoModelOptions {
    /// Create empty Google video options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set polling interval in milliseconds.
    pub const fn with_poll_interval_ms(mut self, value: u64) -> Self {
        self.poll_interval_ms = Some(value);
        self
    }

    /// Set polling timeout in milliseconds.
    pub const fn with_poll_timeout_ms(mut self, value: u64) -> Self {
        self.poll_timeout_ms = Some(value);
        self
    }

    /// Set person generation policy.
    pub fn with_person_generation(mut self, value: impl Into<String>) -> Self {
        self.person_generation = Some(value.into());
        self
    }

    /// Set negative prompt.
    pub fn with_negative_prompt(mut self, value: impl Into<String>) -> Self {
        self.negative_prompt = Some(value.into());
        self
    }

    /// Set reference images.
    pub fn with_reference_images(mut self, value: Vec<GoogleReferenceImage>) -> Self {
        self.reference_images = Some(value);
        self
    }

    /// Add one extra provider-owned field.
    pub fn with_extra_field(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_fields.insert(key.into(), value);
        self
    }
}

/// Deprecated AI SDK compatibility alias for Google video-model options.
#[deprecated(note = "Use `GoogleVideoModelOptions` instead.")]
pub type GoogleGenerativeAIVideoProviderOptions = GoogleVideoModelOptions;

/// AI SDK-style alias for Google video model ids.
pub type GoogleVideoModelId = String;

/// Deprecated AI SDK compatibility alias for Google video model ids.
#[deprecated(note = "Use `GoogleVideoModelId` instead.")]
pub type GoogleGenerativeAIVideoModelId = GoogleVideoModelId;

/// Provider-owned upload options for Google files API.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct GoogleFilesUploadOptions {
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "displayName",
        alias = "display_name"
    )]
    pub display_name: Option<String>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "pollIntervalMs",
        alias = "poll_interval_ms"
    )]
    pub poll_interval_ms: Option<u64>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "pollTimeoutMs",
        alias = "poll_timeout_ms"
    )]
    pub poll_timeout_ms: Option<u64>,
    #[serde(flatten, default, skip_serializing_if = "HashMap::is_empty")]
    pub extra_fields: HashMap<String, serde_json::Value>,
}

impl GoogleFilesUploadOptions {
    /// Create empty Google files upload options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set display name.
    pub fn with_display_name(mut self, value: impl Into<String>) -> Self {
        self.display_name = Some(value.into());
        self
    }

    /// Set polling interval in milliseconds.
    pub const fn with_poll_interval_ms(mut self, value: u64) -> Self {
        self.poll_interval_ms = Some(value);
        self
    }

    /// Set polling timeout in milliseconds.
    pub const fn with_poll_timeout_ms(mut self, value: u64) -> Self {
        self.poll_timeout_ms = Some(value);
        self
    }

    /// Add one extra provider-owned field.
    pub fn with_extra_field(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_fields.insert(key.into(), value);
        self
    }
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

    /// Set the top-level threshold compatibility field.
    pub fn with_threshold(mut self, threshold: GeminiHarmBlockThreshold) -> Self {
        self.threshold = Some(threshold);
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

    /// Enable or disable streamed function-call arguments on Vertex streaming paths.
    pub const fn with_stream_function_call_arguments(mut self, enabled: bool) -> Self {
        self.stream_function_call_arguments = Some(enabled);
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

    /// Set service tier (`standard`, `flex`, `priority`).
    pub fn with_service_tier(mut self, tier: impl Into<String>) -> Self {
        self.service_tier = Some(tier.into());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemini_image_options_serialize_to_google_shape() {
        let value = serde_json::to_value(
            GeminiImageOptions::new()
                .with_aspect_ratio("16:9")
                .with_person_generation("allow_all")
                .with_extra_field("style", serde_json::json!("cinematic")),
        )
        .expect("serialize GeminiImageOptions");

        assert_eq!(
            value,
            serde_json::json!({
                "aspectRatio": "16:9",
                "personGeneration": "allow_all",
                "style": "cinematic"
            })
        );
    }

    #[test]
    fn gemini_image_options_accept_aliases() {
        let options: GeminiImageOptions = serde_json::from_value(serde_json::json!({
            "aspect_ratio": "9:16",
            "personGeneration": "dont_allow"
        }))
        .expect("deserialize GeminiImageOptions");

        assert_eq!(options.aspect_ratio.as_deref(), Some("9:16"));
        assert_eq!(options.person_generation.as_deref(), Some("dont_allow"));
    }

    #[test]
    #[allow(deprecated)]
    fn google_image_option_aliases_resolve_to_same_type() {
        let options: GoogleImageModelOptions = GeminiImageOptions::new().with_aspect_ratio("1:1");
        let deprecated: GoogleGenerativeAIImageProviderOptions = options.clone();

        assert_eq!(options.aspect_ratio, deprecated.aspect_ratio);
    }

    #[test]
    #[allow(deprecated)]
    fn google_language_option_aliases_resolve_to_same_type() {
        let options: GoogleLanguageModelOptions = GeminiOptions::new()
            .with_service_tier("flex")
            .with_stream_function_call_arguments(true);
        let deprecated: GoogleGenerativeAIProviderOptions = options.clone();

        assert_eq!(options.service_tier, deprecated.service_tier);
        assert_eq!(
            options.stream_function_call_arguments,
            deprecated.stream_function_call_arguments
        );
    }

    #[test]
    fn google_embedding_options_serialize_to_google_shape() {
        let value = serde_json::to_value(
            GoogleEmbeddingModelOptions::new()
                .with_output_dimensionality(256)
                .with_task_type(crate::types::EmbeddingTaskType::SemanticSimilarity)
                .with_content(vec![Some(vec![
                    GoogleEmbeddingContentPart::InlineData {
                        inline_data: GoogleEmbeddingInlineData {
                            mime_type: "image/png".to_string(),
                            data: "Zm9v".to_string(),
                        },
                    },
                    GoogleEmbeddingContentPart::Text {
                        text: "caption".to_string(),
                    },
                ])]),
        )
        .expect("serialize GoogleEmbeddingModelOptions");

        assert_eq!(
            value,
            serde_json::json!({
                "outputDimensionality": 256,
                "taskType": "SEMANTIC_SIMILARITY",
                "content": [[
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": "Zm9v"
                        }
                    },
                    {
                        "text": "caption"
                    }
                ]]
            })
        );
    }

    #[test]
    #[allow(deprecated)]
    fn google_embedding_option_aliases_resolve_to_same_type() {
        let options = GoogleEmbeddingModelOptions::new().with_output_dimensionality(768);
        let deprecated: GoogleGenerativeAIEmbeddingProviderOptions = options.clone();

        assert_eq!(
            options.output_dimensionality,
            deprecated.output_dimensionality
        );
    }

    #[test]
    #[allow(deprecated)]
    fn google_video_aliases_resolve_to_same_type() {
        let options = GoogleVideoModelOptions::new()
            .with_negative_prompt("no cats")
            .with_person_generation("allow_all");
        let deprecated_options: GoogleGenerativeAIVideoProviderOptions = options.clone();
        let model_id: GoogleVideoModelId = "veo-3.1-generate-preview".to_string();
        let deprecated_model_id: GoogleGenerativeAIVideoModelId = model_id.clone();

        assert_eq!(options.negative_prompt, deprecated_options.negative_prompt);
        assert_eq!(model_id, deprecated_model_id);
    }

    #[test]
    fn google_files_upload_options_serialize_to_google_shape() {
        let value = serde_json::to_value(
            GoogleFilesUploadOptions::new()
                .with_display_name("spec.pdf")
                .with_poll_interval_ms(250)
                .with_poll_timeout_ms(30_000)
                .with_extra_field("customFlag", serde_json::json!(true)),
        )
        .expect("serialize GoogleFilesUploadOptions");

        assert_eq!(
            value,
            serde_json::json!({
                "displayName": "spec.pdf",
                "pollIntervalMs": 250,
                "pollTimeoutMs": 30000,
                "customFlag": true
            })
        );
    }
}
