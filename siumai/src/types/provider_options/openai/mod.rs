//! OpenAI-specific Provider Options
//!
//! This module contains types for OpenAI-specific features including:
//! - Audio input/output (multimodal)
//! - Responses API configuration
//! - Built-in tools (web search, file search, computer use)
//! - Reasoning effort settings
//! - Service tier preferences

use serde::{Deserialize, Serialize};

// Sub-modules
pub mod audio;
pub mod enums;
pub mod prediction;
pub mod responses_api;
pub mod web_search;

// Re-exports
pub use audio::{
    ChatCompletionAudio, ChatCompletionAudioFormat, ChatCompletionAudioVoice,
    ChatCompletionModalities, InputAudio, InputAudioFormat,
};
pub use enums::{ReasoningEffort, ServiceTier, TextVerbosity, Truncation};
pub use prediction::{PredictionContent, PredictionContentData};
pub use responses_api::ResponsesApiConfig;
pub use web_search::{OpenAiWebSearchOptions, UserLocationWrapper, WebSearchLocation};

// Re-export OpenAiBuiltInTool from tools module to avoid duplication
pub use crate::types::tools::OpenAiBuiltInTool;

/// OpenAI-specific options
///
/// Type-safe configuration for OpenAI-specific features.
/// Use this with `ChatRequest::with_openai_options()` or `ChatRequestBuilder::openai_options()`.
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
    /// Modalities for multimodal output (e.g., text and audio)
    pub modalities: Option<Vec<ChatCompletionModalities>>,
    /// Audio output configuration (required when audio modality is requested)
    pub audio: Option<ChatCompletionAudio>,
    /// Predicted output content (for faster response times)
    pub prediction: Option<PredictionContent>,
    /// Web search options (context size and user location)
    pub web_search_options: Option<OpenAiWebSearchOptions>,
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

    /// Set modalities (e.g., for audio output)
    ///
    /// # Example
    /// ```ignore
    /// use siumai::types::provider_options::openai::{OpenAiOptions, ChatCompletionModalities};
    ///
    /// let options = OpenAiOptions::new()
    ///     .with_modalities(vec![
    ///         ChatCompletionModalities::Text,
    ///         ChatCompletionModalities::Audio,
    ///     ]);
    /// ```
    pub fn with_modalities(mut self, modalities: Vec<ChatCompletionModalities>) -> Self {
        self.modalities = Some(modalities);
        self
    }

    /// Enable audio output with configuration
    ///
    /// # Example
    /// ```ignore
    /// use siumai::types::provider_options::openai::{
    ///     OpenAiOptions, ChatCompletionAudio, ChatCompletionAudioVoice,
    ///     ChatCompletionAudioFormat, ChatCompletionModalities,
    /// };
    ///
    /// let options = OpenAiOptions::new()
    ///     .with_modalities(vec![
    ///         ChatCompletionModalities::Text,
    ///         ChatCompletionModalities::Audio,
    ///     ])
    ///     .with_audio(ChatCompletionAudio::new(
    ///         ChatCompletionAudioVoice::Ash,
    ///         ChatCompletionAudioFormat::Mp3,
    ///     ));
    /// ```
    pub fn with_audio(mut self, audio: ChatCompletionAudio) -> Self {
        self.audio = Some(audio);
        self
    }

    /// Enable audio output with voice (uses MP3 format by default)
    ///
    /// This is a convenience method that automatically sets modalities to include audio.
    ///
    /// # Example
    /// ```ignore
    /// use siumai::types::provider_options::openai::{OpenAiOptions, ChatCompletionAudioVoice};
    ///
    /// let options = OpenAiOptions::new()
    ///     .with_audio_voice(ChatCompletionAudioVoice::Ash);
    /// ```
    pub fn with_audio_voice(mut self, voice: ChatCompletionAudioVoice) -> Self {
        self.modalities = Some(vec![
            ChatCompletionModalities::Text,
            ChatCompletionModalities::Audio,
        ]);
        self.audio = Some(ChatCompletionAudio::with_voice(voice));
        self
    }

    /// Set predicted output content
    ///
    /// Configuration for Predicted Outputs, which can greatly improve response times
    /// when large parts of the model response are known ahead of time.
    ///
    /// # Example
    /// ```ignore
    /// use siumai::types::provider_options::openai::{OpenAiOptions, PredictionContent};
    ///
    /// let options = OpenAiOptions::new()
    ///     .with_prediction(PredictionContent::text("The original file content..."));
    /// ```
    pub fn with_prediction(mut self, prediction: PredictionContent) -> Self {
        self.prediction = Some(prediction);
        self
    }

    /// Set web search options
    ///
    /// Configure web search context size and user location.
    ///
    /// # Example
    /// ```ignore
    /// use siumai::types::provider_options::openai::{OpenAiOptions, OpenAiWebSearchOptions};
    ///
    /// let options = OpenAiOptions::new()
    ///     .with_web_search_options(
    ///         OpenAiWebSearchOptions::new()
    ///             .with_context_size("high")
    ///     );
    /// ```
    pub fn with_web_search_options(mut self, options: OpenAiWebSearchOptions) -> Self {
        self.web_search_options = Some(options);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_options_default() {
        let options = OpenAiOptions::default();
        assert!(options.responses_api.is_none());
        assert!(options.built_in_tools.is_empty());
        assert!(options.reasoning_effort.is_none());
        assert!(options.service_tier.is_none());
        assert!(options.modalities.is_none());
        assert!(options.audio.is_none());
    }

    #[test]
    fn test_openai_options_with_audio() {
        let options = OpenAiOptions::new().with_audio_voice(ChatCompletionAudioVoice::Ash);

        assert!(options.modalities.is_some());
        let modalities = options.modalities.unwrap();
        assert_eq!(modalities.len(), 2);
        assert!(modalities.contains(&ChatCompletionModalities::Text));
        assert!(modalities.contains(&ChatCompletionModalities::Audio));

        assert!(options.audio.is_some());
        let audio = options.audio.unwrap();
        assert_eq!(audio.voice, ChatCompletionAudioVoice::Ash);
        assert_eq!(audio.format, ChatCompletionAudioFormat::Mp3);
    }

    #[test]
    fn test_openai_options_builder() {
        let options = OpenAiOptions::new()
            .with_reasoning_effort(ReasoningEffort::High)
            .with_service_tier(ServiceTier::Default)
            .with_web_search();

        assert_eq!(options.reasoning_effort, Some(ReasoningEffort::High));
        assert_eq!(options.service_tier, Some(ServiceTier::Default));
        assert_eq!(options.built_in_tools.len(), 1);
    }
}
