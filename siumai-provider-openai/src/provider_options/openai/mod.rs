//! OpenAI-specific provider options.
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

use crate::types::Tool;

/// OpenAI-specific options
///
/// Type-safe configuration for OpenAI-specific features.
///
/// This is usually carried via the open `providerOptions` JSON map (`provider_id = "openai"`),
/// and should be carried via `providerOptions["openai"]`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpenAiOptions {
    /// Responses API configuration
    pub responses_api: Option<ResponsesApiConfig>,
    /// Provider-defined tools for OpenAI (web_search, file_search, computer use, etc.)
    /// Use `siumai::hosted_tools::openai::*` helpers to construct these.
    #[deprecated(
        note = "Prefer `ChatRequest::with_tools` with `Tool::ProviderDefined` (use `siumai::hosted_tools::openai::*`). This field is kept as a compatibility layer."
    )]
    pub provider_tools: Vec<Tool>,
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
    #[deprecated(
        note = "Prefer OpenAI Responses API + provider-defined tool `siumai::hosted_tools::openai::web_search()`."
    )]
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

    /// Add a provider-defined tool (see `hosted_tools::openai`)
    #[deprecated(
        note = "Prefer `ChatRequest::with_tools` with `Tool::ProviderDefined` (use `siumai::hosted_tools::openai::*`). This method is kept as a compatibility layer."
    )]
    #[allow(deprecated)]
    pub fn with_provider_tool(mut self, tool: Tool) -> Self {
        self.provider_tools.push(tool);
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
    pub fn with_modalities(mut self, modalities: Vec<ChatCompletionModalities>) -> Self {
        self.modalities = Some(modalities);
        self
    }

    /// Enable audio output with configuration
    pub fn with_audio(mut self, audio: ChatCompletionAudio) -> Self {
        self.audio = Some(audio);
        self
    }

    /// Enable audio output with voice (uses MP3 format by default)
    ///
    /// This is a convenience method that automatically sets modalities to include audio.
    pub fn with_audio_voice(mut self, voice: ChatCompletionAudioVoice) -> Self {
        self.modalities = Some(vec![
            ChatCompletionModalities::Text,
            ChatCompletionModalities::Audio,
        ]);
        self.audio = Some(ChatCompletionAudio::with_voice(voice));
        self
    }

    /// Set predicted output content
    pub fn with_prediction(mut self, prediction: PredictionContent) -> Self {
        self.prediction = Some(prediction);
        self
    }

    /// Set web search options
    #[deprecated(
        note = "Prefer OpenAI Responses API + provider-defined tool `siumai::hosted_tools::openai::web_search()`."
    )]
    #[allow(deprecated)]
    pub fn with_web_search_options(mut self, options: OpenAiWebSearchOptions) -> Self {
        self.web_search_options = Some(options);
        self
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_options_default() {
        let options = OpenAiOptions::default();
        assert!(options.responses_api.is_none());
        assert!(options.provider_tools.is_empty());
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
}
