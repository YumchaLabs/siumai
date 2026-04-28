//! Alibaba/Qwen provider options.
//!
//! These typed option structs mirror AI SDK `@ai-sdk/alibaba` language-model options.
//! The OpenAI-compatible preset historically uses provider id `qwen`, so both
//! `providerOptions["alibaba"]` and `providerOptions["qwen"]` are supported by the runtime.

use serde::{Deserialize, Serialize};

/// Alibaba prompt cache-control marker used on message/part provider options.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlibabaCacheControl {
    /// Cache-control marker type. Alibaba currently documents `ephemeral`.
    #[serde(rename = "type")]
    pub r#type: String,
}

impl AlibabaCacheControl {
    /// Create an ephemeral prompt-cache marker.
    pub fn ephemeral() -> Self {
        Self {
            r#type: "ephemeral".to_string(),
        }
    }
}

/// Typed Alibaba/Qwen chat/language-model options.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AlibabaChatOptions {
    /// Enable thinking/reasoning mode for supported Qwen models.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "enable_thinking"
    )]
    pub enable_thinking: Option<bool>,
    /// Maximum number of reasoning tokens to generate.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "thinking_budget"
    )]
    pub thinking_budget: Option<u32>,
    /// Whether to allow parallel tool calls.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "parallel_tool_calls"
    )]
    pub parallel_tool_calls: Option<bool>,
}

impl AlibabaChatOptions {
    /// Create empty Alibaba/Qwen chat options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable provider-side thinking.
    pub const fn with_enable_thinking(mut self, enable_thinking: bool) -> Self {
        self.enable_thinking = Some(enable_thinking);
        self
    }

    /// Set the thinking budget.
    pub const fn with_thinking_budget(mut self, thinking_budget: u32) -> Self {
        self.thinking_budget = Some(thinking_budget);
        self
    }

    /// Control parallel tool calls.
    pub const fn with_parallel_tool_calls(mut self, parallel_tool_calls: bool) -> Self {
        self.parallel_tool_calls = Some(parallel_tool_calls);
        self
    }
}

/// AI SDK-aligned alias for Alibaba language-model options.
pub type AlibabaLanguageModelOptions = AlibabaChatOptions;

/// Deprecated AI SDK-compatible alias for Alibaba language-model options.
#[deprecated(note = "Use AlibabaLanguageModelOptions instead.")]
pub type AlibabaProviderOptions = AlibabaLanguageModelOptions;

/// Local preset alias for Qwen chat options.
pub type QwenChatOptions = AlibabaChatOptions;

/// Local preset alias for Qwen language-model options.
pub type QwenLanguageModelOptions = AlibabaLanguageModelOptions;

/// Deprecated local preset alias for Qwen provider options.
#[deprecated(note = "Use QwenLanguageModelOptions instead.")]
pub type QwenProviderOptions = QwenLanguageModelOptions;

/// Typed Alibaba video-model options aligned with
/// `repo-ref/ai/packages/alibaba/src/alibaba-video-model.ts`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AlibabaVideoModelOptions {
    /// Negative prompt to specify what to avoid.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "negative_prompt"
    )]
    pub negative_prompt: Option<String>,
    /// URL to audio file for audio-video sync.
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "audio_url")]
    pub audio_url: Option<String>,
    /// Enable provider prompt extension.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "prompt_extend"
    )]
    pub prompt_extend: Option<bool>,
    /// Shot type: `single` or `multi`.
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "shot_type")]
    pub shot_type: Option<String>,
    /// Whether to add a watermark.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub watermark: Option<bool>,
    /// Enable audio generation for supported video models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<bool>,
    /// Reference URLs for reference-to-video mode.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "reference_urls"
    )]
    pub reference_urls: Option<Vec<String>>,
    /// Polling interval in milliseconds for high-level video helpers.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "poll_interval_ms"
    )]
    pub poll_interval_ms: Option<u64>,
    /// Polling timeout in milliseconds for high-level video helpers.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "poll_timeout_ms"
    )]
    pub poll_timeout_ms: Option<u64>,
}

impl AlibabaVideoModelOptions {
    /// Create empty Alibaba video options.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_negative_prompt(mut self, negative_prompt: impl Into<String>) -> Self {
        self.negative_prompt = Some(negative_prompt.into());
        self
    }

    pub fn with_audio_url(mut self, audio_url: impl Into<String>) -> Self {
        self.audio_url = Some(audio_url.into());
        self
    }

    pub const fn with_prompt_extend(mut self, prompt_extend: bool) -> Self {
        self.prompt_extend = Some(prompt_extend);
        self
    }

    pub fn with_shot_type(mut self, shot_type: impl Into<String>) -> Self {
        self.shot_type = Some(shot_type.into());
        self
    }

    pub const fn with_watermark(mut self, watermark: bool) -> Self {
        self.watermark = Some(watermark);
        self
    }

    pub const fn with_audio(mut self, audio: bool) -> Self {
        self.audio = Some(audio);
        self
    }

    pub fn with_reference_urls<I, S>(mut self, reference_urls: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.reference_urls = Some(reference_urls.into_iter().map(Into::into).collect());
        self
    }

    pub const fn with_poll_interval_ms(mut self, poll_interval_ms: u64) -> Self {
        self.poll_interval_ms = Some(poll_interval_ms);
        self
    }

    pub const fn with_poll_timeout_ms(mut self, poll_timeout_ms: u64) -> Self {
        self.poll_timeout_ms = Some(poll_timeout_ms);
        self
    }
}

/// Deprecated AI SDK-compatible alias for Alibaba video options.
#[deprecated(note = "Use AlibabaVideoModelOptions instead.")]
pub type AlibabaVideoProviderOptions = AlibabaVideoModelOptions;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alibaba_options_serialize_to_ai_sdk_shape() {
        let value = serde_json::to_value(
            AlibabaChatOptions::new()
                .with_enable_thinking(true)
                .with_thinking_budget(2048)
                .with_parallel_tool_calls(false),
        )
        .expect("options serialize");

        assert_eq!(
            value,
            serde_json::json!({
                "enableThinking": true,
                "thinkingBudget": 2048,
                "parallelToolCalls": false
            })
        );
    }

    #[test]
    fn alibaba_options_accept_snake_case_aliases() {
        let options: AlibabaChatOptions = serde_json::from_value(serde_json::json!({
            "enable_thinking": false,
            "thinking_budget": 1024,
            "parallel_tool_calls": true
        }))
        .expect("options deserialize");

        assert_eq!(options.enable_thinking, Some(false));
        assert_eq!(options.thinking_budget, Some(1024));
        assert_eq!(options.parallel_tool_calls, Some(true));
    }

    #[test]
    fn alibaba_cache_control_serializes_to_ai_sdk_shape() {
        let value = serde_json::to_value(AlibabaCacheControl::ephemeral())
            .expect("cache control serialize");

        assert_eq!(value, serde_json::json!({ "type": "ephemeral" }));
    }

    #[test]
    fn alibaba_video_options_serialize_to_ai_sdk_shape() {
        let value = serde_json::to_value(
            AlibabaVideoModelOptions::new()
                .with_negative_prompt("no rain")
                .with_audio_url("https://example.com/audio.mp3")
                .with_prompt_extend(true)
                .with_shot_type("multi")
                .with_watermark(false)
                .with_audio(true)
                .with_reference_urls(["https://example.com/ref.png"])
                .with_poll_interval_ms(5000)
                .with_poll_timeout_ms(600000),
        )
        .expect("options serialize");

        assert_eq!(
            value,
            serde_json::json!({
                "negativePrompt": "no rain",
                "audioUrl": "https://example.com/audio.mp3",
                "promptExtend": true,
                "shotType": "multi",
                "watermark": false,
                "audio": true,
                "referenceUrls": ["https://example.com/ref.png"],
                "pollIntervalMs": 5000,
                "pollTimeoutMs": 600000
            })
        );
    }

    #[test]
    fn alibaba_video_options_accept_snake_case_aliases() {
        let options: AlibabaVideoModelOptions = serde_json::from_value(serde_json::json!({
            "negative_prompt": "no blur",
            "audio_url": "https://example.com/audio.wav",
            "prompt_extend": false,
            "shot_type": "single",
            "reference_urls": ["https://example.com/ref.mp4"],
            "poll_interval_ms": 1000,
            "poll_timeout_ms": 2000
        }))
        .expect("options deserialize");

        assert_eq!(options.negative_prompt.as_deref(), Some("no blur"));
        assert_eq!(
            options.audio_url.as_deref(),
            Some("https://example.com/audio.wav")
        );
        assert_eq!(options.prompt_extend, Some(false));
        assert_eq!(options.shot_type.as_deref(), Some("single"));
        assert_eq!(
            options.reference_urls,
            Some(vec!["https://example.com/ref.mp4".to_string()])
        );
        assert_eq!(options.poll_interval_ms, Some(1000));
        assert_eq!(options.poll_timeout_ms, Some(2000));
    }
}
