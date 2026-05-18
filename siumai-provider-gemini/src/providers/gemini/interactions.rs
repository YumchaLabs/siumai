use async_trait::async_trait;
use std::sync::Arc;

use crate::LlmError;
use crate::streaming::ChatStream;
use crate::traits::ChatCapability;
use crate::types::{ChatMessage, ChatRequest, ChatResponse, Tool};

use super::{GeminiConfig, SharedIdGenerator};

/// Model selector for the Google Interactions API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GoogleInteractionsModelInput {
    Model(String),
    Agent(String),
}

impl GoogleInteractionsModelInput {
    pub fn model(model_id: impl Into<String>) -> Self {
        Self::Model(model_id.into())
    }

    pub fn agent(agent_name: impl Into<String>) -> Self {
        Self::Agent(agent_name.into())
    }

    pub fn id(&self) -> &str {
        match self {
            Self::Model(model_id) | Self::Agent(model_id) => model_id,
        }
    }

    pub const fn is_agent(&self) -> bool {
        matches!(self, Self::Agent(_))
    }
}

impl From<String> for GoogleInteractionsModelInput {
    fn from(value: String) -> Self {
        Self::Model(value)
    }
}

impl From<&str> for GoogleInteractionsModelInput {
    fn from(value: &str) -> Self {
        Self::Model(value.to_string())
    }
}

/// Provider-owned handle for `google.interactions(...)`.
///
/// This type is intentionally not wired into ordinary Gemini chat execution.
/// The Interactions API uses `/v1beta/interactions`, background agent polling,
/// per-block signatures, and interaction ids, so it needs a dedicated runtime
/// slice before it can honestly execute requests.
#[derive(Clone)]
pub struct GoogleInteractionsLanguageModel {
    config: GeminiConfig,
    model_input: GoogleInteractionsModelInput,
}

impl std::fmt::Debug for GoogleInteractionsLanguageModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GoogleInteractionsLanguageModel")
            .field("provider", &self.provider())
            .field("base_url", &self.config.base_url)
            .field("model_input", &self.model_input)
            .finish()
    }
}

impl GoogleInteractionsLanguageModel {
    pub fn new(config: GeminiConfig, model_input: impl Into<GoogleInteractionsModelInput>) -> Self {
        Self {
            config,
            model_input: model_input.into(),
        }
    }

    pub fn config(&self) -> &GeminiConfig {
        &self.config
    }

    pub fn model_input(&self) -> &GoogleInteractionsModelInput {
        &self.model_input
    }

    pub fn model_id(&self) -> &str {
        self.model_input.id()
    }

    pub fn agent(&self) -> Option<&str> {
        match &self.model_input {
            GoogleInteractionsModelInput::Agent(agent) => Some(agent),
            GoogleInteractionsModelInput::Model(_) => None,
        }
    }

    pub fn provider(&self) -> String {
        format!("{}.interactions", self.config.provider_name())
    }

    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    pub fn generate_id(&self) -> String {
        self.config.generate_id()
    }

    fn unsupported_runtime_error(&self) -> LlmError {
        LlmError::UnsupportedOperation(
            "google.interactions runtime is not implemented yet; this handle only locks the package surface while the dedicated /interactions execution lane is tracked"
                .to_string(),
        )
    }
}

#[async_trait]
impl ChatCapability for GoogleInteractionsLanguageModel {
    async fn chat_with_tools(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        Err(self.unsupported_runtime_error())
    }

    async fn chat_stream(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        Err(self.unsupported_runtime_error())
    }

    async fn chat_request(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
        Err(self.unsupported_runtime_error())
    }

    async fn chat_stream_request(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
        Err(self.unsupported_runtime_error())
    }
}

/// Curated model-id constants for the audited Google Interactions package surface.
pub mod interactions {
    pub const GEMINI_2_5_COMPUTER_USE_PREVIEW_10_2025: &str =
        "gemini-2.5-computer-use-preview-10-2025";
    pub const GEMINI_2_5_FLASH: &str = "gemini-2.5-flash";
    pub const GEMINI_2_5_FLASH_IMAGE: &str = "gemini-2.5-flash-image";
    pub const GEMINI_2_5_FLASH_LITE: &str = "gemini-2.5-flash-lite";
    pub const GEMINI_2_5_FLASH_LITE_PREVIEW_09_2025: &str = "gemini-2.5-flash-lite-preview-09-2025";
    pub const GEMINI_2_5_FLASH_NATIVE_AUDIO_PREVIEW_12_2025: &str =
        "gemini-2.5-flash-native-audio-preview-12-2025";
    pub const GEMINI_2_5_FLASH_PREVIEW_09_2025: &str = "gemini-2.5-flash-preview-09-2025";
    pub const GEMINI_2_5_FLASH_PREVIEW_TTS: &str = "gemini-2.5-flash-preview-tts";
    pub const GEMINI_2_5_PRO: &str = "gemini-2.5-pro";
    pub const GEMINI_2_5_PRO_PREVIEW_TTS: &str = "gemini-2.5-pro-preview-tts";
    pub const GEMINI_3_FLASH_PREVIEW: &str = "gemini-3-flash-preview";
    pub const GEMINI_3_PRO_IMAGE_PREVIEW: &str = "gemini-3-pro-image-preview";
    pub const GEMINI_3_PRO_PREVIEW: &str = "gemini-3-pro-preview";
    pub const GEMINI_3_1_PRO_PREVIEW: &str = "gemini-3.1-pro-preview";
    pub const GEMINI_3_1_FLASH_IMAGE_PREVIEW: &str = "gemini-3.1-flash-image-preview";
    pub const GEMINI_3_1_FLASH_LITE_PREVIEW: &str = "gemini-3.1-flash-lite-preview";
    pub const GEMINI_3_1_FLASH_TTS_PREVIEW: &str = "gemini-3.1-flash-tts-preview";
    pub const LYRIA_3_CLIP_PREVIEW: &str = "lyria-3-clip-preview";
    pub const LYRIA_3_PRO_PREVIEW: &str = "lyria-3-pro-preview";

    pub const ALL: &[&str] = &[
        GEMINI_2_5_COMPUTER_USE_PREVIEW_10_2025,
        GEMINI_2_5_FLASH,
        GEMINI_2_5_FLASH_IMAGE,
        GEMINI_2_5_FLASH_LITE,
        GEMINI_2_5_FLASH_LITE_PREVIEW_09_2025,
        GEMINI_2_5_FLASH_NATIVE_AUDIO_PREVIEW_12_2025,
        GEMINI_2_5_FLASH_PREVIEW_09_2025,
        GEMINI_2_5_FLASH_PREVIEW_TTS,
        GEMINI_2_5_PRO,
        GEMINI_2_5_PRO_PREVIEW_TTS,
        GEMINI_3_FLASH_PREVIEW,
        GEMINI_3_PRO_IMAGE_PREVIEW,
        GEMINI_3_PRO_PREVIEW,
        GEMINI_3_1_PRO_PREVIEW,
        GEMINI_3_1_FLASH_IMAGE_PREVIEW,
        GEMINI_3_1_FLASH_LITE_PREVIEW,
        GEMINI_3_1_FLASH_TTS_PREVIEW,
        LYRIA_3_CLIP_PREVIEW,
        LYRIA_3_PRO_PREVIEW,
    ];
}

/// Curated agent-name constants for the audited Google Interactions package surface.
pub mod agents {
    pub const DEEP_RESEARCH_PRO_PREVIEW_12_2025: &str = "deep-research-pro-preview-12-2025";
    pub const DEEP_RESEARCH_PREVIEW_04_2026: &str = "deep-research-preview-04-2026";
    pub const DEEP_RESEARCH_MAX_PREVIEW_04_2026: &str = "deep-research-max-preview-04-2026";

    pub const ALL: &[&str] = &[
        DEEP_RESEARCH_PRO_PREVIEW_12_2025,
        DEEP_RESEARCH_PREVIEW_04_2026,
        DEEP_RESEARCH_MAX_PREVIEW_04_2026,
    ];
}

pub(super) fn interactions_config_from_builder_parts(
    mut config: GeminiConfig,
    model_input: GoogleInteractionsModelInput,
    generate_id: Option<SharedIdGenerator>,
) -> GoogleInteractionsLanguageModel {
    if let Some(generate_id) = generate_id {
        config = config.with_shared_generate_id(generate_id);
    }
    GoogleInteractionsLanguageModel::new(config, model_input)
}

pub(super) fn clone_shared_id_generator(
    generate_id: &Option<SharedIdGenerator>,
) -> Option<SharedIdGenerator> {
    generate_id.as_ref().map(Arc::clone)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ChatCapability;

    #[tokio::test]
    async fn interactions_handle_is_explicitly_deferred_at_runtime() {
        let model = GoogleInteractionsLanguageModel::new(
            GeminiConfig::new("test-key").with_provider_name("google.generative-ai"),
            GoogleInteractionsModelInput::agent(agents::DEEP_RESEARCH_PREVIEW_04_2026),
        );

        assert_eq!(model.provider(), "google.generative-ai.interactions");
        assert_eq!(model.agent(), Some(agents::DEEP_RESEARCH_PREVIEW_04_2026));

        let err = model
            .chat_request(ChatRequest::new(vec![ChatMessage::user("hi").build()]))
            .await
            .expect_err("runtime is deferred");

        match err {
            LlmError::UnsupportedOperation(message) => {
                assert!(message.contains("google.interactions runtime is not implemented yet"));
            }
            other => panic!("expected UnsupportedOperation, got {other:?}"),
        }
    }
}
