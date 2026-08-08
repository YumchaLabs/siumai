//! Volcengine ARK provider integration for Siumai.
//!
//! This crate owns Volcengine identity, credentials, endpoint selection, typed ARK options, model
//! advisories, and the verified ARK OpenAI dialect. Network execution is delegated to the shared
//! OpenAI-compatible runtime without exposing that engine as the public provider owner.
#![deny(unsafe_code)]

mod image;
mod language;
pub mod models;
mod native;
pub mod options;
mod provider;
mod video;

pub use image::{
    ARK_IMAGE_API_MODE, ArkGeneratedImage, ArkImageInput, ArkImageModel, ArkImageOptions,
    ArkImageOutputFormat, ArkImageRequest, ArkImageResponse, ArkImageResponseFormat, ArkImages,
    ArkOptimizePromptMode, ArkSequentialImageGeneration,
};
pub use language::{
    CHAT_SOURCE, DEFAULT_BASE_URL, MODEL_SOURCE, PLATFORM_ID, PROVIDER_ID, RESPONSES_SOURCE,
    VERIFIED_ON, VolcengineProfileError,
};
pub use options::{
    ARK_BETA_IMAGE_PROCESS_HEADER, ARK_BETA_KNOWLEDGE_SEARCH_HEADER, ARK_BETA_MCP_HEADER,
    ArkCaching, ArkCachingType, ArkChatOptions, ArkMcpApproval, ArkMcpTool, ArkResponsesOptions,
    ArkResponsesTool, ArkThinking, ArkThinkingType,
};
pub use provider::{
    VolcengineConfigError, VolcengineCredential, VolcengineLanguageApi, VolcengineLanguageModel,
    VolcengineProvider, VolcengineProviderBuilder,
};
pub use siumai_openai_compatible::{
    BearerCredential, CredentialRequest, CredentialSourceError, DynamicCredentialSource,
};
pub use video::{
    ArkMediaSource, ArkMediaUrlWire, ArkVideoAudioRole, ArkVideoContent, ArkVideoCreateRequest,
    ArkVideoCreateResult, ArkVideoDeleteResult, ArkVideoImageRole, ArkVideoResolution,
    ArkVideoServiceTier, ArkVideoTask, ArkVideoTaskContent, ArkVideoTaskId, ArkVideoTaskList,
    ArkVideoTaskListQuery, ArkVideoTaskStatus, ArkVideoTasks, ArkVideoUsage, ArkVideoVideoRole,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
