//! Native Anthropic provider integration for Siumai.
//!
//! This crate owns Anthropic identity, authentication, official endpoint policy,
//! typed Messages options, durable prompt-cache annotations, model catalog introspection,
//! and provider-native resource APIs. Canonical Anthropic Messages wire semantics
//! and reusable execution live in the protocol and compatible-engine crates.
#![deny(unsafe_code)]

mod annotations;
mod auth;
mod metadata;
mod models;
mod options;
mod profile;
mod provider;
mod request_policy;
pub mod resources;

pub use annotations::{
    AnthropicAnnotationResolver, AnthropicCacheTtl, AnthropicContentOptions, AnthropicMessageCache,
    AnthropicToolOptions, AnthropicToolSpecError,
};
pub use auth::{AnthropicCredential, AnthropicCredentialError};
pub use metadata::{
    AnthropicAssignedInferenceGeo, AnthropicAssignedServiceTier, AnthropicAssignedSpeed,
    AnthropicLanguageResponseExt, AnthropicResponseMetadata, AnthropicResponseMetadataError,
    AnthropicResponseUsage,
};
pub use models::{
    CLAUDE_FABLE_5, CLAUDE_HAIKU_4_5, CLAUDE_HAIKU_4_5_20251001, CLAUDE_MYTHOS_5,
    CLAUDE_MYTHOS_PREVIEW, CLAUDE_OPUS_4_1_20250805, CLAUDE_OPUS_4_6, CLAUDE_OPUS_4_7,
    CLAUDE_OPUS_4_8, CLAUDE_OPUS_5, CLAUDE_SONNET_4_6, CLAUDE_SONNET_5, current_models,
};
pub use options::{AnthropicMessagesOptions, AnthropicThinking, AnthropicTokenCountOptions};
pub use profile::AnthropicProfileError;
pub use provider::{
    AnthropicConfigError, AnthropicLanguageModel, AnthropicProvider, AnthropicProviderBuilder,
};
pub use siumai_protocol_anthropic::messages::MessagesMetadata;
pub use siumai_protocol_anthropic::messages::{
    AdvisorToolOptions, AnthropicTool, AnthropicToolReference, ClearThinkingEdit,
    ClearThinkingKeep, ClearToolInputs, ClearToolUsesEdit, CompactionEdit, ComputerToolOptions,
    ContainerSkill, ContainerSkillType, ContextManagement, ContextManagementEdit,
    ContextManagementTrigger, FallbackOutputConfig, InferenceGeo, InferenceSpeed,
    McpAuthorizationToken, McpServer, McpToolConfig, McpToolsetOptions, MessagesContainer,
    MessagesServiceTierPreference, MidConversationToolChange, OutputEffort, ResponseInclusion,
    ServerFallback, ServerFallbacks, TextEditorToolOptions, ThinkingDisplay, TokenTaskBudget,
    ToolCaller, UserLocation, WebFetchToolOptions, WebSearchToolOptions,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests;
