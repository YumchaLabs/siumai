//! Configured execution engine for Anthropic Messages-compatible APIs.
//!
//! This crate owns reusable protocol execution, endpoint isolation, authentication,
//! option merging, and evidence-backed compatibility profiles. It deliberately does
//! not own any branded provider identity, provider annotation namespace, or native
//! resource API.
#![deny(unsafe_code)]

mod auth;
mod model;
mod options;
mod profile;
mod projection;
mod provider;

pub use auth::{AnthropicCompatibleCredential, CredentialError};
pub use model::AnthropicCompatibleLanguageModel;
pub use options::MessagesCallOptions;
pub use profile::{
    AnthropicCompatibleProfile, MessagesRequestPolicy, MessagesRequestRequirements,
    NoMessagesRequestPolicy,
};
pub use projection::{
    MessagesRequestProjection, MessagesRequestProjectionContext, NativeMessagesRequestProjection,
    ProjectedMessagesRequest,
};
pub use provider::{
    AnthropicCompatibleConfigError, AnthropicCompatibleProvider, AnthropicCompatibleProviderBuilder,
};
pub use siumai_protocol_anthropic::messages::{
    CacheControl, CacheControlWireStyle, CacheTtl, ClearThinkingEdit, ClearThinkingKeep,
    ClearToolInputs, ClearToolUsesEdit, CompactionEdit, ContainerSkill, ContainerSkillType,
    ContextManagement, ContextManagementEdit, ContextManagementTrigger, InferenceGeo,
    InferenceSpeed, McpAuthorizationToken, McpServer, MessagesAssignedServiceTier,
    MessagesContainer, MessagesEncodingRuleError, MessagesEncodingRules,
    MessagesServiceTierPreference, MessagesTokenCountOptions, MidConversationSystemEncoding,
    TemperatureEncodingRule, TokenTaskBudget,
};

#[cfg(test)]
mod tests;
