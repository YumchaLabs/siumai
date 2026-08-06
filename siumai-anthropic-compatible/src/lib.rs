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
mod policy;
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
    CacheControlWireStyle, MessagesEncodingRuleError, MessagesEncodingRules, MessagesServiceTier,
    MidConversationSystemEncoding, TemperatureEncodingRule,
};

#[cfg(test)]
mod tests;
