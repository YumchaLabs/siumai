//! OpenAI-compatible vendor presets (OpenAI-like providers).
//!
//! This module centralizes vendor identifiers and re-exports the OpenAI-compatible
//! adapter-based client/builder so users can treat "OpenAI" as a protocol family.

/// A stable, compile-time vendor id.
///
/// This is intentionally small and string-backed so it can be used in const contexts
/// and does not require allocating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpenAiVendorId(&'static str);

impl OpenAiVendorId {
    pub const SILICONFLOW: Self = Self("siliconflow");
    pub const DEEPSEEK: Self = Self("deepseek");
    pub const OPENROUTER: Self = Self("openrouter");
    pub const TOGETHER: Self = Self("together");
    pub const FIREWORKS: Self = Self("fireworks");
    pub const GITHUB_COPILOT: Self = Self("github_copilot");
    pub const PERPLEXITY: Self = Self("perplexity");

    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

pub use crate::providers::openai_compatible::{
    OpenAiCompatibleBuilder as OpenAiVendorBuilder,
    OpenAiCompatibleClient as OpenAiVendorClient,
    ProviderConfig,
    get_provider_config,
    list_provider_ids,
    provider_supports_capability,
};

pub use crate::providers::openai_compatible::providers::models;

