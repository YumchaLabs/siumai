/// Lower-level MoonshotAI text-family compat client/config aliases.
///
/// These map to the shared OpenAI-compatible runtime used by the audited MoonshotAI
/// chat-only wrapper. For the unified AI SDK-style provider surface, use [`moonshotai()`],
/// [`create_moonshotai()`], [`crate::compat::Provider::moonshotai()`], or
/// [`SiumaiBuilder::moonshotai()`].
pub use siumai_provider_openai_compatible::providers::openai_compatible::{
    MOONSHOTAI_VERSION as VERSION, MoonshotAIChatModelId, MoonshotAIClient, MoonshotAIConfig,
    MoonshotAIProviderSettings,
};
use siumai_registry::provider::SiumaiBuilder;

/// Curated MoonshotAI model constants aligned with the audited AI SDK package subset.
pub mod models {
    pub use siumai_provider_openai_compatible::providers::openai_compatible::moonshotai::{
        self, recommended,
    };
}

/// Create the unified MoonshotAI provider builder.
///
/// This mirrors the AI SDK package-level `moonshotai` export more closely than the lower
/// level `MoonshotAIClient`/`MoonshotAIConfig` compat aliases.
pub fn moonshotai() -> SiumaiBuilder {
    SiumaiBuilder::new().moonshotai()
}

/// Create the unified MoonshotAI provider builder.
///
/// This is the Rust package-surface analogue of AI SDK `createMoonshotAI()`.
pub fn create_moonshotai() -> SiumaiBuilder {
    moonshotai()
}

/// Typed provider options (`provider_options_map["moonshotai"]`).
pub mod options {
    #[allow(deprecated)]
    pub use siumai_provider_openai_compatible::provider_options::{
        MoonshotAIChatOptions, MoonshotAILanguageModelOptions, MoonshotAIProviderOptions,
        MoonshotAIReasoningHistory, MoonshotAIThinkingConfig, MoonshotAIThinkingType,
    };
    pub use siumai_provider_openai_compatible::providers::openai_compatible::ext::MoonshotAIChatRequestExt;
}

pub use models::{moonshotai as model_sets, recommended};
#[allow(deprecated)]
pub use options::{
    MoonshotAIChatOptions, MoonshotAIChatRequestExt, MoonshotAILanguageModelOptions,
    MoonshotAIProviderOptions, MoonshotAIReasoningHistory, MoonshotAIThinkingConfig,
    MoonshotAIThinkingType,
};
