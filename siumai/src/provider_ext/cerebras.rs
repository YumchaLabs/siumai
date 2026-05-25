/// Lower-level Cerebras text-family compat client/config aliases.
///
/// These map to the shared OpenAI-compatible runtime used by the audited Cerebras
/// chat/language-model lane. For the unified AI SDK-style provider surface, use
/// [`cerebras()`], [`create_cerebras()`], [`crate::compat::Provider::cerebras()`], or
/// [`SiumaiBuilder::cerebras()`].
pub use siumai_provider_openai_compatible::providers::openai_compatible::{
    CEREBRAS_VERSION as VERSION, CerebrasChatModelId, CerebrasClient, CerebrasConfig,
    CerebrasProviderSettings,
};
use siumai_registry::provider::SiumaiBuilder;

/// Curated Cerebras model constants aligned with the audited AI SDK package subset.
pub mod models {
    pub use siumai_provider_openai_compatible::providers::openai_compatible::cerebras::{
        self, chat,
    };
}

/// Create the unified Cerebras provider builder.
///
/// This mirrors the AI SDK package-level `cerebras` export while continuing to reuse the shared
/// OpenAI-compatible runtime internally.
pub fn cerebras() -> SiumaiBuilder {
    SiumaiBuilder::new().cerebras()
}

/// Create the unified Cerebras provider builder.
///
/// This is the Rust package-surface analogue of AI SDK `createCerebras()`.
pub fn create_cerebras() -> SiumaiBuilder {
    cerebras()
}

pub use models::{cerebras as model_sets, chat};
