/// Lower-level Perplexity text-family compat client/config aliases.
///
/// These map to the shared OpenAI-compatible runtime used by the audited Perplexity chat
/// wrapper. For the unified AI SDK-style provider surface, use [`perplexity()`],
/// [`create_perplexity()`], [`crate::Provider::perplexity()`], or
/// [`crate::provider::SiumaiBuilder::perplexity()`].
pub use siumai_provider_openai_compatible::providers::openai_compatible::{
    PERPLEXITY_VERSION as VERSION, PerplexityClient, PerplexityConfig, PerplexityProviderSettings,
};

/// Curated Perplexity model constants aligned with the audited AI SDK package subset.
pub mod models {
    pub use siumai_provider_openai_compatible::providers::openai_compatible::perplexity::{
        self, chat,
    };
}

/// Create the unified Perplexity provider builder.
///
/// This mirrors the AI SDK package-level `perplexity` export more closely than the lower
/// level `PerplexityClient`/`PerplexityConfig` compat aliases.
pub fn perplexity() -> crate::provider::SiumaiBuilder {
    crate::Provider::perplexity()
}

/// Create the unified Perplexity provider builder.
///
/// This is the Rust package-surface analogue of AI SDK `createPerplexity()`.
pub fn create_perplexity() -> crate::provider::SiumaiBuilder {
    perplexity()
}

/// Typed response metadata helpers (`ChatResponse.provider_metadata["perplexity"]`).
pub mod metadata {
    pub use siumai_provider_openai_compatible::providers::openai_compatible::{
        PerplexityChatResponseExt, PerplexityImage, PerplexityMetadata, PerplexityUsage,
    };
}

/// Typed provider options (`provider_options_map["perplexity"]`).
pub mod options {
    pub use siumai_provider_openai_compatible::provider_options::{
        PerplexityOptions, PerplexitySearchContextSize, PerplexitySearchMode,
        PerplexitySearchRecencyFilter, PerplexityUserLocation, PerplexityWebSearchOptions,
    };
    pub use siumai_provider_openai_compatible::providers::openai_compatible::ext::PerplexityChatRequestExt;
}

pub use metadata::{
    PerplexityChatResponseExt, PerplexityImage, PerplexityMetadata, PerplexityUsage,
};
pub use models::{chat, perplexity as model_sets};
pub use options::{
    PerplexityChatRequestExt, PerplexityOptions, PerplexitySearchContextSize, PerplexitySearchMode,
    PerplexitySearchRecencyFilter, PerplexityUserLocation, PerplexityWebSearchOptions,
};
