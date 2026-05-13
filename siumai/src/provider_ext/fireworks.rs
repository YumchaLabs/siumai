/// Lower-level Fireworks text-family compat client/config aliases.
///
/// These map to the shared OpenAI-compatible runtime used by Fireworks chat,
/// completion, embedding, and transcription lanes. For the unified AI SDK-style provider
/// surface that also owns image generation/edit routing, use [`fireworks()`],
/// [`create_fireworks()`], [`crate::Provider::fireworks()`], or
/// [`crate::provider::SiumaiBuilder::fireworks()`].
pub use siumai_provider_openai_compatible::providers::openai_compatible::{
    FIREWORKS_VERSION as VERSION, FireworksClient, FireworksConfig, FireworksEmbeddingModelId,
    FireworksErrorData, FireworksImageModelId, FireworksProviderSettings,
};

/// Curated Fireworks model constants aligned with the audited AI SDK package subset.
pub mod models {
    pub use siumai_provider_openai_compatible::providers::openai_compatible::fireworks::{
        self, chat, completion, embedding, image,
    };
}

/// Create the unified Fireworks provider builder.
///
/// This mirrors the AI SDK package-level `fireworks` export more closely than the lower
/// level `FireworksClient`/`FireworksConfig` compat aliases.
pub fn fireworks() -> crate::provider::SiumaiBuilder {
    crate::Provider::fireworks()
}

/// Create the unified Fireworks provider builder.
///
/// This is the Rust package-surface analogue of AI SDK `createFireworks()`.
pub fn create_fireworks() -> crate::provider::SiumaiBuilder {
    fireworks()
}

/// Typed provider options (`provider_options_map["fireworks"]`).
pub mod options {
    #[allow(deprecated)]
    pub use siumai_provider_openai_compatible::provider_options::{
        FireworksChatOptions, FireworksEmbeddingModelOptions, FireworksEmbeddingProviderOptions,
        FireworksLanguageModelOptions, FireworksProviderOptions, FireworksReasoningHistory,
        FireworksThinkingConfig, FireworksThinkingType,
    };
    pub use siumai_provider_openai_compatible::providers::openai_compatible::ext::FireworksChatRequestExt;
}

pub use models::{chat, completion, embedding, fireworks as model_sets, image};
#[allow(deprecated)]
pub use options::{
    FireworksChatOptions, FireworksChatRequestExt, FireworksEmbeddingModelOptions,
    FireworksEmbeddingProviderOptions, FireworksLanguageModelOptions, FireworksProviderOptions,
    FireworksReasoningHistory, FireworksThinkingConfig, FireworksThinkingType,
};
