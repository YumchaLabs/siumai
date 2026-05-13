/// Lower-level DeepInfra text-family compat client/config aliases.
///
/// These map to the shared OpenAI-compatible runtime used by DeepInfra chat,
/// completion, and embedding lanes. For the unified AI SDK-style provider surface
/// that also owns image generation/edit routing, use [`deepinfra()`],
/// [`create_deepinfra()`], [`crate::Provider::deepinfra()`], or
/// [`crate::provider::SiumaiBuilder::deepinfra()`].
pub use siumai_provider_openai_compatible::providers::openai_compatible::{
    DEEPINFRA_VERSION as VERSION, DeepInfraChatModelId, DeepInfraClient,
    DeepInfraCompletionModelId, DeepInfraConfig, DeepInfraEmbeddingModelId, DeepInfraErrorData,
    DeepInfraImageModelId, DeepInfraProviderSettings,
};

/// Curated DeepInfra model constants aligned with the audited AI SDK package subset.
pub mod models {
    pub use siumai_provider_openai_compatible::providers::openai_compatible::deepinfra::{
        self, chat, completion, embedding, image,
    };
}

/// Create the unified DeepInfra provider builder.
///
/// This mirrors the AI SDK package-level `deepinfra` export more closely than the lower
/// level `DeepInfraClient`/`DeepInfraConfig` compat aliases.
pub fn deepinfra() -> crate::provider::SiumaiBuilder {
    crate::Provider::deepinfra()
}

/// Create the unified DeepInfra provider builder.
///
/// This is the Rust package-surface analogue of AI SDK `createDeepInfra()`.
pub fn create_deepinfra() -> crate::provider::SiumaiBuilder {
    deepinfra()
}

pub use models::{chat, completion, deepinfra as model_sets, embedding, image};
