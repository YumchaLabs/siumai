/// Lower-level Mistral text-family compat client/config aliases.
///
/// These map to the shared OpenAI-compatible runtime used by the audited Mistral
/// `chat` and `embedding` lanes. For the unified AI SDK-style provider surface, use
/// [`mistral()`], [`create_mistral()`], [`crate::Provider::mistral()`], or
/// [`crate::provider::SiumaiBuilder::mistral()`].
pub use siumai_provider_openai_compatible::providers::openai_compatible::{
    MISTRAL_VERSION as VERSION, MistralClient, MistralConfig, MistralProviderSettings,
};

/// Curated Mistral model constants aligned with the audited AI SDK package subset.
pub mod models {
    pub use siumai_provider_openai_compatible::providers::openai_compatible::mistral::{
        self, chat, embedding,
    };
}

/// Create the unified Mistral provider builder.
///
/// This mirrors the AI SDK package-level `mistral` export more closely than the lower
/// level `MistralClient`/`MistralConfig` compat aliases.
pub fn mistral() -> crate::provider::SiumaiBuilder {
    crate::Provider::mistral()
}

/// Create the unified Mistral provider builder.
///
/// This is the Rust package-surface analogue of AI SDK `createMistral()`.
pub fn create_mistral() -> crate::provider::SiumaiBuilder {
    mistral()
}

/// Typed provider options (`provider_options_map["mistral"]`).
pub mod options {
    pub use siumai_provider_openai_compatible::provider_options::{
        MistralChatOptions, MistralLanguageModelOptions, MistralReasoningEffort,
    };
    pub use siumai_provider_openai_compatible::providers::openai_compatible::ext::MistralChatRequestExt;
}

pub use models::{chat, embedding, mistral as model_sets};
pub use options::{
    MistralChatOptions, MistralChatRequestExt, MistralLanguageModelOptions, MistralReasoningEffort,
};
