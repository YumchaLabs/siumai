pub use siumai_provider_groq::providers::groq::{
    GroqBuilder, GroqClient, GroqConfig, GroqProviderSettings, VERSION,
};

/// Create the Groq provider builder.
pub fn groq() -> GroqBuilder {
    crate::Provider::groq()
}

/// Create the Groq provider builder.
///
/// This is the Rust package-surface analogue of AI SDK `createGroq()`.
pub fn create_groq() -> GroqBuilder {
    groq()
}

/// Typed response metadata helpers (`ChatResponse.provider_metadata["groq"]`).
pub mod metadata {
    pub use siumai_provider_groq::provider_metadata::groq::{
        GroqChatResponseExt, GroqMetadata, GroqSource, GroqSourceExt, GroqSourceMetadata,
    };
}

pub use metadata::{
    GroqChatResponseExt, GroqMetadata, GroqSource, GroqSourceExt, GroqSourceMetadata,
};

/// Provider tool factories that return `Tool` directly (Vercel-aligned).
pub mod tools {
    pub use crate::tools::groq::*;
}

/// Vercel-style provider tool factories that return `Tool` directly.
pub mod provider_tools {
    pub use crate::tools::groq::*;
}

/// Typed provider options (`provider_options_map["groq"]`).
pub mod options {
    #[allow(deprecated)]
    pub use siumai_provider_groq::provider_options::{
        GroqLanguageModelOptions, GroqOptions, GroqProviderOptions, GroqReasoningEffort,
        GroqReasoningFormat, GroqServiceTier, GroqTranscriptionModelOptions,
    };
    pub use siumai_provider_groq::providers::groq::{GroqChatRequestExt, GroqSttRequestExt};
}

// Provider-owned typed options (kept out of `siumai-core`).
#[allow(deprecated)]
pub use options::{
    GroqChatRequestExt, GroqLanguageModelOptions, GroqOptions, GroqProviderOptions,
    GroqReasoningEffort, GroqReasoningFormat, GroqServiceTier, GroqSttRequestExt,
    GroqTranscriptionModelOptions,
};

/// Non-unified Groq extension APIs (escape hatches).
pub mod ext {
    pub use siumai_provider_groq::providers::groq::ext::*;
}
