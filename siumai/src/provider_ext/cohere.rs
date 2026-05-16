pub use siumai_provider_cohere::providers::cohere::{
    CohereBuilder, CohereClient, CohereConfig, CohereProviderSettings, VERSION,
};

/// Create the Cohere provider builder.
pub fn cohere() -> CohereBuilder {
    crate::compat::Provider::cohere()
}

/// Create the Cohere provider builder.
///
/// This is the Rust package-surface analogue of AI SDK `createCohere()`.
pub fn create_cohere() -> CohereBuilder {
    cohere()
}

pub mod models {
    pub use siumai_provider_cohere::providers::cohere::models::{
        self as model_sets, chat, embedding, rerank,
    };
}

/// Typed provider options (`provider_options_map["cohere"]`).
pub mod options {
    #[allow(deprecated)]
    pub use siumai_provider_cohere::provider_options::{
        CohereChatModelOptions, CohereChatOptions, CohereEmbeddingInputType,
        CohereEmbeddingModelOptions, CohereEmbeddingOptions, CohereEmbeddingTruncate,
        CohereLanguageModelOptions, CohereRerankOptions, CohereRerankingModelOptions,
        CohereRerankingOptions, CohereThinkingConfig, CohereThinkingType,
    };
    pub use siumai_provider_cohere::providers::cohere::{
        CohereChatRequestExt, CohereEmbeddingRequestExt, CohereRerankRequestExt,
    };
}

pub use models::{chat, embedding, model_sets, rerank};
#[allow(deprecated)]
pub use options::{
    CohereChatModelOptions, CohereChatOptions, CohereChatRequestExt, CohereEmbeddingInputType,
    CohereEmbeddingModelOptions, CohereEmbeddingOptions, CohereEmbeddingRequestExt,
    CohereEmbeddingTruncate, CohereLanguageModelOptions, CohereRerankOptions,
    CohereRerankRequestExt, CohereRerankingModelOptions, CohereRerankingOptions,
    CohereThinkingConfig, CohereThinkingType,
};
