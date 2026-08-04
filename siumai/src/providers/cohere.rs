//! Curated Cohere embedding and rerank provider facade.

pub use siumai_provider_cohere::{
    CohereConfigError, CohereEmbeddingModel, CohereProvider, CohereProviderBuilder,
    CohereRerankModel,
};

pub mod options {
    pub use siumai_provider_cohere::provider_options::{
        CohereEmbeddingInputType, CohereEmbeddingOptions, CohereEmbeddingTruncate,
        CohereRerankOptions,
    };
}
