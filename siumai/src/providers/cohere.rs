//! Curated Cohere embedding and rerank provider facade.

pub use siumai_provider_cohere::{
    CohereConfigError, CohereEmbeddingModel, CohereProfile, CohereProfileError, CohereProvider,
    CohereProviderBuilder, CohereRerankModel,
};

pub mod models {
    pub use siumai_provider_cohere::models::*;
}

pub mod options {
    pub use siumai_provider_cohere::provider_options::{
        CohereEmbeddingInputType, CohereEmbeddingOptions, CohereEmbeddingTruncate,
        CohereRerankOptions,
    };
}
