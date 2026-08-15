//! Curated Cohere embedding, rerank, and audio transcription provider facade.

pub use siumai_provider_cohere::{
    CohereConfigError, CohereEmbeddingModel, CohereProfile, CohereProfileError, CohereProvider,
    CohereProviderBuilder, CohereRerankModel, CohereTranscriptionRequest, CohereTranscriptions,
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
