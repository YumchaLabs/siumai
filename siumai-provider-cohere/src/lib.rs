//! siumai-provider-cohere
//!
//! Cohere embedding, rerank, and audio transcription provider for siumai.
#![deny(unsafe_code)]

mod configured;
pub mod models;

pub use configured::{
    CohereConfigError, CohereEmbeddingModel, CohereProfile, CohereProfileError, CohereProvider,
    CohereProviderBuilder, CohereRerankModel, CohereTranscriptionRequest, CohereTranscriptions,
};

/// Provider-owned typed option structs (Cohere-specific).
pub mod provider_options;
