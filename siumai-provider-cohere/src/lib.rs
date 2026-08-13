//! siumai-provider-cohere
//!
//! Cohere embedding and rerank provider for siumai.
#![deny(unsafe_code)]

mod configured;
pub mod models;

pub use configured::{
    CohereConfigError, CohereEmbeddingModel, CohereProfile, CohereProfileError, CohereProvider,
    CohereProviderBuilder, CohereRerankModel,
};

/// Provider-owned typed option structs (Cohere-specific).
pub mod provider_options;
