//! Provider-owned typed option structs for Cohere.

pub mod cohere;

pub use cohere::{
    CohereEmbeddingInputType, CohereEmbeddingOptions, CohereEmbeddingTruncate, CohereRerankOptions,
};
