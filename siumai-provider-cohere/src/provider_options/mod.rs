//! Provider-owned typed option structs for Cohere.

pub mod cohere;

#[allow(deprecated)]
pub use cohere::{
    CohereChatModelOptions, CohereChatOptions, CohereEmbeddingInputType,
    CohereEmbeddingModelOptions, CohereEmbeddingOptions, CohereEmbeddingTruncate,
    CohereLanguageModelOptions, CohereRerankOptions, CohereRerankingModelOptions,
    CohereRerankingOptions, CohereThinkingConfig, CohereThinkingType,
};
