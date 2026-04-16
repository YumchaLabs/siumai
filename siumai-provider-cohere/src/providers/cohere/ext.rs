use crate::provider_options::{CohereChatOptions, CohereEmbeddingOptions, CohereRerankOptions};

/// Typed chat request helpers for Cohere.
pub trait CohereChatRequestExt {
    /// Store typed options under `provider_options_map["cohere"]`.
    fn with_cohere_options(self, options: CohereChatOptions) -> Self;
}

impl CohereChatRequestExt for crate::types::ChatRequest {
    fn with_cohere_options(mut self, options: CohereChatOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize CohereChatOptions");
        self.provider_options_map.insert("cohere", value);
        self
    }
}

/// Typed embedding request helpers for Cohere.
pub trait CohereEmbeddingRequestExt {
    /// Store typed options under `provider_options_map["cohere"]`.
    fn with_cohere_options(self, options: CohereEmbeddingOptions) -> Self;
}

impl CohereEmbeddingRequestExt for crate::types::EmbeddingRequest {
    fn with_cohere_options(mut self, options: CohereEmbeddingOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize CohereEmbeddingOptions");
        self.provider_options_map.insert("cohere", value);
        self
    }
}

/// Typed rerank request helpers for Cohere.
pub trait CohereRerankRequestExt {
    /// Store typed options under `provider_options_map["cohere"]`.
    fn with_cohere_options(self, options: CohereRerankOptions) -> Self;
}

impl CohereRerankRequestExt for crate::types::RerankRequest {
    fn with_cohere_options(mut self, options: CohereRerankOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize CohereRerankOptions");
        self.provider_options_map.insert("cohere", value);
        self
    }
}
