use crate::provider_options::CohereRerankOptions;

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
