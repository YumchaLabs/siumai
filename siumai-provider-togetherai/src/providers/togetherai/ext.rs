use crate::provider_options::TogetherAiRerankOptions;

/// Typed rerank request helpers for TogetherAI.
pub trait TogetherAiRerankRequestExt {
    /// Store typed options under `provider_options_map["togetherai"]`.
    fn with_togetherai_options(self, options: TogetherAiRerankOptions) -> Self;
}

impl TogetherAiRerankRequestExt for crate::types::RerankRequest {
    fn with_togetherai_options(mut self, options: TogetherAiRerankOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize TogetherAiRerankOptions");
        self.provider_options_map.insert("togetherai", value);
        self
    }
}
