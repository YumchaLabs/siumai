use crate::provider_options::{BedrockChatOptions, BedrockRerankOptions};

/// Typed chat request helpers for Amazon Bedrock.
pub trait BedrockChatRequestExt {
    /// Store typed options under `provider_options_map["bedrock"]`.
    fn with_bedrock_chat_options(self, options: BedrockChatOptions) -> Self;
}

impl BedrockChatRequestExt for crate::types::ChatRequest {
    fn with_bedrock_chat_options(mut self, options: BedrockChatOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize BedrockChatOptions");
        self.provider_options_map.insert("bedrock", value);
        self
    }
}

/// Typed rerank request helpers for Amazon Bedrock.
pub trait BedrockRerankRequestExt {
    /// Store typed options under `provider_options_map["bedrock"]`.
    fn with_bedrock_rerank_options(self, options: BedrockRerankOptions) -> Self;
}

impl BedrockRerankRequestExt for crate::types::RerankRequest {
    fn with_bedrock_rerank_options(mut self, options: BedrockRerankOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize BedrockRerankOptions");
        self.provider_options_map.insert("bedrock", value);
        self
    }
}
