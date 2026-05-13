pub use siumai_provider_amazon_bedrock::providers::bedrock::{
    AmazonBedrockProviderSettings, BedrockBuilder, BedrockClient, BedrockConfig, VERSION,
};

/// Create the Bedrock provider builder.
pub fn bedrock() -> BedrockBuilder {
    crate::Provider::bedrock()
}

/// Create the Bedrock provider builder.
///
/// This is the Rust package-surface analogue of AI SDK `createAmazonBedrock()`.
pub fn create_amazon_bedrock() -> BedrockBuilder {
    bedrock()
}

/// Anthropic provider tool factories re-exported on the Bedrock surface like AI SDK.
#[cfg(feature = "anthropic")]
pub mod tools {
    pub use crate::tools::anthropic::*;
}

/// Compatibility alias for older imports.
#[cfg(feature = "anthropic")]
pub mod provider_tools {
    pub use crate::tools::anthropic::*;
}

/// Typed response metadata helpers (`ChatResponse.provider_metadata["bedrock"]`).
pub mod metadata {
    pub use siumai_provider_amazon_bedrock::provider_metadata::bedrock::{
        BedrockChatResponseExt, BedrockContentPartExt, BedrockMetadata,
        BedrockReasoningContentPartMetadata,
    };
}

/// Typed provider options (`provider_options_map["bedrock"]`).
pub mod options {
    #[allow(deprecated)]
    pub use siumai_provider_amazon_bedrock::provider_options::{
        AmazonBedrockEmbeddingModelOptions, AmazonBedrockLanguageModelOptions,
        AmazonBedrockRerankingModelOptions, BedrockCachePoint, BedrockCachePointConfig,
        BedrockCachePointType, BedrockCacheTtl, BedrockChatOptions, BedrockEmbeddingInputType,
        BedrockEmbeddingOptions, BedrockEmbeddingPurpose, BedrockEmbeddingTruncate,
        BedrockFilePartCitations, BedrockFilePartProviderOptions, BedrockProviderOptions,
        BedrockReasoningConfig, BedrockReasoningEffort, BedrockReasoningType, BedrockRerankOptions,
        BedrockRerankingOptions, BedrockServiceTier,
    };
    pub use siumai_provider_amazon_bedrock::providers::bedrock::{
        BedrockChatRequestExt, BedrockEmbeddingRequestExt, BedrockMessageExt,
        BedrockRequestContentPartExt, BedrockRerankRequestExt,
    };
    #[cfg(feature = "anthropic")]
    #[allow(deprecated)]
    pub use siumai_provider_anthropic::provider_options::anthropic::AnthropicProviderOptions;
}

pub use metadata::{
    BedrockChatResponseExt, BedrockContentPartExt, BedrockMetadata,
    BedrockReasoningContentPartMetadata,
};
#[cfg(feature = "anthropic")]
#[allow(deprecated)]
pub use options::AnthropicProviderOptions;
#[allow(deprecated)]
pub use options::{
    AmazonBedrockEmbeddingModelOptions, AmazonBedrockLanguageModelOptions,
    AmazonBedrockRerankingModelOptions, BedrockCachePoint, BedrockCachePointConfig,
    BedrockCachePointType, BedrockCacheTtl, BedrockChatOptions, BedrockChatRequestExt,
    BedrockEmbeddingInputType, BedrockEmbeddingOptions, BedrockEmbeddingPurpose,
    BedrockEmbeddingRequestExt, BedrockEmbeddingTruncate, BedrockFilePartCitations,
    BedrockFilePartProviderOptions, BedrockMessageExt, BedrockProviderOptions,
    BedrockReasoningConfig, BedrockReasoningEffort, BedrockReasoningType,
    BedrockRequestContentPartExt, BedrockRerankOptions, BedrockRerankRequestExt,
    BedrockRerankingOptions, BedrockServiceTier,
};
pub use siumai_provider_amazon_bedrock::providers::bedrock::assistant_message_with_reasoning_metadata;
