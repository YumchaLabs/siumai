//! Provider-owned typed option structs for Amazon Bedrock.

pub mod bedrock;

#[allow(deprecated)]
pub use bedrock::{
    AmazonBedrockEmbeddingModelOptions, AmazonBedrockLanguageModelOptions,
    AmazonBedrockRerankingModelOptions, BedrockCachePoint, BedrockCachePointConfig,
    BedrockCachePointType, BedrockCacheTtl, BedrockChatOptions, BedrockEmbeddingInputType,
    BedrockEmbeddingOptions, BedrockEmbeddingPurpose, BedrockEmbeddingTruncate,
    BedrockFilePartCitations, BedrockFilePartProviderOptions, BedrockProviderOptions,
    BedrockReasoningConfig, BedrockReasoningEffort, BedrockReasoningType, BedrockRerankOptions,
    BedrockRerankingOptions, BedrockServiceTier,
};
