//! OpenAI-compatible vendor extension helpers.

pub mod metadata;
pub mod request_options;

pub use metadata::{
    OpenRouterChatResponseExt, OpenRouterContentPartExt, OpenRouterContentPartMetadata,
    OpenRouterMetadata, OpenRouterSource, OpenRouterSourceExt, OpenRouterSourceMetadata,
    PerplexityChatResponseExt, PerplexityCost, PerplexityImage, PerplexityMetadata,
    PerplexityUsage,
};
pub use request_options::{
    AlibabaChatRequestExt, DeepSeekChatRequestExt, FireworksChatRequestExt, GroqChatRequestExt,
    GroqTranscriptionRequestExt, MistralChatRequestExt, MoonshotAIChatRequestExt,
    OpenAiCompatibleChatRequestExt, OpenAiCompatibleCompletionRequestExt,
    OpenAiCompatibleEmbeddingRequestExt, OpenRouterChatRequestExt, PerplexityChatRequestExt,
    QwenChatRequestExt, TogetherAIImageRequestExt, TogetherAIRerankRequestExt, XaiChatRequestExt,
    XaiImageRequestExt,
};
