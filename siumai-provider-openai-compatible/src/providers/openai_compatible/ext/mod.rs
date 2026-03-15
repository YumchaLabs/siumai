//! OpenAI-compatible vendor extension helpers.

pub mod metadata;
pub mod request_options;

pub use metadata::{
    OpenRouterChatResponseExt, OpenRouterContentPartExt, OpenRouterContentPartMetadata,
    OpenRouterMetadata, OpenRouterSource, OpenRouterSourceExt, OpenRouterSourceMetadata,
    PerplexityChatResponseExt, PerplexityImage, PerplexityMetadata, PerplexityUsage,
};
pub use request_options::{OpenRouterChatRequestExt, PerplexityChatRequestExt};
