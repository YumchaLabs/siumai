/// Typed response metadata helpers (`ChatResponse.provider_metadata["openrouter"]`).
pub mod metadata {
    pub use siumai_provider_openai_compatible::providers::openai_compatible::{
        OpenRouterChatResponseExt, OpenRouterContentPartExt, OpenRouterContentPartMetadata,
        OpenRouterMetadata, OpenRouterSource, OpenRouterSourceExt, OpenRouterSourceMetadata,
    };
}

/// Typed provider options (`provider_options_map["openrouter"]`).
pub mod options {
    pub use siumai_provider_openai_compatible::provider_options::{
        OpenRouterOptions, OpenRouterTransform,
    };
    pub use siumai_provider_openai_compatible::providers::openai_compatible::ext::OpenRouterChatRequestExt;
}

pub use metadata::{
    OpenRouterChatResponseExt, OpenRouterContentPartExt, OpenRouterContentPartMetadata,
    OpenRouterMetadata, OpenRouterSource, OpenRouterSourceExt, OpenRouterSourceMetadata,
};
pub use options::{OpenRouterChatRequestExt, OpenRouterOptions, OpenRouterTransform};
