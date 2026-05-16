pub use siumai_provider_deepseek::providers::deepseek::{
    DeepSeekBuilder, DeepSeekClient, DeepSeekConfig, DeepSeekErrorData, DeepSeekProviderSettings,
    VERSION,
};

/// Create the DeepSeek provider builder.
pub fn deepseek() -> DeepSeekBuilder {
    crate::compat::Provider::deepseek()
}

/// Create the DeepSeek provider builder.
///
/// This is the Rust package-surface analogue of AI SDK `createDeepSeek()`.
pub fn create_deepseek() -> DeepSeekBuilder {
    deepseek()
}

/// Curated DeepSeek model constants aligned with the audited AI SDK chat surface.
pub mod models {
    pub use siumai_provider_deepseek::providers::deepseek::models::{self as model_sets, chat};
}

/// Typed response metadata helpers (`ChatResponse.provider_metadata["deepseek"]`).
pub mod metadata {
    pub use siumai_provider_deepseek::provider_metadata::deepseek::{
        DeepSeekChatResponseExt, DeepSeekMetadata, DeepSeekSource, DeepSeekSourceExt,
        DeepSeekSourceMetadata,
    };
}

pub use metadata::{
    DeepSeekChatResponseExt, DeepSeekMetadata, DeepSeekSource, DeepSeekSourceExt,
    DeepSeekSourceMetadata,
};

/// Typed provider options (`provider_options_map["deepseek"]`).
pub mod options {
    #[allow(deprecated)]
    pub use siumai_provider_deepseek::provider_options::{
        DeepSeekChatOptions, DeepSeekLanguageModelOptions,
    };
    pub use siumai_provider_deepseek::providers::deepseek::DeepSeekOptions;
    pub use siumai_provider_deepseek::providers::deepseek::ext::DeepSeekChatRequestExt;
}

pub use models::{chat, model_sets};
#[allow(deprecated)]
pub use options::{
    DeepSeekChatOptions, DeepSeekChatRequestExt, DeepSeekLanguageModelOptions, DeepSeekOptions,
};

/// Non-unified DeepSeek extension APIs (escape hatches).
pub mod ext {
    pub use siumai_provider_deepseek::providers::deepseek::ext::*;
}
