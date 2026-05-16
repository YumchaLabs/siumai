pub use siumai_provider_togetherai::providers::togetherai::{
    TogetherAIErrorData, TogetherAIProviderSettings, TogetherAiBuilder, TogetherAiClient,
    TogetherAiConfig, VERSION,
};
use siumai_registry::provider::SiumaiBuilder;

/// Create the unified TogetherAI provider builder.
pub fn togetherai() -> SiumaiBuilder {
    SiumaiBuilder::new().togetherai()
}

/// Create the unified TogetherAI provider builder.
///
/// This is the Rust package-surface analogue of AI SDK `createTogetherAI()`.
pub fn create_togetherai() -> SiumaiBuilder {
    togetherai()
}

pub mod models {
    pub use siumai_provider_togetherai::providers::togetherai::models::{
        self as model_sets, chat, completion, embedding, image, rerank,
    };
}

/// Typed provider options (`provider_options_map["togetherai"]`).
pub mod options {
    #[allow(deprecated)]
    pub use siumai_provider_togetherai::provider_options::{
        TogetherAIImageModelOptions, TogetherAIImageProviderOptions,
        TogetherAIRerankingModelOptions, TogetherAIRerankingOptions, TogetherAiImageModelOptions,
        TogetherAiImageOptions, TogetherAiImageProviderOptions, TogetherAiRerankOptions,
        TogetherAiRerankingModelOptions, TogetherAiRerankingOptions,
    };
    pub use siumai_provider_togetherai::providers::togetherai::{
        TogetherAiImageRequestExt, TogetherAiRerankRequestExt,
    };
}

pub use models::{chat, completion, embedding, image, model_sets, rerank};
#[allow(deprecated)]
pub use options::{
    TogetherAIImageModelOptions, TogetherAIImageProviderOptions, TogetherAIRerankingModelOptions,
    TogetherAIRerankingOptions, TogetherAiImageModelOptions, TogetherAiImageOptions,
    TogetherAiImageProviderOptions, TogetherAiImageRequestExt, TogetherAiRerankOptions,
    TogetherAiRerankRequestExt, TogetherAiRerankingModelOptions, TogetherAiRerankingOptions,
};
