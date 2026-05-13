pub use siumai_provider_xai::providers::xai::{
    VERSION, XaiBuilder, XaiClient, XaiConfig, XaiErrorData, XaiProviderSettings, XaiVideoModelId,
};

/// Create the xAI provider builder.
pub fn xai() -> XaiBuilder {
    crate::Provider::xai()
}

/// Create the xAI provider builder.
///
/// This is the Rust package-surface analogue of AI SDK `createXai()`.
pub fn create_xai() -> XaiBuilder {
    xai()
}

/// Typed response metadata helpers (`ChatResponse.provider_metadata["xai"]`).
pub mod metadata {
    pub use siumai_provider_xai::provider_metadata::xai::{
        XaiChatResponseExt, XaiMetadata, XaiSource, XaiSourceExt, XaiSourceMetadata,
    };
}

pub use metadata::{XaiChatResponseExt, XaiMetadata, XaiSource, XaiSourceExt, XaiSourceMetadata};

/// Provider tool factories that return `Tool` directly (Vercel-aligned).
pub mod tools {
    pub use crate::tools::xai::*;
}

/// Vercel-style provider tool factories that return `Tool` directly.
pub mod provider_tools {
    pub use crate::tools::xai::*;
}

/// Typed provider options (`provider_options_map["xai"]`).
pub mod options {
    #[allow(deprecated)]
    pub use siumai_provider_xai::providers::xai::ext::{
        XaiChatRequestExt, XaiImageRequestExt, XaiTtsRequestExt, XaiVideoRequestExt,
    };
    #[allow(deprecated)]
    pub use siumai_provider_xai::providers::xai::{
        NewsSearchSource, RssSearchSource, SearchMode, SearchSource, WebSearchSource,
        XSearchSource, XaiChatOptions, XaiChatReasoningEffort, XaiFilesOptions,
        XaiImageModelOptions, XaiImageOptions, XaiImageProviderOptions, XaiImageQuality,
        XaiImageResolution, XaiLanguageModelChatOptions, XaiLanguageModelResponsesOptions,
        XaiOptions, XaiProviderOptions, XaiReasoningSummary, XaiResponseInclude,
        XaiResponsesOptions, XaiResponsesProviderOptions, XaiResponsesReasoningEffort,
        XaiSearchParameters, XaiTtsOptions, XaiVideoMode, XaiVideoModelOptions, XaiVideoOptions,
        XaiVideoProviderOptions, XaiVideoResolution,
    };
}

// Provider-owned typed options (kept out of `siumai-core`).
#[allow(deprecated)]
pub use options::{
    NewsSearchSource, RssSearchSource, SearchMode, SearchSource, WebSearchSource, XSearchSource,
    XaiChatOptions, XaiChatReasoningEffort, XaiChatRequestExt, XaiFilesOptions,
    XaiImageModelOptions, XaiImageOptions, XaiImageProviderOptions, XaiImageQuality,
    XaiImageRequestExt, XaiImageResolution, XaiLanguageModelChatOptions,
    XaiLanguageModelResponsesOptions, XaiOptions, XaiProviderOptions, XaiReasoningSummary,
    XaiResponseInclude, XaiResponsesOptions, XaiResponsesProviderOptions,
    XaiResponsesReasoningEffort, XaiSearchParameters, XaiTtsOptions, XaiTtsRequestExt,
    XaiVideoMode, XaiVideoModelOptions, XaiVideoOptions, XaiVideoProviderOptions,
    XaiVideoRequestExt, XaiVideoResolution,
};

/// Non-unified xAI extension APIs (escape hatches).
pub mod ext {
    pub use siumai_provider_xai::providers::xai::ext::*;
}
