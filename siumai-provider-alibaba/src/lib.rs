//! Rust-first Alibaba Cloud Model Studio provider.
//!
//! Alibaba is the public provider identity. The implementation may use DashScope
//! endpoints and native wire protocols internally, but those service details do
//! not create a second provider, route, region, or model-catalog API.

#![deny(unsafe_code)]

mod embedding;
mod language;
mod native_error;
pub mod options;
mod provider;
mod video;

pub use embedding::{
    AlibabaEmbeddingModel, AlibabaEmbeddingOptions, AlibabaEmbeddingOutputType,
    AlibabaEmbeddingProfileError, AlibabaEmbeddingTextType, EMBEDDING_API_MODE_ID,
    EMBEDDING_PROTOCOL_ID, EMBEDDING_SOURCE, EMBEDDING_VERIFIED_ON,
    LEGACY_SINGAPORE_EMBEDDING_BASE_URL, TEXT_EMBEDDING_V3, TEXT_EMBEDDING_V4,
};
pub use language::{
    CHAT_SOURCE, LEGACY_SINGAPORE_LANGUAGE_BASE_URL, PLATFORM_ID, PROVIDER_ID, RESPONSES_SOURCE,
    VERIFIED_ON,
};
pub use options::{
    ALIBABA_SESSION_CACHE_HEADER, AlibabaChatOptions, AlibabaPromptCacheBreakpoint,
    AlibabaReasoningEffort, AlibabaResponsesOptions, AlibabaResponsesTool, AlibabaSearchOptions,
    AlibabaSearchStrategy,
};
pub use provider::{
    AlibabaConfigError, AlibabaCredential, AlibabaLanguageApi, AlibabaLanguageModel,
    AlibabaProvider, AlibabaProviderBuilder, AlibabaWorkspaceEndpoint,
    AlibabaWorkspaceEndpointError, LEGACY_SINGAPORE_ORIGIN,
};
pub use siumai_openai_compatible::{
    BearerCredential, CredentialRequest, CredentialSourceError, DynamicCredentialSource,
};

/// Explicit opt-in surface for the unstable asynchronous video-job contract.
pub mod experimental {
    pub use crate::provider::{AlibabaVideoProviderBuilderExt, AlibabaVideoProviderExt};
    pub use crate::video::{
        AlibabaVideoDownloadPolicy, AlibabaVideoJob, AlibabaVideoMedia, AlibabaVideoMediaType,
        AlibabaVideoModel, AlibabaVideoParameters, AlibabaVideoRequest, AlibabaVideoRequestError,
        AlibabaVideoShotType, AlibabaVideoUsage, LEGACY_SINGAPORE_VIDEO_BASE_URL,
        VIDEO_API_MODE_ID, VIDEO_CANCEL_SOURCE, VIDEO_IMAGE_SOURCE, VIDEO_PROTOCOL_ID,
        VIDEO_REFERENCE_SOURCE, VIDEO_TEXT_SOURCE, WAN_2_7_I2V, WAN_2_7_I2V_SNAPSHOT, WAN_2_7_R2V,
        WAN_2_7_R2V_SNAPSHOT, WAN_2_7_T2V, WAN_2_7_T2V_SNAPSHOT,
    };
}
