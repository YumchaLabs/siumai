//! Alibaba Cloud Model Studio provider facade.

pub use siumai_provider_alibaba::{
    AlibabaConfigError, AlibabaContentCache, AlibabaCredential, AlibabaEmbeddingModel,
    AlibabaEmbeddingOptions, AlibabaEmbeddingOutputType, AlibabaEmbeddingTextType,
    AlibabaLanguageApi, AlibabaLanguageModel, AlibabaMessageCache, AlibabaMessagesOptions,
    AlibabaMessagesThinking, AlibabaPromptCacheBreakpoint, AlibabaProvider, AlibabaProviderBuilder,
    AlibabaToolCache, AlibabaWorkspaceEndpoint, AlibabaWorkspaceEndpointError, BearerCredential,
    CredentialRequest, CredentialSourceError, DynamicCredentialSource, EMBEDDING_API_MODE_ID,
    EMBEDDING_PROTOCOL_ID, EMBEDDING_SOURCE, EMBEDDING_VERIFIED_ON,
    LEGACY_SINGAPORE_EMBEDDING_BASE_URL, LEGACY_SINGAPORE_LANGUAGE_BASE_URL,
    LEGACY_SINGAPORE_MESSAGES_BASE_URL, LEGACY_SINGAPORE_ORIGIN, MESSAGES_SOURCE,
    MESSAGES_VERIFIED_ON,
};

/// Explicit opt-in surface for Alibaba's unstable asynchronous video jobs.
pub mod experimental {
    pub use siumai_provider_alibaba::experimental::{
        AlibabaVideoDownloadPolicy, AlibabaVideoJob, AlibabaVideoJobId, AlibabaVideoJobIdError,
        AlibabaVideoJobStatus, AlibabaVideoMedia, AlibabaVideoMediaType, AlibabaVideoModel,
        AlibabaVideoParameters, AlibabaVideoProviderBuilderExt, AlibabaVideoProviderExt,
        AlibabaVideoRequest, AlibabaVideoRequestError, AlibabaVideoShotType, AlibabaVideoUsage,
        LEGACY_SINGAPORE_VIDEO_BASE_URL, VIDEO_API_MODE_ID, VIDEO_CANCEL_SOURCE,
        VIDEO_IMAGE_SOURCE, VIDEO_PROTOCOL_ID, VIDEO_REFERENCE_SOURCE, VIDEO_TEXT_SOURCE,
        WAN_2_7_I2V, WAN_2_7_I2V_SNAPSHOT, WAN_2_7_R2V, WAN_2_7_R2V_SNAPSHOT, WAN_2_7_T2V,
        WAN_2_7_T2V_SNAPSHOT,
    };
}

pub mod options {
    pub use siumai_provider_alibaba::options::{
        ALIBABA_SESSION_CACHE_HEADER, AlibabaChatOptions, AlibabaMessagesOptions,
        AlibabaMessagesThinking, AlibabaPromptCacheBreakpoint, AlibabaReasoningEffort,
        AlibabaResponsesOptions, AlibabaResponsesTool, AlibabaSearchOptions, AlibabaSearchStrategy,
    };
}
