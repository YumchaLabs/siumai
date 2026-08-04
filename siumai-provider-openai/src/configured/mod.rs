//! Rust-first configured OpenAI provider surface.
//!
//! Responses and Chat Completions are explicit model types backed by one
//! clone-cheap provider runtime. The default language-model path is Responses.

mod catalog;
mod credential;
mod mode;
mod model;
mod options;
mod policy;
mod profile;
mod provider;
#[cfg(feature = "openai-realtime")]
mod realtime;
#[cfg(feature = "openai-realtime")]
mod realtime_resource;
mod responses_resource;

pub use catalog::{
    GPT_5_6, GPT_5_6_LUNA, GPT_5_6_SOL, GPT_5_6_TERRA, OpenAiModelClass, classify_model,
};
pub use credential::{OpenAiCredential, OpenAiCredentialError};
pub use mode::OpenAiApiMode;
pub use model::{OpenAiChatCompletionsModel, OpenAiResponsesModel};
pub use options::{
    OpenAiChatCompletionsOptions, OpenAiContextManagement, OpenAiFunctionToolOptions,
    OpenAiPromptCacheBreakpoint, OpenAiPromptCacheMode, OpenAiPromptCacheOptions,
    OpenAiPromptCacheTtl, OpenAiProviderTool, OpenAiReasoning, OpenAiReasoningContext,
    OpenAiReasoningEffort, OpenAiReasoningMode, OpenAiReasoningSummary, OpenAiResponseInclude,
    OpenAiResponsesOptions, OpenAiServiceTier, OpenAiTextVerbosity, OpenAiToolCaller,
    OpenAiTruncation,
};
pub use profile::OpenAiProfile;
pub use provider::{OpenAiConfigError, OpenAiProvider, OpenAiProviderBuilder};
#[cfg(feature = "openai-realtime")]
pub use realtime::{
    OPENAI_REALTIME_CLIENT_SECRETS_URL, OPENAI_REALTIME_MODEL,
    OPENAI_REALTIME_TRANSLATION_CLIENT_SECRETS_URL, OPENAI_REALTIME_TRANSLATION_MODEL,
    OPENAI_REALTIME_TRANSLATION_SOURCE_URL, OPENAI_REALTIME_TRANSLATION_WEBRTC_CALLS_URL,
    OPENAI_REALTIME_TRANSLATION_WEBSOCKET_URL, OPENAI_REALTIME_WEBRTC_CALLS_URL,
    OPENAI_REALTIME_WEBRTC_DATA_CHANNEL, OPENAI_REALTIME_WEBSOCKET_SOURCE_URL,
    OPENAI_REALTIME_WEBSOCKET_URL, OpenAiRealtimeBootstrapMetadata,
    OpenAiRealtimeClientSecretExpiry, OpenAiRealtimeClientSecretExpiryAnchor,
    OpenAiRealtimeClientSecretRequest, OpenAiRealtimeClientSecretResource, OpenAiRealtimeConfig,
    OpenAiRealtimeConfigError, OpenAiRealtimeEndpoint, OpenAiRealtimeEndpointKind,
    OpenAiRealtimeInbound, OpenAiRealtimeResourceError, OpenAiRealtimeRoute, OpenAiRealtimeSession,
    OpenAiTranslationConfig, OpenAiTranslationInbound, OpenAiTranslationSession,
};
#[cfg(feature = "openai-realtime")]
pub use realtime_resource::OpenAiRealtimeResource;
pub use responses_resource::{
    OpenAiBackgroundResponse, OpenAiDeletedResponse, OpenAiResponsesCompactRequest,
    OpenAiResponsesCompaction, OpenAiResponsesInputItemsOptions, OpenAiResponsesInputItemsOrder,
    OpenAiResponsesInputItemsPage, OpenAiResponsesResource, OpenAiResponsesRetrieveOptions,
};

/// Provider-faithful Responses models, options, resources, and native wire values.
pub mod responses {
    pub use crate::configured::{
        OpenAiBackgroundResponse, OpenAiDeletedResponse, OpenAiFunctionToolOptions,
        OpenAiPromptCacheBreakpoint, OpenAiPromptCacheMode, OpenAiPromptCacheOptions,
        OpenAiPromptCacheTtl, OpenAiProviderTool, OpenAiReasoning, OpenAiReasoningContext,
        OpenAiReasoningEffort, OpenAiReasoningMode, OpenAiReasoningSummary, OpenAiResponseInclude,
        OpenAiResponsesCompactRequest, OpenAiResponsesCompaction, OpenAiResponsesInputItemsOptions,
        OpenAiResponsesInputItemsOrder, OpenAiResponsesInputItemsPage, OpenAiResponsesModel,
        OpenAiResponsesOptions, OpenAiResponsesResource, OpenAiResponsesRetrieveOptions,
        OpenAiServiceTier, OpenAiTextVerbosity, OpenAiToolCaller, OpenAiTruncation,
    };
    pub use siumai_protocol_openai::responses_next::{
        AnnotationWire, CustomToolCallItemWire, FunctionCallItemWire, IncompleteDetailsWire,
        InputTokenDetailsWire, ItemStatus, MessageItemWire, OutputContentPart, OutputItem,
        OutputRefusalWire, OutputTextWire, OutputTokenDetailsWire, ProgramItemWire,
        ProgramOutputItemWire, ProviderToolItemWire, ReasoningItemWire, ReasoningTextWire,
        ResponseErrorWire, ResponseReasoningConfigWire, ResponseStatus, ResponseUsageWire,
        ResponseWire, ToolCallerWire, UnknownContentPartWire, UnknownOutputItemWire,
    };
}

/// Experimental provider-native contracts that are intentionally outside the stable families.
#[cfg(feature = "openai-realtime")]
pub mod experimental {
    /// Typed OpenAI Realtime and Realtime Translation sessions.
    pub mod realtime {
        pub use crate::configured::{
            OPENAI_REALTIME_CLIENT_SECRETS_URL, OPENAI_REALTIME_MODEL,
            OPENAI_REALTIME_TRANSLATION_CLIENT_SECRETS_URL, OPENAI_REALTIME_TRANSLATION_MODEL,
            OPENAI_REALTIME_TRANSLATION_SOURCE_URL, OPENAI_REALTIME_TRANSLATION_WEBRTC_CALLS_URL,
            OPENAI_REALTIME_TRANSLATION_WEBSOCKET_URL, OPENAI_REALTIME_WEBRTC_CALLS_URL,
            OPENAI_REALTIME_WEBRTC_DATA_CHANNEL, OPENAI_REALTIME_WEBSOCKET_SOURCE_URL,
            OPENAI_REALTIME_WEBSOCKET_URL, OpenAiRealtimeBootstrapMetadata,
            OpenAiRealtimeClientSecretExpiry, OpenAiRealtimeClientSecretExpiryAnchor,
            OpenAiRealtimeClientSecretRequest, OpenAiRealtimeClientSecretResource,
            OpenAiRealtimeConfig, OpenAiRealtimeConfigError, OpenAiRealtimeEndpoint,
            OpenAiRealtimeEndpointKind, OpenAiRealtimeInbound, OpenAiRealtimeResource,
            OpenAiRealtimeResourceError, OpenAiRealtimeRoute, OpenAiRealtimeSession,
            OpenAiTranslationConfig, OpenAiTranslationInbound, OpenAiTranslationSession,
        };
        pub use siumai_protocol_openai::realtime::{
            DecodedRealtimeEvent, FunctionCallArgumentsDeltaEvent, FunctionCallArgumentsDoneEvent,
            IncompleteFunctionCall, OpenAiRealtimeClientEvent, OpenAiRealtimeServerError,
            OpenAiRealtimeServerEvent, OpenAiTranslationClientEvent, OpenAiTranslationServerEvent,
            RealtimeCodecError, RealtimeCodecLimits, RealtimeResponseDoneEvent,
            RealtimeResponseStatus, TranslationAudioDeltaEvent, TranslationSessionState,
            UnknownRealtimeEvent,
        };

        /// Connector seams for custom transports and deterministic integration tests.
        pub mod advanced {
            pub use crate::configured::realtime::{
                OpenAiRealtimeConnectRequest, OpenAiRealtimeConnector, OpenAiRealtimeSocket,
                OpenAiRealtimeSocketReceiver, OpenAiRealtimeSocketSender, OpenAiWebSocketConnector,
            };
        }
    }
}
