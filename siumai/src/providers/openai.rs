//! Curated OpenAI provider facade.

pub use siumai_provider_openai::configured::{
    OpenAiApiMode, OpenAiConfigError, OpenAiCredential, OpenAiCredentialError, OpenAiProfile,
    OpenAiProvider, OpenAiProviderBuilder,
};

pub mod models {
    pub use siumai_provider_openai::configured::{
        GPT_5_5, GPT_5_5_PRO, GPT_5_6, GPT_5_6_LUNA, GPT_5_6_SOL, GPT_5_6_TERRA, OpenAiModelClass,
        classify_model,
    };
}

pub mod chat_completions {
    pub use siumai_provider_openai::configured::{
        OpenAiChatCompletionsModel, OpenAiChatCompletionsOptions,
    };
}

pub mod responses {
    pub use siumai_provider_openai::configured::{
        OpenAiBackgroundResponse, OpenAiDeletedResponse, OpenAiFunctionToolOptions,
        OpenAiPromptCacheBreakpoint, OpenAiPromptCacheMode, OpenAiPromptCacheOptions,
        OpenAiPromptCacheRetention, OpenAiPromptCacheTtl, OpenAiProviderTool, OpenAiReasoning,
        OpenAiReasoningContext, OpenAiReasoningEffort, OpenAiReasoningMode, OpenAiReasoningSummary,
        OpenAiResponseInclude, OpenAiResponsesCompactRequest, OpenAiResponsesCompaction,
        OpenAiResponsesInputItemsOptions, OpenAiResponsesInputItemsOrder,
        OpenAiResponsesInputItemsPage, OpenAiResponsesInputTokenCount,
        OpenAiResponsesInputTokenCountRequest, OpenAiResponsesModel, OpenAiResponsesOptions,
        OpenAiResponsesResource, OpenAiResponsesRetrieveOptions, OpenAiServiceTier,
        OpenAiTextVerbosity, OpenAiToolCaller, OpenAiTruncation,
    };
}

#[cfg(feature = "openai-realtime")]
pub mod experimental {
    pub mod realtime {
        pub use siumai_provider_openai::configured::experimental::realtime::{
            DecodedRealtimeEvent, FunctionCallArgumentsDeltaEvent, FunctionCallArgumentsDoneEvent,
            IncompleteFunctionCall, OPENAI_REALTIME_CLIENT_SECRETS_URL, OPENAI_REALTIME_MODEL,
            OPENAI_REALTIME_TRANSLATION_CLIENT_SECRETS_URL, OPENAI_REALTIME_TRANSLATION_MODEL,
            OPENAI_REALTIME_TRANSLATION_SOURCE_URL, OPENAI_REALTIME_TRANSLATION_WEBRTC_CALLS_URL,
            OPENAI_REALTIME_TRANSLATION_WEBSOCKET_URL, OPENAI_REALTIME_WEBRTC_CALLS_URL,
            OPENAI_REALTIME_WEBRTC_DATA_CHANNEL, OPENAI_REALTIME_WEBSOCKET_SOURCE_URL,
            OPENAI_REALTIME_WEBSOCKET_URL, OpenAiRealtimeBootstrapMetadata,
            OpenAiRealtimeClientEvent, OpenAiRealtimeClientSecretExpiry,
            OpenAiRealtimeClientSecretExpiryAnchor, OpenAiRealtimeClientSecretRequest,
            OpenAiRealtimeClientSecretResource, OpenAiRealtimeConfig, OpenAiRealtimeConfigError,
            OpenAiRealtimeEndpoint, OpenAiRealtimeEndpointKind, OpenAiRealtimeInbound,
            OpenAiRealtimeResource, OpenAiRealtimeResourceError, OpenAiRealtimeRoute,
            OpenAiRealtimeServerError, OpenAiRealtimeServerEvent, OpenAiRealtimeSession,
            OpenAiTranslationClientEvent, OpenAiTranslationConfig, OpenAiTranslationInbound,
            OpenAiTranslationServerEvent, OpenAiTranslationSession, RealtimeCodecError,
            RealtimeCodecLimits, RealtimeResponseDoneEvent, RealtimeResponseStatus,
            TranslationAudioDeltaEvent, TranslationSessionState, UnknownRealtimeEvent,
        };

        pub mod advanced {
            pub use siumai_provider_openai::configured::experimental::realtime::advanced::{
                OpenAiRealtimeConnectRequest, OpenAiRealtimeConnector, OpenAiRealtimeSocket,
                OpenAiRealtimeSocketReceiver, OpenAiRealtimeSocketSender, OpenAiWebSocketConnector,
            };
        }
    }
}
