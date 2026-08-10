//! Rust-first configured OpenAI provider surface.
//!
//! Responses and Chat Completions are explicit language-model types backed by one
//! clone-cheap provider runtime. The same provider owns portable embedding, image,
//! buffered speech, and final-result transcription adapters plus typed native
//! resources. The default language-model path is Responses.

mod annotations;
mod catalog;
mod credential;
mod embedding;
mod http_error;
mod image;
mod mode;
mod model;
mod options;
mod profile;
mod provider;
#[cfg(feature = "openai-realtime")]
mod realtime;
#[cfg(feature = "openai-realtime")]
mod realtime_resource;
mod resources;
mod responses_native;
mod responses_resource;
#[cfg(feature = "openai-responses-websocket")]
mod responses_websocket;
mod speech;
mod tools;
mod transcription;

pub use annotations::{OpenAiAnnotationError, OpenAiContentOptions, OpenAiPromptCacheMarker};
pub use catalog::{
    GPT_5_5, GPT_5_5_PRO, GPT_5_6, GPT_5_6_LUNA, GPT_5_6_SOL, GPT_5_6_TERRA, OpenAiModelClass,
    classify_model,
};
pub use credential::{OpenAiCredential, OpenAiCredentialError};
pub use embedding::{
    OpenAiEmbeddingModel, OpenAiEmbeddingOptions, TEXT_EMBEDDING_3_LARGE, TEXT_EMBEDDING_3_SMALL,
    TEXT_EMBEDDING_ADA_002,
};
pub use image::{
    CHATGPT_IMAGE_LATEST, DALL_E_2, DALL_E_3, GPT_IMAGE_1, GPT_IMAGE_1_5, GPT_IMAGE_1_MINI,
    GPT_IMAGE_2, OpenAiImageGenerationQuality, OpenAiImageModel, OpenAiImageOptions,
    OpenAiImageResponseFormat, OpenAiImageStyle,
};
pub use mode::OpenAiApiMode;
pub use model::{OpenAiChatCompletionsModel, OpenAiResponsesModel};
pub use options::{
    OpenAiChatCompletionsOptions, OpenAiContextManagement, OpenAiFunctionToolOptions,
    OpenAiPromptCacheMode, OpenAiPromptCacheOptions, OpenAiPromptCacheRetention,
    OpenAiPromptCacheTtl, OpenAiReasoning, OpenAiReasoningContext, OpenAiReasoningEffort,
    OpenAiReasoningMode, OpenAiReasoningSummary, OpenAiResponseInclude, OpenAiResponsesOptions,
    OpenAiServiceTier, OpenAiTextVerbosity, OpenAiTruncation,
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
pub use resources::*;
pub use responses_native::{
    OpenAiResponsesResponse, OpenAiResponsesStream, OpenAiResponsesStreamFrame,
};
pub use responses_resource::{
    OpenAiBackgroundResponse, OpenAiDeletedResponse, OpenAiResponsesCompactRequest,
    OpenAiResponsesCompaction, OpenAiResponsesInputItemsOptions, OpenAiResponsesInputItemsOrder,
    OpenAiResponsesInputItemsPage, OpenAiResponsesInputTokenCount,
    OpenAiResponsesInputTokenCountRequest, OpenAiResponsesResource, OpenAiResponsesRetrieveOptions,
};
#[cfg(feature = "openai-responses-websocket")]
pub use responses_websocket::{
    OPENAI_RESPONSES_WEBSOCKET_URL, OpenAiResponsesWarmUpFrame, OpenAiResponsesWarmUpOutcome,
    OpenAiResponsesWebSocketConfig, OpenAiResponsesWebSocketConfigError,
    OpenAiResponsesWebSocketEvent, OpenAiResponsesWebSocketSession, OpenAiResponsesWebSocketTurn,
    OpenAiResponsesWebSocketTurnKind,
};
pub use speech::{
    GPT_4O_MINI_TTS, GPT_4O_MINI_TTS_2025_03_20, GPT_4O_MINI_TTS_2025_12_15, OpenAiSpeechModel,
    OpenAiSpeechOptions, TTS_1, TTS_1_1106, TTS_1_HD, TTS_1_HD_1106,
};
pub use tools::{
    OpenAiApplyPatchTool, OpenAiApproximateLocation, OpenAiCodeInterpreterAutoContainer,
    OpenAiCodeInterpreterContainer, OpenAiCodeInterpreterTool, OpenAiContainerMemoryLimit,
    OpenAiContainerNetworkPolicy, OpenAiCustomTool, OpenAiCustomToolFormat, OpenAiDomainSecret,
    OpenAiFileSearchFilter, OpenAiFileSearchFilterList, OpenAiFileSearchFilterScalar,
    OpenAiFileSearchHybridSearch, OpenAiFileSearchRankingOptions, OpenAiFileSearchTool,
    OpenAiGrammarSyntax, OpenAiImageAction, OpenAiImageBackground, OpenAiImageGenerationTool,
    OpenAiImageInputFidelity, OpenAiImageInputMask, OpenAiImageModeration, OpenAiImageOutputFormat,
    OpenAiImageQuality, OpenAiImageSize, OpenAiInlineSkillSource, OpenAiLocalShellSkill,
    OpenAiMcpAllowedTools, OpenAiMcpApproval, OpenAiMcpApprovalFilter, OpenAiMcpEndpoint,
    OpenAiMcpTool, OpenAiRawTool, OpenAiResponsesTool, OpenAiShellEnvironment, OpenAiShellSkill,
    OpenAiShellTool, OpenAiToolCaller, OpenAiToolSearchExecution, OpenAiToolSearchTool,
    OpenAiWebSearchContentType, OpenAiWebSearchContextSize, OpenAiWebSearchFilters,
    OpenAiWebSearchImageSettings, OpenAiWebSearchReturnTokenBudget, OpenAiWebSearchTool,
};
pub use transcription::{
    GPT_4O_MINI_TRANSCRIBE, GPT_4O_MINI_TRANSCRIBE_2025_03_20, GPT_4O_MINI_TRANSCRIBE_2025_12_15,
    GPT_4O_TRANSCRIBE, GPT_4O_TRANSCRIBE_DIARIZE, OpenAiTranscriptionModel,
    OpenAiTranscriptionOptions, OpenAiTranscriptionResponseFormat,
    OpenAiTranscriptionTimestampGranularity, WHISPER_1,
};

/// Provider-faithful Responses models, options, resources, and native wire values.
pub mod responses {
    pub use crate::configured::{
        OpenAiApplyPatchTool, OpenAiApproximateLocation, OpenAiCodeInterpreterAutoContainer,
        OpenAiCodeInterpreterContainer, OpenAiCodeInterpreterTool, OpenAiContainerMemoryLimit,
        OpenAiContainerNetworkPolicy, OpenAiCustomTool, OpenAiCustomToolFormat, OpenAiDomainSecret,
        OpenAiFileSearchFilter, OpenAiFileSearchFilterList, OpenAiFileSearchFilterScalar,
        OpenAiFileSearchHybridSearch, OpenAiFileSearchRankingOptions, OpenAiFileSearchTool,
        OpenAiGrammarSyntax, OpenAiImageAction, OpenAiImageBackground, OpenAiImageGenerationTool,
        OpenAiImageInputFidelity, OpenAiImageInputMask, OpenAiImageModeration,
        OpenAiImageOutputFormat, OpenAiImageQuality, OpenAiImageSize, OpenAiInlineSkillSource,
        OpenAiLocalShellSkill, OpenAiMcpAllowedTools, OpenAiMcpApproval, OpenAiMcpApprovalFilter,
        OpenAiMcpEndpoint, OpenAiMcpTool, OpenAiRawTool, OpenAiResponsesTool,
        OpenAiShellEnvironment, OpenAiShellSkill, OpenAiShellTool, OpenAiToolSearchExecution,
        OpenAiToolSearchTool, OpenAiWebSearchContentType, OpenAiWebSearchContextSize,
        OpenAiWebSearchFilters, OpenAiWebSearchImageSettings, OpenAiWebSearchReturnTokenBudget,
        OpenAiWebSearchTool,
    };
    pub use crate::configured::{
        OpenAiBackgroundResponse, OpenAiDeletedResponse, OpenAiFunctionToolOptions,
        OpenAiReasoning, OpenAiReasoningContext, OpenAiReasoningEffort, OpenAiReasoningMode,
        OpenAiReasoningSummary, OpenAiResponseInclude, OpenAiResponsesCompactRequest,
        OpenAiResponsesCompaction, OpenAiResponsesInputItemsOptions,
        OpenAiResponsesInputItemsOrder, OpenAiResponsesInputItemsPage,
        OpenAiResponsesInputTokenCount, OpenAiResponsesInputTokenCountRequest,
        OpenAiResponsesModel, OpenAiResponsesOptions, OpenAiResponsesResource,
        OpenAiResponsesResponse, OpenAiResponsesRetrieveOptions, OpenAiResponsesStream,
        OpenAiResponsesStreamFrame, OpenAiServiceTier, OpenAiTextVerbosity, OpenAiToolCaller,
        OpenAiTruncation,
    };
    pub use siumai_protocol_openai::responses::{
        AnnotationWire, CustomToolCallItemWire, FunctionCallItemWire, IncompleteDetailsWire,
        InputTokenDetailsWire, ItemStatus, MessageItemWire, OutputContentPart, OutputItem,
        OutputRefusalWire, OutputTextWire, OutputTokenDetailsWire, ProgramItemWire,
        ProgramOutputItemWire, ProviderToolItemWire, ReasoningItemWire, ReasoningTextWire,
        ResponseErrorWire, ResponseReasoningConfigWire, ResponseStatus, ResponseUsageWire,
        ResponseWire, ResponsesReplayStatus, ResponsesStreamEvent, ResponsesStreamEventKind,
        StreamEventWire, ToolCallerWire, UnknownContentPartWire, UnknownOutputItemWire,
    };
}

/// Experimental provider-native contracts that are intentionally outside the stable families.
pub mod experimental {
    /// Provider-native OpenAI Skills directory lifecycle.
    pub mod skills {
        pub use crate::configured::resources::skills::{
            OpenAiSkillFile, OpenAiSkillListOptions, OpenAiSkillUpload, OpenAiSkillVersionUpload,
            OpenAiSkills, OpenAiSkillsProviderExt,
        };
        pub use siumai_protocol_openai::experimental::skills::{
            OpenAiDeletedSkill, OpenAiDeletedSkillVersion, OpenAiSkill, OpenAiSkillUpdateRequest,
            OpenAiSkillVersion,
        };
    }

    /// Typed OpenAI Realtime and Realtime Translation sessions.
    #[cfg(feature = "openai-realtime")]
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

    /// Persistent provider-owned Responses WebSocket sessions.
    #[cfg(feature = "openai-responses-websocket")]
    pub mod responses_websocket {
        pub use crate::configured::{
            OPENAI_RESPONSES_WEBSOCKET_URL, OpenAiResponsesWarmUpFrame,
            OpenAiResponsesWarmUpOutcome, OpenAiResponsesWebSocketConfig,
            OpenAiResponsesWebSocketConfigError, OpenAiResponsesWebSocketEvent,
            OpenAiResponsesWebSocketSession, OpenAiResponsesWebSocketTurn,
            OpenAiResponsesWebSocketTurnKind,
        };

        /// Connector seams for custom transports and deterministic integration tests.
        pub mod advanced {
            pub use crate::configured::responses_websocket::{
                OpenAiResponsesWebSocketConnectRequest, OpenAiResponsesWebSocketConnector,
                OpenAiResponsesWebSocketSocket, OpenAiResponsesWebSocketSocketReceiver,
                OpenAiResponsesWebSocketSocketSender, OpenAiResponsesWebSocketTransportConnector,
            };
        }
    }
}
