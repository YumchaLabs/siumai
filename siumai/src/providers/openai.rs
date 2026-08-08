//! Curated OpenAI provider facade.

pub use siumai_provider_openai::{
    OpenAiApiMode, OpenAiConfigError, OpenAiCredential, OpenAiCredentialError, OpenAiProfile,
    OpenAiProvider, OpenAiProviderBuilder,
};

pub mod models {
    pub use siumai_provider_openai::{
        GPT_5_5, GPT_5_5_PRO, GPT_5_6, GPT_5_6_LUNA, GPT_5_6_SOL, GPT_5_6_TERRA, OpenAiModelClass,
        classify_model,
    };
}

pub mod chat_completions {
    pub use siumai_provider_openai::{OpenAiChatCompletionsModel, OpenAiChatCompletionsOptions};
}

pub mod embeddings {
    pub use siumai_provider_openai::{
        OpenAiEmbeddingModel, OpenAiEmbeddingOptions, TEXT_EMBEDDING_3_LARGE,
        TEXT_EMBEDDING_3_SMALL, TEXT_EMBEDDING_ADA_002,
    };
}

pub mod images {
    pub use siumai_provider_openai::{
        CHATGPT_IMAGE_LATEST, DALL_E_2, DALL_E_3, GPT_IMAGE_1, GPT_IMAGE_1_5, GPT_IMAGE_1_MINI,
        GPT_IMAGE_2, OpenAiImageBackground, OpenAiImageGenerationQuality, OpenAiImageModel,
        OpenAiImageModeration, OpenAiImageOptions, OpenAiImageOutputFormat,
        OpenAiImageResponseFormat, OpenAiImageStyle,
    };
}

pub mod audio {
    pub mod speech {
        pub use siumai_provider_openai::{
            GPT_4O_MINI_TTS, GPT_4O_MINI_TTS_2025_03_20, GPT_4O_MINI_TTS_2025_12_15,
            OpenAiSpeechModel, OpenAiSpeechOptions, TTS_1, TTS_1_1106, TTS_1_HD, TTS_1_HD_1106,
        };
    }

    pub mod transcription {
        pub use siumai_provider_openai::{
            GPT_4O_MINI_TRANSCRIBE, GPT_4O_MINI_TRANSCRIBE_2025_03_20,
            GPT_4O_MINI_TRANSCRIBE_2025_12_15, GPT_4O_TRANSCRIBE, GPT_4O_TRANSCRIBE_DIARIZE,
            OpenAiTranscriptionModel, OpenAiTranscriptionOptions,
            OpenAiTranscriptionResponseFormat, OpenAiTranscriptionTimestampGranularity, WHISPER_1,
        };
    }
}

pub mod resources {
    pub mod conversations {
        pub use siumai_provider_openai::{
            OpenAiConversation, OpenAiConversationCreateRequest, OpenAiConversationDeleted,
            OpenAiConversationInputItem, OpenAiConversationItem,
            OpenAiConversationItemsCreateRequest, OpenAiConversationItemsListOptions,
            OpenAiConversationRole, OpenAiConversationUpdateRequest, OpenAiConversations,
        };
    }

    pub mod files {
        pub use siumai_provider_openai::{
            OpenAiBinaryContent, OpenAiFile, OpenAiFileDeleted, OpenAiFileExpirationAnchor,
            OpenAiFileExpiresAfter, OpenAiFileListOptions, OpenAiFilePurpose, OpenAiFileUpload,
            OpenAiFiles,
        };
    }

    pub mod vector_stores {
        pub use siumai_provider_openai::{
            OpenAiChunkingStrategy, OpenAiStaticChunkingSettings, OpenAiVectorStore,
            OpenAiVectorStoreCreateRequest, OpenAiVectorStoreDeleted, OpenAiVectorStoreExpiration,
            OpenAiVectorStoreExpirationAnchor, OpenAiVectorStoreFile,
            OpenAiVectorStoreFileAttachRequest, OpenAiVectorStoreFileCounts,
            OpenAiVectorStoreFileDeleted, OpenAiVectorStoreFileError,
            OpenAiVectorStoreFileListOptions, OpenAiVectorStoreFileStatusFilter,
            OpenAiVectorStoreListOptions, OpenAiVectorStoreUpdateRequest, OpenAiVectorStores,
        };
    }

    pub mod skills {
        pub use siumai_provider_openai::{
            OpenAiDeletedSkill, OpenAiDeletedSkillVersion, OpenAiSkill, OpenAiSkillFile,
            OpenAiSkillListOptions, OpenAiSkillUpdateRequest, OpenAiSkillUpload,
            OpenAiSkillVersion, OpenAiSkillVersionUpload, OpenAiSkills,
        };
    }

    pub use siumai_provider_openai::{OpenAiCursorPage, OpenAiListOrder, OpenAiMetadata};
}

pub mod responses {
    pub use siumai_provider_openai::responses::{
        ResponsesStreamEvent, ResponsesStreamEventKind, StreamEventWire,
    };
    pub use siumai_provider_openai::{
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
    pub use siumai_provider_openai::{
        OpenAiBackgroundResponse, OpenAiDeletedResponse, OpenAiFunctionToolOptions,
        OpenAiPromptCacheBreakpoint, OpenAiPromptCacheMode, OpenAiPromptCacheOptions,
        OpenAiPromptCacheRetention, OpenAiPromptCacheTtl, OpenAiReasoning, OpenAiReasoningContext,
        OpenAiReasoningEffort, OpenAiReasoningMode, OpenAiReasoningSummary, OpenAiResponseInclude,
        OpenAiResponsesCompactRequest, OpenAiResponsesCompaction, OpenAiResponsesInputItemsOptions,
        OpenAiResponsesInputItemsOrder, OpenAiResponsesInputItemsPage,
        OpenAiResponsesInputTokenCount, OpenAiResponsesInputTokenCountRequest,
        OpenAiResponsesModel, OpenAiResponsesOptions, OpenAiResponsesResource,
        OpenAiResponsesResponse, OpenAiResponsesRetrieveOptions, OpenAiResponsesStream,
        OpenAiResponsesStreamFrame, OpenAiServiceTier, OpenAiTextVerbosity, OpenAiToolCaller,
        OpenAiTruncation,
    };
}

#[cfg(feature = "openai-realtime")]
pub mod experimental {
    pub mod realtime {
        pub use siumai_provider_openai::experimental::realtime::{
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
            pub use siumai_provider_openai::experimental::realtime::advanced::{
                OpenAiRealtimeConnectRequest, OpenAiRealtimeConnector, OpenAiRealtimeSocket,
                OpenAiRealtimeSocketReceiver, OpenAiRealtimeSocketSender, OpenAiWebSocketConnector,
            };
        }
    }
}
