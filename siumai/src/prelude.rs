//! Convenient pre-import module.

/// Default prelude: Vercel-aligned unified surface.
///
/// - Stable model families: `siumai::prelude::*` / `siumai::prelude::unified::*`
/// - Non-family capabilities: `siumai::prelude::extensions::*`
/// - Provider-specific APIs: `siumai::provider_ext::<provider>::*`
pub use self::unified::*;

/// Vercel-aligned unified surface (recommended for new code).
///
/// This module centers the seven stable model families:
/// Language/Embedding/Image/Reranking/Speech/Transcription/Video.
///
/// Compatibility-oriented construction aliases remain source-compatible under
/// `siumai::compat` and `prelude::compat`, not through this stable prelude.
pub mod unified {
    /// Directional generated-output content namespace.
    pub use crate::content::output;
    /// Directional request content namespace.
    pub use crate::content::prompt;

    pub use crate::structured_output::{
        GenerateObjectOptions, GenerateObjectResult, GenerateObjectSchema, PartialJsonParseResult,
        PartialJsonParseState, PartialJsonValueStream, PartialJsonValueStreamEvent,
        RepairTextContext, RepairTextFunction, RepairTextFuture, fix_partial_json, generate_array,
        generate_choice, generate_enum, generate_json, generate_object, parse_partial_json,
        partial_json_value_stream,
    };
    pub use crate::tools;
    pub use crate::{
        ExecutableTool, ExecutableTools, ProviderDefinedToolFactory,
        ProviderDefinedToolFactoryWithOutputSchema, ProviderExecutedToolFactory,
        ToolExecuteFunction, ToolExecutionOptions, ToolExecutionResult, ToolExecutionStream,
        ToolModelOutputContext, ToolSet, create_provider_defined_tool_factory,
        create_provider_defined_tool_factory_with_output_schema,
        create_provider_executed_tool_factory, dynamic_tool, execute_tool, is_executable_tool,
        model_messages_from_chat_messages,
    };
    pub use crate::{
        IdGenerator, IdGeneratorOptions, SerialJobExecutor, ToolNameMapping, create_id_generator,
        create_tool_name_mapping, generate_id,
    };
    pub use crate::{assistant, conversation, conversation_with_system, messages, quick_chat};
    pub use crate::{
        completion, embedding, files, image, rerank, skills, speech, structured_output, text,
        transcription, video,
    };
    pub use crate::{
        embed, embed_many, generate_image, generate_speech, generate_text, transcribe,
    };
    pub use crate::{system, tool, user, user_with_image};
    pub use siumai_core::completion::CompletionModel;
    pub use siumai_core::embedding::EmbeddingModel;
    pub use siumai_core::error::{ErrorCategory, LlmError, LlmErrorExt};
    pub use siumai_core::image::{ImageModel, ImageModelV4};
    pub use siumai_core::rerank::RerankingModel;
    pub use siumai_core::speech::SpeechModel;
    pub use siumai_core::streaming::{
        ChatStream, ChatStreamCustomContent, ChatStreamEvent, ChatStreamFileData,
        ChatStreamFilePart, ChatStreamFinishInfo, ChatStreamHandle, ChatStreamPart,
        ChatStreamReplay, ChatStreamToolApprovalRequest, ChatStreamToolCall, ChatStreamToolResult,
        StreamProviderMetadata,
    };
    pub use siumai_core::text::{
        LanguageModel, LanguageModelV4, LanguageModelV4DoStreamResult, LanguageModelV4Stream,
    };
    pub use siumai_core::traits::{
        ChatCapability, CompletionCapability, EmbeddingExtensions, ImageGenerationCapability,
        ModelMetadata, ProviderCapabilities, SpeechCapability, TranscriptionCapability,
    };
    pub use siumai_core::transcription::TranscriptionModel;
    pub use siumai_core::video::{VideoModel, VideoModelV4};

    // Core request/response types for the seven stable model families.
    #[allow(deprecated)]
    pub use siumai_core::types::{
        AISDKError, APICallError, AssistantContent, AssistantContentPart, AssistantModelMessage,
        AudioStreamEvent, CacheControl, CallWarning, CallbackModelInfo, CancelHandle, ChatInit,
        ChatMessage, ChatRequest, ChatRequestBuilder, ChatRequestOptions, ChatResponse, ChatState,
        ChatStatus, ChatTransportReconnectToStreamOptions, ChatTransportSendMessagesOptions,
        ChatTransportTrigger, CommonParams, CompletionRequest, CompletionRequestOptions,
        CompletionResponse, CompletionStreamProtocol, CompletionTokensDetails, Context,
        CreateUIMessage, CustomContentUIPart, CustomOutput, CustomPart, CustomProviderOptions,
        DataContent, DataUIMessageChunk, DataUIPart, DefaultGeneratedAudioFile,
        DefaultGeneratedAudioFileWithType, DefaultGeneratedFile, DefaultGeneratedFileWithType,
        DefaultStepResult, DownloadError, DynamicToolCall, DynamicToolError, DynamicToolResult,
        DynamicToolUIPart, EmbedEndEvent, EmbedManyResult, EmbedOutput, EmbedResponseData,
        EmbedResult, EmbedStartEvent, EmbedValue, Embedding, EmbeddingModelCallEndEvent,
        EmbeddingModelCallStartEvent, EmbeddingModelUsage, EmbeddingRequest, EmbeddingResponse,
        EmbeddingTaskType, EmptyResponseBodyError, FileOutput, FilePart, FilePartSource,
        FileUIPart, FinishReason, FlexibleSchema, GenerateImagePrompt, GenerateImageRequest,
        GenerateImageResult, GenerateObjectEndEvent, GenerateObjectOutputStrategy,
        GenerateObjectResponseMetadata, GenerateObjectStartEvent, GenerateObjectStepEndEvent,
        GenerateObjectStepStartEvent, GenerateTextContentPart, GenerateTextEndEvent,
        GenerateTextModelInfo, GenerateTextReasoningPart, GenerateTextResponseMetadata,
        GenerateTextResult, GenerateTextStartEvent, GenerateTextStepEndEvent,
        GenerateTextStepReasoningPart, GenerateTextStepResult, GenerateTextStepStartEvent,
        GenerateVideoResult, GeneratedAudioFile, GeneratedFile, GeneratedImage,
        HttpChatTransportInitOptions, HttpConfig, ImageDetail, ImageGenerationRequest,
        ImageGenerationResponse, ImageModelProviderMetadata, ImageModelResponseMetadata,
        ImageModelUsage, ImagePart, InferUIDataParts, InferUIMessageChunk, InferUIMessageData,
        InferUIMessageMetadata, InferUIMessagePart, InferUIMessageToolCall,
        InferUIMessageToolOutputs, InferUIMessageTools, InferUITool, InferUITools,
        InvalidArgumentError, InvalidDataContentError, InvalidMessageRoleError, InvalidPromptError,
        InvalidResponseDataError, InvalidStreamPartError, InvalidToolApprovalError,
        InvalidToolInputError, JSONParseError, JSONSchema7, JSONValue, LanguageModelCallOptions,
        LanguageModelInputTokenDetails, LanguageModelOutputTokenDetails, LanguageModelReasoning,
        LanguageModelRequestMetadata, LanguageModelResponseMetadata,
        LanguageModelStreamModelCallEndPart, LanguageModelStreamModelCallResponseMetadataPart,
        LanguageModelStreamModelCallStartPart, LanguageModelStreamPart, LanguageModelUsage,
        LanguageModelV4AssistantContentPart, LanguageModelV4AssistantMessage,
        LanguageModelV4CallOptions, LanguageModelV4Content, LanguageModelV4CustomContent,
        LanguageModelV4CustomPart, LanguageModelV4DataContent, LanguageModelV4File,
        LanguageModelV4FilePart, LanguageModelV4FilePartData, LanguageModelV4FinishReason,
        LanguageModelV4FunctionTool, LanguageModelV4FunctionToolInputExample,
        LanguageModelV4GenerateResponseMetadata, LanguageModelV4GenerateResult,
        LanguageModelV4InputTokens, LanguageModelV4Message, LanguageModelV4OutputTokens,
        LanguageModelV4Prompt, LanguageModelV4ProviderTool, LanguageModelV4Reasoning,
        LanguageModelV4ReasoningFile, LanguageModelV4ReasoningFilePart,
        LanguageModelV4ReasoningPart, LanguageModelV4RequestMetadata,
        LanguageModelV4ResponseMetadata, LanguageModelV4Source,
        LanguageModelV4StreamResponseMetadata, LanguageModelV4StreamResult,
        LanguageModelV4SystemMessage, LanguageModelV4Text, LanguageModelV4TextPart,
        LanguageModelV4Tool, LanguageModelV4ToolApprovalRequest,
        LanguageModelV4ToolApprovalResponsePart, LanguageModelV4ToolCall,
        LanguageModelV4ToolCallPart, LanguageModelV4ToolChoice, LanguageModelV4ToolContentPart,
        LanguageModelV4ToolMessage, LanguageModelV4ToolResult,
        LanguageModelV4ToolResultContentPart, LanguageModelV4ToolResultOutput,
        LanguageModelV4ToolResultPart, LanguageModelV4Usage, LanguageModelV4UserContentPart,
        LanguageModelV4UserMessage, LazySchema, LoadAPIKeyError, LoadSettingError, MediaSource,
        MessageContent, MessageConversionError, MessageMetadata, MessageRole,
        MissingToolResultsError, ModelCallResponseData, ModelInfo, ModelMessage,
        ModelMessageConversionError, ModelMessageRole, NoContentGeneratedError,
        NoImageGeneratedError, NoObjectGeneratedError, NoOutputGeneratedError,
        NoSpeechGeneratedError, NoSuchModelError, NoSuchModelType, NoSuchProviderError,
        NoSuchProviderReferenceError, NoSuchToolError, NoTranscriptGeneratedError,
        NoVideoGeneratedError, ObjectStreamErrorPart, ObjectStreamFinishPart,
        ObjectStreamObjectPart, ObjectStreamPart, ObjectStreamTextDeltaPart, OnChunkEvent,
        OnFinishEvent, OnStartEvent, OnStepFinishEvent, OnStepStartEvent, OnToolCallFinishEvent,
        OnToolCallStartEvent, OutputSchema, PrepareReconnectToStreamRequestOptions,
        PrepareSendMessagesRequestOptions, PrepareStepOptions, PrepareStepResult,
        PreparedReconnectToStreamRequest, PreparedSendMessagesRequest, Prompt,
        PromptExecutionError, PromptInput, PromptTokensDetails, PromptValidationError,
        ProviderDefinedTool, ProviderMetadata, ProviderOptions, ProviderOptionsMap,
        ProviderReference, ProviderType, PruneEmptyMessagesMode, PruneMessagesOptions,
        PruneReasoningMode, PruneToolCallMode, PruneToolCallRule, ReasoningFileOutput,
        ReasoningFilePart, ReasoningFileUIPart, ReasoningOutput, ReasoningPart, ReasoningUIPart,
        RequestCredentials, RequestOptions, RerankEndEvent, RerankRanking, RerankRankingEntry,
        RerankRequest, RerankResponse, RerankResponseMetadata, RerankResult, RerankStartEvent,
        RerankingModelCallEndEvent, RerankingModelCallRanking, RerankingModelCallStartEvent,
        ResponseFormat, ResponseMessage, ResponseMetadata, RetryError, RetryErrorReason, Schema,
        SchemaValidator, Source, SourceDocumentUIPart, SourceUrlUIPart,
        SpeechModelResponseMetadata, SpeechResult, StandardizedPrompt, StaticToolCall,
        StaticToolError, StaticToolOutputDenied, StaticToolResult, StepResult, StepStartUIPart,
        StopCondition, StreamRequestOptions, StreamTextChunk, StreamTextChunkEvent,
        StreamTextLifecycleChunk, StreamTextLifecycleChunkType, SttRequest, SttResponse,
        SystemModelMessage, SystemPrompt, TelemetryOptions, TextOutput, TextPart,
        TextStreamAbortPart, TextStreamCustomPart, TextStreamErrorPart, TextStreamFilePart,
        TextStreamFinishPart, TextStreamFinishStepPart, TextStreamPart, TextStreamRawPart,
        TextStreamReasoningDeltaPart, TextStreamReasoningEndPart, TextStreamReasoningFilePart,
        TextStreamReasoningStartPart, TextStreamSourcePart, TextStreamStartPart,
        TextStreamStartStepPart, TextStreamTextDeltaPart, TextStreamTextEndPart,
        TextStreamTextStartPart, TextStreamToolApprovalRequestPart,
        TextStreamToolApprovalResponsePart, TextStreamToolCallPart, TextStreamToolErrorPart,
        TextStreamToolInputDeltaPart, TextStreamToolInputEndPart, TextStreamToolInputStartPart,
        TextStreamToolOutputDeniedPart, TextStreamToolResultPart, TextUIPart, TimeoutConfiguration,
        TimeoutConfigurationSettings, TooManyEmbeddingValuesForCallError, Tool,
        ToolApprovalConfiguration, ToolApprovalDecisionContext, ToolApprovalRequest,
        ToolApprovalRequestOutput, ToolApprovalResponse, ToolApprovalResponseOutput,
        ToolApprovalStatus, ToolApprovalStatusDetails, ToolApprovalStatusType, ToolCall,
        ToolCallNotFoundForApprovalError, ToolCallPart, ToolCallRepairContext, ToolCallRepairError,
        ToolCallRepairFunctionError, ToolCallRepairResult, ToolChoice, ToolContent,
        ToolContentPart, ToolError, ToolExecutionEndEvent, ToolExecutionStartEvent,
        ToolModelMessage, ToolOutput, ToolOutputDenied, ToolResult, ToolResultContentPart,
        ToolResultOutput, ToolResultPart, ToolUIPart, TranscriptionModelResponseMetadata,
        TranscriptionResult, TranscriptionSegment, TtsRequest, TtsResponse, TypeValidationContext,
        TypeValidationError, TypedToolCall, TypedToolError, TypedToolOutputDenied, TypedToolResult,
        UI_MESSAGE_STREAM_HEADERS, UIDataPartSchemas, UIDataTypes, UIDataTypesToSchemas, UIMessage,
        UIMessageChunk, UIMessagePart, UIMessageStreamError, UIMessageStreamOptions, UITool,
        UIToolInvocation, UITools, UiCustomPart, UiDataPart, UiFilePart, UiMessage,
        UiMessageAbortChunk, UiMessageChunk, UiMessageCustomChunk, UiMessageDataChunk,
        UiMessageErrorChunk, UiMessageFileChunk, UiMessageFinishChunk, UiMessageFinishStepChunk,
        UiMessageMetadataChunk, UiMessagePart, UiMessageReasoningDeltaChunk,
        UiMessageReasoningEndChunk, UiMessageReasoningFileChunk, UiMessageReasoningStartChunk,
        UiMessageRole, UiMessageSourceDocumentChunk, UiMessageSourceUrlChunk, UiMessageStartChunk,
        UiMessageStartStepChunk, UiMessageStreamOptions, UiMessageTextDeltaChunk,
        UiMessageTextEndChunk, UiMessageTextStartChunk, UiMessageToolApprovalRequestChunk,
        UiMessageToolApprovalResponseChunk, UiMessageToolInputAvailableChunk,
        UiMessageToolInputDeltaChunk, UiMessageToolInputErrorChunk, UiMessageToolInputStartChunk,
        UiMessageToolOutputAvailableChunk, UiMessageToolOutputDeniedChunk,
        UiMessageToolOutputErrorChunk, UiMessageWithoutId, UiPartState, UiProviderMetadata,
        UiReasoningFilePart, UiReasoningPart, UiSourceDocumentPart, UiSourceUrlPart, UiTextPart,
        UiToolApproval, UiToolApprovalDecision, UiToolApprovalRequest, UiToolApprovedApproval,
        UiToolDeniedApproval, UiToolInvocation, UiToolInvocationState, UiToolKind, UiToolPart,
        UiToolPartState, UnsupportedFunctionalityError, UnsupportedModelVersionError, Usage,
        UsageInputTokens, UsageOutputTokens, UseCompletionOptions, UserContent, UserContentPart,
        UserModelMessage, ValidationResult, VideoModelProviderMetadata, VideoModelResponseMetadata,
        Warning, add_image_model_usage, add_language_model_usage, as_language_model_usage,
        as_schema, as_schema_or_empty, create_null_language_model_usage, empty_json_schema,
        filter_active_tools, get_chunk_timeout_ms, get_static_tool_name, get_step_timeout_ms,
        get_tool_name, get_tool_or_dynamic_tool_name, get_tool_timeout_ms, get_total_timeout_ms,
        has_tool_call, is_custom_content_ui_part, is_data_ui_message_chunk, is_data_ui_part,
        is_dynamic_tool_ui_part, is_file_ui_part, is_loop_finished, is_reasoning_file_ui_part,
        is_reasoning_ui_part, is_static_tool_ui_part, is_step_count, is_stop_condition_met,
        is_text_ui_part, is_tool_ui_part, json_schema, json_schema_with_validator,
        last_assistant_message_is_complete_with_approval_responses,
        last_assistant_message_is_complete_with_tool_calls, lazy_schema,
        prepare_language_model_v4_prompt, prepare_tool_choice, prune_messages,
    };

    pub mod registry {
        pub use crate::registry::{
            BuildContext, CompletionModelHandle, EmbeddingModelHandle, ImageModelHandle,
            LanguageModelHandle, ProviderBuildOverrides, ProviderFactory, ProviderRegistryHandle,
            RegistryOptions, RerankingModelHandle, SpeechModelHandle, TranscriptionModelHandle,
            VideoModelHandle, create_bare_registry, create_empty_registry,
            create_provider_registry,
        };

        #[cfg(any(
            feature = "openai",
            feature = "azure",
            feature = "anthropic",
            feature = "google",
            feature = "google-vertex",
            feature = "ollama",
            feature = "xai",
            feature = "groq",
            feature = "minimaxi",
            feature = "deepseek",
            feature = "deepinfra",
            feature = "cohere",
            feature = "togetherai",
            feature = "bedrock",
            feature = "gateway"
        ))]
        pub use crate::registry::{
            builtin_provider_factory, create_registry_with_defaults, global,
            openai_compatible_provider_factory,
        };

        #[cfg(feature = "azure")]
        pub use crate::registry::azure_provider_factory_with_options;
    }
}

/// Explicit compatibility prelude for migration-oriented imports.
///
/// Prefer `prelude::unified::*` for new code.
pub mod compat {
    #[allow(deprecated)]
    pub use crate::compat::{
        CallSettings, Experimental_GenerateImageResult, Experimental_GeneratedImage,
        Experimental_LanguageModelStreamPart, Experimental_SpeechResult,
        Experimental_TranscriptionResult, ExperimentalLanguageModelStreamPart, Provider, Siumai,
        SiumaiBuilder, StreamingToolCallDelta, StreamingToolCallFunctionDelta,
        StreamingToolCallTracker, StreamingToolCallTrackerOptions, StreamingToolCallTypeValidation,
        experimental_filter_active_tools, step_count_is,
    };

    /// Narrow legacy type namespace for migration-oriented imports.
    pub mod types {
        pub use crate::compat::types::{ChatMessage, StopCondition, Tool, Warning};

        /// Historical catch-all type namespace for last-resort migrations.
        pub mod legacy_all {
            pub use crate::compat::types::legacy_all::*;
        }
    }

    /// Legacy chat content payloads.
    pub mod content {
        pub use crate::compat::content::*;
    }
}

/// Non-unified extension capabilities (provider-specific or non-family features).
pub mod extensions {
    pub use crate::extensions::*;
}
