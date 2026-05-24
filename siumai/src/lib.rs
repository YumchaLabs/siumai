//! # Siumai - A Unified LLM Interface Library
//!
//! Siumai is a unified LLM interface library for Rust, supporting multiple AI providers.
//! It adopts a trait-separated architectural pattern and provides a type-safe API.
//!
#![deny(unsafe_code)]

//! ## Features
//!
//! - **Capability Separation**: Uses traits to distinguish different AI capabilities (chat, audio, vision, etc.)
//! - **Shared Parameters**: AI parameters are shared as much as possible, with extension points for provider-specific parameters.
//! - **Construction**: Prefer registry/config-first construction; builder-style construction remains available under `siumai::compat` as a time-bounded compatibility convenience.
//! - **Type Safety**: Leverages Rust's type system to ensure compile-time safety.
//! - **HTTP Customization**: Supports passing in a reqwest client and custom HTTP configurations.
//! - **Library First**: Focuses on core library functionality, avoiding application-layer features.
//! - **Flexible Capability Access**: Capability checks serve as hints rather than restrictions, allowing users to try new model features.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use siumai::prelude::unified::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Recommended construction: resolve a model handle from the registry.
//!     // Note: API key is automatically read from `OPENAI_API_KEY`.
//!     let model = registry::global().language_model("openai:gpt-4o-mini")?;
//!
//!     // Recommended invocation style: model-family APIs.
//!     let request = ChatRequest::new(vec![user!("Hello, world!")]);
//!     let response = siumai::text::generate(&model, request, siumai::text::GenerateOptions::default())
//!         .await?;
//!     println!("Response: {}", response.content_text().unwrap_or_default());
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Capability Access Philosophy
//!
//! Siumai takes a **permissive and quiet approach** to capability access. It never blocks operations
//! based on static capability information, and doesn't generate noise with automatic warnings.
//! The actual API determines what's supported:
//!
//! ```rust,no_run
//! use siumai::prelude::unified::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Recommended construction: resolve a model handle from the registry.
//!     // Note: API key is automatically read from `OPENAI_API_KEY`.
//!     let model = registry::global().language_model("openai:gpt-4o")?; // Vision-capable model
//!
//!     // Vercel-aligned approach: image understanding is done via multimodal Chat messages.
//!     // (No separate "VisionCapability" unified surface.)
//!     let messages = vec![user_with_image!("Describe this image", "https://example.com/a.png")];
//!     let resp = siumai::text::generate(
//!         &model,
//!         ChatRequest::new(messages),
//!         siumai::text::GenerateOptions::default(),
//!     )
//!     .await?;
//!     println!("Answer: {}", resp.content_text().unwrap_or_default());
//!
//!     Ok(())
//! }
//! ```

/// Enabled providers at compile time
pub const ENABLED_PROVIDERS: &str = env!("SIUMAI_ENABLED_PROVIDERS");

/// Number of enabled providers at compile time
pub const PROVIDER_COUNT: &str = env!("SIUMAI_PROVIDER_COUNT");

// Workspace split facade (beta.5):
// - siumai-core: provider-agnostic runtime + types
// - siumai-registry: registry handle + factories
// Stable facade modules (recommended):
// Prefer `siumai::prelude::unified::*` + `siumai::provider_ext::<provider>::*`.

/// Internal re-exports used by `#[macro_export]` macros.
///
/// These are intentionally not part of the stable public API surface.
#[doc(hidden)]
pub mod __private {
    /// Attach Anthropic prompt-cache request options for legacy cache-control macros.
    ///
    /// This keeps macro expansion on the facade-private path now that the former
    /// `ChatMessageBuilder::cache_control(...)` helper has been removed from `siumai-spec`,
    /// while preserving the macro's historical output shape.
    pub fn with_anthropic_cache_control(
        mut message: types::ChatMessage,
        cache: types::CacheControl,
    ) -> types::ChatMessage {
        let cache_json = match &cache {
            types::CacheControl::Ephemeral => serde_json::json!({ "type": "ephemeral" }),
            types::CacheControl::Persistent { ttl } => {
                let mut value = serde_json::json!({ "type": "ephemeral" });
                if let Some(duration) = ttl {
                    value["ttl"] = serde_json::json!(duration.as_secs());
                }
                value
            }
        };

        message.metadata.cache_control = Some(cache);
        message.provider_options.insert(
            "anthropic",
            serde_json::json!({ "cacheControl": cache_json }),
        );
        message
    }

    pub use siumai_core::types;
}

/// Hosted tools are part of the stable unified experience (Vercel-aligned).
pub mod hosted_tools;
pub use siumai_core::types::{
    convert_data_content_to_base64_string, convert_data_content_to_uint8_array,
    convert_uint8_array_to_text,
};
/// AI SDK-style utility helpers.
pub use siumai_core::utils::{delay, is_abort_error};
pub use siumai_provider_utils::standards::{ToolNameMapping, create_tool_name_mapping};
/// Provider-utils owned AI SDK-style utility helpers.
pub use siumai_provider_utils::{
    Arrayable, DEFAULT_ID_ALPHABET, DEFAULT_ID_SIZE, DEFAULT_JSON_GENERIC_SUFFIX,
    DEFAULT_JSON_SCHEMA_PREFIX, DEFAULT_JSON_SCHEMA_SUFFIX, DEFAULT_MAX_DOWNLOAD_SIZE,
    DEFAULT_REASONING_BUDGET_PERCENTAGES, Download, DownloadOptions, DownloadedFile, HeaderRecord,
    IdGenerator, IdGeneratorOptions, JsonInstructionMessageOptions, JsonInstructionOptions,
    JsonParseResult, LoadApiKeyOptions, LoadOptionalSettingOptions, LoadSettingOptions,
    ReasoningBudgetOptions, ReasoningLevel, ReasoningLevelConversionError, SerialJobExecutor,
    SupportedUrlMap, TypeValidationResult, UrlSupportRegex, VERSION, as_array, combine_headers,
    convert_base64_to_uint8_array, convert_image_model_file_to_data_uri, convert_to_base64,
    convert_uint8_array_to_base64, cosine_similarity, create_download, create_id_generator,
    download_url, extract_response_headers, filter_nullable, generate_id, get_error_message,
    get_runtime_environment_user_agent, get_text_from_data_url, inject_json_instruction,
    inject_json_instruction_into_messages, is_custom_reasoning, is_deep_equal_data,
    is_non_nullable, is_parsable_json, is_provider_reference, is_url_supported, load_api_key,
    load_optional_setting, load_setting, map_reasoning_to_provider_budget,
    map_reasoning_to_provider_effort, media_type_to_extension, normalize_header_map,
    normalize_headers, normalize_optional_headers, parse_json, parse_json_with_schema,
    parse_provider_options, read_response_with_size_limit, remove_undefined_entries,
    resolve_provider_reference, safe_parse_json, safe_parse_json_with_schema, safe_validate_types,
    strip_file_extension, validate_download_url, validate_types, with_user_agent_suffix,
    without_trailing_slash,
};

/// Protocol mapping facade (stable imports for protocol standards).
pub mod protocol;
/// Provider-defined tool factories (Vercel-aligned).
pub mod tools;

// Unified retry facade (siumai-core re-export + provider-aware defaults)
mod request_options;
pub mod retry_api;

/// Model families (recommended Rust-first surface).
pub mod completion;
pub mod embedding;
/// High-level file upload helper aligned with AI SDK `uploadFile`.
pub mod files;
pub mod image;
pub mod rerank;
/// High-level skill upload helper aligned with AI SDK `uploadSkill`.
pub mod skills;
pub mod speech;
/// Structured output helpers (JSON extraction + parsing).
pub mod structured_output;
pub use structured_output::{
    GenerateObjectOptions, GenerateObjectResult, GenerateObjectSchema, PartialJsonParseResult,
    PartialJsonParseState, PartialJsonValueStream, PartialJsonValueStreamEvent, RepairTextContext,
    RepairTextFunction, RepairTextFuture, fix_partial_json, generate_array, generate_choice,
    generate_enum, generate_json, generate_object, parse_partial_json, partial_json_value_stream,
};
pub mod text;
pub use text::generate_text;
pub mod transcription;
/// AI SDK-style `UIMessage` validation and conversion helpers.
pub mod ui;
/// Task-oriented video generation family helpers.
pub mod video;

/// Embed one text value through the high-level AI SDK-style helper surface.
pub async fn embed<M, V>(
    model: &M,
    value: V,
    options: embedding::EmbedOptions,
) -> Result<siumai_core::types::EmbedResult, siumai_core::error::LlmError>
where
    M: embedding::EmbeddingModel + ?Sized,
    V: Into<String>,
{
    embedding::embed_value(model, value, options).await
}

/// Embed several text values through the high-level AI SDK-style helper surface.
pub async fn embed_many<M>(
    model: &M,
    values: Vec<String>,
    options: embedding::EmbedOptions,
) -> Result<siumai_core::types::EmbedManyResult, siumai_core::error::LlmError>
where
    M: embedding::EmbeddingModel + ?Sized,
{
    embedding::embed_values(model, values, options).await
}

/// Rerank documents through the high-level AI SDK-style helper surface.
///
/// The JSON projection preserves both text and structured-document requests. Use
/// `siumai::rerank::rerank(...)` when you need the raw Rust-first `RerankResponse`.
pub async fn rerank<M>(
    model: &M,
    request: rerank::RerankRequest,
    options: rerank::RerankOptions,
) -> Result<
    siumai_core::types::RerankResult<siumai_core::types::JSONValue>,
    siumai_core::error::LlmError,
>
where
    M: rerank::RerankingModel + ?Sized,
{
    rerank::rerank_result(model, request, options).await
}

/// Generate images through the high-level AI SDK-style helper surface.
///
/// This root helper returns `GenerateImageResult`. Use
/// `siumai::image::generate_image(...)` when you need the raw Rust-first
/// `ImageGenerationResponse`.
pub async fn generate_image<M>(
    model: &M,
    request: image::GenerateImageRequest,
    options: image::GenerateOptions,
) -> Result<siumai_core::types::GenerateImageResult, siumai_core::error::LlmError>
where
    M: image::ImageModel + siumai_core::traits::ImageExtras + ?Sized,
{
    image::generate_image_result(model, request, options).await
}

/// Generate speech audio through the high-level AI SDK-style helper surface.
pub async fn generate_speech<M>(
    model: &M,
    request: speech::TtsRequest,
    options: speech::SynthesizeOptions,
) -> Result<speech::SpeechResult, siumai_core::error::LlmError>
where
    M: speech::SpeechModel + ?Sized,
{
    speech::synthesize(model, request, options).await
}

/// Transcribe audio through the high-level AI SDK-style helper surface.
pub async fn transcribe<M>(
    model: &M,
    request: transcription::SttRequest,
    options: transcription::TranscribeOptions,
) -> Result<transcription::TranscriptionResult, siumai_core::error::LlmError>
where
    M: transcription::TranscriptionModel + ?Sized,
{
    transcription::transcribe(model, request, options).await
}

/// Generate videos through the AI SDK-style experimental helper surface.
///
/// This returns the passive `GenerateVideoResult` envelope over generated files.
/// Use `siumai::video::generate(...)` when you need the Rust-first task-oriented result.
pub async fn experimental_generate_video<M>(
    model: &M,
    request: video::VideoGenerationRequest,
    options: video::GenerateOptions,
) -> Result<siumai_core::types::GenerateVideoResult, siumai_core::error::LlmError>
where
    M: video::VideoModel + ?Sized,
{
    video::experimental_generate_video_result(model, request, options).await
}

/// Upload a file through the high-level AI SDK-style helper surface.
pub async fn upload_file<A, D>(
    api: &A,
    data: D,
    options: files::UploadFileOptions,
) -> Result<files::UploadFileResult, siumai_core::error::LlmError>
where
    A: files::UploadFileApi + ?Sized,
    D: Into<siumai_core::types::DataContent>,
{
    files::upload(api, data, options).await
}

/// Upload a skill through the high-level AI SDK-style helper surface.
pub async fn upload_skill<A>(
    api: &A,
    files: Vec<skills::UploadSkillFile>,
    options: skills::UploadSkillOptions,
) -> Result<skills::UploadSkillResult, siumai_core::error::LlmError>
where
    A: skills::UploadSkillApi + ?Sized,
{
    skills::upload(api, files, options).await
}

/// Tool runtime (schema + execution binding).
pub mod tooling;

/// AI SDK-style tool runtime helpers.
pub use siumai_core::tooling::{
    ExecutableTool, ExecutableTools, ProviderDefinedToolFactory,
    ProviderDefinedToolFactoryWithOutputSchema, ProviderExecutedToolFactory, ToolExecuteFunction,
    ToolExecutionOptions, ToolExecutionResult, ToolExecutionStream, ToolModelOutputContext,
    ToolSet, create_provider_defined_tool_factory,
    create_provider_defined_tool_factory_with_output_schema, create_provider_executed_tool_factory,
    dynamic_tool, execute_tool, is_executable_tool, model_messages_from_chat_messages, tool,
};

/// AI SDK-style JSON event stream parser.
pub use siumai_core::streaming::parse_json_event_stream;

/// Compatibility surface for legacy, method-style APIs (time-bounded).
pub mod compat;

/// Directional content facade.
///
/// Prefer `content::prompt` for request input and `content::output` for generated response output.
/// Legacy serde-facing chat payloads remain explicit under `content::compat` / `compat::content`.
pub mod content;
// Compatibility / internal modules (kept but hidden to reduce accidental coupling).
//
// NOTE: These low-level modules are intentionally NOT re-exported at the top-level.
// Use `siumai::experimental::*` for advanced integrations and internal building blocks.

/// Experimental low-level APIs (advanced use only).
///
/// This module exposes lower-level building blocks from `siumai-core` (executors, middleware,
/// auth providers, etc.) without making them part of the stable facade surface.
///
/// Prefer `siumai::prelude::unified::*`, `siumai::hosted_tools::*`, and `siumai::provider_ext::*`
/// unless you are building integrations or custom providers.
pub mod experimental;

pub use siumai_registry::registry;

/// Stable alias for provider-specific extension surface.
///
/// This is a naming convenience for Vercel AI SDK alignment: provider packages expose
/// provider-owned helpers (tools/options/metadata) under a `providers::*` namespace.
pub use crate::provider_ext as providers;

/// Extension capabilities (non-unified surface).
///
/// These are capability/adapter-level traits and payloads rather than the stable family-model
/// entrypoints. Prefer `siumai::prelude::unified` for stable family execution. Video's stable
/// surface is `siumai::video::*` / `VideoModel`; the low-level `VideoGenerationCapability`
/// remains here for provider adapters and compatibility code. Music remains extension-only.
pub mod extensions;
/// Provider extension APIs (non-unified surface).
///
/// These are stable module paths for provider-specific endpoints/resources.
pub mod provider_ext;
// Model constants (simplified access)
pub use model_catalog::model_constants as models;

// Model constants (detailed access)
pub use model_catalog::constants;

/// Convenient pre-import module
pub mod prelude {
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
            GenerateObjectOptions, GenerateObjectResult, GenerateObjectSchema,
            PartialJsonParseResult, PartialJsonParseState, PartialJsonValueStream,
            PartialJsonValueStreamEvent, RepairTextContext, RepairTextFunction, RepairTextFuture,
            fix_partial_json, generate_array, generate_choice, generate_enum, generate_json,
            generate_object, parse_partial_json, partial_json_value_stream,
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
            IdGenerator, IdGeneratorOptions, SerialJobExecutor, ToolNameMapping,
            create_id_generator, create_tool_name_mapping, generate_id,
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
        pub use siumai_core::error::{ErrorCategory, LlmError, LlmErrorExt};
        pub use siumai_core::image::{ImageModel, ImageModelV4};
        pub use siumai_core::rerank::RerankingModel;
        pub use siumai_core::speech::SpeechModel;
        pub use siumai_core::streaming::{
            ChatStream, ChatStreamCustomContent, ChatStreamEvent, ChatStreamFileData,
            ChatStreamFilePart, ChatStreamFinishInfo, ChatStreamHandle, ChatStreamPart,
            ChatStreamReplay, ChatStreamToolApprovalRequest, ChatStreamToolCall,
            ChatStreamToolResult, StreamProviderMetadata,
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

        pub use siumai_core::embedding::EmbeddingModel;
        // Core request/response types for the seven stable model families.
        #[allow(deprecated)]
        pub use siumai_core::types::{
            AISDKError, APICallError, AssistantContent, AssistantContentPart,
            AssistantModelMessage, AudioStreamEvent, CacheControl, CallWarning, CallbackModelInfo,
            CancelHandle, ChatInit, ChatMessage, ChatRequest, ChatRequestBuilder,
            ChatRequestOptions, ChatResponse, ChatState, ChatStatus,
            ChatTransportReconnectToStreamOptions, ChatTransportSendMessagesOptions,
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
            InvalidArgumentError, InvalidDataContentError, InvalidMessageRoleError,
            InvalidPromptError, InvalidResponseDataError, InvalidStreamPartError,
            InvalidToolApprovalError, InvalidToolInputError, JSONParseError, JSONSchema7,
            JSONValue, LanguageModelCallOptions, LanguageModelInputTokenDetails,
            LanguageModelOutputTokenDetails, LanguageModelReasoning, LanguageModelRequestMetadata,
            LanguageModelResponseMetadata, LanguageModelStreamModelCallEndPart,
            LanguageModelStreamModelCallResponseMetadataPart,
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
            OnFinishEvent, OnStartEvent, OnStepFinishEvent, OnStepStartEvent,
            OnToolCallFinishEvent, OnToolCallStartEvent, OutputSchema,
            PrepareReconnectToStreamRequestOptions, PrepareSendMessagesRequestOptions,
            PrepareStepOptions, PrepareStepResult, PreparedReconnectToStreamRequest,
            PreparedSendMessagesRequest, Prompt, PromptExecutionError, PromptInput,
            PromptTokensDetails, PromptValidationError, ProviderDefinedTool, ProviderMetadata,
            ProviderOptions, ProviderOptionsMap, ProviderReference, ProviderType,
            PruneEmptyMessagesMode, PruneMessagesOptions, PruneReasoningMode, PruneToolCallMode,
            PruneToolCallRule, ReasoningFileOutput, ReasoningFilePart, ReasoningFileUIPart,
            ReasoningOutput, ReasoningPart, ReasoningUIPart, RequestCredentials, RequestOptions,
            RerankEndEvent, RerankRanking, RerankRankingEntry, RerankRequest, RerankResponse,
            RerankResponseMetadata, RerankResult, RerankStartEvent, RerankingModelCallEndEvent,
            RerankingModelCallRanking, RerankingModelCallStartEvent, ResponseFormat,
            ResponseMessage, ResponseMetadata, RetryError, RetryErrorReason, Schema,
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
            TextStreamToolOutputDeniedPart, TextStreamToolResultPart, TextUIPart,
            TimeoutConfiguration, TimeoutConfigurationSettings, TooManyEmbeddingValuesForCallError,
            Tool, ToolApprovalConfiguration, ToolApprovalDecisionContext, ToolApprovalRequest,
            ToolApprovalRequestOutput, ToolApprovalResponse, ToolApprovalResponseOutput,
            ToolApprovalStatus, ToolApprovalStatusDetails, ToolApprovalStatusType, ToolCall,
            ToolCallNotFoundForApprovalError, ToolCallPart, ToolCallRepairContext,
            ToolCallRepairError, ToolCallRepairFunctionError, ToolCallRepairResult, ToolChoice,
            ToolContent, ToolContentPart, ToolError, ToolExecutionEndEvent,
            ToolExecutionStartEvent, ToolModelMessage, ToolOutput, ToolOutputDenied, ToolResult,
            ToolResultContentPart, ToolResultOutput, ToolResultPart, ToolUIPart,
            TranscriptionModelResponseMetadata, TranscriptionResult, TranscriptionSegment,
            TtsRequest, TtsResponse, TypeValidationContext, TypeValidationError, TypedToolCall,
            TypedToolError, TypedToolOutputDenied, TypedToolResult, UI_MESSAGE_STREAM_HEADERS,
            UIDataPartSchemas, UIDataTypes, UIDataTypesToSchemas, UIMessage, UIMessageChunk,
            UIMessagePart, UIMessageStreamError, UIMessageStreamOptions, UITool, UIToolInvocation,
            UITools, UiCustomPart, UiDataPart, UiFilePart, UiMessage, UiMessageAbortChunk,
            UiMessageChunk, UiMessageCustomChunk, UiMessageDataChunk, UiMessageErrorChunk,
            UiMessageFileChunk, UiMessageFinishChunk, UiMessageFinishStepChunk,
            UiMessageMetadataChunk, UiMessagePart, UiMessageReasoningDeltaChunk,
            UiMessageReasoningEndChunk, UiMessageReasoningFileChunk, UiMessageReasoningStartChunk,
            UiMessageRole, UiMessageSourceDocumentChunk, UiMessageSourceUrlChunk,
            UiMessageStartChunk, UiMessageStartStepChunk, UiMessageStreamOptions,
            UiMessageTextDeltaChunk, UiMessageTextEndChunk, UiMessageTextStartChunk,
            UiMessageToolApprovalRequestChunk, UiMessageToolApprovalResponseChunk,
            UiMessageToolInputAvailableChunk, UiMessageToolInputDeltaChunk,
            UiMessageToolInputErrorChunk, UiMessageToolInputStartChunk,
            UiMessageToolOutputAvailableChunk, UiMessageToolOutputDeniedChunk,
            UiMessageToolOutputErrorChunk, UiMessageWithoutId, UiPartState, UiProviderMetadata,
            UiReasoningFilePart, UiReasoningPart, UiSourceDocumentPart, UiSourceUrlPart,
            UiTextPart, UiToolApproval, UiToolApprovalDecision, UiToolApprovalRequest,
            UiToolApprovedApproval, UiToolDeniedApproval, UiToolInvocation, UiToolInvocationState,
            UiToolKind, UiToolPart, UiToolPartState, UnsupportedFunctionalityError,
            UnsupportedModelVersionError, Usage, UsageInputTokens, UsageOutputTokens,
            UseCompletionOptions, UserContent, UserContentPart, UserModelMessage, ValidationResult,
            VideoModelProviderMetadata, VideoModelResponseMetadata, Warning, add_image_model_usage,
            add_language_model_usage, as_language_model_usage, as_schema, as_schema_or_empty,
            create_null_language_model_usage, empty_json_schema, filter_active_tools,
            get_chunk_timeout_ms, get_static_tool_name, get_step_timeout_ms, get_tool_name,
            get_tool_or_dynamic_tool_name, get_tool_timeout_ms, get_total_timeout_ms,
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
                LanguageModelHandle, ProviderBuildOverrides, ProviderFactory,
                ProviderRegistryHandle, RegistryOptions, RerankingModelHandle, SpeechModelHandle,
                TranscriptionModelHandle, VideoModelHandle, create_bare_registry,
                create_empty_registry, create_provider_registry,
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
                feature = "bedrock"
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
            Experimental_TranscriptionResult, ExperimentalLanguageModelStreamPart, Provider,
            Siumai, SiumaiBuilder, StreamingToolCallDelta, StreamingToolCallFunctionDelta,
            StreamingToolCallTracker, StreamingToolCallTrackerOptions,
            StreamingToolCallTypeValidation, experimental_filter_active_tools, step_count_is,
        };

        /// Historical broad type namespace for migration-oriented imports.
        pub mod types {
            pub use crate::compat::types::*;
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
}

// Macros moved to a dedicated module for cleanliness
mod macros;

mod model_catalog;

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(any(feature = "openai", feature = "anthropic"))]
    use crate::compat::Provider;
    use crate::compat::Siumai;
    use crate::prelude::unified::*;

    #[test]
    fn test_macros() {
        // Test simple macros that return ChatMessage directly
        let user_msg = user!("Hello");
        assert_eq!(user_msg.role, MessageRole::User);

        let system_msg = system!("You are helpful");
        assert_eq!(system_msg.role, MessageRole::System);

        let assistant_msg = assistant!("I can help");
        assert_eq!(assistant_msg.role, MessageRole::Assistant);

        // Test that content is correctly set
        match user_msg.content {
            MessageContent::Text(text) => assert_eq!(text, "Hello"),
            _ => panic!("Expected text content"),
        }
    }

    #[test]
    #[allow(deprecated)]
    fn test_provider_builder() {
        #[cfg(feature = "openai")]
        let _openai_builder = Provider::openai();
        #[cfg(feature = "anthropic")]
        let _anthropic_builder = Provider::anthropic();
        let _siumai_builder = Siumai::builder();
        // Basic test for builder creation
        // Placeholder test
    }
}
