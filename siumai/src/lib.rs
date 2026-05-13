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
    pub use siumai_core::types;
}

/// Shared stable data structures.
///
/// This keeps the historical `siumai::types::*` import path available while the
/// workspace remains split across `siumai-spec` and `siumai-core`.
pub mod types {
    pub use siumai_core::types::*;
}

/// Hosted tools are part of the stable unified experience (Vercel-aligned).
pub use siumai_core::hosted_tools;

pub use siumai_core::standards::{ToolNameMapping, create_tool_name_mapping};
/// AI SDK-style utility helpers.
pub use siumai_core::utils::{
    Arrayable, DEFAULT_ID_ALPHABET, DEFAULT_ID_SIZE, DEFAULT_JSON_GENERIC_SUFFIX,
    DEFAULT_JSON_SCHEMA_PREFIX, DEFAULT_JSON_SCHEMA_SUFFIX, DEFAULT_MAX_DOWNLOAD_SIZE,
    DEFAULT_REASONING_BUDGET_PERCENTAGES, Download, DownloadOptions, DownloadedFile, HeaderRecord,
    IdGenerator, IdGeneratorOptions, JsonInstructionMessageOptions, JsonInstructionOptions,
    JsonParseResult, LoadApiKeyOptions, LoadOptionalSettingOptions, LoadSettingOptions,
    ReasoningBudgetOptions, ReasoningLevel, ReasoningLevelConversionError, SerialJobExecutor,
    StreamingToolCallDelta, StreamingToolCallFunctionDelta, StreamingToolCallTracker,
    StreamingToolCallTrackerOptions, StreamingToolCallTypeValidation, SupportedUrlMap,
    TypeValidationResult, UrlSupportRegex, VERSION, as_array, combine_headers,
    convert_base64_to_uint8_array, convert_image_model_file_to_data_uri, convert_to_base64,
    convert_uint8_array_to_base64, cosine_similarity, create_download, create_id_generator, delay,
    download_url, extract_response_headers, filter_nullable, generate_id, get_error_message,
    get_runtime_environment_user_agent, get_text_from_data_url, inject_json_instruction,
    inject_json_instruction_into_messages, is_abort_error, is_custom_reasoning, is_deep_equal_data,
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
///
/// This module exists to decouple downstream code from internal crate names.
/// Over time we may rename protocol crates (e.g. move away from `*-compatible` naming),
/// but `siumai::protocol::*` should remain stable.
pub mod protocol {
    /// OpenAI-like protocol standard mapping (Chat/Embedding/Image/Rerank).
    ///
    /// Backed by `siumai-protocol-openai` (preferred; wraps the legacy `*-compatible` crate name).
    #[cfg(any(feature = "openai", feature = "protocol-openai"))]
    pub mod openai {
        pub use siumai_protocol_openai::standards::openai::*;
    }

    /// Anthropic Messages protocol standard mapping (Chat + streaming).
    ///
    /// Backed by `siumai-protocol-anthropic` (preferred; wraps the legacy `*-compatible` crate name).
    #[cfg(any(feature = "anthropic", feature = "protocol-anthropic"))]
    pub mod anthropic {
        pub use siumai_protocol_anthropic::standards::anthropic::*;
    }

    /// Google Gemini protocol standard mapping (GenerateContent + streaming).
    ///
    /// Backed by `siumai-protocol-gemini`.
    #[cfg(any(feature = "google", feature = "protocol-gemini"))]
    pub mod gemini {
        pub use siumai_protocol_gemini::standards::gemini::*;
    }
}

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

// Compatibility / internal modules (kept but hidden to reduce accidental coupling).
//
// NOTE: These low-level modules are intentionally NOT re-exported at the top-level.
// Use `siumai::experimental::*` for advanced integrations and internal building blocks.

/// Legacy builder module (provider construction internals).
///
/// Prefer `registry::global()` (or config-first provider constructors) for stable construction.
#[doc(hidden)]
pub mod builder {
    pub use siumai_core::builder::*;
}

/// Experimental low-level APIs (advanced use only).
///
/// This module exposes lower-level building blocks from `siumai-core` (executors, middleware,
/// auth providers, etc.) without making them part of the stable facade surface.
///
/// Prefer `siumai::prelude::unified::*`, `siumai::hosted_tools::*`, and `siumai::provider_ext::*`
/// unless you are building integrations or custom providers.
pub mod experimental {
    pub mod core {
        pub use siumai_core::core::*;
    }

    /// Protocol bridge contracts (advanced API).
    ///
    /// This exposes bridge decision/report types used by cross-protocol request,
    /// response, and stream adapters.
    pub mod bridge {
        pub use siumai_bridge::*;
    }

    /// Non-streaming response encoding utilities (advanced API).
    ///
    /// This exposes protocol-level JSON encoders used by gateways/proxies to re-serialize
    /// unified `ChatResponse` objects into provider-native JSON response bodies.
    pub mod encoding {
        pub use siumai_core::encoding::{
            JsonEncodeOptions, JsonResponseConverter, encode_chat_response_as_json,
        };
    }

    /// Streaming utilities (advanced API).
    ///
    /// This exposes low-level building blocks from `siumai-core` that are useful when building
    /// gateways/proxies that need to re-serialize streams into provider-native wire formats.
    pub mod streaming {
        pub use siumai_core::streaming::{
            ChatByteStream, LanguageModelV4StreamCustomContent, LanguageModelV4StreamFile,
            LanguageModelV4StreamFileData, LanguageModelV4StreamFinishReason,
            LanguageModelV4StreamInputTokens, LanguageModelV4StreamOutputTokens,
            LanguageModelV4StreamPart, LanguageModelV4StreamReasoningFile,
            LanguageModelV4StreamResponseMetadata, LanguageModelV4StreamSource,
            LanguageModelV4StreamToolApprovalRequest, LanguageModelV4StreamToolCall,
            LanguageModelV4StreamToolResult, LanguageModelV4StreamUsage,
            OpenAiResponsesStreamPartsBridge, SharedV4ProviderMetadata, SharedV4Warning,
            StreamPartNamespace, TypedStreamPart, UnsupportedStreamPartBehavior,
            encode_chat_stream_as_jsonl, encode_chat_stream_as_sse, ensure_stream_end,
            transform_chat_event_stream,
        };
    }

    /// Custom provider API (advanced).
    pub mod custom_provider {
        pub use siumai_core::custom_provider::*;
    }

    /// Provider implementation internals (advanced API).
    ///
    /// If you find yourself importing from here, consider depending on the relevant
    /// provider crate directly instead of going through the facade.
    pub mod providers {
        pub use siumai_core as core;

        #[cfg(feature = "bedrock")]
        pub use siumai_provider_amazon_bedrock as amazon_bedrock;
        #[cfg(feature = "anthropic")]
        pub use siumai_provider_anthropic as anthropic;
        #[cfg(feature = "azure")]
        pub use siumai_provider_azure as azure;
        #[cfg(feature = "cohere")]
        pub use siumai_provider_cohere as cohere;
        #[cfg(feature = "deepseek")]
        pub use siumai_provider_deepseek as deepseek;
        #[cfg(feature = "google")]
        pub use siumai_provider_gemini as gemini;
        #[cfg(feature = "google-vertex")]
        pub use siumai_provider_google_vertex as google_vertex;
        #[cfg(feature = "groq")]
        pub use siumai_provider_groq as groq;
        #[cfg(feature = "minimaxi")]
        pub use siumai_provider_minimaxi as minimaxi;
        #[cfg(feature = "ollama")]
        pub use siumai_provider_ollama as ollama;
        #[cfg(feature = "openai")]
        pub use siumai_provider_openai as openai;
        #[cfg(feature = "togetherai")]
        pub use siumai_provider_togetherai as togetherai;
        #[cfg(feature = "xai")]
        pub use siumai_provider_xai as xai;
    }

    /// Protocol mapping and adapter helpers (advanced API).
    ///
    /// Prefer `siumai::prelude::unified::*` unless you are building integrations or custom providers.
    pub mod standards {
        #[cfg(feature = "bedrock")]
        pub use siumai_provider_amazon_bedrock::standards::bedrock;
        #[cfg(feature = "anthropic")]
        pub use siumai_provider_anthropic::standards::anthropic;
        #[cfg(feature = "cohere")]
        pub use siumai_provider_cohere::standards::cohere;
        #[cfg(feature = "google")]
        pub use siumai_provider_gemini::standards::gemini;
        #[cfg(feature = "ollama")]
        pub use siumai_provider_ollama::standards::ollama;
        #[cfg(feature = "openai")]
        pub use siumai_provider_openai::standards::openai;
        #[cfg(feature = "togetherai")]
        pub use siumai_provider_togetherai::standards::togetherai;
    }

    pub use siumai_core::{auth, client, defaults, execution, observability, params, retry, utils};
}

#[cfg(feature = "openai")]
pub use siumai_provider_openai_compatible::siumai_for_each_openai_compatible_provider;
pub use siumai_registry::registry;

/// Stable alias for provider-specific extension surface.
///
/// This is a naming convenience for Vercel AI SDK alignment: provider packages expose
/// provider-owned helpers (tools/options/metadata) under a `providers::*` namespace.
pub use crate::provider_ext as providers;

/// Provider extension APIs (non-unified surface).
///
/// These are stable module paths for provider-specific endpoints/resources.
pub mod provider_ext;

// Unified interface (`Siumai`) and builder
pub mod provider;
pub mod provider_catalog;

/// Extension capabilities (non-unified surface).
///
/// These are intentionally *not* part of the Vercel-aligned unified model families.
/// Prefer `siumai::prelude::unified` for the stable unified surface.
pub mod extensions {
    pub use siumai_core::traits::{
        AudioCapability, EmbeddingCapability, FileManagementCapability, ImageExtras,
        ModelListingCapability, ModerationCapability, MusicGenerationCapability, RerankCapability,
        SkillsCapability, SpeechExtras, TimeoutCapability, TranscriptionExtras,
        VideoGenerationCapability,
    };

    /// Types used by non-unified extension capabilities.
    pub mod types {
        pub use siumai_core::types::{
            FileDeleteResponse, FileListQuery, FileListResponse, FileObject, FileUploadRequest,
            ImageEditInput, ImageEditRequest, ImageVariationRequest, ModerationRequest,
            ModerationResponse, SkillFileContent, SkillProviderMetadata, SkillUploadFile,
            SkillUploadRequest, SkillUploadResult, VideoGenerationInput, VideoGenerationRequest,
            VideoGenerationResponse, VideoTaskStatus, VideoTaskStatusResponse,
        };
    }
}

/// Global registry handle with built-in provider factories.
///
/// Only available when at least one provider feature is enabled.
#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "cohere",
    feature = "togetherai",
    feature = "ollama",
    feature = "xai",
    feature = "groq",
    feature = "minimaxi"
))]
pub use registry::global as registry_global;

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
    /// This module centers the six stable model families:
    /// Language/Embedding/Image/Reranking/Speech/Transcription.
    ///
    /// Compatibility-oriented construction aliases remain source-compatible for now,
    /// but are hidden from docs and also exposed explicitly under `prelude::compat`.
    pub mod unified {
        #[doc(hidden)]
        pub use crate::Provider;
        pub use crate::experimental_generate_video;
        pub use crate::files::{
            FileUploadProvider, UploadFileApi, UploadFileOptions, UploadFileProviderMetadata,
            UploadFileResult,
        };
        #[allow(deprecated)]
        pub use crate::image::experimental_generate_image;
        pub use crate::parse_json_event_stream;
        #[doc(hidden)]
        pub use crate::provider::Siumai;
        pub use crate::registry::ProviderFactory;
        pub use crate::retry_api::*;
        pub use crate::skills::{
            UploadSkillApi, UploadSkillFile, UploadSkillFileContent, UploadSkillOptions,
            UploadSkillProviderMetadata, UploadSkillResult,
        };
        #[allow(deprecated)]
        pub use crate::speech::experimental_generate_speech;
        pub use crate::structured_output::{
            GenerateObjectOptions, GenerateObjectResult, GenerateObjectSchema,
            PartialJsonParseResult, PartialJsonParseState, PartialJsonValueStream,
            PartialJsonValueStreamEvent, RepairTextContext, RepairTextFunction, RepairTextFuture,
            fix_partial_json, generate_array, generate_choice, generate_enum, generate_json,
            generate_object, parse_partial_json, partial_json_value_stream,
        };
        pub use crate::tooling;
        pub use crate::tools;
        #[allow(deprecated)]
        pub use crate::transcription::experimental_transcribe;
        pub use crate::{
            Arrayable, DEFAULT_ID_ALPHABET, DEFAULT_ID_SIZE, DEFAULT_JSON_GENERIC_SUFFIX,
            DEFAULT_JSON_SCHEMA_PREFIX, DEFAULT_JSON_SCHEMA_SUFFIX, DEFAULT_MAX_DOWNLOAD_SIZE,
            DEFAULT_REASONING_BUDGET_PERCENTAGES, Download, DownloadOptions, DownloadedFile,
            HeaderRecord, IdGenerator, IdGeneratorOptions, JsonInstructionMessageOptions,
            JsonInstructionOptions, JsonParseResult, LoadApiKeyOptions, LoadOptionalSettingOptions,
            LoadSettingOptions, ReasoningBudgetOptions, ReasoningLevel,
            ReasoningLevelConversionError, SerialJobExecutor, StreamingToolCallDelta,
            StreamingToolCallFunctionDelta, StreamingToolCallTracker,
            StreamingToolCallTrackerOptions, StreamingToolCallTypeValidation, SupportedUrlMap,
            ToolNameMapping, TypeValidationResult, UrlSupportRegex, VERSION, as_array,
            combine_headers, convert_base64_to_uint8_array, convert_image_model_file_to_data_uri,
            convert_to_base64, convert_uint8_array_to_base64, cosine_similarity, create_download,
            create_id_generator, create_tool_name_mapping, delay, download_url,
            extract_response_headers, filter_nullable, generate_id, get_error_message,
            get_runtime_environment_user_agent, get_text_from_data_url, inject_json_instruction,
            inject_json_instruction_into_messages, is_abort_error, is_custom_reasoning,
            is_deep_equal_data, is_non_nullable, is_parsable_json, is_provider_reference,
            is_url_supported, load_api_key, load_optional_setting, load_setting,
            map_reasoning_to_provider_budget, map_reasoning_to_provider_effort,
            media_type_to_extension, normalize_header_map, normalize_headers,
            normalize_optional_headers, parse_json, parse_json_with_schema, parse_provider_options,
            read_response_with_size_limit, remove_undefined_entries, resolve_provider_reference,
            safe_parse_json, safe_parse_json_with_schema, safe_validate_types,
            strip_file_extension, validate_download_url, validate_types, with_user_agent_suffix,
            without_trailing_slash,
        };
        pub use crate::{
            ExecutableTool, ExecutableTools, ProviderDefinedToolFactory,
            ProviderDefinedToolFactoryWithOutputSchema, ProviderExecutedToolFactory,
            ToolExecuteFunction, ToolExecutionOptions, ToolExecutionResult, ToolExecutionStream,
            ToolModelOutputContext, ToolSet, create_provider_defined_tool_factory,
            create_provider_defined_tool_factory_with_output_schema,
            create_provider_executed_tool_factory, dynamic_tool, execute_tool, is_executable_tool,
            model_messages_from_chat_messages,
        };
        pub use crate::{assistant, conversation, conversation_with_system, messages, quick_chat};
        pub use crate::{
            completion, embedding, files, image, rerank, skills, speech, structured_output, text,
            transcription, video,
        };
        pub use crate::{
            embed, embed_many, generate_image, generate_speech, generate_text, transcribe,
            upload_file, upload_skill,
        };
        pub use crate::{system, tool, user, user_with_image};
        pub use siumai_core::completion::CompletionModel;
        pub use siumai_core::error::{ErrorCategory, LlmError};
        pub use siumai_core::execution::middleware::LanguageModelMiddleware;
        pub use siumai_core::image::{ImageModel, ImageModelV4};
        pub use siumai_core::rerank::RerankingModel;
        pub use siumai_core::speech::SpeechModel;
        pub use siumai_core::streaming::*;
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
        // Core request/response types for the six stable model families.
        #[allow(deprecated)]
        pub use siumai_core::types::{
            AISDKError, APICallError, AssistantContent, AssistantContentPart,
            AssistantModelMessage, AudioStreamEvent, CacheControl, CallSettings, CallWarning,
            CallbackModelInfo, CancelHandle, ChatInit, ChatMessage, ChatRequest,
            ChatRequestBuilder, ChatRequestOptions, ChatResponse, ChatState, ChatStatus,
            ChatTransportReconnectToStreamOptions, ChatTransportSendMessagesOptions,
            ChatTransportTrigger, CommonParams, CompletionRequest, CompletionRequestOptions,
            CompletionResponse, CompletionStreamProtocol, CompletionTokensDetails, ContentPart,
            Context, CreateUIMessage, CustomContentUIPart, CustomOutput, CustomPart,
            CustomProviderOptions, DataContent, DataUIMessageChunk, DataUIPart,
            DefaultGeneratedAudioFile, DefaultGeneratedAudioFileWithType, DefaultGeneratedFile,
            DefaultGeneratedFileWithType, DefaultStepResult, DownloadError, DynamicToolCall,
            DynamicToolError, DynamicToolResult, DynamicToolUIPart, EmbedEndEvent, EmbedManyResult,
            EmbedOutput, EmbedResponseData, EmbedResult, EmbedStartEvent, EmbedValue, Embedding,
            EmbeddingModelCallEndEvent, EmbeddingModelCallStartEvent, EmbeddingModelUsage,
            EmbeddingRequest, EmbeddingResponse, EmptyResponseBodyError,
            Experimental_GenerateImageResult, Experimental_GeneratedImage,
            Experimental_LanguageModelStreamPart, Experimental_SpeechResult,
            Experimental_TranscriptionResult, ExperimentalLanguageModelStreamPart, FileOutput,
            FilePart, FilePartSource, FileUIPart, FinishReason, FlexibleSchema,
            GenerateImagePrompt, GenerateImageRequest, GenerateImageResult, GenerateObjectEndEvent,
            GenerateObjectOutputStrategy, GenerateObjectResponseMetadata, GenerateObjectStartEvent,
            GenerateObjectStepEndEvent, GenerateObjectStepStartEvent, GenerateTextContentPart,
            GenerateTextEndEvent, GenerateTextModelInfo, GenerateTextReasoningPart,
            GenerateTextResponseMetadata, GenerateTextResult, GenerateTextStartEvent,
            GenerateTextStepEndEvent, GenerateTextStepReasoningPart, GenerateTextStepResult,
            GenerateTextStepStartEvent, GenerateVideoResult, GeneratedAudioFile, GeneratedFile,
            GeneratedImage, HttpChatTransportInitOptions, HttpConfig, ImageDetail,
            ImageGenerationRequest, ImageGenerationResponse, ImageModelProviderMetadata,
            ImageModelResponseMetadata, ImageModelUsage, ImagePart, InferUIDataParts,
            InferUIMessageChunk, InferUIMessageData, InferUIMessageMetadata, InferUIMessagePart,
            InferUIMessageToolCall, InferUIMessageToolOutputs, InferUIMessageTools, InferUITool,
            InferUITools, InvalidArgumentError, InvalidDataContentError, InvalidMessageRoleError,
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
            ToolResultOutput, ToolResultPart, ToolUIPart, TranscriptionModelResponseMetadata,
            TranscriptionResult, TranscriptionSegment, TtsRequest, TtsResponse,
            TypeValidationContext, TypeValidationError, TypedToolCall, TypedToolError,
            TypedToolOutputDenied, TypedToolResult, UI_MESSAGE_STREAM_HEADERS, UIDataPartSchemas,
            UIDataTypes, UIDataTypesToSchemas, UIMessage, UIMessageChunk, UIMessagePart,
            UIMessageStreamError, UIMessageStreamOptions, UITool, UIToolInvocation, UITools,
            UiCustomPart, UiDataPart, UiFilePart, UiMessage, UiMessageAbortChunk, UiMessageChunk,
            UiMessageCustomChunk, UiMessageDataChunk, UiMessageErrorChunk, UiMessageFileChunk,
            UiMessageFinishChunk, UiMessageFinishStepChunk, UiMessageMetadataChunk, UiMessagePart,
            UiMessageReasoningDeltaChunk, UiMessageReasoningEndChunk, UiMessageReasoningFileChunk,
            UiMessageReasoningStartChunk, UiMessageRole, UiMessageSourceDocumentChunk,
            UiMessageSourceUrlChunk, UiMessageStartChunk, UiMessageStartStepChunk,
            UiMessageStreamOptions, UiMessageTextDeltaChunk, UiMessageTextEndChunk,
            UiMessageTextStartChunk, UiMessageToolApprovalRequestChunk,
            UiMessageToolApprovalResponseChunk, UiMessageToolInputAvailableChunk,
            UiMessageToolInputDeltaChunk, UiMessageToolInputErrorChunk,
            UiMessageToolInputStartChunk, UiMessageToolOutputAvailableChunk,
            UiMessageToolOutputDeniedChunk, UiMessageToolOutputErrorChunk, UiMessageWithoutId,
            UiPartState, UiProviderMetadata, UiReasoningFilePart, UiReasoningPart,
            UiSourceDocumentPart, UiSourceUrlPart, UiTextPart, UiToolApproval,
            UiToolApprovalDecision, UiToolApprovalRequest, UiToolApprovedApproval,
            UiToolDeniedApproval, UiToolInvocation, UiToolInvocationState, UiToolKind, UiToolPart,
            UiToolPartState, UnsupportedFunctionalityError, UnsupportedModelVersionError, Usage,
            UsageInputTokens, UsageOutputTokens, UseCompletionOptions, UserContent,
            UserContentPart, UserModelMessage, ValidationResult, VideoModelProviderMetadata,
            VideoModelResponseMetadata, Warning, add_image_model_usage, add_language_model_usage,
            as_language_model_usage, as_schema, as_schema_or_empty,
            convert_data_content_to_base64_string, convert_data_content_to_uint8_array,
            convert_uint8_array_to_text, create_null_language_model_usage, empty_json_schema,
            experimental_filter_active_tools, filter_active_tools, get_chunk_timeout_ms,
            get_static_tool_name, get_step_timeout_ms, get_tool_name,
            get_tool_or_dynamic_tool_name, get_tool_timeout_ms, get_total_timeout_ms,
            has_tool_call, is_custom_content_ui_part, is_data_ui_message_chunk, is_data_ui_part,
            is_dynamic_tool_ui_part, is_file_ui_part, is_loop_finished, is_reasoning_file_ui_part,
            is_reasoning_ui_part, is_static_tool_ui_part, is_step_count, is_stop_condition_met,
            is_text_ui_part, is_tool_ui_part, json_schema, json_schema_with_validator,
            last_assistant_message_is_complete_with_approval_responses,
            last_assistant_message_is_complete_with_tool_calls, lazy_schema,
            prepare_language_model_v4_prompt, prepare_tool_choice, prune_messages, step_count_is,
        };

        pub mod registry {
            pub use crate::registry::{
                CompletionModelHandle, EmbeddingModelHandle, ImageModelHandle, LanguageModelHandle,
                ProviderFactory, ProviderRegistryHandle, RegistryOptions, RerankingModelHandle,
                SpeechModelHandle, TranscriptionModelHandle, VideoModelHandle,
                create_bare_registry, create_empty_registry, create_provider_registry,
            };

            #[cfg(any(
                feature = "openai",
                feature = "anthropic",
                feature = "google",
                feature = "ollama",
                feature = "xai",
                feature = "groq",
                feature = "minimaxi",
                feature = "deepseek",
                feature = "cohere",
                feature = "togetherai",
                feature = "bedrock"
            ))]
            pub use crate::registry::{create_registry_with_defaults, global};
        }
    }

    /// Explicit compatibility prelude for migration-oriented imports.
    ///
    /// Prefer `prelude::unified::*` for new code.
    pub mod compat {
        pub use crate::Provider;
        pub use crate::compat::{Siumai, SiumaiBuilder};
    }

    /// Non-unified extension capabilities (provider-specific or non-family features).
    pub mod extensions {
        pub use crate::extensions::*;
    }

    /// Registry module kept for compatibility with historical imports.
    ///
    /// Prefer `siumai::prelude::unified::registry::*` for the Vercel-aligned surface.
    pub mod registry {
        pub use super::unified::registry::*;
    }
}

/// Provider-specific builder convenience entry point.
///
/// Prefer `siumai::prelude::unified::registry::*` or config-first provider clients for stable
/// construction. Use `Provider::*` when you need a provider-owned builder or compatibility
/// construction path.
///
/// # Example
/// ```rust,no_run
/// use siumai::prelude::*;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Get a client specifically for OpenAI
///     let openai_client = Provider::openai()
///         .api_key("your-openai-key")
///         .model("gpt-4")
///         .build()
///         .await?;
///
///     // You can now call both standard and OpenAI-specific methods
///     let messages = vec![user!("Hello!")];
///     let response = openai_client.chat(messages).await?;
///     println!("OpenAI says: {}", response.text().unwrap_or_default());
///     // let assistant = openai_client.create_assistant(...).await?; // Example of specific feature
///
///     Ok(())
/// }
/// ```
pub struct Provider;

impl Provider {
    /// Create an `OpenAI` client builder
    #[cfg(feature = "openai")]
    pub fn openai() -> siumai_provider_openai::providers::openai::OpenAiBuilder {
        siumai_provider_openai::providers::openai::OpenAiBuilder::new(
            crate::builder::BuilderBase::default(),
        )
    }

    /// Create an explicit `OpenAI Responses` client builder.
    #[cfg(feature = "openai")]
    pub fn openai_responses() -> siumai_provider_openai::providers::openai::OpenAiBuilder {
        Self::openai().use_responses_api(true)
    }

    /// Create an explicit `OpenAI Chat Completions` client builder.
    #[cfg(feature = "openai")]
    pub fn openai_chat() -> siumai_provider_openai::providers::openai::OpenAiBuilder {
        Self::openai().use_responses_api(false)
    }

    /// Create an `Azure OpenAI` unified builder (Responses API by default).
    #[cfg(feature = "azure")]
    pub fn azure() -> crate::provider::SiumaiBuilder {
        crate::provider::SiumaiBuilder::new().azure()
    }

    /// Create an `Azure OpenAI Chat Completions` unified builder.
    #[cfg(feature = "azure")]
    pub fn azure_chat() -> crate::provider::SiumaiBuilder {
        crate::provider::SiumaiBuilder::new().azure_chat()
    }

    /// Create an Anthropic client builder
    #[cfg(feature = "anthropic")]
    pub fn anthropic() -> siumai_provider_anthropic::providers::anthropic::AnthropicBuilder {
        siumai_provider_anthropic::providers::anthropic::AnthropicBuilder::new(
            crate::builder::BuilderBase::default(),
        )
    }

    /// Create an Amazon Bedrock client builder
    #[cfg(feature = "bedrock")]
    pub fn bedrock() -> siumai_provider_amazon_bedrock::providers::bedrock::BedrockBuilder {
        siumai_provider_amazon_bedrock::providers::bedrock::BedrockBuilder::new(
            crate::builder::BuilderBase::default(),
        )
    }

    /// Create a Cohere client builder
    #[cfg(feature = "cohere")]
    pub fn cohere() -> siumai_provider_cohere::providers::cohere::CohereBuilder {
        siumai_provider_cohere::providers::cohere::CohereBuilder::new(
            crate::builder::BuilderBase::default(),
        )
    }

    /// Create a Gemini client builder
    #[cfg(feature = "google")]
    pub fn gemini() -> siumai_provider_gemini::providers::gemini::GeminiBuilder {
        siumai_provider_gemini::providers::gemini::GeminiBuilder::new(
            crate::builder::BuilderBase::default(),
        )
    }

    /// Create a Google client builder alias (AI SDK package-aligned).
    #[cfg(feature = "google")]
    pub fn google() -> siumai_provider_gemini::providers::gemini::GeminiBuilder {
        Self::gemini().name("google.generative-ai")
    }

    /// Create a TogetherAI unified builder.
    ///
    /// This aligns with the AI SDK-style provider surface:
    /// chat/completion/embedding/speech/transcription route through the shared
    /// OpenAI-compatible TogetherAI path, while image and rerank use provider-owned TogetherAI
    /// implementations.
    #[cfg(feature = "togetherai")]
    pub fn togetherai() -> crate::provider::SiumaiBuilder {
        crate::provider::SiumaiBuilder::new().togetherai()
    }

    /// Create a DeepInfra unified builder.
    ///
    /// This aligns with the AI SDK-style provider surface:
    /// text/completion/embedding reuse the shared OpenAI-compatible runtime, while image
    /// generation and editing use the provider-owned DeepInfra `/inference` and
    /// `/openai/images/edits` routes.
    #[cfg(feature = "deepinfra")]
    pub fn deepinfra() -> crate::provider::SiumaiBuilder {
        crate::provider::SiumaiBuilder::new().deepinfra()
    }

    /// Create a Mistral unified builder.
    ///
    /// This mirrors the AI SDK `mistral` provider package surface while continuing to reuse the
    /// shared OpenAI-compatible runtime internally.
    #[cfg(feature = "openai")]
    pub fn mistral() -> crate::provider::SiumaiBuilder {
        crate::provider::SiumaiBuilder::new().mistral()
    }

    /// Create a Fireworks unified builder.
    ///
    /// This mirrors the AI SDK `fireworks` provider package surface: chat/completion/embedding/
    /// transcription use the shared OpenAI-compatible runtime, while image generation/edit use the
    /// provider-owned Fireworks workflow routes.
    #[cfg(feature = "openai")]
    pub fn fireworks() -> crate::provider::SiumaiBuilder {
        crate::provider::SiumaiBuilder::new().fireworks()
    }

    /// Create a Perplexity unified builder.
    ///
    /// This mirrors the AI SDK `perplexity` provider package surface while continuing to reuse the
    /// shared OpenAI-compatible runtime internally.
    #[cfg(feature = "openai")]
    pub fn perplexity() -> crate::provider::SiumaiBuilder {
        crate::provider::SiumaiBuilder::new().perplexity()
    }

    /// Create a MoonshotAI unified builder.
    ///
    /// This mirrors the AI SDK `moonshotai` provider package surface while continuing to reuse the
    /// shared OpenAI-compatible runtime internally.
    #[cfg(feature = "openai")]
    pub fn moonshotai() -> crate::provider::SiumaiBuilder {
        crate::provider::SiumaiBuilder::new().moonshotai()
    }

    /// Create an Ollama client builder
    #[cfg(feature = "ollama")]
    pub fn ollama() -> siumai_provider_ollama::providers::ollama::OllamaBuilder {
        siumai_provider_ollama::providers::ollama::OllamaBuilder::new(
            crate::builder::BuilderBase::default(),
        )
    }

    /// Create an xAI client builder
    #[cfg(feature = "xai")]
    pub fn xai() -> siumai_provider_xai::providers::xai::XaiBuilder {
        siumai_provider_xai::providers::xai::XaiBuilder::new(crate::builder::BuilderBase::default())
    }

    /// Create a Groq client builder
    #[cfg(feature = "groq")]
    pub fn groq() -> siumai_provider_groq::providers::groq::GroqBuilder {
        siumai_provider_groq::providers::groq::GroqBuilder::new(
            crate::builder::BuilderBase::default(),
        )
    }

    /// Create a MiniMaxi client builder
    #[cfg(feature = "minimaxi")]
    pub fn minimaxi() -> siumai_provider_minimaxi::providers::minimaxi::MinimaxiBuilder {
        siumai_provider_minimaxi::providers::minimaxi::MinimaxiBuilder::new(
            crate::builder::BuilderBase::default(),
        )
    }

    /// Create a Google Vertex client builder
    #[cfg(feature = "google-vertex")]
    pub fn vertex() -> siumai_provider_google_vertex::providers::vertex::GoogleVertexBuilder {
        siumai_provider_google_vertex::providers::vertex::GoogleVertexBuilder::new(
            crate::builder::BuilderBase::default(),
        )
    }

    /// Create a Google Vertex MaaS unified builder.
    ///
    /// This aligns with the AI SDK `vertexMaas` surface:
    /// chat/completion/embedding route through the shared OpenAI-compatible runtime on
    /// Vertex's `/endpoints/openapi` base URL, authenticated with Google-style Bearer tokens.
    #[cfg(feature = "google-vertex")]
    pub fn vertex_maas() -> crate::provider::SiumaiBuilder {
        crate::provider::SiumaiBuilder::new().vertex_maas()
    }

    /// Create an Anthropic on Vertex client builder
    #[cfg(feature = "google-vertex")]
    pub fn anthropic_vertex()
    -> siumai_provider_google_vertex::providers::anthropic_vertex::VertexAnthropicBuilder {
        siumai_provider_google_vertex::providers::anthropic_vertex::VertexAnthropicBuilder::new(
            crate::builder::BuilderBase::default(),
        )
    }

    /// AI SDK package-aligned alias for `anthropic_vertex()`.
    #[cfg(feature = "google-vertex")]
    pub fn vertex_anthropic()
    -> siumai_provider_google_vertex::providers::anthropic_vertex::VertexAnthropicBuilder {
        Self::anthropic_vertex()
    }

    /// Create a DeepSeek client builder
    #[cfg(feature = "deepseek")]
    pub fn deepseek() -> siumai_provider_deepseek::providers::deepseek::DeepSeekBuilder {
        siumai_provider_deepseek::providers::deepseek::DeepSeekBuilder::new(
            crate::builder::BuilderBase::default(),
        )
    }

    // Provider convenience functions live on `LlmBuilder` / `Siumai::builder()` / `Provider::*`.
}

// Macros moved to a dedicated module for cleanliness
mod macros;

mod model_catalog;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::unified::*;
    use crate::provider::Siumai;

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
