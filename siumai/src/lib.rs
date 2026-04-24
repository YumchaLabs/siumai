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

mod experimental_bridge;

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

/// AI SDK-style ID generation helpers.
pub use siumai_core::utils::{
    DEFAULT_ID_ALPHABET, DEFAULT_ID_SIZE, IdGenerator, IdGeneratorOptions, create_id_generator,
    generate_id,
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
    PartialJsonParseState, RepairTextContext, RepairTextFunction, RepairTextFuture,
    fix_partial_json, generate_array, generate_choice, generate_enum, generate_json,
    generate_object, parse_partial_json,
};
pub mod text;
pub mod transcription;
/// AI SDK-style `UIMessage` validation and conversion helpers.
pub mod ui;
/// Task-oriented video generation family helpers.
pub mod video;

/// Tool runtime (schema + execution binding).
pub mod tooling;

/// AI SDK-style tool runtime helpers.
pub use siumai_core::tooling::{
    ExecutableTool, ExecutableTools, ToolExecuteFunction, ToolExecutionOptions,
    ToolExecutionResult, ToolExecutionStream, ToolModelOutputContext, ToolSet, dynamic_tool,
    execute_tool, is_executable_tool, model_messages_from_chat_messages, tool,
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
        pub use crate::experimental_bridge::*;
        pub use siumai_core::bridge::*;
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
            ChatByteStream, LanguageModelV3StreamPart, LanguageModelV4CustomContent,
            LanguageModelV4File, LanguageModelV4FileData, LanguageModelV4FinishReason,
            LanguageModelV4InputTokens, LanguageModelV4OutputTokens, LanguageModelV4ReasoningFile,
            LanguageModelV4ResponseMetadata, LanguageModelV4Source, LanguageModelV4StreamPart,
            LanguageModelV4ToolApprovalRequest, LanguageModelV4ToolCall, LanguageModelV4ToolResult,
            LanguageModelV4Usage, OpenAiResponsesStreamPartsBridge, SharedV4ProviderMetadata,
            SharedV4Warning, StreamPartNamespace, V3UnsupportedPartBehavior,
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
pub mod provider_ext {
    #[cfg(feature = "openai")]
    pub mod openai {
        #[cfg(feature = "openai-websocket")]
        pub use siumai_provider_openai::providers::openai::OpenAiIncrementalWebSocketSession;
        #[cfg(feature = "openai-websocket")]
        pub use siumai_provider_openai::providers::openai::OpenAiWebSocketRecoveryConfig;
        #[cfg(feature = "openai-websocket")]
        pub use siumai_provider_openai::providers::openai::OpenAiWebSocketSession;
        #[cfg(feature = "openai-websocket")]
        pub use siumai_provider_openai::providers::openai::OpenAiWebSocketTransport;
        pub use siumai_provider_openai::providers::openai::{
            OpenAIProviderSettings, OpenAiBuilder, OpenAiClient, OpenAiConfig, VERSION,
        };

        /// Create the OpenAI provider builder.
        pub fn openai() -> OpenAiBuilder {
            crate::Provider::openai()
        }

        /// Create the OpenAI provider builder.
        ///
        /// This is the Rust package-surface analogue of AI SDK `createOpenAI()`.
        pub fn create_openai() -> OpenAiBuilder {
            openai()
        }

        /// Provider tool factories that return `Tool` directly (Vercel-aligned).
        ///
        /// This mirrors the Vercel AI SDK `{ type: "provider", id, name, args }` convention.
        pub mod tools {
            pub use crate::tools::openai::*;
        }

        /// Provider-executed tool builders (typed args).
        ///
        /// Prefer this module when you need provider-specific configuration helpers and want `.build()`.
        pub mod hosted_tools {
            pub use crate::hosted_tools::openai::*;
        }

        /// Compatibility alias for older imports.
        ///
        /// Prefer `siumai::providers::openai::tools`.
        pub mod provider_tools {
            pub use crate::tools::openai::*;
        }

        /// Typed response metadata helpers (`ChatResponse.provider_metadata["openai"]`).
        pub mod metadata {
            pub use siumai_provider_openai::provider_metadata::openai::{
                OpenAiChatResponseExt, OpenAiContentPartExt, OpenAiContentPartMetadata,
                OpenAiMetadata, OpenAiSource, OpenAiSourceExt, OpenAiSourceMetadata,
            };
        }
        pub use metadata::{
            OpenAiChatResponseExt, OpenAiContentPartExt, OpenAiContentPartMetadata, OpenAiMetadata,
            OpenAiSource, OpenAiSourceExt, OpenAiSourceMetadata,
        };

        /// Typed provider options (`provider_options_map["openai"]`).
        pub mod options {
            pub use siumai_provider_openai::provider_options::openai::{
                ChatCompletionAudio, ChatCompletionAudioFormat, ChatCompletionAudioVoice,
                ChatCompletionModalities, InputAudio, InputAudioFormat,
                OpenAIContextManagementConfig, OpenAIContextManagementType,
                OpenAIEmbeddingModelOptions, OpenAIFilesOptions, OpenAILanguageModelChatOptions,
                OpenAILanguageModelCompletionOptions, OpenAILanguageModelResponsesOptions,
                OpenAISpeechModelOptions, OpenAITranscriptionModelOptions, OpenAiOptions,
                OpenAiWebSearchOptions, PredictionContent, PredictionContentData, ReasoningEffort,
                ResponsesApiConfig, ServiceTier, SystemMessageMode, TextVerbosity, Truncation,
                UserLocationWrapper, WebSearchLocation,
            };
            #[allow(deprecated)]
            pub use siumai_provider_openai::provider_options::openai::{
                OpenAIChatLanguageModelOptions, OpenAIResponsesProviderOptions,
            };
            pub use siumai_provider_openai::providers::openai::ext::OpenAiChatRequestExt;
            pub use siumai_provider_openai::providers::openai::ext::audio_options::{
                OpenAiSttOptions, OpenAiSttRequestExt, OpenAiTtsOptions,
            };
            pub use siumai_provider_openai::providers::openai::types::{
                OpenAiEmbeddingOptions, OpenAiEmbeddingRequestExt,
            };
        }
        pub use options::{
            ChatCompletionAudio, ChatCompletionAudioFormat, ChatCompletionAudioVoice,
            ChatCompletionModalities, InputAudio, InputAudioFormat, OpenAIContextManagementConfig,
            OpenAIContextManagementType, OpenAIEmbeddingModelOptions, OpenAIFilesOptions,
            OpenAILanguageModelChatOptions, OpenAILanguageModelCompletionOptions,
            OpenAILanguageModelResponsesOptions, OpenAISpeechModelOptions,
            OpenAITranscriptionModelOptions, OpenAiChatRequestExt, OpenAiEmbeddingOptions,
            OpenAiEmbeddingRequestExt, OpenAiOptions, OpenAiSttOptions, OpenAiSttRequestExt,
            OpenAiTtsOptions, OpenAiWebSearchOptions, PredictionContent, PredictionContentData,
            ReasoningEffort, ResponsesApiConfig, ServiceTier, SystemMessageMode, TextVerbosity,
            Truncation, UserLocationWrapper, WebSearchLocation,
        };
        #[allow(deprecated)]
        pub use options::{OpenAIChatLanguageModelOptions, OpenAIResponsesProviderOptions};

        /// Non-unified OpenAI extension APIs (streaming helpers, moderation/files resources, etc.).
        pub mod ext {
            pub use siumai_provider_openai::providers::openai::ext::{
                moderation, responses, speech_streaming, transcription_streaming,
            };
            pub use siumai_provider_openai::providers::openai::responses::OpenAiResponsesEventConverter;
        }

        /// Provider-specific resources not covered by the unified families.
        pub mod resources {
            pub use siumai_provider_openai::providers::openai::{
                OpenAiFiles, OpenAiModels, OpenAiModeration, OpenAiRerank, OpenAiSkills,
            };
        }

        /// Legacy OpenAI parameter structs (client-level defaults).
        ///
        /// Prefer request-level provider options (`OpenAiOptions`) for new code.
        pub mod legacy_params {
            pub use siumai_provider_openai::params::openai::{
                FunctionChoice, OpenAiParams, OpenAiParamsBuilder, ResponseFormat, ToolChoice,
            };
        }
    }

    /// OpenAI-compatible vendors (DeepSeek/OpenRouter/Moonshot/etc.) via the OpenAI-like protocol family.
    #[cfg(feature = "openai")]
    pub mod openai_compatible {
        pub use siumai_provider_openai_compatible::providers::openai_compatible::{
            ConfigurableAdapter, MetadataExtractor, OPENAI_COMPATIBLE_VERSION as VERSION,
            OpenAICompatibleChatModelId, OpenAICompatibleClient, OpenAICompatibleCompletionModelId,
            OpenAICompatibleConfig, OpenAICompatibleEmbeddingModelId, OpenAICompatibleErrorData,
            OpenAICompatibleImageModelId, OpenAICompatibleProviderSettings,
            OpenAICompatibleRequestSettings, OpenAiCompatibleChatModelId, OpenAiCompatibleClient,
            OpenAiCompatibleCompletionModelId, OpenAiCompatibleConfig,
            OpenAiCompatibleEmbeddingModelId, OpenAiCompatibleErrorData,
            OpenAiCompatibleImageModelId, OpenAiCompatibleRequestSettings, ProviderAdapter,
            ProviderCompatibility, ProviderConfig, ProviderErrorStructure, RequestBodyTransformer,
            ResponseMetadataExtractor, deepinfra, deepseek, fireworks, generic_provider_config,
            get_provider_config, groq, list_provider_ids, moonshot, moonshotai, openrouter,
            provider_supports_capability, siliconflow, vertex_maas, xai,
        };

        /// Typed generic OpenAI-compatible provider options (`provider_options_map["openaiCompatible"]`).
        pub mod options {
            #[allow(deprecated)]
            pub use siumai_provider_openai_compatible::provider_options::{
                OpenAICompatibleCompletionProviderOptions, OpenAICompatibleEmbeddingModelOptions,
                OpenAICompatibleEmbeddingProviderOptions, OpenAICompatibleLanguageModelChatOptions,
                OpenAICompatibleLanguageModelCompletionOptions, OpenAICompatibleProviderOptions,
                OpenAiCompatibleCompletionProviderOptions, OpenAiCompatibleEmbeddingModelOptions,
                OpenAiCompatibleEmbeddingProviderOptions, OpenAiCompatibleLanguageModelChatOptions,
                OpenAiCompatibleLanguageModelCompletionOptions, OpenAiCompatibleProviderOptions,
            };
            pub use siumai_provider_openai_compatible::providers::openai_compatible::ext::{
                OpenAiCompatibleChatRequestExt, OpenAiCompatibleCompletionRequestExt,
                OpenAiCompatibleEmbeddingRequestExt,
            };
        }

        #[allow(deprecated)]
        pub use options::{
            OpenAICompatibleCompletionProviderOptions, OpenAICompatibleEmbeddingModelOptions,
            OpenAICompatibleEmbeddingProviderOptions, OpenAICompatibleLanguageModelChatOptions,
            OpenAICompatibleLanguageModelCompletionOptions, OpenAICompatibleProviderOptions,
            OpenAiCompatibleChatRequestExt, OpenAiCompatibleCompletionProviderOptions,
            OpenAiCompatibleCompletionRequestExt, OpenAiCompatibleEmbeddingModelOptions,
            OpenAiCompatibleEmbeddingProviderOptions, OpenAiCompatibleEmbeddingRequestExt,
            OpenAiCompatibleLanguageModelChatOptions,
            OpenAiCompatibleLanguageModelCompletionOptions, OpenAiCompatibleProviderOptions,
        };
    }

    #[cfg(feature = "openai")]
    pub mod openrouter {
        /// Typed response metadata helpers (`ChatResponse.provider_metadata["openrouter"]`).
        pub mod metadata {
            pub use siumai_provider_openai_compatible::providers::openai_compatible::{
                OpenRouterChatResponseExt, OpenRouterContentPartExt, OpenRouterContentPartMetadata,
                OpenRouterMetadata, OpenRouterSource, OpenRouterSourceExt,
                OpenRouterSourceMetadata,
            };
        }

        /// Typed provider options (`provider_options_map["openrouter"]`).
        pub mod options {
            pub use siumai_provider_openai_compatible::provider_options::{
                OpenRouterOptions, OpenRouterTransform,
            };
            pub use siumai_provider_openai_compatible::providers::openai_compatible::ext::OpenRouterChatRequestExt;
        }

        pub use metadata::{
            OpenRouterChatResponseExt, OpenRouterContentPartExt, OpenRouterContentPartMetadata,
            OpenRouterMetadata, OpenRouterSource, OpenRouterSourceExt, OpenRouterSourceMetadata,
        };
        pub use options::{OpenRouterChatRequestExt, OpenRouterOptions, OpenRouterTransform};
    }

    #[cfg(feature = "openai")]
    pub mod mistral {
        /// Lower-level Mistral text-family compat client/config aliases.
        ///
        /// These map to the shared OpenAI-compatible runtime used by the audited Mistral
        /// `chat` and `embedding` lanes. For the unified AI SDK-style provider surface, use
        /// [`mistral()`], [`create_mistral()`], [`crate::Provider::mistral()`], or
        /// [`crate::provider::SiumaiBuilder::mistral()`].
        pub use siumai_provider_openai_compatible::providers::openai_compatible::{
            MISTRAL_VERSION as VERSION, MistralClient, MistralConfig, MistralProviderSettings,
        };

        /// Curated Mistral model constants aligned with the audited AI SDK package subset.
        pub mod models {
            pub use siumai_provider_openai_compatible::providers::openai_compatible::mistral::{
                self, chat, embedding,
            };
        }

        /// Create the unified Mistral provider builder.
        ///
        /// This mirrors the AI SDK package-level `mistral` export more closely than the lower
        /// level `MistralClient`/`MistralConfig` compat aliases.
        pub fn mistral() -> crate::provider::SiumaiBuilder {
            crate::Provider::mistral()
        }

        /// Create the unified Mistral provider builder.
        ///
        /// This is the Rust package-surface analogue of AI SDK `createMistral()`.
        pub fn create_mistral() -> crate::provider::SiumaiBuilder {
            mistral()
        }

        /// Typed provider options (`provider_options_map["mistral"]`).
        pub mod options {
            pub use siumai_provider_openai_compatible::provider_options::{
                MistralChatOptions, MistralLanguageModelOptions, MistralReasoningEffort,
            };
            pub use siumai_provider_openai_compatible::providers::openai_compatible::ext::MistralChatRequestExt;
        }

        pub use models::{chat, embedding, mistral as model_sets};
        pub use options::{
            MistralChatOptions, MistralChatRequestExt, MistralLanguageModelOptions,
            MistralReasoningEffort,
        };
    }

    #[cfg(feature = "openai")]
    pub mod perplexity {
        /// Lower-level Perplexity text-family compat client/config aliases.
        ///
        /// These map to the shared OpenAI-compatible runtime used by the audited Perplexity chat
        /// wrapper. For the unified AI SDK-style provider surface, use [`perplexity()`],
        /// [`create_perplexity()`], [`crate::Provider::perplexity()`], or
        /// [`crate::provider::SiumaiBuilder::perplexity()`].
        pub use siumai_provider_openai_compatible::providers::openai_compatible::{
            PERPLEXITY_VERSION as VERSION, PerplexityClient, PerplexityConfig,
            PerplexityProviderSettings,
        };

        /// Curated Perplexity model constants aligned with the audited AI SDK package subset.
        pub mod models {
            pub use siumai_provider_openai_compatible::providers::openai_compatible::perplexity::{
                self, chat,
            };
        }

        /// Create the unified Perplexity provider builder.
        ///
        /// This mirrors the AI SDK package-level `perplexity` export more closely than the lower
        /// level `PerplexityClient`/`PerplexityConfig` compat aliases.
        pub fn perplexity() -> crate::provider::SiumaiBuilder {
            crate::Provider::perplexity()
        }

        /// Create the unified Perplexity provider builder.
        ///
        /// This is the Rust package-surface analogue of AI SDK `createPerplexity()`.
        pub fn create_perplexity() -> crate::provider::SiumaiBuilder {
            perplexity()
        }

        /// Typed response metadata helpers (`ChatResponse.provider_metadata["perplexity"]`).
        pub mod metadata {
            pub use siumai_provider_openai_compatible::providers::openai_compatible::{
                PerplexityChatResponseExt, PerplexityImage, PerplexityMetadata, PerplexityUsage,
            };
        }

        /// Typed provider options (`provider_options_map["perplexity"]`).
        pub mod options {
            pub use siumai_provider_openai_compatible::provider_options::{
                PerplexityOptions, PerplexitySearchContextSize, PerplexitySearchMode,
                PerplexitySearchRecencyFilter, PerplexityUserLocation, PerplexityWebSearchOptions,
            };
            pub use siumai_provider_openai_compatible::providers::openai_compatible::ext::PerplexityChatRequestExt;
        }

        pub use metadata::{
            PerplexityChatResponseExt, PerplexityImage, PerplexityMetadata, PerplexityUsage,
        };
        pub use models::{chat, perplexity as model_sets};
        pub use options::{
            PerplexityChatRequestExt, PerplexityOptions, PerplexitySearchContextSize,
            PerplexitySearchMode, PerplexitySearchRecencyFilter, PerplexityUserLocation,
            PerplexityWebSearchOptions,
        };
    }

    #[cfg(feature = "openai")]
    pub mod fireworks {
        /// Lower-level Fireworks text-family compat client/config aliases.
        ///
        /// These map to the shared OpenAI-compatible runtime used by Fireworks chat,
        /// completion, embedding, and transcription lanes. For the unified AI SDK-style provider
        /// surface that also owns image generation/edit routing, use [`fireworks()`],
        /// [`create_fireworks()`], [`crate::Provider::fireworks()`], or
        /// [`crate::provider::SiumaiBuilder::fireworks()`].
        pub use siumai_provider_openai_compatible::providers::openai_compatible::{
            FIREWORKS_VERSION as VERSION, FireworksClient, FireworksConfig,
            FireworksEmbeddingModelId, FireworksErrorData, FireworksImageModelId,
            FireworksProviderSettings,
        };

        /// Curated Fireworks model constants aligned with the audited AI SDK package subset.
        pub mod models {
            pub use siumai_provider_openai_compatible::providers::openai_compatible::fireworks::{
                self, chat, completion, embedding, image,
            };
        }

        /// Create the unified Fireworks provider builder.
        ///
        /// This mirrors the AI SDK package-level `fireworks` export more closely than the lower
        /// level `FireworksClient`/`FireworksConfig` compat aliases.
        pub fn fireworks() -> crate::provider::SiumaiBuilder {
            crate::Provider::fireworks()
        }

        /// Create the unified Fireworks provider builder.
        ///
        /// This is the Rust package-surface analogue of AI SDK `createFireworks()`.
        pub fn create_fireworks() -> crate::provider::SiumaiBuilder {
            fireworks()
        }

        /// Typed provider options (`provider_options_map["fireworks"]`).
        pub mod options {
            #[allow(deprecated)]
            pub use siumai_provider_openai_compatible::provider_options::{
                FireworksChatOptions, FireworksEmbeddingModelOptions,
                FireworksEmbeddingProviderOptions, FireworksLanguageModelOptions,
                FireworksProviderOptions, FireworksReasoningHistory, FireworksThinkingConfig,
                FireworksThinkingType,
            };
            pub use siumai_provider_openai_compatible::providers::openai_compatible::ext::FireworksChatRequestExt;
        }

        pub use models::{chat, completion, embedding, fireworks as model_sets, image};
        #[allow(deprecated)]
        pub use options::{
            FireworksChatOptions, FireworksChatRequestExt, FireworksEmbeddingModelOptions,
            FireworksEmbeddingProviderOptions, FireworksLanguageModelOptions,
            FireworksProviderOptions, FireworksReasoningHistory, FireworksThinkingConfig,
            FireworksThinkingType,
        };
    }

    #[cfg(feature = "openai")]
    pub mod moonshotai {
        /// Lower-level MoonshotAI text-family compat client/config aliases.
        ///
        /// These map to the shared OpenAI-compatible runtime used by the audited MoonshotAI
        /// chat-only wrapper. For the unified AI SDK-style provider surface, use [`moonshotai()`],
        /// [`create_moonshotai()`], [`crate::Provider::moonshotai()`], or
        /// [`crate::provider::SiumaiBuilder::moonshotai()`].
        pub use siumai_provider_openai_compatible::providers::openai_compatible::{
            MOONSHOTAI_VERSION as VERSION, MoonshotAIChatModelId, MoonshotAIClient,
            MoonshotAIConfig, MoonshotAIProviderSettings,
        };

        /// Curated MoonshotAI model constants aligned with the audited AI SDK package subset.
        pub mod models {
            pub use siumai_provider_openai_compatible::providers::openai_compatible::moonshotai::{
                self, recommended,
            };
        }

        /// Create the unified MoonshotAI provider builder.
        ///
        /// This mirrors the AI SDK package-level `moonshotai` export more closely than the lower
        /// level `MoonshotAIClient`/`MoonshotAIConfig` compat aliases.
        pub fn moonshotai() -> crate::provider::SiumaiBuilder {
            crate::Provider::moonshotai()
        }

        /// Create the unified MoonshotAI provider builder.
        ///
        /// This is the Rust package-surface analogue of AI SDK `createMoonshotAI()`.
        pub fn create_moonshotai() -> crate::provider::SiumaiBuilder {
            moonshotai()
        }

        /// Typed provider options (`provider_options_map["moonshotai"]`).
        pub mod options {
            #[allow(deprecated)]
            pub use siumai_provider_openai_compatible::provider_options::{
                MoonshotAIChatOptions, MoonshotAILanguageModelOptions, MoonshotAIProviderOptions,
                MoonshotAIReasoningHistory, MoonshotAIThinkingConfig, MoonshotAIThinkingType,
            };
            pub use siumai_provider_openai_compatible::providers::openai_compatible::ext::MoonshotAIChatRequestExt;
        }

        pub use models::{moonshotai as model_sets, recommended};
        #[allow(deprecated)]
        pub use options::{
            MoonshotAIChatOptions, MoonshotAIChatRequestExt, MoonshotAILanguageModelOptions,
            MoonshotAIProviderOptions, MoonshotAIReasoningHistory, MoonshotAIThinkingConfig,
            MoonshotAIThinkingType,
        };
    }

    #[cfg(feature = "deepinfra")]
    pub mod deepinfra {
        /// Lower-level DeepInfra text-family compat client/config aliases.
        ///
        /// These map to the shared OpenAI-compatible runtime used by DeepInfra chat,
        /// completion, and embedding lanes. For the unified AI SDK-style provider surface
        /// that also owns image generation/edit routing, use [`deepinfra()`],
        /// [`create_deepinfra()`], [`crate::Provider::deepinfra()`], or
        /// [`crate::provider::SiumaiBuilder::deepinfra()`].
        pub use siumai_provider_openai_compatible::providers::openai_compatible::{
            DEEPINFRA_VERSION as VERSION, DeepInfraChatModelId, DeepInfraClient,
            DeepInfraCompletionModelId, DeepInfraConfig, DeepInfraEmbeddingModelId,
            DeepInfraErrorData, DeepInfraImageModelId, DeepInfraProviderSettings,
        };

        /// Curated DeepInfra model constants aligned with the audited AI SDK package subset.
        pub mod models {
            pub use siumai_provider_openai_compatible::providers::openai_compatible::deepinfra::{
                self, chat, completion, embedding, image,
            };
        }

        /// Create the unified DeepInfra provider builder.
        ///
        /// This mirrors the AI SDK package-level `deepinfra` export more closely than the lower
        /// level `DeepInfraClient`/`DeepInfraConfig` compat aliases.
        pub fn deepinfra() -> crate::provider::SiumaiBuilder {
            crate::Provider::deepinfra()
        }

        /// Create the unified DeepInfra provider builder.
        ///
        /// This is the Rust package-surface analogue of AI SDK `createDeepInfra()`.
        pub fn create_deepinfra() -> crate::provider::SiumaiBuilder {
            deepinfra()
        }

        pub use models::{chat, completion, deepinfra as model_sets, embedding, image};
    }

    #[cfg(feature = "google-vertex")]
    pub mod vertex_maas {
        /// Lower-level Vertex MaaS text-family compat client/config aliases.
        ///
        /// These map to the shared OpenAI-compatible runtime used by the audited Vertex MaaS
        /// chat, completion, and embedding lanes. For the unified AI SDK-style provider surface,
        /// use [`vertex_maas()`], [`create_vertex_maas()`], [`crate::Provider::vertex_maas()`], or
        /// [`crate::provider::SiumaiBuilder::vertex_maas()`].
        pub use siumai_provider_openai_compatible::providers::openai_compatible::{
            GOOGLE_VERTEX_MAAS_VERSION as VERSION, GoogleVertexMaasClient, GoogleVertexMaasConfig,
            GoogleVertexMaasModelId, GoogleVertexMaasProviderSettings,
        };

        /// Curated Vertex MaaS model constants aligned with the audited AI SDK package subset.
        pub mod models {
            pub use siumai_provider_openai_compatible::providers::openai_compatible::vertex_maas::{
                self, chat, completion, embedding,
            };
        }

        /// Create the unified Google Vertex MaaS provider builder.
        pub fn vertex_maas() -> crate::provider::SiumaiBuilder {
            crate::Provider::vertex_maas()
        }

        /// Create the unified Google Vertex MaaS provider builder.
        ///
        /// This is the Rust package-surface analogue of AI SDK `createVertexMaas()`.
        pub fn create_vertex_maas() -> crate::provider::SiumaiBuilder {
            vertex_maas()
        }

        pub use models::{chat, completion, embedding, vertex_maas as model_sets};
    }

    #[cfg(feature = "bedrock")]
    pub mod bedrock {
        pub use siumai_provider_amazon_bedrock::providers::bedrock::{
            AmazonBedrockProviderSettings, BedrockBuilder, BedrockClient, BedrockConfig, VERSION,
        };

        /// Create the Bedrock provider builder.
        pub fn bedrock() -> BedrockBuilder {
            crate::Provider::bedrock()
        }

        /// Create the Bedrock provider builder.
        ///
        /// This is the Rust package-surface analogue of AI SDK `createAmazonBedrock()`.
        pub fn create_amazon_bedrock() -> BedrockBuilder {
            bedrock()
        }

        /// Anthropic provider tool factories re-exported on the Bedrock surface like AI SDK.
        #[cfg(feature = "anthropic")]
        pub mod tools {
            pub use crate::tools::anthropic::*;
        }

        /// Compatibility alias for older imports.
        #[cfg(feature = "anthropic")]
        pub mod provider_tools {
            pub use crate::tools::anthropic::*;
        }

        /// Typed response metadata helpers (`ChatResponse.provider_metadata["bedrock"]`).
        pub mod metadata {
            pub use siumai_provider_amazon_bedrock::provider_metadata::bedrock::{
                BedrockChatResponseExt, BedrockContentPartExt, BedrockMetadata,
                BedrockReasoningContentPartMetadata,
            };
        }

        /// Typed provider options (`provider_options_map["bedrock"]`).
        pub mod options {
            #[allow(deprecated)]
            pub use siumai_provider_amazon_bedrock::provider_options::{
                AmazonBedrockEmbeddingModelOptions, AmazonBedrockLanguageModelOptions,
                AmazonBedrockRerankingModelOptions, BedrockCachePoint, BedrockCachePointConfig,
                BedrockCachePointType, BedrockCacheTtl, BedrockChatOptions,
                BedrockEmbeddingInputType, BedrockEmbeddingOptions, BedrockEmbeddingPurpose,
                BedrockEmbeddingTruncate, BedrockFilePartCitations, BedrockFilePartProviderOptions,
                BedrockProviderOptions, BedrockReasoningConfig, BedrockReasoningEffort,
                BedrockReasoningType, BedrockRerankOptions, BedrockRerankingOptions,
                BedrockServiceTier,
            };
            pub use siumai_provider_amazon_bedrock::providers::bedrock::{
                BedrockChatRequestExt, BedrockEmbeddingRequestExt, BedrockMessageExt,
                BedrockRequestContentPartExt, BedrockRerankRequestExt,
            };
            #[cfg(feature = "anthropic")]
            #[allow(deprecated)]
            pub use siumai_provider_anthropic::provider_options::anthropic::AnthropicProviderOptions;
        }

        pub use metadata::{
            BedrockChatResponseExt, BedrockContentPartExt, BedrockMetadata,
            BedrockReasoningContentPartMetadata,
        };
        #[cfg(feature = "anthropic")]
        #[allow(deprecated)]
        pub use options::AnthropicProviderOptions;
        #[allow(deprecated)]
        pub use options::{
            AmazonBedrockEmbeddingModelOptions, AmazonBedrockLanguageModelOptions,
            AmazonBedrockRerankingModelOptions, BedrockCachePoint, BedrockCachePointConfig,
            BedrockCachePointType, BedrockCacheTtl, BedrockChatOptions, BedrockChatRequestExt,
            BedrockEmbeddingInputType, BedrockEmbeddingOptions, BedrockEmbeddingPurpose,
            BedrockEmbeddingRequestExt, BedrockEmbeddingTruncate, BedrockFilePartCitations,
            BedrockFilePartProviderOptions, BedrockMessageExt, BedrockProviderOptions,
            BedrockReasoningConfig, BedrockReasoningEffort, BedrockReasoningType,
            BedrockRequestContentPartExt, BedrockRerankOptions, BedrockRerankRequestExt,
            BedrockRerankingOptions, BedrockServiceTier,
        };
        pub use siumai_provider_amazon_bedrock::providers::bedrock::assistant_message_with_reasoning_metadata;
    }

    #[cfg(feature = "cohere")]
    pub mod cohere {
        pub use siumai_provider_cohere::providers::cohere::{
            CohereBuilder, CohereClient, CohereConfig, CohereProviderSettings, VERSION,
        };

        /// Create the Cohere provider builder.
        pub fn cohere() -> CohereBuilder {
            crate::Provider::cohere()
        }

        /// Create the Cohere provider builder.
        ///
        /// This is the Rust package-surface analogue of AI SDK `createCohere()`.
        pub fn create_cohere() -> CohereBuilder {
            cohere()
        }

        pub mod models {
            pub use siumai_provider_cohere::providers::cohere::models::{
                self as model_sets, chat, embedding, rerank,
            };
        }

        /// Typed provider options (`provider_options_map["cohere"]`).
        pub mod options {
            #[allow(deprecated)]
            pub use siumai_provider_cohere::provider_options::{
                CohereChatModelOptions, CohereChatOptions, CohereEmbeddingInputType,
                CohereEmbeddingModelOptions, CohereEmbeddingOptions, CohereEmbeddingTruncate,
                CohereLanguageModelOptions, CohereRerankOptions, CohereRerankingModelOptions,
                CohereRerankingOptions, CohereThinkingConfig, CohereThinkingType,
            };
            pub use siumai_provider_cohere::providers::cohere::{
                CohereChatRequestExt, CohereEmbeddingRequestExt, CohereRerankRequestExt,
            };
        }

        pub use models::{chat, embedding, model_sets, rerank};
        #[allow(deprecated)]
        pub use options::{
            CohereChatModelOptions, CohereChatOptions, CohereChatRequestExt,
            CohereEmbeddingInputType, CohereEmbeddingModelOptions, CohereEmbeddingOptions,
            CohereEmbeddingRequestExt, CohereEmbeddingTruncate, CohereLanguageModelOptions,
            CohereRerankOptions, CohereRerankRequestExt, CohereRerankingModelOptions,
            CohereRerankingOptions, CohereThinkingConfig, CohereThinkingType,
        };
    }

    #[cfg(feature = "togetherai")]
    pub mod togetherai {
        pub use siumai_provider_togetherai::providers::togetherai::{
            TogetherAIErrorData, TogetherAIProviderSettings, TogetherAiBuilder, TogetherAiClient,
            TogetherAiConfig, VERSION,
        };

        /// Create the unified TogetherAI provider builder.
        pub fn togetherai() -> crate::provider::SiumaiBuilder {
            crate::Provider::togetherai()
        }

        /// Create the unified TogetherAI provider builder.
        ///
        /// This is the Rust package-surface analogue of AI SDK `createTogetherAI()`.
        pub fn create_togetherai() -> crate::provider::SiumaiBuilder {
            togetherai()
        }

        pub mod models {
            pub use siumai_provider_togetherai::providers::togetherai::models::{
                self as model_sets, chat, completion, embedding, image, rerank,
            };
        }

        /// Typed provider options (`provider_options_map["togetherai"]`).
        pub mod options {
            #[allow(deprecated)]
            pub use siumai_provider_togetherai::provider_options::{
                TogetherAIImageModelOptions, TogetherAIImageProviderOptions,
                TogetherAIRerankingModelOptions, TogetherAIRerankingOptions,
                TogetherAiImageModelOptions, TogetherAiImageOptions,
                TogetherAiImageProviderOptions, TogetherAiRerankOptions,
                TogetherAiRerankingModelOptions, TogetherAiRerankingOptions,
            };
            pub use siumai_provider_togetherai::providers::togetherai::{
                TogetherAiImageRequestExt, TogetherAiRerankRequestExt,
            };
        }

        pub use models::{chat, completion, embedding, image, model_sets, rerank};
        #[allow(deprecated)]
        pub use options::{
            TogetherAIImageModelOptions, TogetherAIImageProviderOptions,
            TogetherAIRerankingModelOptions, TogetherAIRerankingOptions,
            TogetherAiImageModelOptions, TogetherAiImageOptions, TogetherAiImageProviderOptions,
            TogetherAiImageRequestExt, TogetherAiRerankOptions, TogetherAiRerankRequestExt,
            TogetherAiRerankingModelOptions, TogetherAiRerankingOptions,
        };
    }

    #[cfg(feature = "azure")]
    pub mod azure {
        pub use siumai_provider_azure::providers::azure_openai::{
            AzureChatMode, AzureOpenAIProviderSettings, AzureOpenAiBuilder, AzureOpenAiClient,
            AzureOpenAiConfig, AzureOpenAiSpec, AzureUrlConfig, VERSION,
        };

        /// Create the unified Azure provider builder.
        pub fn azure() -> crate::provider::SiumaiBuilder {
            crate::Provider::azure()
        }

        /// Create the unified Azure provider builder.
        ///
        /// This is the Rust package-surface analogue of AI SDK `createAzure()`.
        pub fn create_azure() -> crate::provider::SiumaiBuilder {
            azure()
        }

        /// Typed provider options (`provider_options_map["azure"]`).
        pub mod options {
            pub use siumai_provider_azure::provider_options::{
                AzureOpenAiOptions, AzureReasoningEffort, AzureResponsesApiConfig,
                OpenAIContextManagementConfig, OpenAIContextManagementType,
                OpenAILanguageModelChatOptions, OpenAILanguageModelResponsesOptions,
                SystemMessageMode,
            };
            #[allow(deprecated)]
            pub use siumai_provider_azure::provider_options::{
                OpenAIChatLanguageModelOptions, OpenAIResponsesProviderOptions,
            };
            pub use siumai_provider_azure::providers::azure_openai::AzureOpenAiChatRequestExt;
        }
        pub use options::{
            AzureOpenAiChatRequestExt, AzureOpenAiOptions, AzureReasoningEffort,
            AzureResponsesApiConfig, OpenAIContextManagementConfig, OpenAIContextManagementType,
            OpenAILanguageModelChatOptions, OpenAILanguageModelResponsesOptions, SystemMessageMode,
        };
        #[allow(deprecated)]
        pub use options::{OpenAIChatLanguageModelOptions, OpenAIResponsesProviderOptions};

        /// Typed response metadata helpers (`ChatResponse.provider_metadata["azure"]`).
        pub mod metadata {
            pub use siumai_provider_azure::provider_metadata::azure::{
                AzureChatResponseExt, AzureContentPartExt, AzureContentPartMetadata, AzureMetadata,
                AzureSource, AzureSourceExt, AzureSourceMetadata,
            };
        }
        pub use metadata::{
            AzureChatResponseExt, AzureContentPartExt, AzureContentPartMetadata, AzureMetadata,
            AzureSource, AzureSourceExt, AzureSourceMetadata,
        };
    }

    #[cfg(feature = "anthropic")]
    pub mod anthropic {
        pub use siumai_provider_anthropic::providers::anthropic::{
            AnthropicBuilder, AnthropicClient, AnthropicConfig, AnthropicProviderSettings, VERSION,
        };

        /// Create the Anthropic provider builder.
        pub fn anthropic() -> AnthropicBuilder {
            crate::Provider::anthropic()
        }

        /// Create the Anthropic provider builder.
        ///
        /// This is the Rust package-surface analogue of AI SDK `createAnthropic()`.
        pub fn create_anthropic() -> AnthropicBuilder {
            anthropic()
        }

        /// Provider tool factories that return `Tool` directly (Vercel-aligned).
        pub mod tools {
            pub use crate::tools::anthropic::*;
        }

        /// Provider-executed tool builders (typed args).
        pub mod hosted_tools {
            pub use crate::hosted_tools::anthropic::*;
        }

        /// Compatibility alias for older imports.
        pub mod provider_tools {
            pub use crate::tools::anthropic::*;
        }

        /// Typed provider options (`provider_options_map["anthropic"]`).
        pub mod options {
            #[allow(deprecated)]
            pub use siumai_provider_anthropic::provider_options::anthropic::{
                AnthropicCacheControl, AnthropicCacheType, AnthropicContainerConfig,
                AnthropicContainerSkill, AnthropicContainerSkillType,
                AnthropicContextManagementAllKeep, AnthropicContextManagementConfig,
                AnthropicContextManagementEdit, AnthropicContextManagementInputTokensValue,
                AnthropicContextManagementThinkingKeep,
                AnthropicContextManagementThinkingTurnsKeep,
                AnthropicContextManagementThinkingTurnsKeepKind,
                AnthropicContextManagementToolUsesKeep, AnthropicContextManagementTrigger,
                AnthropicEffort, AnthropicInferenceGeo, AnthropicLanguageModelOptions,
                AnthropicMcpServer, AnthropicMcpServerType, AnthropicMcpToolConfiguration,
                AnthropicOptions, AnthropicProviderOptions, AnthropicRequestCacheControl,
                AnthropicRequestCacheControlTtl, AnthropicRequestCacheControlType,
                AnthropicRequestMetadata, AnthropicResponseFormat, AnthropicSpeed,
                AnthropicStructuredOutputMode, AnthropicTaskBudget, AnthropicTaskBudgetType,
                AnthropicThinkingConfig, AnthropicThinkingDisplay, AnthropicToolAllowedCaller,
                AnthropicToolOptions, PromptCachingConfig, ThinkingModeConfig,
            };
            pub use siumai_provider_anthropic::providers::anthropic::ext::AnthropicChatRequestExt;
        }
        #[allow(deprecated)]
        pub use options::{
            AnthropicCacheControl, AnthropicCacheType, AnthropicChatRequestExt,
            AnthropicContainerConfig, AnthropicContainerSkill, AnthropicContainerSkillType,
            AnthropicContextManagementAllKeep, AnthropicContextManagementConfig,
            AnthropicContextManagementEdit, AnthropicContextManagementInputTokensValue,
            AnthropicContextManagementThinkingKeep, AnthropicContextManagementThinkingTurnsKeep,
            AnthropicContextManagementThinkingTurnsKeepKind,
            AnthropicContextManagementToolUsesKeep, AnthropicContextManagementTrigger,
            AnthropicEffort, AnthropicInferenceGeo, AnthropicLanguageModelOptions,
            AnthropicMcpServer, AnthropicMcpServerType, AnthropicMcpToolConfiguration,
            AnthropicOptions, AnthropicProviderOptions, AnthropicRequestCacheControl,
            AnthropicRequestCacheControlTtl, AnthropicRequestCacheControlType,
            AnthropicRequestMetadata, AnthropicResponseFormat, AnthropicSpeed,
            AnthropicStructuredOutputMode, AnthropicTaskBudget, AnthropicTaskBudgetType,
            AnthropicThinkingConfig, AnthropicThinkingDisplay, AnthropicToolAllowedCaller,
            AnthropicToolOptions, PromptCachingConfig, ThinkingModeConfig,
        };

        /// Typed response metadata helpers (`ChatResponse.provider_metadata["anthropic"]`).
        pub mod metadata {
            pub use siumai_provider_anthropic::provider_metadata::anthropic::{
                AnthropicChatResponseExt, AnthropicCitation, AnthropicCitationsBlock,
                AnthropicContentPartExt, AnthropicMessageContainerMetadata,
                AnthropicMessageContainerSkill, AnthropicMessageMetadata, AnthropicMetadata,
                AnthropicServerToolUse, AnthropicSource, AnthropicToolCallMetadata,
                AnthropicToolCaller, AnthropicUsageIteration,
            };
        }
        pub use metadata::{
            AnthropicChatResponseExt, AnthropicCitation, AnthropicCitationsBlock,
            AnthropicContentPartExt, AnthropicMessageContainerMetadata,
            AnthropicMessageContainerSkill, AnthropicMessageMetadata, AnthropicMetadata,
            AnthropicServerToolUse, AnthropicSource, AnthropicToolCallMetadata,
            AnthropicToolCaller, AnthropicUsageIteration,
        };
        pub use siumai_provider_anthropic::providers::anthropic::{
            find_anthropic_container_id_from_last_step,
            forward_anthropic_container_id_from_last_step,
        };

        /// Non-unified Anthropic extension APIs (request extensions, tool helpers, thinking, etc.).
        pub mod ext {
            pub use siumai_provider_anthropic::providers::anthropic::ext::{
                AnthropicToolExt, structured_output, thinking, tools,
            };
        }
        pub use ext::AnthropicToolExt;

        /// Provider-specific resources not covered by the unified families.
        pub mod resources {
            pub use siumai_provider_anthropic::providers::anthropic::{
                AnthropicCountTokensResponse, AnthropicCreateMessageBatchRequest, AnthropicFiles,
                AnthropicListMessageBatchesResponse, AnthropicMessageBatch,
                AnthropicMessageBatchRequest, AnthropicMessageBatches, AnthropicSkills,
                AnthropicTokens,
            };
        }

        // Legacy Anthropic parameter structs (provider-owned).
        pub use siumai_provider_anthropic::params::anthropic::{AnthropicParams, CacheControl};
    }

    #[cfg(feature = "google")]
    pub mod gemini {
        #[allow(deprecated)]
        pub use siumai_provider_gemini::providers::gemini::GoogleGenerativeAIProviderSettings;
        pub use siumai_provider_gemini::providers::gemini::types::GeminiConfig;
        pub use siumai_provider_gemini::providers::gemini::{
            GeminiBuilder, GeminiClient, GoogleProviderSettings, SharedIdGenerator, VERSION,
        };

        /// Curated model-id groups aligned with the audited `@ai-sdk/google` package surface.
        pub mod models {
            /// Google/Gemini language-model ids exported by the audited AI SDK package.
            pub mod chat {
                pub const GEMINI_2_0_FLASH: &str = "gemini-2.0-flash";
                pub const GEMINI_2_0_FLASH_001: &str = "gemini-2.0-flash-001";
                pub const GEMINI_2_0_FLASH_LITE: &str = "gemini-2.0-flash-lite";
                pub const GEMINI_2_0_FLASH_LITE_001: &str = "gemini-2.0-flash-lite-001";
                pub const GEMINI_2_5_PRO: &str = "gemini-2.5-pro";
                pub const GEMINI_2_5_FLASH: &str = "gemini-2.5-flash";
                pub const GEMINI_2_5_FLASH_IMAGE: &str = "gemini-2.5-flash-image";
                pub const GEMINI_2_5_FLASH_LITE: &str = "gemini-2.5-flash-lite";
                pub const GEMINI_2_5_FLASH_PREVIEW_TTS: &str = "gemini-2.5-flash-preview-tts";
                pub const GEMINI_2_5_PRO_PREVIEW_TTS: &str = "gemini-2.5-pro-preview-tts";
                pub const GEMINI_2_5_FLASH_NATIVE_AUDIO_LATEST: &str =
                    "gemini-2.5-flash-native-audio-latest";
                pub const GEMINI_2_5_FLASH_NATIVE_AUDIO_PREVIEW_09_2025: &str =
                    "gemini-2.5-flash-native-audio-preview-09-2025";
                pub const GEMINI_2_5_FLASH_NATIVE_AUDIO_PREVIEW_12_2025: &str =
                    "gemini-2.5-flash-native-audio-preview-12-2025";
                pub const GEMINI_2_5_COMPUTER_USE_PREVIEW_10_2025: &str =
                    "gemini-2.5-computer-use-preview-10-2025";
                pub const GEMINI_3_PRO_PREVIEW: &str = "gemini-3-pro-preview";
                pub const GEMINI_3_PRO_IMAGE_PREVIEW: &str = "gemini-3-pro-image-preview";
                pub const GEMINI_3_FLASH_PREVIEW: &str = "gemini-3-flash-preview";
                pub const GEMINI_3_1_PRO_PREVIEW: &str = "gemini-3.1-pro-preview";
                pub const GEMINI_3_1_PRO_PREVIEW_CUSTOMTOOLS: &str =
                    "gemini-3.1-pro-preview-customtools";
                pub const GEMINI_3_1_FLASH_IMAGE_PREVIEW: &str = "gemini-3.1-flash-image-preview";
                pub const GEMINI_3_1_FLASH_LITE_PREVIEW: &str = "gemini-3.1-flash-lite-preview";
                pub const GEMINI_3_1_FLASH_TTS_PREVIEW: &str = "gemini-3.1-flash-tts-preview";
                pub const GEMINI_PRO_LATEST: &str = "gemini-pro-latest";
                pub const GEMINI_FLASH_LATEST: &str = "gemini-flash-latest";
                pub const GEMINI_FLASH_LITE_LATEST: &str = "gemini-flash-lite-latest";
                pub const DEEP_RESEARCH_PRO_PREVIEW_12_2025: &str =
                    "deep-research-pro-preview-12-2025";
                pub const NANO_BANANA_PRO_PREVIEW: &str = "nano-banana-pro-preview";
                pub const AQA: &str = "aqa";
                pub const GEMINI_ROBOTICS_ER_1_5_PREVIEW: &str = "gemini-robotics-er-1.5-preview";
                pub const GEMMA_3_1B_IT: &str = "gemma-3-1b-it";
                pub const GEMMA_3_4B_IT: &str = "gemma-3-4b-it";
                pub const GEMMA_3N_E4B_IT: &str = "gemma-3n-e4b-it";
                pub const GEMMA_3N_E2B_IT: &str = "gemma-3n-e2b-it";
                pub const GEMMA_3_12B_IT: &str = "gemma-3-12b-it";
                pub const GEMMA_3_27B_IT: &str = "gemma-3-27b-it";

                pub const ALL: &[&str] = &[
                    GEMINI_2_0_FLASH,
                    GEMINI_2_0_FLASH_001,
                    GEMINI_2_0_FLASH_LITE,
                    GEMINI_2_0_FLASH_LITE_001,
                    GEMINI_2_5_PRO,
                    GEMINI_2_5_FLASH,
                    GEMINI_2_5_FLASH_IMAGE,
                    GEMINI_2_5_FLASH_LITE,
                    GEMINI_2_5_FLASH_PREVIEW_TTS,
                    GEMINI_2_5_PRO_PREVIEW_TTS,
                    GEMINI_2_5_FLASH_NATIVE_AUDIO_LATEST,
                    GEMINI_2_5_FLASH_NATIVE_AUDIO_PREVIEW_09_2025,
                    GEMINI_2_5_FLASH_NATIVE_AUDIO_PREVIEW_12_2025,
                    GEMINI_2_5_COMPUTER_USE_PREVIEW_10_2025,
                    GEMINI_3_PRO_PREVIEW,
                    GEMINI_3_PRO_IMAGE_PREVIEW,
                    GEMINI_3_FLASH_PREVIEW,
                    GEMINI_3_1_PRO_PREVIEW,
                    GEMINI_3_1_PRO_PREVIEW_CUSTOMTOOLS,
                    GEMINI_3_1_FLASH_IMAGE_PREVIEW,
                    GEMINI_3_1_FLASH_LITE_PREVIEW,
                    GEMINI_3_1_FLASH_TTS_PREVIEW,
                    GEMINI_PRO_LATEST,
                    GEMINI_FLASH_LATEST,
                    GEMINI_FLASH_LITE_LATEST,
                    DEEP_RESEARCH_PRO_PREVIEW_12_2025,
                    NANO_BANANA_PRO_PREVIEW,
                    AQA,
                    GEMINI_ROBOTICS_ER_1_5_PREVIEW,
                    GEMMA_3_1B_IT,
                    GEMMA_3_4B_IT,
                    GEMMA_3N_E4B_IT,
                    GEMMA_3N_E2B_IT,
                    GEMMA_3_12B_IT,
                    GEMMA_3_27B_IT,
                ];
            }

            /// Google embedding-model ids exported by the audited AI SDK package.
            pub mod embedding {
                pub const GEMINI_EMBEDDING_001: &str = "gemini-embedding-001";
                pub const GEMINI_EMBEDDING_2_PREVIEW: &str = "gemini-embedding-2-preview";

                pub const ALL: &[&str] = &[GEMINI_EMBEDDING_001, GEMINI_EMBEDDING_2_PREVIEW];
            }

            /// Google image-model ids exported by the audited AI SDK package.
            pub mod image {
                pub const IMAGEN_4_0_GENERATE_001: &str = "imagen-4.0-generate-001";
                pub const IMAGEN_4_0_ULTRA_GENERATE_001: &str = "imagen-4.0-ultra-generate-001";
                pub const IMAGEN_4_0_FAST_GENERATE_001: &str = "imagen-4.0-fast-generate-001";
                pub const GEMINI_2_5_FLASH_IMAGE: &str = "gemini-2.5-flash-image";
                pub const GEMINI_3_PRO_IMAGE_PREVIEW: &str = "gemini-3-pro-image-preview";
                pub const GEMINI_3_1_FLASH_IMAGE_PREVIEW: &str = "gemini-3.1-flash-image-preview";

                pub const ALL: &[&str] = &[
                    IMAGEN_4_0_GENERATE_001,
                    IMAGEN_4_0_ULTRA_GENERATE_001,
                    IMAGEN_4_0_FAST_GENERATE_001,
                    GEMINI_2_5_FLASH_IMAGE,
                    GEMINI_3_PRO_IMAGE_PREVIEW,
                    GEMINI_3_1_FLASH_IMAGE_PREVIEW,
                ];
            }

            /// Google video-model ids exported by the audited AI SDK package.
            pub mod video {
                pub const VEO_3_1_FAST_GENERATE_PREVIEW: &str = "veo-3.1-fast-generate-preview";
                pub const VEO_3_1_GENERATE_PREVIEW: &str = "veo-3.1-generate-preview";
                pub const VEO_3_1_GENERATE: &str = "veo-3.1-generate";
                pub const VEO_3_1_LITE_GENERATE_PREVIEW: &str = "veo-3.1-lite-generate-preview";
                pub const VEO_3_0_GENERATE_001: &str = "veo-3.0-generate-001";
                pub const VEO_3_0_FAST_GENERATE_001: &str = "veo-3.0-fast-generate-001";
                pub const VEO_2_0_GENERATE_001: &str = "veo-2.0-generate-001";

                pub const ALL: &[&str] = &[
                    VEO_3_1_FAST_GENERATE_PREVIEW,
                    VEO_3_1_GENERATE_PREVIEW,
                    VEO_3_1_GENERATE,
                    VEO_3_1_LITE_GENERATE_PREVIEW,
                    VEO_3_0_GENERATE_001,
                    VEO_3_0_FAST_GENERATE_001,
                    VEO_2_0_GENERATE_001,
                ];
            }

            /// Group aliases that mirror the common public provider-facade convention.
            pub mod model_sets {
                pub const ALL_CHAT: &[&str] = super::chat::ALL;
                pub const ALL_EMBEDDING: &[&str] = super::embedding::ALL;
                pub const ALL_IMAGE: &[&str] = super::image::ALL;
                pub const ALL_VIDEO: &[&str] = super::video::ALL;

                pub const CHAT: &[&str] = ALL_CHAT;
                pub const EMBEDDING: &[&str] = ALL_EMBEDDING;
                pub const IMAGE: &[&str] = ALL_IMAGE;
                pub const VIDEO: &[&str] = ALL_VIDEO;
            }
        }

        pub use models::{chat, embedding, image, model_sets, video};

        /// Provider tool factories that return `Tool` directly (Vercel-aligned).
        pub mod tools {
            pub use crate::tools::google::*;
        }

        /// Provider-executed tool builders (typed args).
        pub mod hosted_tools {
            pub use crate::hosted_tools::google::*;
        }

        /// Compatibility alias for older imports.
        pub mod provider_tools {
            pub use crate::tools::google::*;
        }

        /// Typed provider options (`provider_options_map["google"]`).
        pub mod options {
            pub use siumai_provider_gemini::provider_options::gemini::{
                GeminiHarmBlockThreshold, GeminiHarmCategory, GeminiImageOptions, GeminiOptions,
                GeminiResponseModality, GeminiSafetySetting, GeminiThinkingConfig,
                GeminiThinkingLevel, GoogleEmbeddingContentPart, GoogleEmbeddingInlineData,
                GoogleEmbeddingModelOptions, GoogleFilesUploadOptions, GoogleImageModelOptions,
                GoogleLanguageModelOptions, GoogleVideoModelId, GoogleVideoModelOptions,
            };
            #[allow(deprecated)]
            pub use siumai_provider_gemini::provider_options::gemini::{
                GoogleGenerativeAIEmbeddingProviderOptions, GoogleGenerativeAIImageProviderOptions,
                GoogleGenerativeAIProviderOptions, GoogleGenerativeAIVideoModelId,
                GoogleGenerativeAIVideoProviderOptions,
            };
            pub use siumai_provider_gemini::providers::gemini::ext::{
                GeminiChatRequestExt, GeminiImageRequestExt, GoogleChatRequestExt,
                GoogleEmbeddingRequestExt, GoogleImageRequestExt, GoogleVideoRequestExt,
            };
            pub use siumai_provider_gemini::providers::gemini::types::{
                GeminiEmbeddingOptions, GeminiEmbeddingRequestExt,
            };
        }

        // Provider-owned typed options (kept out of `siumai-core`).
        pub use options::{
            GeminiChatRequestExt, GeminiEmbeddingOptions, GeminiEmbeddingRequestExt,
            GeminiHarmBlockThreshold, GeminiHarmCategory, GeminiImageOptions,
            GeminiImageRequestExt, GeminiOptions, GeminiResponseModality, GeminiSafetySetting,
            GeminiThinkingConfig, GeminiThinkingLevel, GoogleChatRequestExt,
            GoogleEmbeddingContentPart, GoogleEmbeddingInlineData, GoogleEmbeddingModelOptions,
            GoogleEmbeddingRequestExt, GoogleFilesUploadOptions, GoogleImageModelOptions,
            GoogleImageRequestExt, GoogleLanguageModelOptions, GoogleVideoModelId,
            GoogleVideoModelOptions, GoogleVideoRequestExt,
        };
        #[allow(deprecated)]
        pub use options::{
            GoogleGenerativeAIEmbeddingProviderOptions, GoogleGenerativeAIImageProviderOptions,
            GoogleGenerativeAIProviderOptions, GoogleGenerativeAIVideoModelId,
            GoogleGenerativeAIVideoProviderOptions,
        };

        /// Typed response metadata helpers (`ChatResponse.provider_metadata["google"]`).
        pub mod metadata {
            #[allow(deprecated)]
            pub use siumai_provider_gemini::provider_metadata::gemini::{
                GeminiChatResponseExt, GeminiContentPartExt, GeminiMetadata, GeminiSource,
                GoogleGenerativeAIProviderMetadata, GoogleProviderMetadata,
            };
        }
        #[allow(deprecated)]
        pub use metadata::{
            GeminiChatResponseExt, GeminiContentPartExt, GeminiMetadata, GeminiSource,
            GoogleGenerativeAIProviderMetadata, GoogleProviderMetadata,
        };
        pub use siumai_provider_gemini::providers::gemini::{GoogleErrorBody, GoogleErrorData};

        /// Non-unified Gemini extension APIs (escape hatches).
        pub mod ext {
            pub use siumai_provider_gemini::providers::gemini::ext::{
                code_execution, file_search_stores, tools,
            };
        }

        /// Provider-specific resources not covered by the unified families.
        pub mod resources {
            pub use siumai_provider_gemini::providers::gemini::{
                GeminiCachedContents, GeminiFileSearchStores, GeminiFiles, GeminiModels,
                GeminiTokens, GeminiVideo, GoogleErrorBody, GoogleErrorData,
            };
        }

        // Legacy Gemini parameter structs (provider-owned).
        pub use siumai_provider_gemini::params::gemini::{
            GeminiParams, GeminiParamsBuilder, GenerationConfig, SafetyCategory, SafetySetting,
            SafetyThreshold,
        };
    }

    /// Vercel alignment: the AI SDK uses `@ai-sdk/google` for Gemini.
    ///
    /// This is a stable alias for the Gemini provider extension surface.
    #[cfg(feature = "google")]
    pub mod google {
        /// Create the Google provider builder.
        pub fn google() -> super::gemini::GeminiBuilder {
            crate::Provider::google()
        }

        /// Create the Google provider builder.
        ///
        /// This is the Rust package-surface analogue of AI SDK `createGoogle()`.
        pub fn create_google() -> super::gemini::GeminiBuilder {
            google()
        }

        /// Create the Google provider builder.
        ///
        /// Deprecated analogue of AI SDK `createGoogleGenerativeAI()`.
        #[allow(deprecated)]
        #[deprecated(note = "Use `create_google` instead.")]
        pub fn create_google_generative_ai() -> super::gemini::GeminiBuilder {
            create_google()
        }

        pub use super::gemini::*;
    }

    #[cfg(feature = "google-vertex")]
    pub mod google_vertex {
        pub use siumai_provider_google_vertex::providers::vertex::{
            GoogleVertexBuilder, GoogleVertexClient, GoogleVertexConfig,
            GoogleVertexProviderSettings, SharedIdGenerator, VERSION,
        };

        /// Create the Google Vertex provider builder.
        pub fn vertex() -> GoogleVertexBuilder {
            crate::Provider::vertex()
        }

        /// Create the Google Vertex provider builder.
        ///
        /// This is the Rust package-surface analogue of AI SDK `createVertex()`.
        pub fn create_vertex() -> GoogleVertexBuilder {
            vertex()
        }

        /// Curated Google Vertex model constants aligned with the audited public subset.
        pub mod models {
            pub use siumai_provider_google_vertex::providers::vertex::models::{
                self as model_sets, chat, embedding, image, video,
            };
        }

        /// Provider tool factories that return `Tool` directly (Vercel-aligned `googleVertexTools`).
        pub mod tools {
            use siumai_core::types::Tool;

            pub use crate::tools::google::{
                code_execution, enterprise_web_search, google_maps, url_context,
            };

            pub fn google_search() -> Tool {
                crate::hosted_tools::google::google_search().build()
            }

            pub fn file_search(file_search_store_names: Vec<String>) -> Tool {
                crate::hosted_tools::google::file_search()
                    .with_file_search_store_names(file_search_store_names)
                    .build()
            }

            pub fn vertex_rag_store(rag_corpus: impl Into<String>) -> Tool {
                crate::hosted_tools::google::vertex_rag_store(rag_corpus).build()
            }
        }

        /// Provider-executed tool builders (typed args).
        pub mod hosted_tools {
            pub use crate::hosted_tools::google::{
                FileSearchConfig, GoogleSearchConfig, VertexRagStoreConfig, code_execution,
                enterprise_web_search, file_search, google_maps, google_search, url_context,
                vertex_rag_store,
            };
        }

        /// Typed provider options (`provider_options_map["vertex"]`).
        pub mod options {
            #[allow(deprecated)]
            pub use siumai_provider_google_vertex::provider_options::vertex::GoogleVertexImageProviderOptions;
            #[allow(deprecated)]
            pub use siumai_provider_google_vertex::provider_options::vertex::GoogleVertexVideoProviderOptions;
            pub use siumai_provider_google_vertex::provider_options::vertex::{
                GoogleVertexEmbeddingModelOptions, GoogleVertexImageModelOptions,
                GoogleVertexReferenceImage, GoogleVertexVideoModelId,
                GoogleVertexVideoModelOptions, VertexEmbeddingOptions, VertexImagenEditMode,
                VertexImagenEditOptions, VertexImagenInlineImage, VertexImagenMaskImageConfig,
                VertexImagenMaskMode, VertexImagenOptions, VertexImagenReferenceImage,
                VertexImagenSafetySetting, VertexImagenSampleImageSize, VertexPersonGeneration,
            };
            pub use siumai_provider_google_vertex::providers::vertex::{
                VertexEmbeddingRequestExt, VertexImagenRequestExt, VertexVideoRequestExt,
            };
        }

        pub use models::{chat, embedding, image, model_sets, video};
        #[allow(deprecated)]
        pub use options::GoogleVertexImageProviderOptions;
        #[allow(deprecated)]
        pub use options::GoogleVertexVideoProviderOptions;
        pub use options::{
            GoogleVertexEmbeddingModelOptions, GoogleVertexImageModelOptions,
            GoogleVertexReferenceImage, GoogleVertexVideoModelId, GoogleVertexVideoModelOptions,
            VertexEmbeddingOptions, VertexEmbeddingRequestExt, VertexImagenEditMode,
            VertexImagenEditOptions, VertexImagenInlineImage, VertexImagenMaskImageConfig,
            VertexImagenMaskMode, VertexImagenOptions, VertexImagenReferenceImage,
            VertexImagenRequestExt, VertexImagenSafetySetting, VertexImagenSampleImageSize,
            VertexPersonGeneration, VertexVideoRequestExt,
        };

        /// Typed response metadata helpers (`ChatResponse.provider_metadata["vertex"]`).
        pub mod metadata {
            pub use siumai_provider_google_vertex::provider_metadata::vertex::{
                VertexChatResponseExt, VertexContentPartExt, VertexGroundingMetadata,
                VertexLogprobsResult, VertexMetadata, VertexPromptFeedback, VertexSafetyRating,
                VertexSource, VertexUrlContextMetadata, VertexUsageMetadata,
            };
        }
        pub use metadata::{
            VertexChatResponseExt, VertexContentPartExt, VertexGroundingMetadata,
            VertexLogprobsResult, VertexMetadata, VertexPromptFeedback, VertexSafetyRating,
            VertexSource, VertexUrlContextMetadata, VertexUsageMetadata,
        };
    }

    #[cfg(feature = "minimaxi")]
    pub mod minimaxi {
        pub use siumai_provider_minimaxi::providers::minimaxi::{MinimaxiBuilder, MinimaxiClient};

        /// Curated MiniMaxi model constants for the public provider surface.
        pub mod models {
            pub use siumai_provider_minimaxi::providers::minimaxi::models::{
                self as model_sets, chat, image, music, speech, video,
            };
        }

        /// Typed response metadata helpers (`ChatResponse.provider_metadata["minimaxi"]`).
        pub mod metadata {
            pub use siumai_provider_minimaxi::provider_metadata::minimaxi::{
                MinimaxiChatResponseExt, MinimaxiCitation, MinimaxiCitationsBlock,
                MinimaxiContentPartExt, MinimaxiMetadata, MinimaxiServerToolUse, MinimaxiSource,
                MinimaxiToolCallMetadata, MinimaxiToolCaller,
            };
        }
        pub use metadata::{
            MinimaxiChatResponseExt, MinimaxiCitation, MinimaxiCitationsBlock,
            MinimaxiContentPartExt, MinimaxiMetadata, MinimaxiServerToolUse, MinimaxiSource,
            MinimaxiToolCallMetadata, MinimaxiToolCaller,
        };

        /// Typed provider options (`provider_options_map["minimaxi"]`).
        pub mod options {
            pub use siumai_provider_minimaxi::provider_options::{
                MinimaxiOptions, MinimaxiResponseFormat, MinimaxiThinkingModeConfig,
                MinimaxiTtsOptions, MinimaxiVideoOptions,
            };
            pub use siumai_provider_minimaxi::providers::minimaxi::ext::tts::MinimaxiTtsRequestBuilder;
            pub use siumai_provider_minimaxi::providers::minimaxi::ext::tts_options::MinimaxiTtsRequestExt;
            pub use siumai_provider_minimaxi::providers::minimaxi::ext::{
                MinimaxiChatRequestExt, MinimaxiVideoRequestExt,
            };
        }

        // Provider-owned typed options (kept out of `siumai-core`).
        pub use models::{chat, image, model_sets, music, speech, video};
        pub use options::{
            MinimaxiChatRequestExt, MinimaxiOptions, MinimaxiResponseFormat,
            MinimaxiThinkingModeConfig, MinimaxiTtsOptions, MinimaxiTtsRequestBuilder,
            MinimaxiTtsRequestExt, MinimaxiVideoOptions, MinimaxiVideoRequestExt,
        };

        /// Non-unified MiniMaxi extension APIs (escape hatches).
        pub mod ext {
            pub use siumai_provider_minimaxi::providers::minimaxi::ext::{
                music, structured_output, thinking, video,
            };
        }

        /// Provider-specific resources not covered by the unified families.
        pub mod resources {
            /// MiniMaxi file management API client (extension resource).
            pub use siumai_provider_minimaxi::providers::minimaxi::files::MinimaxiFiles;
        }

        /// MiniMaxi low-level config (for advanced use; prefer the builder for most cases).
        pub use siumai_provider_minimaxi::providers::minimaxi::config::MinimaxiConfig;
    }

    #[cfg(feature = "ollama")]
    pub mod ollama {
        pub use siumai_provider_ollama::providers::ollama::{
            OllamaBuilder, OllamaClient, OllamaConfig,
        };

        /// Curated Ollama model constants for the public provider surface.
        pub mod models {
            pub use siumai_provider_ollama::providers::ollama::models::{
                self as model_sets, chat, embedding,
            };
        }

        /// Typed response metadata helpers (`ChatResponse.provider_metadata["ollama"]`).
        pub mod metadata {
            pub use siumai_provider_ollama::provider_metadata::ollama::{
                OllamaChatResponseExt, OllamaMetadata,
            };
        }

        pub use metadata::{OllamaChatResponseExt, OllamaMetadata};

        /// Typed provider options (`provider_options_map["ollama"]`).
        pub mod options {
            pub use siumai_provider_ollama::provider_options::OllamaOptions;
            pub use siumai_provider_ollama::providers::ollama::ext::OllamaChatRequestExt;
            pub use siumai_provider_ollama::providers::ollama::types::{
                OllamaEmbeddingOptions, OllamaEmbeddingRequestExt,
            };
        }

        // Provider-owned typed options (kept out of `siumai-core`).
        pub use models::{chat, embedding, model_sets};
        pub use options::{
            OllamaChatRequestExt, OllamaEmbeddingOptions, OllamaEmbeddingRequestExt, OllamaOptions,
        };

        /// Non-unified Ollama extension APIs (escape hatches).
        pub mod ext {
            pub use siumai_provider_ollama::providers::ollama::ext::request_options;
        }

        /// Provider-owned Ollama default parameter struct.
        pub use siumai_provider_ollama::providers::ollama::config::OllamaParams;
    }

    #[cfg(feature = "google-vertex")]
    pub mod anthropic_vertex {
        pub use siumai_provider_google_vertex::providers::anthropic_vertex::{
            GoogleVertexAnthropicMessagesModelId, GoogleVertexAnthropicProviderSettings,
            VertexAnthropicBuilder, VertexAnthropicClient, VertexAnthropicConfig,
        };

        /// Create the Anthropic-on-Vertex provider builder.
        pub fn vertex_anthropic() -> VertexAnthropicBuilder {
            crate::Provider::vertex_anthropic()
        }

        /// Create the Anthropic-on-Vertex provider builder.
        ///
        /// This is the Rust package-surface analogue of AI SDK `createVertexAnthropic()`.
        pub fn create_vertex_anthropic() -> VertexAnthropicBuilder {
            vertex_anthropic()
        }

        /// Curated Anthropic-on-Vertex model constants aligned with the audited public subset.
        pub mod models {
            pub use siumai_provider_google_vertex::providers::anthropic_vertex::models::{
                self as model_sets, chat,
            };
        }

        /// Provider tool factories that return `Tool` directly (Vertex Anthropic supported subset).
        pub mod tools {
            pub use siumai_provider_google_vertex::providers::anthropic_vertex::tools::*;
        }

        /// Provider-executed tool builders and typed helper inputs.
        pub mod hosted_tools {
            pub use siumai_provider_google_vertex::providers::anthropic_vertex::hosted_tools::*;
        }

        /// Compatibility alias for older imports.
        pub mod provider_tools {
            pub use siumai_provider_google_vertex::providers::anthropic_vertex::provider_tools::*;
        }

        /// Typed response metadata helpers (`ChatResponse.provider_metadata["anthropic"]`).
        pub mod metadata {
            pub use siumai_provider_google_vertex::providers::anthropic_vertex::{
                AnthropicChatResponseExt, AnthropicCitation, AnthropicCitationsBlock,
                AnthropicContentPartExt, AnthropicMessageContainerMetadata,
                AnthropicMessageContainerSkill, AnthropicMessageMetadata, AnthropicMetadata,
                AnthropicServerToolUse, AnthropicSource, AnthropicToolCallMetadata,
                AnthropicToolCaller, AnthropicUsageIteration,
            };
        }

        pub use metadata::{
            AnthropicChatResponseExt, AnthropicCitation, AnthropicCitationsBlock,
            AnthropicContentPartExt, AnthropicMessageContainerMetadata,
            AnthropicMessageContainerSkill, AnthropicMessageMetadata, AnthropicMetadata,
            AnthropicServerToolUse, AnthropicSource, AnthropicToolCallMetadata,
            AnthropicToolCaller, AnthropicUsageIteration,
        };

        /// Typed provider options (`provider_options_map["anthropic"]` on the Vertex wrapper path).
        pub mod options {
            pub use siumai_provider_google_vertex::providers::anthropic_vertex::{
                VertexAnthropicChatRequestExt, VertexAnthropicOptions,
                VertexAnthropicStructuredOutputMode, VertexAnthropicThinkingMode,
            };
        }

        pub use models::{chat, model_sets};
        pub use options::{
            VertexAnthropicChatRequestExt, VertexAnthropicOptions,
            VertexAnthropicStructuredOutputMode, VertexAnthropicThinkingMode,
        };

        /// Non-unified Anthropic-on-Vertex extension APIs (escape hatches).
        pub mod ext {
            pub use siumai_provider_google_vertex::providers::anthropic_vertex::ext::*;
        }
    }

    #[cfg(feature = "deepseek")]
    pub mod deepseek {
        pub use siumai_provider_deepseek::providers::deepseek::{
            DeepSeekBuilder, DeepSeekClient, DeepSeekConfig, DeepSeekErrorData,
            DeepSeekProviderSettings, VERSION,
        };

        /// Create the DeepSeek provider builder.
        pub fn deepseek() -> DeepSeekBuilder {
            crate::Provider::deepseek()
        }

        /// Create the DeepSeek provider builder.
        ///
        /// This is the Rust package-surface analogue of AI SDK `createDeepSeek()`.
        pub fn create_deepseek() -> DeepSeekBuilder {
            deepseek()
        }

        /// Curated DeepSeek model constants aligned with the audited AI SDK chat surface.
        pub mod models {
            pub use siumai_provider_deepseek::providers::deepseek::models::{
                self as model_sets, chat,
            };
        }

        /// Typed response metadata helpers (`ChatResponse.provider_metadata["deepseek"]`).
        pub mod metadata {
            pub use siumai_provider_deepseek::provider_metadata::deepseek::{
                DeepSeekChatResponseExt, DeepSeekMetadata, DeepSeekSource, DeepSeekSourceExt,
                DeepSeekSourceMetadata,
            };
        }

        pub use metadata::{
            DeepSeekChatResponseExt, DeepSeekMetadata, DeepSeekSource, DeepSeekSourceExt,
            DeepSeekSourceMetadata,
        };

        /// Typed provider options (`provider_options_map["deepseek"]`).
        pub mod options {
            #[allow(deprecated)]
            pub use siumai_provider_deepseek::provider_options::{
                DeepSeekChatOptions, DeepSeekLanguageModelOptions,
            };
            pub use siumai_provider_deepseek::providers::deepseek::DeepSeekOptions;
            pub use siumai_provider_deepseek::providers::deepseek::ext::DeepSeekChatRequestExt;
        }

        pub use models::{chat, model_sets};
        #[allow(deprecated)]
        pub use options::{
            DeepSeekChatOptions, DeepSeekChatRequestExt, DeepSeekLanguageModelOptions,
            DeepSeekOptions,
        };

        /// Non-unified DeepSeek extension APIs (escape hatches).
        pub mod ext {
            pub use siumai_provider_deepseek::providers::deepseek::ext::*;
        }
    }

    #[cfg(feature = "xai")]
    pub mod xai {
        pub use siumai_provider_xai::providers::xai::{
            VERSION, XaiBuilder, XaiClient, XaiConfig, XaiErrorData, XaiProviderSettings,
            XaiVideoModelId,
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

        pub use metadata::{
            XaiChatResponseExt, XaiMetadata, XaiSource, XaiSourceExt, XaiSourceMetadata,
        };

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
                XaiSearchParameters, XaiTtsOptions, XaiVideoMode, XaiVideoModelOptions,
                XaiVideoOptions, XaiVideoProviderOptions, XaiVideoResolution,
            };
        }

        // Provider-owned typed options (kept out of `siumai-core`).
        #[allow(deprecated)]
        pub use options::{
            NewsSearchSource, RssSearchSource, SearchMode, SearchSource, WebSearchSource,
            XSearchSource, XaiChatOptions, XaiChatReasoningEffort, XaiChatRequestExt,
            XaiFilesOptions, XaiImageModelOptions, XaiImageOptions, XaiImageProviderOptions,
            XaiImageQuality, XaiImageRequestExt, XaiImageResolution, XaiLanguageModelChatOptions,
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
    }

    #[cfg(feature = "groq")]
    pub mod groq {
        pub use siumai_provider_groq::providers::groq::{
            GroqBuilder, GroqClient, GroqConfig, GroqProviderSettings, VERSION,
        };

        /// Create the Groq provider builder.
        pub fn groq() -> GroqBuilder {
            crate::Provider::groq()
        }

        /// Create the Groq provider builder.
        ///
        /// This is the Rust package-surface analogue of AI SDK `createGroq()`.
        pub fn create_groq() -> GroqBuilder {
            groq()
        }

        /// Typed response metadata helpers (`ChatResponse.provider_metadata["groq"]`).
        pub mod metadata {
            pub use siumai_provider_groq::provider_metadata::groq::{
                GroqChatResponseExt, GroqMetadata, GroqSource, GroqSourceExt, GroqSourceMetadata,
            };
        }

        pub use metadata::{
            GroqChatResponseExt, GroqMetadata, GroqSource, GroqSourceExt, GroqSourceMetadata,
        };

        /// Provider tool factories that return `Tool` directly (Vercel-aligned).
        pub mod tools {
            pub use crate::tools::groq::*;
        }

        /// Vercel-style provider tool factories that return `Tool` directly.
        pub mod provider_tools {
            pub use crate::tools::groq::*;
        }

        /// Typed provider options (`provider_options_map["groq"]`).
        pub mod options {
            #[allow(deprecated)]
            pub use siumai_provider_groq::provider_options::{
                GroqLanguageModelOptions, GroqOptions, GroqProviderOptions, GroqReasoningEffort,
                GroqReasoningFormat, GroqServiceTier, GroqTranscriptionModelOptions,
            };
            pub use siumai_provider_groq::providers::groq::{
                GroqChatRequestExt, GroqSttRequestExt,
            };
        }

        // Provider-owned typed options (kept out of `siumai-core`).
        #[allow(deprecated)]
        pub use options::{
            GroqChatRequestExt, GroqLanguageModelOptions, GroqOptions, GroqProviderOptions,
            GroqReasoningEffort, GroqReasoningFormat, GroqServiceTier, GroqSttRequestExt,
            GroqTranscriptionModelOptions,
        };

        /// Non-unified Groq extension APIs (escape hatches).
        pub mod ext {
            pub use siumai_provider_groq::providers::groq::ext::*;
        }
    }
}

// Unified interface (`Siumai`) and builder
pub mod provider;
pub mod provider_catalog;

/// Extension capabilities (non-unified surface).
///
/// These are intentionally *not* part of the Vercel-aligned unified model families.
/// Prefer `siumai::prelude::unified` for the stable unified surface.
pub mod extensions {
    pub use siumai_core::traits::{
        AudioCapability, FileManagementCapability, ImageExtras, ModelListingCapability,
        ModerationCapability, MusicGenerationCapability, SkillsCapability, SpeechExtras,
        TimeoutCapability, TranscriptionExtras, VideoGenerationCapability,
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
        pub use crate::files::{
            FileUploadProvider, UploadFileApi, UploadFileOptions, UploadFileProviderMetadata,
            UploadFileResult,
        };
        pub use crate::parse_json_event_stream;
        #[doc(hidden)]
        pub use crate::provider::Siumai;
        pub use crate::registry::ProviderFactory;
        pub use crate::retry_api::*;
        pub use crate::skills::{
            UploadSkillApi, UploadSkillFile, UploadSkillFileContent, UploadSkillOptions,
            UploadSkillProviderMetadata, UploadSkillResult,
        };
        pub use crate::structured_output::{
            GenerateObjectOptions, GenerateObjectResult, GenerateObjectSchema,
            PartialJsonParseResult, PartialJsonParseState, RepairTextContext, RepairTextFunction,
            RepairTextFuture, fix_partial_json, generate_array, generate_choice, generate_enum,
            generate_json, generate_object, parse_partial_json,
        };
        pub use crate::tooling;
        pub use crate::tools;
        pub use crate::{
            DEFAULT_ID_ALPHABET, DEFAULT_ID_SIZE, IdGenerator, IdGeneratorOptions,
            create_id_generator, generate_id,
        };
        pub use crate::{
            ExecutableTool, ExecutableTools, ToolExecuteFunction, ToolExecutionOptions,
            ToolExecutionResult, ToolExecutionStream, ToolModelOutputContext, ToolSet,
            dynamic_tool, execute_tool, is_executable_tool, model_messages_from_chat_messages,
        };
        pub use crate::{assistant, conversation, conversation_with_system, messages, quick_chat};
        pub use crate::{
            completion, embedding, files, image, rerank, skills, speech, structured_output, text,
            transcription, video,
        };
        pub use crate::{system, tool, user, user_with_image};
        pub use siumai_core::completion::CompletionModel;
        pub use siumai_core::error::{ErrorCategory, LlmError};
        pub use siumai_core::execution::middleware::LanguageModelMiddleware;
        pub use siumai_core::image::{ImageModel, ImageModelV4};
        pub use siumai_core::rerank::RerankingModel;
        pub use siumai_core::speech::SpeechModel;
        pub use siumai_core::streaming::*;
        pub use siumai_core::text::LanguageModel;
        pub use siumai_core::traits::{
            ChatCapability, CompletionCapability, EmbeddingCapability, EmbeddingExtensions,
            ImageGenerationCapability, ModelMetadata, ProviderCapabilities, RerankCapability,
            SpeechCapability, TranscriptionCapability,
        };
        pub use siumai_core::transcription::TranscriptionModel;
        pub use siumai_core::video::{VideoModel, VideoModelV3, VideoModelV4};

        pub use siumai_core::embedding::EmbeddingModel;
        // Core request/response types for the six stable model families.
        #[allow(deprecated)]
        pub use siumai_core::types::{
            AssistantContent, AssistantContentPart, AssistantModelMessage, AudioStreamEvent,
            CacheControl, CallSettings, CallWarning, CancelHandle, ChatMessage, ChatRequest,
            ChatRequestBuilder, ChatResponse, CommonParams, CompletionRequest, CompletionResponse,
            CompletionTokensDetails, ContentPart, Context, CustomPart, CustomProviderOptions,
            DataContent, Embedding, EmbeddingModelUsage, EmbeddingRequest, EmbeddingResponse,
            FilePart, FilePartSource, FinishReason, FlexibleSchema, GenerateImageRequest,
            GeneratedImage, HttpConfig, ImageDetail, ImageGenerationRequest,
            ImageGenerationResponse, ImageModelProviderMetadata, ImageModelResponseMetadata,
            ImageModelUsage, ImagePart, InvalidDataContentError, JSONSchema7, JSONValue,
            LanguageModelCallOptions, LanguageModelInputTokenDetails,
            LanguageModelOutputTokenDetails, LanguageModelReasoning, LanguageModelRequestMetadata,
            LanguageModelResponseMetadata, LanguageModelUsage, LazySchema, MediaSource,
            MessageContent, MessageMetadata, MessageRole, MissingToolResultsError, ModelInfo,
            ModelMessage, ModelMessageConversionError, ModelMessageRole, OutputSchema, Prompt,
            PromptExecutionError, PromptInput, PromptTokensDetails, PromptValidationError,
            ProviderDefinedTool, ProviderMetadata, ProviderOptions, ProviderOptionsMap,
            ProviderReference, ProviderType, ReasoningFilePart, ReasoningPart, RequestOptions,
            RerankRequest, RerankResponse, ResponseFormat, ResponseMetadata, Schema,
            SchemaValidator, Source, SpeechModelResponseMetadata, StandardizedPrompt,
            StreamRequestOptions, SttRequest, SttResponse, SystemModelMessage, SystemPrompt,
            TextPart, TimeoutConfiguration, TimeoutConfigurationSettings, Tool,
            ToolApprovalRequest, ToolApprovalResponse, ToolCall, ToolCallPart, ToolChoice,
            ToolContent, ToolContentPart, ToolModelMessage, ToolResult, ToolResultOutput,
            ToolResultPart, TranscriptionModelResponseMetadata, TtsRequest, TtsResponse, Usage,
            UserContent, UserContentPart, UserModelMessage, ValidationResult,
            VideoModelProviderMetadata, VideoModelResponseMetadata, Warning, add_image_model_usage,
            add_language_model_usage, as_schema, as_schema_or_empty,
            convert_data_content_to_base64_string, create_null_language_model_usage,
            empty_json_schema, get_chunk_timeout_ms, get_step_timeout_ms, get_tool_timeout_ms,
            get_total_timeout_ms, json_schema, json_schema_with_validator, lazy_schema,
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

/// Provider entry point for creating specific provider clients
///
/// This is the main entry point for creating provider-specific clients.
/// Use this when you need access to provider-specific features and APIs.
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
