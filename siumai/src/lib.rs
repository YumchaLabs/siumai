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
//! - **Builder Pattern**: Supports a builder pattern for chained method calls.
//! - **Type Safety**: Leverages Rust's type system to ensure compile-time safety.
//! - **HTTP Customization**: Supports passing in a reqwest client and custom HTTP configurations.
//! - **Library First**: Focuses on core library functionality, avoiding application-layer features.
//! - **Flexible Capability Access**: Capability checks serve as hints rather than restrictions, allowing users to try new model features.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use siumai::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a unified client via the Siumai builder
//!     let client = Siumai::builder()
//!         .openai()
//!         .api_key("your-api-key")
//!         .model("gpt-4")
//!         .temperature(0.7)
//!         .build()
//!         .await?;
//!
//!     // Send a chat request
//!     let messages = vec![user!("Hello, world!")];
//!     let response = client.chat(messages).await?;
//!     if let Some(text) = response.content_text() {
//!         println!("Response: {}", text);
//!     }
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
//! use siumai::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = Siumai::builder()
//!         .openai()
//!         .api_key("your-api-key")
//!         .model("gpt-4o")  // This model supports vision
//!         .build()
//!         .await?;
//!
//!     // Vercel-aligned approach: image understanding is done via multimodal Chat messages.
//!     // (No separate "VisionCapability" unified surface.)
//!     let messages = vec![user_with_image!("Describe this image", "https://example.com/a.png")];
//!     let resp = client.chat(messages).await?;
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

/// Hosted tools are part of the stable unified experience (Vercel-aligned).
pub use siumai_core::hosted_tools;

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
pub mod retry_api;

// Compatibility / internal modules (kept but hidden to reduce accidental coupling).
//
// NOTE: These low-level modules are intentionally NOT re-exported at the top-level.
// Use `siumai::experimental::*` for advanced integrations and internal building blocks.

/// Legacy builder module (provider construction internals).
///
/// Prefer `Siumai::builder()` / `Provider::...()` / `registry::global()` for stable construction.
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

        #[cfg(feature = "anthropic")]
        pub use siumai_provider_anthropic as anthropic;
        #[cfg(feature = "azure")]
        pub use siumai_provider_azure as azure;
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
        #[cfg(feature = "xai")]
        pub use siumai_provider_xai as xai;
    }

    /// Protocol mapping and adapter helpers (advanced API).
    ///
    /// Prefer `siumai::prelude::unified::*` unless you are building integrations or custom providers.
    pub mod standards {
        #[cfg(feature = "anthropic")]
        pub use siumai_provider_anthropic::standards::anthropic;
        #[cfg(feature = "google")]
        pub use siumai_provider_gemini::standards::gemini;
        #[cfg(feature = "ollama")]
        pub use siumai_provider_ollama::standards::ollama;
        #[cfg(feature = "openai")]
        pub use siumai_provider_openai::standards::openai;
    }

    pub use siumai_core::{auth, client, defaults, execution, observability, params, retry, utils};
}

pub use siumai_core::siumai_for_each_openai_compatible_provider;
pub use siumai_registry::registry;

/// Provider extension APIs (non-unified surface).
///
/// These are stable module paths for provider-specific endpoints/resources.
pub mod provider_ext {
    #[cfg(feature = "openai")]
    pub mod openai {
        pub use siumai_provider_openai::providers::openai::{OpenAiClient, OpenAiConfig};

        /// Provider-executed tool factories (Vercel-aligned).
        ///
        /// These tools are executed on the provider side (e.g. OpenAI Responses hosted tools).
        pub mod tools {
            pub use crate::hosted_tools::openai::*;
        }

        /// Typed response metadata helpers (`ChatResponse.provider_metadata["openai"]`).
        pub mod metadata {
            pub use siumai_provider_openai::provider_metadata::openai::{
                OpenAiChatResponseExt, OpenAiMetadata, OpenAiSource,
            };
        }
        pub use metadata::{OpenAiChatResponseExt, OpenAiMetadata, OpenAiSource};

        /// Typed provider options (`provider_options_map["openai"]`).
        pub mod options {
            pub use siumai_provider_openai::provider_options::openai::{
                ChatCompletionAudio, ChatCompletionAudioFormat, ChatCompletionAudioVoice,
                ChatCompletionModalities, InputAudio, InputAudioFormat, OpenAiOptions,
                OpenAiWebSearchOptions, PredictionContent, PredictionContentData, ReasoningEffort,
                ResponsesApiConfig, ServiceTier, TextVerbosity, Truncation, UserLocationWrapper,
                WebSearchLocation,
            };
            pub use siumai_provider_openai::providers::openai::ext::OpenAiChatRequestExt;
            pub use siumai_provider_openai::providers::openai::ext::audio_options::{
                OpenAiSttOptions, OpenAiTtsOptions,
            };
            pub use siumai_provider_openai::providers::openai::types::{
                OpenAiEmbeddingOptions, OpenAiEmbeddingRequestExt,
            };
        }
        pub use options::{
            ChatCompletionAudio, ChatCompletionAudioFormat, ChatCompletionAudioVoice,
            ChatCompletionModalities, InputAudio, InputAudioFormat, OpenAiChatRequestExt,
            OpenAiEmbeddingOptions, OpenAiEmbeddingRequestExt, OpenAiOptions, OpenAiSttOptions,
            OpenAiTtsOptions, OpenAiWebSearchOptions, PredictionContent, PredictionContentData,
            ReasoningEffort, ResponsesApiConfig, ServiceTier, TextVerbosity, Truncation,
            UserLocationWrapper, WebSearchLocation,
        };

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
                OpenAiFiles, OpenAiModels, OpenAiModeration, OpenAiRerank,
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

    #[cfg(feature = "azure")]
    pub mod azure {
        pub use siumai_provider_azure::providers::azure_openai::{
            AzureChatMode, AzureOpenAiClient, AzureOpenAiConfig, AzureOpenAiSpec, AzureUrlConfig,
        };
    }

    #[cfg(feature = "anthropic")]
    pub mod anthropic {
        pub use siumai_provider_anthropic::providers::anthropic::AnthropicClient;

        /// Provider-executed tool factories (Vercel-aligned).
        pub mod tools {
            pub use crate::hosted_tools::anthropic::*;
        }

        /// Typed provider options (`provider_options_map["anthropic"]`).
        pub mod options {
            pub use siumai_provider_anthropic::provider_options::anthropic::{
                AnthropicCacheControl, AnthropicCacheType, AnthropicOptions,
                AnthropicResponseFormat, PromptCachingConfig, ThinkingModeConfig,
            };
            pub use siumai_provider_anthropic::providers::anthropic::ext::AnthropicChatRequestExt;
        }
        pub use options::{
            AnthropicCacheControl, AnthropicCacheType, AnthropicChatRequestExt, AnthropicOptions,
            AnthropicResponseFormat, PromptCachingConfig, ThinkingModeConfig,
        };

        /// Typed response metadata helpers (`ChatResponse.provider_metadata["anthropic"]`).
        pub mod metadata {
            pub use siumai_provider_anthropic::provider_metadata::anthropic::{
                AnthropicChatResponseExt, AnthropicCitation, AnthropicCitationsBlock,
                AnthropicMetadata, AnthropicServerToolUse, AnthropicSource,
            };
        }
        pub use metadata::{
            AnthropicChatResponseExt, AnthropicCitation, AnthropicCitationsBlock,
            AnthropicMetadata, AnthropicServerToolUse, AnthropicSource,
        };

        /// Non-unified Anthropic extension APIs (request extensions, tool helpers, thinking, etc.).
        pub mod ext {
            pub use siumai_provider_anthropic::providers::anthropic::ext::{
                structured_output, thinking, tools,
            };
        }

        // Legacy Anthropic parameter structs (provider-owned).
        pub use siumai_provider_anthropic::params::anthropic::{AnthropicParams, CacheControl};
    }

    #[cfg(feature = "google")]
    pub mod gemini {
        pub use siumai_provider_gemini::providers::gemini::GeminiClient;
        pub use siumai_provider_gemini::providers::gemini::types::GeminiConfig;

        /// Provider-executed tool factories (Vercel-aligned).
        pub mod tools {
            pub use crate::hosted_tools::google::*;
        }

        /// Typed provider options (`provider_options_map["google"]`).
        pub mod options {
            pub use siumai_provider_gemini::provider_options::gemini::{
                GeminiHarmBlockThreshold, GeminiHarmCategory, GeminiOptions,
                GeminiResponseModality, GeminiSafetySetting, GeminiThinkingConfig,
                GeminiThinkingLevel,
            };
            pub use siumai_provider_gemini::providers::gemini::ext::GeminiChatRequestExt;
            pub use siumai_provider_gemini::providers::gemini::types::{
                GeminiEmbeddingOptions, GeminiEmbeddingRequestExt,
            };
        }

        // Provider-owned typed options (kept out of `siumai-core`).
        pub use options::{
            GeminiChatRequestExt, GeminiEmbeddingOptions, GeminiEmbeddingRequestExt,
            GeminiHarmBlockThreshold, GeminiHarmCategory, GeminiOptions, GeminiResponseModality,
            GeminiSafetySetting, GeminiThinkingConfig, GeminiThinkingLevel,
        };

        /// Typed response metadata helpers (`ChatResponse.provider_metadata["google"]`).
        pub mod metadata {
            pub use siumai_provider_gemini::provider_metadata::gemini::{
                GeminiChatResponseExt, GeminiMetadata, GeminiSource,
            };
        }
        pub use metadata::{GeminiChatResponseExt, GeminiMetadata, GeminiSource};

        /// Non-unified Gemini extension APIs (escape hatches).
        pub mod ext {
            pub use siumai_provider_gemini::providers::gemini::ext::{
                code_execution, file_search_stores, tools,
            };
        }

        /// Provider-specific resources not covered by the unified families.
        pub mod resources {
            pub use siumai_provider_gemini::providers::gemini::{
                GeminiFileSearchStores, GeminiFiles, GeminiModels,
            };
        }

        // Legacy Gemini parameter structs (provider-owned).
        pub use siumai_provider_gemini::params::gemini::{
            GeminiParams, GeminiParamsBuilder, GenerationConfig, SafetyCategory, SafetySetting,
            SafetyThreshold,
        };
    }

    #[cfg(feature = "google-vertex")]
    pub mod google_vertex {
        pub use siumai_provider_google_vertex::providers::vertex::{
            GoogleVertexClient, GoogleVertexConfig,
        };

        /// Provider-hosted tools (Vercel-aligned `googleVertexTools`).
        pub mod tools {
            pub use siumai_provider_google_vertex::tools::{
                VertexRagStoreConfig, code_execution, enterprise_web_search, file_search,
                google_maps, google_search, url_context, vertex_rag_store,
            };
        }

        /// Typed provider options (`provider_options_map["vertex"]`).
        pub mod options {
            pub use siumai_provider_google_vertex::provider_options::vertex::{
                VertexEmbeddingOptions, VertexImagenEditOptions, VertexImagenInlineImage,
                VertexImagenMaskImageConfig, VertexImagenOptions, VertexImagenReferenceImage,
            };
            pub use siumai_provider_google_vertex::providers::vertex::{
                VertexEmbeddingRequestExt, VertexImagenRequestExt,
            };
        }

        pub use options::{
            VertexEmbeddingOptions, VertexEmbeddingRequestExt, VertexImagenEditOptions,
            VertexImagenInlineImage, VertexImagenMaskImageConfig, VertexImagenOptions,
            VertexImagenReferenceImage, VertexImagenRequestExt,
        };
    }

    #[cfg(feature = "minimaxi")]
    pub mod minimaxi {
        pub use siumai_provider_minimaxi::providers::minimaxi::MinimaxiClient;

        /// Typed provider options (`provider_options_map["minimaxi"]`).
        pub mod options {
            pub use siumai_provider_minimaxi::provider_options::MinimaxiTtsOptions;
            pub use siumai_provider_minimaxi::providers::minimaxi::ext::tts::MinimaxiTtsRequestBuilder;
            pub use siumai_provider_minimaxi::providers::minimaxi::ext::tts_options::MinimaxiTtsRequestExt;
        }

        // Provider-owned typed options (kept out of `siumai-core`).
        pub use options::{MinimaxiTtsOptions, MinimaxiTtsRequestBuilder, MinimaxiTtsRequestExt};

        /// Non-unified MiniMaxi extension APIs (escape hatches).
        pub mod ext {
            pub use siumai_provider_minimaxi::providers::minimaxi::ext::{music, video};
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
        pub use siumai_provider_ollama::providers::ollama::{OllamaClient, OllamaConfig};

        /// Typed provider options (`provider_options_map["ollama"]`).
        pub mod options {
            pub use siumai_provider_ollama::provider_options::OllamaOptions;
            pub use siumai_provider_ollama::providers::ollama::ext::OllamaChatRequestExt;
            pub use siumai_provider_ollama::providers::ollama::types::{
                OllamaEmbeddingOptions, OllamaEmbeddingRequestExt,
            };
        }

        // Provider-owned typed options (kept out of `siumai-core`).
        pub use options::{
            OllamaChatRequestExt, OllamaEmbeddingOptions, OllamaEmbeddingRequestExt, OllamaOptions,
        };

        /// Non-unified Ollama extension APIs (escape hatches).
        pub mod ext {
            pub use siumai_provider_ollama::providers::ollama::ext::request_options;
        }
    }

    #[cfg(feature = "anthropic")]
    pub mod anthropic_vertex {
        pub use siumai_provider_anthropic::providers::anthropic_vertex::client::{
            VertexAnthropicClient, VertexAnthropicConfig,
        };
    }

    #[cfg(feature = "xai")]
    pub mod xai {
        pub use siumai_provider_xai::providers::xai::XaiClient;

        /// Typed provider options (`provider_options_map["xai"]`).
        pub mod options {
            pub use siumai_provider_xai::providers::xai::ext::XaiChatRequestExt;
            pub use siumai_provider_xai::providers::xai::{
                SearchMode, SearchSource, SearchSourceType, XaiOptions, XaiSearchParameters,
            };
        }

        // Provider-owned typed options (kept out of `siumai-core`).
        pub use options::{
            SearchMode, SearchSource, SearchSourceType, XaiChatRequestExt, XaiOptions,
            XaiSearchParameters,
        };

        /// Non-unified xAI extension APIs (escape hatches).
        pub mod ext {
            pub use siumai_provider_xai::providers::xai::ext::*;
        }
    }

    #[cfg(feature = "groq")]
    pub mod groq {
        pub use siumai_provider_groq::providers::groq::GroqClient;

        /// Typed provider options (`provider_options_map["groq"]`).
        pub mod options {
            pub use siumai_provider_groq::provider_options::GroqOptions;
            pub use siumai_provider_groq::providers::groq::ext::GroqChatRequestExt;
        }

        // Provider-owned typed options (kept out of `siumai-core`).
        pub use options::{GroqChatRequestExt, GroqOptions};

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
        ModerationCapability, MusicGenerationCapability, SpeechExtras, TimeoutCapability,
        TranscriptionExtras, VideoGenerationCapability,
    };

    /// Types used by non-unified extension capabilities.
    pub mod types {
        pub use siumai_core::types::{
            FileDeleteResponse, FileListQuery, FileListResponse, FileObject, FileUploadRequest,
            ImageEditRequest, ImageVariationRequest, ModerationRequest, ModerationResponse,
            VideoGenerationRequest, VideoGenerationResponse, VideoTaskStatus,
            VideoTaskStatusResponse,
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
    /// This module intentionally exports only the six stable model families:
    /// Language/Embedding/Image/Reranking/Speech/Transcription.
    pub mod unified {
        pub use crate::Provider;
        pub use crate::provider::Siumai;
        pub use crate::retry_api::*;
        pub use crate::tools;
        pub use crate::{assistant, conversation, conversation_with_system, messages, quick_chat};
        pub use crate::{system, tool, user, user_with_image};
        pub use siumai_core::error::{ErrorCategory, LlmError};
        pub use siumai_core::streaming::*;
        pub use siumai_core::traits::{
            ChatCapability, EmbeddingCapability, EmbeddingExtensions, ImageGenerationCapability,
            ProviderCapabilities, RerankCapability, SpeechCapability, TranscriptionCapability,
        };

        // Core request/response types for the six stable model families.
        pub use siumai_core::types::{
            AudioStreamEvent, CacheControl, ChatMessage, ChatRequest, ChatRequestBuilder,
            ChatResponse, CommonParams, CompletionTokensDetails, ContentPart,
            CustomProviderOptions, EmbeddingRequest, EmbeddingResponse, FinishReason,
            GeneratedImage, HttpConfig, ImageDetail, ImageGenerationRequest,
            ImageGenerationResponse, MediaSource, MessageContent, MessageMetadata, MessageRole,
            ModelInfo, OutputSchema, PromptTokensDetails, ProviderDefinedTool, ProviderOptionsMap,
            ProviderType, RerankRequest, RerankResponse, ResponseFormat, ResponseMetadata,
            SchemaValidator, SttRequest, SttResponse, Tool, ToolChoice, TtsRequest, TtsResponse,
            Usage, Warning,
        };

        pub mod registry {
            pub use crate::registry::{
                EmbeddingModelHandle, ImageModelHandle, LanguageModelHandle, ProviderFactory,
                ProviderRegistryHandle, RegistryOptions, RerankingModelHandle, SpeechModelHandle,
                TranscriptionModelHandle, create_bare_registry, create_empty_registry,
                create_provider_registry,
            };

            #[cfg(any(
                feature = "openai",
                feature = "anthropic",
                feature = "google",
                feature = "ollama",
                feature = "xai",
                feature = "groq",
                feature = "minimaxi"
            ))]
            pub use crate::registry::{create_registry_with_defaults, global};
        }
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

    /// Create an Anthropic client builder
    #[cfg(feature = "anthropic")]
    pub fn anthropic() -> siumai_provider_anthropic::providers::anthropic::AnthropicBuilder {
        siumai_provider_anthropic::providers::anthropic::AnthropicBuilder::new(
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

    /// Create a Google Vertex client builder
    #[cfg(feature = "google-vertex")]
    pub fn vertex() -> siumai_provider_google_vertex::providers::vertex::GoogleVertexBuilder {
        siumai_provider_google_vertex::providers::vertex::GoogleVertexBuilder::new(
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
