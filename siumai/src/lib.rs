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

pub mod embedding;
pub mod image;
pub mod rerank;
pub mod speech;
/// Structured output helpers (JSON extraction + parsing).
pub mod structured_output;
/// Model families (recommended Rust-first surface).
pub mod text;
pub mod transcription;

/// Tool runtime (schema + execution binding).
pub mod tooling;

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
            ChatByteStream, LanguageModelV3StreamPart, OpenAiResponsesStreamPartsBridge,
            StreamPartNamespace, V3UnsupportedPartBehavior, encode_chat_stream_as_jsonl,
            encode_chat_stream_as_sse, transform_chat_event_stream,
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
        pub use siumai_provider_openai::providers::openai::{OpenAiClient, OpenAiConfig};

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

    /// OpenAI-compatible vendors (DeepSeek/OpenRouter/Moonshot/etc.) via the OpenAI-like protocol family.
    #[cfg(feature = "openai")]
    pub mod openai_compatible {
        pub use siumai_provider_openai_compatible::providers::openai_compatible::{
            ConfigurableAdapter, OpenAiCompatibleClient, OpenAiCompatibleConfig, ProviderAdapter,
            ProviderCompatibility, ProviderConfig, get_provider_config, list_provider_ids,
            provider_supports_capability,
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
    pub mod perplexity {
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
        pub use options::{
            PerplexityChatRequestExt, PerplexityOptions, PerplexitySearchContextSize,
            PerplexitySearchMode, PerplexitySearchRecencyFilter, PerplexityUserLocation,
            PerplexityWebSearchOptions,
        };
    }

    #[cfg(feature = "bedrock")]
    pub mod bedrock {
        pub use siumai_provider_amazon_bedrock::providers::bedrock::{
            BedrockClient, BedrockConfig,
        };

        /// Typed response metadata helpers (`ChatResponse.provider_metadata["bedrock"]`).
        pub mod metadata {
            pub use siumai_provider_amazon_bedrock::provider_metadata::bedrock::{
                BedrockChatResponseExt, BedrockMetadata,
            };
        }

        /// Typed provider options (`provider_options_map["bedrock"]`).
        pub mod options {
            pub use siumai_provider_amazon_bedrock::provider_options::{
                BedrockChatOptions, BedrockRerankOptions,
            };
            pub use siumai_provider_amazon_bedrock::providers::bedrock::{
                BedrockChatRequestExt, BedrockRerankRequestExt,
            };
        }

        pub use metadata::{BedrockChatResponseExt, BedrockMetadata};
        pub use options::{
            BedrockChatOptions, BedrockChatRequestExt, BedrockRerankOptions,
            BedrockRerankRequestExt,
        };
    }

    #[cfg(feature = "cohere")]
    pub mod cohere {
        pub use siumai_provider_cohere::providers::cohere::{CohereClient, CohereConfig};

        /// Typed provider options (`provider_options_map["cohere"]`).
        pub mod options {
            pub use siumai_provider_cohere::provider_options::CohereRerankOptions;
            pub use siumai_provider_cohere::providers::cohere::CohereRerankRequestExt;
        }

        pub use options::{CohereRerankOptions, CohereRerankRequestExt};
    }

    #[cfg(feature = "togetherai")]
    pub mod togetherai {
        pub use siumai_provider_togetherai::providers::togetherai::{
            TogetherAiClient, TogetherAiConfig,
        };

        /// Typed provider options (`provider_options_map["togetherai"]`).
        pub mod options {
            pub use siumai_provider_togetherai::provider_options::TogetherAiRerankOptions;
            pub use siumai_provider_togetherai::providers::togetherai::TogetherAiRerankRequestExt;
        }

        pub use options::{TogetherAiRerankOptions, TogetherAiRerankRequestExt};
    }

    #[cfg(feature = "azure")]
    pub mod azure {
        pub use siumai_provider_azure::providers::azure_openai::{
            AzureChatMode, AzureOpenAiBuilder, AzureOpenAiClient, AzureOpenAiConfig,
            AzureOpenAiSpec, AzureUrlConfig,
        };
    }

    #[cfg(feature = "anthropic")]
    pub mod anthropic {
        pub use siumai_provider_anthropic::providers::anthropic::{
            AnthropicClient, AnthropicConfig,
        };

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
            pub use siumai_provider_anthropic::provider_options::anthropic::{
                AnthropicCacheControl, AnthropicCacheType, AnthropicOptions,
                AnthropicResponseFormat, AnthropicStructuredOutputMode, PromptCachingConfig,
                ThinkingModeConfig,
            };
            pub use siumai_provider_anthropic::providers::anthropic::ext::AnthropicChatRequestExt;
        }
        pub use options::{
            AnthropicCacheControl, AnthropicCacheType, AnthropicChatRequestExt, AnthropicOptions,
            AnthropicResponseFormat, AnthropicStructuredOutputMode, PromptCachingConfig,
            ThinkingModeConfig,
        };

        /// Typed response metadata helpers (`ChatResponse.provider_metadata["anthropic"]`).
        pub mod metadata {
            pub use siumai_provider_anthropic::provider_metadata::anthropic::{
                AnthropicChatResponseExt, AnthropicCitation, AnthropicCitationsBlock,
                AnthropicContentPartExt, AnthropicMetadata, AnthropicServerToolUse,
                AnthropicSource, AnthropicToolCallMetadata, AnthropicToolCaller,
            };
        }
        pub use metadata::{
            AnthropicChatResponseExt, AnthropicCitation, AnthropicCitationsBlock,
            AnthropicContentPartExt, AnthropicMetadata, AnthropicServerToolUse, AnthropicSource,
            AnthropicToolCallMetadata, AnthropicToolCaller,
        };

        /// Non-unified Anthropic extension APIs (request extensions, tool helpers, thinking, etc.).
        pub mod ext {
            pub use siumai_provider_anthropic::providers::anthropic::ext::{
                structured_output, thinking, tools,
            };
        }

        /// Provider-specific resources not covered by the unified families.
        pub mod resources {
            pub use siumai_provider_anthropic::providers::anthropic::{
                AnthropicCountTokensResponse, AnthropicCreateMessageBatchRequest, AnthropicFile,
                AnthropicFileDeleteResponse, AnthropicFiles, AnthropicListFilesResponse,
                AnthropicListMessageBatchesResponse, AnthropicMessageBatch,
                AnthropicMessageBatchRequest, AnthropicMessageBatches, AnthropicTokens,
            };
        }

        // Legacy Anthropic parameter structs (provider-owned).
        pub use siumai_provider_anthropic::params::anthropic::{AnthropicParams, CacheControl};
    }

    #[cfg(feature = "google")]
    pub mod gemini {
        pub use siumai_provider_gemini::providers::gemini::GeminiClient;
        pub use siumai_provider_gemini::providers::gemini::types::GeminiConfig;

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
                GeminiChatResponseExt, GeminiContentPartExt, GeminiMetadata, GeminiSource,
            };
        }
        pub use metadata::{
            GeminiChatResponseExt, GeminiContentPartExt, GeminiMetadata, GeminiSource,
        };

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
                GeminiTokens,
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
        pub use super::gemini::*;
    }

    #[cfg(feature = "google-vertex")]
    pub mod google_vertex {
        pub use siumai_provider_google_vertex::providers::vertex::{
            GoogleVertexClient, GoogleVertexConfig,
        };

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
                MinimaxiTtsOptions,
            };
            pub use siumai_provider_minimaxi::providers::minimaxi::ext::MinimaxiChatRequestExt;
            pub use siumai_provider_minimaxi::providers::minimaxi::ext::tts::MinimaxiTtsRequestBuilder;
            pub use siumai_provider_minimaxi::providers::minimaxi::ext::tts_options::MinimaxiTtsRequestExt;
        }

        // Provider-owned typed options (kept out of `siumai-core`).
        pub use options::{
            MinimaxiChatRequestExt, MinimaxiOptions, MinimaxiResponseFormat,
            MinimaxiThinkingModeConfig, MinimaxiTtsOptions, MinimaxiTtsRequestBuilder,
            MinimaxiTtsRequestExt,
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
        pub use siumai_provider_ollama::providers::ollama::{OllamaClient, OllamaConfig};

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
            VertexAnthropicBuilder, VertexAnthropicClient, VertexAnthropicConfig,
        };
    }

    #[cfg(feature = "deepseek")]
    pub mod deepseek {
        pub use siumai_provider_deepseek::providers::deepseek::{DeepSeekClient, DeepSeekConfig};

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
            pub use siumai_provider_deepseek::providers::deepseek::DeepSeekOptions;
            pub use siumai_provider_deepseek::providers::deepseek::ext::DeepSeekChatRequestExt;
        }

        pub use options::{DeepSeekChatRequestExt, DeepSeekOptions};

        /// Non-unified DeepSeek extension APIs (escape hatches).
        pub mod ext {
            pub use siumai_provider_deepseek::providers::deepseek::ext::*;
        }
    }

    #[cfg(feature = "xai")]
    pub mod xai {
        pub use siumai_provider_xai::providers::xai::{XaiClient, XaiConfig};

        /// Typed response metadata helpers (`ChatResponse.provider_metadata["xai"]`).
        pub mod metadata {
            pub use siumai_provider_xai::provider_metadata::xai::{
                XaiChatResponseExt, XaiMetadata, XaiSource, XaiSourceExt, XaiSourceMetadata,
            };
        }

        pub use metadata::{
            XaiChatResponseExt, XaiMetadata, XaiSource, XaiSourceExt, XaiSourceMetadata,
        };

        /// Vercel-style provider tool factories that return `Tool` directly.
        pub mod provider_tools {
            pub use crate::tools::xai::*;
        }

        /// Typed provider options (`provider_options_map["xai"]`).
        pub mod options {
            pub use siumai_provider_xai::providers::xai::ext::{
                XaiChatRequestExt, XaiTtsRequestExt,
            };
            pub use siumai_provider_xai::providers::xai::{
                SearchMode, SearchSource, SearchSourceType, XaiOptions, XaiSearchParameters,
                XaiTtsOptions,
            };
        }

        // Provider-owned typed options (kept out of `siumai-core`).
        pub use options::{
            SearchMode, SearchSource, SearchSourceType, XaiChatRequestExt, XaiOptions,
            XaiSearchParameters, XaiTtsOptions, XaiTtsRequestExt,
        };

        /// Non-unified xAI extension APIs (escape hatches).
        pub mod ext {
            pub use siumai_provider_xai::providers::xai::ext::*;
        }
    }

    #[cfg(feature = "groq")]
    pub mod groq {
        pub use siumai_provider_groq::providers::groq::{GroqClient, GroqConfig};

        /// Typed response metadata helpers (`ChatResponse.provider_metadata["groq"]`).
        pub mod metadata {
            pub use siumai_provider_groq::provider_metadata::groq::{
                GroqChatResponseExt, GroqMetadata, GroqSource, GroqSourceExt, GroqSourceMetadata,
            };
        }

        pub use metadata::{
            GroqChatResponseExt, GroqMetadata, GroqSource, GroqSourceExt, GroqSourceMetadata,
        };

        /// Typed provider options (`provider_options_map["groq"]`).
        pub mod options {
            pub use siumai_provider_groq::provider_options::{
                GroqOptions, GroqReasoningEffort, GroqReasoningFormat, GroqServiceTier,
            };
            pub use siumai_provider_groq::providers::groq::ext::GroqChatRequestExt;
        }

        // Provider-owned typed options (kept out of `siumai-core`).
        pub use options::{
            GroqChatRequestExt, GroqOptions, GroqReasoningEffort, GroqReasoningFormat,
            GroqServiceTier,
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
        #[doc(hidden)]
        pub use crate::provider::Siumai;
        pub use crate::retry_api::*;
        pub use crate::tooling;
        pub use crate::tools;
        pub use crate::{assistant, conversation, conversation_with_system, messages, quick_chat};
        pub use crate::{embedding, image, rerank, speech, text, transcription};
        pub use crate::{system, tool, user, user_with_image};
        pub use siumai_core::error::{ErrorCategory, LlmError};
        pub use siumai_core::rerank::RerankingModel;
        pub use siumai_core::speech::SpeechModel;
        pub use siumai_core::streaming::*;
        pub use siumai_core::text::LanguageModel;
        pub use siumai_core::traits::{
            ChatCapability, EmbeddingCapability, EmbeddingExtensions, ImageGenerationCapability,
            ModelMetadata, ProviderCapabilities, RerankCapability, SpeechCapability,
            TranscriptionCapability,
        };
        pub use siumai_core::transcription::TranscriptionModel;

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

    /// Create a TogetherAI client builder
    #[cfg(feature = "togetherai")]
    pub fn togetherai() -> siumai_provider_togetherai::providers::togetherai::TogetherAiBuilder {
        siumai_provider_togetherai::providers::togetherai::TogetherAiBuilder::new(
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

    /// Create an Anthropic on Vertex client builder
    #[cfg(feature = "google-vertex")]
    pub fn anthropic_vertex()
    -> siumai_provider_google_vertex::providers::anthropic_vertex::VertexAnthropicBuilder {
        siumai_provider_google_vertex::providers::anthropic_vertex::VertexAnthropicBuilder::new(
            crate::builder::BuilderBase::default(),
        )
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
