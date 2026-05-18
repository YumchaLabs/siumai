//! OpenAI-Compatible Provider Interface
//!
//! This module provides model constants for OpenAI-compatible providers.
//! These providers use a dedicated OpenAI-compatible client (`OpenAiCompatibleClient`) that
//! applies provider adapters (field mappings, reasoning extraction, etc.).
//!
//! # Usage
//! ```rust,no_run
//! use siumai_provider_openai_compatible::providers::openai_compatible::{deepseek, OpenAiCompatibleClient};
//! use siumai_core::types::ChatRequest;
//! use siumai_provider_openai_compatible::{text, user};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Config-first construction (recommended):
//!     // Reads `DEEPSEEK_API_KEY` by default for provider_id = "deepseek".
//!     let client = OpenAiCompatibleClient::from_builtin_env("deepseek", Some(deepseek::CHAT)).await?;
//!
//!     // Invocation goes through the stable model-family APIs:
//!     let req = ChatRequest::new(vec![user!("hi")]);
//!     let _resp = text::generate(&client, req, text::GenerateOptions::default()).await?;
//!
//!     Ok(())
//! }
//! ```

pub mod providers;

// New adapter system modules
pub mod alibaba_video;
pub mod builder;
pub mod config;
pub mod default_models;
pub mod ext;
pub mod middleware;
pub mod model_alias;
pub mod openai_client;
pub mod settings;
pub mod spec;
// Macro list for generating builder methods across modules
pub mod builder_list;

// Protocol (standard) modules live under `standards::openai::compat`.
// Keep these module paths for backward compatibility.
pub mod adapter {
    pub use crate::standards::openai::compat::adapter::*;
}
pub mod openai_config {
    pub use crate::standards::openai::compat::openai_config::*;
}
pub mod streaming {
    pub use crate::standards::openai::compat::streaming::*;
}
pub mod transformers {
    pub use crate::standards::openai::compat::transformers::*;
}
pub mod types {
    pub use crate::standards::openai::compat::types::*;
}

// Backward compatible path: `providers::openai_compatible::registry::*`
pub mod registry {
    pub use crate::standards::openai::compat::provider_registry::*;
}

// Re-export model constants for easy access
pub use providers::models::{
    alibaba, deepinfra, deepseek, fireworks, google_vertex_xai, groq, mistral, moonshot,
    moonshotai, openrouter, perplexity, qwen, siliconflow, together, togetherai, vertex_maas, xai,
};

// Re-export new adapter system
pub use crate::provider_options::{
    AlibabaChatOptions, AlibabaLanguageModelOptions, AlibabaVideoModelOptions,
    DeepSeekLanguageModelOptions, DeepSeekThinkingConfig, DeepSeekThinkingType,
    FireworksChatOptions, FireworksLanguageModelOptions, FireworksReasoningHistory,
    FireworksThinkingConfig, FireworksThinkingType, GroqLanguageModelOptions, GroqReasoningEffort,
    GroqReasoningFormat, GroqServiceTier, GroqTranscriptionModelOptions, MistralChatOptions,
    MistralLanguageModelOptions, MistralReasoningEffort, MoonshotAIChatOptions,
    MoonshotAILanguageModelOptions, MoonshotAIReasoningHistory, MoonshotAIThinkingConfig,
    MoonshotAIThinkingType, NewsSearchSource, OpenAICompatibleEmbeddingModelOptions,
    OpenAICompatibleLanguageModelChatOptions, OpenAICompatibleLanguageModelCompletionOptions,
    OpenAiCompatibleEmbeddingModelOptions, OpenAiCompatibleLanguageModelChatOptions,
    OpenAiCompatibleLanguageModelCompletionOptions, OpenRouterOptions, OpenRouterTransform,
    PerplexityOptions, PerplexitySearchContextSize, PerplexitySearchMode,
    PerplexitySearchRecencyFilter, PerplexityUserLocation, PerplexityWebSearchOptions,
    QwenChatOptions, QwenLanguageModelOptions, RssSearchSource, SearchMode, SearchSource,
    TogetherAIImageModelOptions, TogetherAIRerankingModelOptions, WebSearchSource, XSearchSource,
    XaiChatReasoningEffort, XaiFilesOptions, XaiImageModelOptions, XaiImageQuality,
    XaiImageResolution, XaiLanguageModelChatOptions, XaiLanguageModelResponsesOptions, XaiOptions,
    XaiReasoningSummary, XaiResponseInclude, XaiResponsesReasoningEffort, XaiSearchParameters,
    XaiVideoMode, XaiVideoModelOptions, XaiVideoResolution,
};
#[allow(deprecated)]
pub use crate::provider_options::{
    AlibabaProviderOptions, AlibabaVideoProviderOptions, DeepSeekChatOptions,
    DeepSeekProviderOptions, GroqChatOptions, GroqProviderOptions, MoonshotAIProviderOptions,
    OpenAICompatibleCompletionProviderOptions, OpenAICompatibleEmbeddingProviderOptions,
    OpenAICompatibleProviderOptions, OpenAiCompatibleCompletionProviderOptions,
    OpenAiCompatibleEmbeddingProviderOptions, OpenAiCompatibleProviderOptions, QwenProviderOptions,
    TogetherAIImageProviderOptions, TogetherAIRerankingOptions, TogetherAiImageModelOptions,
    TogetherAiImageProviderOptions, TogetherAiRerankingModelOptions, TogetherAiRerankingOptions,
    XaiImageProviderOptions, XaiProviderOptions, XaiResponsesProviderOptions,
    XaiVideoProviderOptions,
};
pub use crate::standards::openai::compat::provider_registry::{
    ConfigurableAdapter, ProviderConfig,
};
pub use adapter::ResponseMetadataExtractor as MetadataExtractor;
pub use adapter::{
    MetadataExtractingAdapter, OpenAiCompatibleRequestSettings, ProviderAdapter,
    ProviderCompatibility, RequestBodyTransformer, RequestTransformingAdapter,
    ResponseMetadataExtractor,
};
pub use alibaba_video::AlibabaVideoModel;
pub use builder::OpenAiCompatibleBuilder;
pub use config::{
    generic_provider_config, get_builtin_providers, get_provider_config, list_provider_ids,
    provider_supports_capability,
};
pub use ext::{
    AlibabaChatRequestExt, DeepSeekChatRequestExt, FireworksChatRequestExt, GroqChatRequestExt,
    GroqTranscriptionRequestExt, MistralChatRequestExt, MoonshotAIChatRequestExt,
    OpenAiCompatibleChatRequestExt, OpenAiCompatibleCompletionRequestExt,
    OpenAiCompatibleEmbeddingRequestExt, OpenRouterChatRequestExt, OpenRouterChatResponseExt,
    OpenRouterContentPartExt, OpenRouterContentPartMetadata, OpenRouterMetadata, OpenRouterSource,
    OpenRouterSourceExt, OpenRouterSourceMetadata, PerplexityChatRequestExt,
    PerplexityChatResponseExt, PerplexityCost, PerplexityImage, PerplexityMetadata,
    PerplexityUsage, QwenChatRequestExt, TogetherAIImageRequestExt, TogetherAIRerankRequestExt,
    XaiChatRequestExt, XaiImageRequestExt,
};
pub use middleware::OpenAiCompatibleToolWarningsMiddleware;
pub use model_alias::normalize_model_id;
pub use openai_client::OpenAiCompatibleClient;
pub use openai_config::OpenAiCompatibleConfig;
pub use settings::{
    AlibabaProviderSettings, DeepInfraProviderSettings, DeepSeekProviderSettings,
    FireworksProviderSettings, GoogleVertexMaasProviderSettings, GoogleVertexXaiProviderSettings,
    GroqProviderSettings, MistralProviderSettings, MoonshotAIProviderSettings,
    OpenAICompatibleProviderSettings, PerplexityProviderSettings, TogetherAIProviderSettings,
    XaiProviderSettings,
};
pub use types::{FieldMappings, ModelConfig, RequestType};

/// AI SDK-aligned OpenAI-compatible error envelope.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OpenAiCompatibleErrorData {
    pub error: OpenAiCompatibleErrorPayload,
}

/// AI SDK-exact-case alias for OpenAI-compatible error envelopes.
pub type OpenAICompatibleErrorData = OpenAiCompatibleErrorData;

/// AI SDK-style generic error-structure helper for OpenAI-compatible providers.
///
/// TypeScript exports `ProviderErrorStructure<T>` as a small public contract for provider-owned
/// error decoding plus message extraction. Rust keeps the same concept as a serde-based helper
/// without forcing callers to replace the compat adapter/runtime.
type ProviderErrorDeserializer<T> =
    std::sync::Arc<dyn Fn(&serde_json::Value) -> serde_json::Result<T> + Send + Sync>;
type ProviderErrorMessageFormatter<T> = std::sync::Arc<dyn Fn(&T) -> String + Send + Sync>;
type ProviderErrorRetryPredicate<T> =
    std::sync::Arc<dyn Fn(reqwest::StatusCode, Option<&T>) -> bool + Send + Sync>;

#[derive(Clone)]
pub struct ProviderErrorStructure<T> {
    deserialize_error: ProviderErrorDeserializer<T>,
    error_to_message: ProviderErrorMessageFormatter<T>,
    is_retryable: Option<ProviderErrorRetryPredicate<T>>,
}

impl<T> std::fmt::Debug for ProviderErrorStructure<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderErrorStructure")
            .field("has_is_retryable", &self.is_retryable.is_some())
            .finish_non_exhaustive()
    }
}

impl<T> ProviderErrorStructure<T> {
    /// Create a custom provider error structure from decode and formatting callbacks.
    pub fn new<D, M>(deserialize_error: D, error_to_message: M) -> Self
    where
        D: Fn(&serde_json::Value) -> serde_json::Result<T> + Send + Sync + 'static,
        M: Fn(&T) -> String + Send + Sync + 'static,
    {
        Self {
            deserialize_error: std::sync::Arc::new(deserialize_error),
            error_to_message: std::sync::Arc::new(error_to_message),
            is_retryable: None,
        }
    }

    /// Create a serde-based provider error structure for a deserializable error envelope.
    pub fn serde_json<M>(error_to_message: M) -> Self
    where
        T: serde::de::DeserializeOwned + 'static,
        M: Fn(&T) -> String + Send + Sync + 'static,
    {
        Self::new(|raw| serde_json::from_value(raw.clone()), error_to_message)
    }

    /// Attach an optional retryability predicate.
    pub fn with_is_retryable<P>(mut self, predicate: P) -> Self
    where
        P: Fn(reqwest::StatusCode, Option<&T>) -> bool + Send + Sync + 'static,
    {
        self.is_retryable = Some(std::sync::Arc::new(predicate));
        self
    }

    /// Decode a provider error envelope from parsed JSON.
    pub fn deserialize(&self, raw: &serde_json::Value) -> serde_json::Result<T> {
        (self.deserialize_error)(raw)
    }

    /// Convert a decoded provider error into a message string.
    pub fn message(&self, error: &T) -> String {
        (self.error_to_message)(error)
    }

    /// Evaluate retryability when a predicate has been configured.
    pub fn is_retryable(&self, status: reqwest::StatusCode, error: Option<&T>) -> Option<bool> {
        self.is_retryable
            .as_ref()
            .map(|predicate| predicate(status, error))
    }
}

/// AI SDK-aligned OpenAI-compatible error payload.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OpenAiCompatibleErrorPayload {
    pub message: String,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub error_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<serde_json::Value>,
}

/// AI SDK-aligned Fireworks error envelope.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FireworksErrorData {
    pub error: String,
}

/// AI SDK-style alias for DeepInfra compat error envelopes.
pub type DeepInfraErrorData = OpenAiCompatibleErrorData;

/// AI SDK-style OpenAI-compatible chat model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type OpenAiCompatibleChatModelId = String;

/// AI SDK-exact-case OpenAI-compatible chat model id alias.
pub type OpenAICompatibleChatModelId = OpenAiCompatibleChatModelId;

/// AI SDK-style OpenAI-compatible completion model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type OpenAiCompatibleCompletionModelId = String;

/// AI SDK-exact-case OpenAI-compatible completion model id alias.
pub type OpenAICompatibleCompletionModelId = OpenAiCompatibleCompletionModelId;

/// AI SDK-style OpenAI-compatible embedding model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type OpenAiCompatibleEmbeddingModelId = String;

/// AI SDK-exact-case OpenAI-compatible embedding model id alias.
pub type OpenAICompatibleEmbeddingModelId = OpenAiCompatibleEmbeddingModelId;

/// AI SDK-style OpenAI-compatible image model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type OpenAiCompatibleImageModelId = String;

/// AI SDK-exact-case OpenAI-compatible image model id alias.
pub type OpenAICompatibleImageModelId = OpenAiCompatibleImageModelId;

/// AI SDK-exact-case alias for generic OpenAI-compatible request settings.
pub type OpenAICompatibleRequestSettings = OpenAiCompatibleRequestSettings;

/// AI SDK-exact-case alias for OpenAI-compatible clients.
pub type OpenAICompatibleClient = OpenAiCompatibleClient;

/// AI SDK-exact-case alias for OpenAI-compatible configs.
pub type OpenAICompatibleConfig = OpenAiCompatibleConfig;

/// Rust package version exposed on the generic OpenAI-compatible package-surface facade.
pub const OPENAI_COMPATIBLE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// AI SDK-style provider-scoped alias for DeepSeek compat clients.
pub type DeepSeekClient = openai_client::OpenAiCompatibleClient;
/// AI SDK-style provider-scoped alias for DeepSeek compat configs.
pub type DeepSeekConfig = openai_config::OpenAiCompatibleConfig;
/// Rust package version exposed on the DeepSeek package-surface facade.
pub const DEEPSEEK_VERSION: &str = env!("CARGO_PKG_VERSION");
/// AI SDK-style DeepSeek chat model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type DeepSeekChatModelId = String;

/// AI SDK-style provider-scoped alias for Groq compat clients.
pub type GroqClient = openai_client::OpenAiCompatibleClient;
/// AI SDK-style provider-scoped alias for Groq compat configs.
pub type GroqConfig = openai_config::OpenAiCompatibleConfig;
/// Rust package version exposed on the Groq package-surface facade.
pub const GROQ_VERSION: &str = env!("CARGO_PKG_VERSION");
/// AI SDK-style Groq chat model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type GroqChatModelId = String;
/// AI SDK-style Groq transcription model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type GroqTranscriptionModelId = String;

/// AI SDK-style provider-scoped alias for Mistral compat clients.
pub type MistralClient = openai_client::OpenAiCompatibleClient;
/// AI SDK-style provider-scoped alias for Mistral compat configs.
pub type MistralConfig = openai_config::OpenAiCompatibleConfig;
/// Rust package version exposed on the Mistral package-surface facade.
pub const MISTRAL_VERSION: &str = env!("CARGO_PKG_VERSION");

/// AI SDK-style provider-scoped alias for Perplexity compat clients.
pub type PerplexityClient = openai_client::OpenAiCompatibleClient;
/// AI SDK-style provider-scoped alias for Perplexity compat configs.
pub type PerplexityConfig = openai_config::OpenAiCompatibleConfig;
/// Rust package version exposed on the Perplexity package-surface facade.
pub const PERPLEXITY_VERSION: &str = env!("CARGO_PKG_VERSION");

/// AI SDK-style provider-scoped alias for Fireworks compat text-family clients.
pub type FireworksClient = openai_client::OpenAiCompatibleClient;
/// AI SDK-style provider-scoped alias for Fireworks compat text-family configs.
pub type FireworksConfig = openai_config::OpenAiCompatibleConfig;
/// Rust package version exposed on the Fireworks package-surface facade.
pub const FIREWORKS_VERSION: &str = env!("CARGO_PKG_VERSION");

/// AI SDK-style Fireworks embedding model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type FireworksEmbeddingModelId = String;

/// AI SDK-style Fireworks image model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type FireworksImageModelId = String;

/// AI SDK-style provider-scoped alias for TogetherAI compat clients.
pub type TogetherAIClient = openai_client::OpenAiCompatibleClient;
/// AI SDK-style provider-scoped alias for TogetherAI compat configs.
pub type TogetherAIConfig = openai_config::OpenAiCompatibleConfig;
/// Rust package version exposed on the TogetherAI package-surface facade.
pub const TOGETHERAI_VERSION: &str = env!("CARGO_PKG_VERSION");
/// AI SDK-style TogetherAI chat model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type TogetherAIChatModelId = String;
/// AI SDK-style TogetherAI completion model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type TogetherAICompletionModelId = String;
/// AI SDK-style TogetherAI embedding model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type TogetherAIEmbeddingModelId = String;
/// AI SDK-style TogetherAI image model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type TogetherAIImageModelId = String;
/// AI SDK-style TogetherAI reranking model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type TogetherAIRerankingModelId = String;

/// AI SDK-style DeepInfra chat model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type DeepInfraChatModelId = String;

/// AI SDK-style DeepInfra completion model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type DeepInfraCompletionModelId = String;

/// AI SDK-style DeepInfra embedding model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type DeepInfraEmbeddingModelId = String;

/// AI SDK-style DeepInfra image model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type DeepInfraImageModelId = String;

/// AI SDK-style provider-scoped alias for DeepInfra compat text-family clients.
pub type DeepInfraClient = openai_client::OpenAiCompatibleClient;
/// AI SDK-style provider-scoped alias for DeepInfra compat text-family configs.
/// Rust package version exposed on the DeepInfra package-surface facade.
pub const DEEPINFRA_VERSION: &str = env!("CARGO_PKG_VERSION");
pub type DeepInfraConfig = openai_config::OpenAiCompatibleConfig;

/// AI SDK-style provider-scoped alias for Alibaba compat language-model clients.
pub type AlibabaClient = openai_client::OpenAiCompatibleClient;
/// AI SDK-style provider-scoped alias for Alibaba compat language-model configs.
pub type AlibabaConfig = openai_config::OpenAiCompatibleConfig;
/// Rust package version exposed on the Alibaba package-surface facade.
pub const ALIBABA_VERSION: &str = env!("CARGO_PKG_VERSION");
/// AI SDK-style Alibaba chat model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type AlibabaChatModelId = String;
/// AI SDK-style Alibaba video model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type AlibabaVideoModelId = String;

/// AI SDK-style provider-scoped alias for MoonshotAI compat language-model clients.
pub type MoonshotAIClient = openai_client::OpenAiCompatibleClient;
/// AI SDK-style provider-scoped alias for MoonshotAI compat language-model configs.
pub type MoonshotAIConfig = openai_config::OpenAiCompatibleConfig;
/// Rust package version exposed on the MoonshotAI package-surface facade.
pub const MOONSHOTAI_VERSION: &str = env!("CARGO_PKG_VERSION");

/// AI SDK-style MoonshotAI chat model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type MoonshotAIChatModelId = String;

/// AI SDK-style provider-scoped alias for xAI compat clients.
pub type XaiClient = openai_client::OpenAiCompatibleClient;
/// AI SDK-style provider-scoped alias for xAI compat configs.
pub type XaiConfig = openai_config::OpenAiCompatibleConfig;
/// Rust package version exposed on the xAI package-surface facade.
pub const XAI_VERSION: &str = env!("CARGO_PKG_VERSION");
/// AI SDK-style xAI chat model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type XaiChatModelId = String;
/// AI SDK-style xAI responses model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type XaiResponsesModelId = String;
/// AI SDK-style xAI image model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type XaiImageModelId = String;
/// AI SDK-style xAI video model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type XaiVideoModelId = String;

/// AI SDK-style provider-scoped alias for Google Vertex MaaS compat text-family clients.
pub type GoogleVertexMaasClient = openai_client::OpenAiCompatibleClient;
/// AI SDK-style provider-scoped alias for Google Vertex MaaS compat text-family configs.
pub type GoogleVertexMaasConfig = openai_config::OpenAiCompatibleConfig;
/// Rust package version exposed on the Google Vertex MaaS package-surface facade.
pub const GOOGLE_VERTEX_MAAS_VERSION: &str = env!("CARGO_PKG_VERSION");
/// AI SDK-style Google Vertex MaaS model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type GoogleVertexMaasModelId = String;

/// AI SDK-style provider-scoped alias for Google Vertex xAI compat text-family clients.
pub type GoogleVertexXaiClient = openai_client::OpenAiCompatibleClient;
/// AI SDK-style provider-scoped alias for Google Vertex xAI compat text-family configs.
pub type GoogleVertexXaiConfig = openai_config::OpenAiCompatibleConfig;
/// Rust package version exposed on the Google Vertex xAI package-surface facade.
pub const GOOGLE_VERTEX_XAI_VERSION: &str = env!("CARGO_PKG_VERSION");
/// AI SDK-style Google Vertex xAI model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type GoogleVertexXaiModelId = String;

// Test modules
#[cfg(test)]
mod tests {
    pub mod base_url_tests;

    use super::{
        DeepInfraChatModelId, DeepInfraCompletionModelId, DeepInfraEmbeddingModelId,
        DeepInfraImageModelId, FireworksErrorData, OpenAICompatibleChatModelId,
        OpenAICompatibleClient, OpenAICompatibleCompletionModelId, OpenAICompatibleConfig,
        OpenAICompatibleEmbeddingModelId, OpenAICompatibleErrorData, OpenAICompatibleImageModelId,
        OpenAICompatibleRequestSettings, OpenAiCompatibleErrorData, ProviderErrorStructure,
    };

    #[test]
    fn openai_compatible_error_data_deserializes_ai_sdk_shape() {
        let data: OpenAiCompatibleErrorData = serde_json::from_value(serde_json::json!({
            "error": {
                "message": "bad request",
                "type": "invalid_request_error",
                "param": null,
                "code": "invalid_prompt"
            }
        }))
        .expect("error data should deserialize");

        assert_eq!(data.error.message, "bad request");
        assert_eq!(
            data.error.error_type.as_deref(),
            Some("invalid_request_error")
        );
        assert_eq!(data.error.code, Some(serde_json::json!("invalid_prompt")));
    }

    #[test]
    fn fireworks_error_data_deserializes_ai_sdk_shape() {
        let data: FireworksErrorData = serde_json::from_value(serde_json::json!({
            "error": "rate limit exceeded"
        }))
        .expect("error data should deserialize");

        assert_eq!(data.error, "rate limit exceeded");
    }

    #[test]
    fn openai_compatible_exact_case_aliases_remain_available() {
        let _: OpenAICompatibleChatModelId = "gpt-4o".to_string();
        let _: OpenAICompatibleCompletionModelId = "gpt-4o-mini-instruct".to_string();
        let _: OpenAICompatibleEmbeddingModelId = "text-embedding-3-small".to_string();
        let _: OpenAICompatibleImageModelId = "black-forest-labs/FLUX.1-schnell".to_string();
        let _ = std::mem::size_of::<OpenAICompatibleClient>();
        let _ = std::mem::size_of::<OpenAICompatibleConfig>();
        let _ = std::mem::size_of::<OpenAICompatibleRequestSettings>();

        let data: OpenAICompatibleErrorData = serde_json::from_value(serde_json::json!({
            "error": {
                "message": "bad request"
            }
        }))
        .expect("exact-case error alias should deserialize");

        assert_eq!(data.error.message, "bad request");
    }

    #[test]
    fn deepinfra_model_id_aliases_remain_available() {
        let _ = std::mem::size_of::<DeepInfraChatModelId>();
        let _ = std::mem::size_of::<DeepInfraCompletionModelId>();
        let _ = std::mem::size_of::<DeepInfraEmbeddingModelId>();
        let _ = std::mem::size_of::<DeepInfraImageModelId>();
    }

    #[test]
    fn provider_error_structure_deserializes_formats_and_marks_retryable() {
        let structure = ProviderErrorStructure::<OpenAiCompatibleErrorData>::serde_json(|data| {
            data.error.message.clone()
        })
        .with_is_retryable(|status, _| {
            status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
        });

        let data = structure
            .deserialize(&serde_json::json!({
                "error": {
                    "message": "rate limited",
                    "type": "rate_limit_error"
                }
            }))
            .expect("deserialize provider error structure");

        assert_eq!(structure.message(&data), "rate limited");
        assert_eq!(
            structure.is_retryable(reqwest::StatusCode::BAD_REQUEST, Some(&data)),
            Some(false)
        );
        assert_eq!(
            structure.is_retryable(reqwest::StatusCode::TOO_MANY_REQUESTS, Some(&data)),
            Some(true)
        );
    }
}
