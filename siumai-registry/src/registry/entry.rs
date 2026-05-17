//! Provider Registry - Vercel AI SDK Aligned Architecture
//!
//! This module provides a provider registry system that aligns with Vercel AI SDK's design:
//! - ProviderFactory trait for creating family model objects
//! - Registry stores factory instances (not hardcoded logic)
//! - Handles delegate to factories for family-model construction
//! - Easy to extend with new providers

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

#[cfg(test)]
use crate::client::LlmClient;
use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::retry_api::RetryOptions;
#[cfg(test)]
use crate::streaming::{ChatStream, ChatStreamHandle};
#[cfg(test)]
use crate::traits::ImageGenerationCapability;
#[cfg(test)]
use crate::traits::{
    ChatCapability, FileManagementCapability, MusicGenerationCapability, ProviderCapabilities,
    SkillsCapability, VideoGenerationCapability,
};
#[cfg(test)]
use crate::traits::{EmbeddingCapability, EmbeddingExtensions};
#[cfg(test)]
use crate::types::EmbeddingResponse;
#[cfg(test)]
use crate::types::{ChatMessage, ChatRequest, ChatResponse};
#[cfg(test)]
use crate::types::{ImageGenerationRequest, ImageGenerationResponse};

use lru::LruCache;
use std::num::NonZeroUsize;
use tokio::sync::Mutex as TokioMutex;

#[cfg(test)]
mod alias_tests;
#[cfg(test)]
mod boundary_tests;
mod build_context;
#[cfg(test)]
mod build_context_tests;
mod cache;
#[cfg(test)]
mod cache_tests;
mod extension_adapters;
mod factory;
mod handles;
#[cfg(test)]
mod interceptor_tests;
#[cfg(test)]
mod language_tests;
#[cfg(test)]
mod test_support;

pub use self::build_context::{BuildContext, ProviderBuildOverrides};
use self::cache::{
    CacheEntry, CompletionCacheEntry, SpeechCacheEntry, TranscriptionCacheEntry, VideoCacheEntry,
};
pub use self::factory::ProviderFactory;
#[cfg(test)]
use self::handles::image_model_handle_max_images_per_call;
use self::handles::image_model_handle_supports_model;
#[cfg(test)]
use self::handles::video_model_handle_max_videos_per_call;
pub use self::handles::{
    CompletionModelHandle, EmbeddingModelHandle, ImageModelHandle, LanguageModelHandle,
    RerankingModelHandle, SpeechModelHandle, TranscriptionModelHandle, VideoModelHandle,
};
#[cfg(test)]
pub use self::test_support::*;

pub struct RegistryOptions {
    pub separator: char,
    pub language_model_middleware: Vec<Arc<dyn LanguageModelMiddleware>>,
    /// HTTP interceptors applied to all clients created via the registry
    pub http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional pre-built HTTP client applied to all clients created via the registry.
    pub http_client: Option<reqwest::Client>,
    /// Optional custom HTTP transport applied to all clients created via the registry.
    pub http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    /// Optional HTTP configuration applied to all clients created via the registry.
    /// When set, this configuration is passed through BuildContext to provider
    /// factories so they can build HTTP clients consistently with SiumaiBuilder.
    pub http_config: Option<crate::types::HttpConfig>,
    /// Optional API key override applied to registry-built provider clients.
    pub api_key: Option<String>,
    /// Optional base URL override applied to registry-built provider clients.
    pub base_url: Option<String>,
    /// Optional unified reasoning enable flag applied to registry-built language models.
    pub reasoning_enabled: Option<bool>,
    /// Optional unified reasoning budget applied to registry-built language models.
    pub reasoning_budget: Option<i32>,
    /// Optional per-provider build overrides applied after global registry overrides.
    pub provider_build_overrides: HashMap<String, ProviderBuildOverrides>,
    /// Unified retry options applied to clients created via the registry (optional)
    pub retry_options: Option<RetryOptions>,
    /// Maximum number of cached clients (LRU eviction when exceeded)
    pub max_cache_entries: Option<usize>,
    /// Time-to-live for cached clients (None = no expiration)
    pub client_ttl: Option<Duration>,
    /// Whether to automatically add model-specific middlewares (e.g., ExtractReasoningMiddleware)
    /// based on provider and model ID. Default: true
    pub auto_middleware: bool,
}

impl Default for RegistryOptions {
    fn default() -> Self {
        Self {
            separator: ':',
            language_model_middleware: Vec::new(),
            http_interceptors: Vec::new(),
            http_client: None,
            http_transport: None,
            http_config: None,
            api_key: None,
            base_url: None,
            reasoning_enabled: None,
            reasoning_budget: None,
            provider_build_overrides: HashMap::new(),
            retry_options: None,
            max_cache_entries: None,
            client_ttl: None,
            auto_middleware: true,
        }
    }
}

/// Provider registry handle - aligned with Vercel AI SDK design
///
/// Stores provider factories and delegates client creation to them.
/// This makes the registry extensible and provider-agnostic.
///
/// Features LRU cache with optional TTL to prevent unbounded growth.
pub struct ProviderRegistryHandle {
    /// Registered provider factories (provider_id -> factory)
    providers: HashMap<String, Arc<dyn ProviderFactory>>,
    /// Separator for parsing "provider:model" identifiers
    separator: char,
    /// Middlewares to apply to all language models
    pub(crate) middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    /// HTTP interceptors to apply to clients created via this registry
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Registry-level pre-built HTTP client propagated via BuildContext.
    http_client: Option<reqwest::Client>,
    /// Registry-level custom HTTP transport propagated via BuildContext.
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    /// Registry-level HTTP configuration propagated via BuildContext.
    http_config: Option<crate::types::HttpConfig>,
    /// Registry-level API key propagated via BuildContext.
    api_key: Option<String>,
    /// Registry-level base URL propagated via BuildContext.
    base_url: Option<String>,
    /// Registry-level unified reasoning flag propagated via BuildContext.
    reasoning_enabled: Option<bool>,
    /// Registry-level unified reasoning budget propagated via BuildContext.
    reasoning_budget: Option<i32>,
    /// Registry-level per-provider build overrides applied after global overrides.
    provider_build_overrides: HashMap<String, ProviderBuildOverrides>,
    /// LRU cache for language model clients (key: "provider:model")
    /// Uses async Mutex for concurrent access and per-key build de-duplication
    language_model_cache: Arc<TokioMutex<LruCache<String, CacheEntry>>>,
    /// LRU cache for completion-family models (key: "provider:model")
    completion_model_cache: Arc<TokioMutex<LruCache<String, CompletionCacheEntry>>>,
    /// LRU cache for speech-family models (key: "provider:model")
    speech_model_cache: Arc<TokioMutex<LruCache<String, SpeechCacheEntry>>>,
    /// LRU cache for transcription-family models (key: "provider:model")
    transcription_model_cache: Arc<TokioMutex<LruCache<String, TranscriptionCacheEntry>>>,
    /// LRU cache for video-family models (key: "provider:model")
    video_model_cache: Arc<TokioMutex<LruCache<String, VideoCacheEntry>>>,
    /// TTL for cached clients
    client_ttl: Option<Duration>,
    /// Whether to automatically add model-specific middlewares
    auto_middleware: bool,
    /// Registry-level retry options applied during client build (optional)
    retry_options: Option<RetryOptions>,
}

impl ProviderRegistryHandle {
    /// Split a registry model id like "provider:model" into (provider, model).
    fn split_id(&self, id: &str) -> Result<(String, String), LlmError> {
        if let Some((p, m)) = id.split_once(self.separator) {
            if p.is_empty() || m.is_empty() {
                return Err(LlmError::InvalidParameter(format!(
                    "Invalid model id for registry: {} (must be 'provider{}model')",
                    id, self.separator
                )));
            }
            Ok((p.to_string(), m.to_string()))
        } else {
            Err(LlmError::InvalidParameter(format!(
                "Invalid model id for registry: {} (must be 'provider{}model')",
                id, self.separator
            )))
        }
    }

    /// Get a provider factory by ID
    fn get_provider(&self, provider_id: &str) -> Result<&Arc<dyn ProviderFactory>, LlmError> {
        self.providers.get(provider_id).ok_or_else(|| {
            LlmError::ConfigurationError(format!(
                "No such provider: {}. Available providers: {:?}",
                provider_id,
                self.providers.keys().collect::<Vec<_>>()
            ))
        })
    }

    fn resolve_provider_build_overrides(&self, provider_id: &str) -> ProviderBuildOverrides {
        ProviderBuildOverrides {
            http_client: self.http_client.clone(),
            http_transport: self.http_transport.clone(),
            http_config: self.http_config.clone(),
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            reasoning_enabled: self.reasoning_enabled,
            reasoning_budget: self.reasoning_budget,
        }
        .merged_with(self.provider_build_overrides.get(provider_id))
    }

    /// Resolve language model - returns a handle that delegates to the factory
    ///
    /// Uses LRU cache with optional TTL to avoid rebuilding clients repeatedly.
    /// Cache key is the full "provider:model" identifier.
    ///
    /// Applies middleware provider_id override if configured.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use siumai_registry::registry::entry::create_provider_registry;
    /// # use std::collections::HashMap;
    /// let registry = create_provider_registry(HashMap::new(), None);
    /// let handle = registry.language_model("openai:gpt-4")?;
    /// # Ok::<(), siumai_registry::error::LlmError>(())
    /// ```
    pub fn language_model(&self, id: &str) -> Result<LanguageModelHandle, LlmError> {
        let (mut provider_id, model_id) = self.split_id(id)?;

        // Normalize common provider id aliases (e.g. "google" -> "gemini") when possible.
        // Only apply normalization when the canonical id is registered and the raw id is not,
        // to avoid surprising overrides for custom registries.
        let normalized = crate::provider::resolver::normalize_provider_id(&provider_id);
        if normalized != provider_id
            && !self.providers.contains_key(&provider_id)
            && self.providers.contains_key(&normalized)
        {
            provider_id = normalized;
        }

        // Combine global middlewares with auto middlewares
        let mut middlewares = self.middlewares.clone();
        if self.auto_middleware {
            let auto_middlewares =
                crate::execution::middleware::build_auto_middlewares_vec(&provider_id, &model_id);
            middlewares.extend(auto_middlewares);
        }

        // Apply middleware provider_id override (aligned with Vercel AI SDK)
        if !middlewares.is_empty() {
            provider_id = crate::execution::middleware::language_model::apply_provider_id_override(
                &middlewares,
                &provider_id,
            );
        }

        let factory = self.get_provider(&provider_id)?;
        let capabilities = factory.capabilities();
        if !capabilities.supports("chat") {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not expose registry language_model/chat handles; use family-specific entries instead",
                provider_id
            )));
        }
        let build_overrides = self.resolve_provider_build_overrides(&provider_id);

        Ok(LanguageModelHandle {
            factory: factory.clone(),
            provider_id,
            model_id,
            middlewares,
            cache: self.language_model_cache.clone(),
            client_ttl: self.client_ttl,
            http_interceptors: self.http_interceptors.clone(),
            http_client: build_overrides.http_client,
            http_transport: build_overrides.http_transport,
            http_config: build_overrides.http_config,
            api_key: build_overrides.api_key,
            base_url: build_overrides.base_url,
            reasoning_enabled: build_overrides.reasoning_enabled,
            reasoning_budget: build_overrides.reasoning_budget,
            retry_options: self.retry_options.clone(),
            capabilities,
        })
    }

    /// Resolve completion model - returns a handle that delegates to the factory.
    pub fn completion_model(&self, id: &str) -> Result<CompletionModelHandle, LlmError> {
        let (mut provider_id, model_id) = self.split_id(id)?;

        let normalized = crate::provider::resolver::normalize_provider_id(&provider_id);
        if normalized != provider_id
            && !self.providers.contains_key(&provider_id)
            && self.providers.contains_key(&normalized)
        {
            provider_id = normalized;
        }

        let mut middlewares = self.middlewares.clone();
        if self.auto_middleware {
            let auto_middlewares =
                crate::execution::middleware::build_auto_middlewares_vec(&provider_id, &model_id);
            middlewares.extend(auto_middlewares);
        }

        if !middlewares.is_empty() {
            provider_id = crate::execution::middleware::language_model::apply_provider_id_override(
                &middlewares,
                &provider_id,
            );
        }

        let factory = self.get_provider(&provider_id)?;
        let capabilities = factory.capabilities();
        if !capabilities.supports("completion") {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not expose completion on the completion_model handle",
                provider_id
            )));
        }
        let build_overrides = self.resolve_provider_build_overrides(&provider_id);

        Ok(CompletionModelHandle {
            factory: factory.clone(),
            provider_id,
            model_id,
            middlewares,
            http_interceptors: self.http_interceptors.clone(),
            http_client: build_overrides.http_client,
            http_transport: build_overrides.http_transport,
            http_config: build_overrides.http_config,
            api_key: build_overrides.api_key,
            base_url: build_overrides.base_url,
            reasoning_enabled: build_overrides.reasoning_enabled,
            reasoning_budget: build_overrides.reasoning_budget,
            cache: self.completion_model_cache.clone(),
            client_ttl: self.client_ttl,
            retry_options: self.retry_options.clone(),
            capabilities,
        })
    }

    /// Resolve embedding model - returns a handle that delegates to the factory
    pub fn embedding_model(&self, id: &str) -> Result<EmbeddingModelHandle, LlmError> {
        let (provider_id, model_id) = self.split_id(id)?;
        let factory = self.get_provider(&provider_id)?;
        let capabilities = factory.capabilities();
        if !capabilities.supports("embedding") {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not expose embedding on the embedding_model handle",
                provider_id
            )));
        }
        let build_overrides = self.resolve_provider_build_overrides(&provider_id);

        Ok(EmbeddingModelHandle {
            factory: factory.clone(),
            provider_id,
            model_id,
            http_interceptors: self.http_interceptors.clone(),
            http_client: build_overrides.http_client,
            http_transport: build_overrides.http_transport,
            http_config: build_overrides.http_config,
            api_key: build_overrides.api_key,
            base_url: build_overrides.base_url,
            retry_options: self.retry_options.clone(),
        })
    }

    /// Resolve image model - returns a handle that delegates to the factory
    pub fn image_model(&self, id: &str) -> Result<ImageModelHandle, LlmError> {
        let (provider_id, model_id) = self.split_id(id)?;
        let factory = self.get_provider(&provider_id)?;
        let capabilities = factory.capabilities();
        if !capabilities.supports("image_generation") {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not expose image_generation on the image_model handle",
                provider_id
            )));
        }
        if !image_model_handle_supports_model(&provider_id, &model_id) {
            return Err(LlmError::UnsupportedOperation(format!(
                "Model '{}' on provider '{}' does not expose image_generation on the image_model handle",
                model_id, provider_id
            )));
        }
        let build_overrides = self.resolve_provider_build_overrides(&provider_id);

        Ok(ImageModelHandle {
            factory: factory.clone(),
            provider_id,
            model_id,
            http_interceptors: self.http_interceptors.clone(),
            http_client: build_overrides.http_client,
            http_transport: build_overrides.http_transport,
            http_config: build_overrides.http_config,
            api_key: build_overrides.api_key,
            base_url: build_overrides.base_url,
            retry_options: self.retry_options.clone(),
        })
    }

    /// Resolve video model - returns a handle that delegates to the factory.
    pub fn video_model(&self, id: &str) -> Result<VideoModelHandle, LlmError> {
        let (provider_id, model_id) = self.split_id(id)?;
        let factory = self.get_provider(&provider_id)?;
        let capabilities = factory.capabilities();
        if !capabilities.supports("video") {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not expose video on the video_model handle",
                provider_id
            )));
        }
        let build_overrides = self.resolve_provider_build_overrides(&provider_id);

        Ok(VideoModelHandle {
            factory: factory.clone(),
            provider_id,
            model_id,
            http_interceptors: self.http_interceptors.clone(),
            http_client: build_overrides.http_client,
            http_transport: build_overrides.http_transport,
            http_config: build_overrides.http_config,
            api_key: build_overrides.api_key,
            base_url: build_overrides.base_url,
            cache: self.video_model_cache.clone(),
            client_ttl: self.client_ttl,
            retry_options: self.retry_options.clone(),
        })
    }

    /// Resolve speech model - returns a handle that delegates to the factory
    pub fn speech_model(&self, id: &str) -> Result<SpeechModelHandle, LlmError> {
        let (provider_id, model_id) = self.split_id(id)?;
        let factory = self.get_provider(&provider_id)?;
        let capabilities = factory.capabilities();
        if !capabilities.supports("speech") {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not expose speech on the speech_model handle",
                provider_id
            )));
        }
        let build_overrides = self.resolve_provider_build_overrides(&provider_id);

        Ok(SpeechModelHandle {
            factory: factory.clone(),
            provider_id,
            model_id,
            http_interceptors: self.http_interceptors.clone(),
            http_client: build_overrides.http_client,
            http_transport: build_overrides.http_transport,
            http_config: build_overrides.http_config,
            api_key: build_overrides.api_key,
            base_url: build_overrides.base_url,
            cache: self.speech_model_cache.clone(),
            client_ttl: self.client_ttl,
            retry_options: self.retry_options.clone(),
        })
    }

    /// Resolve transcription model - returns a handle that delegates to the factory
    pub fn transcription_model(&self, id: &str) -> Result<TranscriptionModelHandle, LlmError> {
        let (provider_id, model_id) = self.split_id(id)?;
        let factory = self.get_provider(&provider_id)?;
        let capabilities = factory.capabilities();
        if !capabilities.supports("transcription") {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not expose transcription on the transcription_model handle",
                provider_id
            )));
        }
        let build_overrides = self.resolve_provider_build_overrides(&provider_id);

        Ok(TranscriptionModelHandle {
            factory: factory.clone(),
            provider_id,
            model_id,
            http_interceptors: self.http_interceptors.clone(),
            http_client: build_overrides.http_client,
            http_transport: build_overrides.http_transport,
            http_config: build_overrides.http_config,
            api_key: build_overrides.api_key,
            base_url: build_overrides.base_url,
            cache: self.transcription_model_cache.clone(),
            client_ttl: self.client_ttl,
            retry_options: self.retry_options.clone(),
        })
    }

    /// Resolve reranking model - returns a handle that delegates to the factory
    pub fn reranking_model(&self, id: &str) -> Result<RerankingModelHandle, LlmError> {
        let (provider_id, model_id) = self.split_id(id)?;
        let factory = self.get_provider(&provider_id)?;
        let capabilities = factory.capabilities();
        if !capabilities.supports("rerank") {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not expose rerank on the reranking_model handle",
                provider_id
            )));
        }
        let build_overrides = self.resolve_provider_build_overrides(&provider_id);

        Ok(RerankingModelHandle {
            factory: factory.clone(),
            provider_id,
            model_id,
            http_interceptors: self.http_interceptors.clone(),
            http_client: build_overrides.http_client,
            http_transport: build_overrides.http_transport,
            http_config: build_overrides.http_config,
            api_key: build_overrides.api_key,
            base_url: build_overrides.base_url,
            retry_options: self.retry_options.clone(),
        })
    }
}

/// Create a provider registry handle - aligned with Vercel AI SDK
///
/// # Arguments
/// * `providers` - Map of provider_id -> ProviderFactory instances
/// * `opts` - Optional registry configuration (separator, middlewares)
///
/// # Example
/// ```rust,no_run
/// use std::collections::HashMap;
/// use std::sync::Arc;
/// use siumai_registry::registry::entry::{create_provider_registry, ProviderFactory};
///
/// let mut providers = HashMap::new();
/// // providers.insert("openai".to_string(), Arc::new(OpenAIProviderFactory) as Arc<dyn ProviderFactory>);
///
/// let registry = create_provider_registry(providers, None);
/// ```
pub fn create_provider_registry(
    providers: HashMap<String, Arc<dyn ProviderFactory>>,
    opts: Option<RegistryOptions>,
) -> ProviderRegistryHandle {
    let o = opts.unwrap_or_default();
    let (
        separator,
        middlewares,
        http_interceptors,
        http_client,
        http_transport,
        http_config,
        api_key,
        base_url,
        reasoning_enabled,
        reasoning_budget,
        provider_build_overrides,
        retry_options,
        max_cache_entries,
        client_ttl,
        auto_middleware,
    ) = (
        o.separator,
        o.language_model_middleware,
        o.http_interceptors,
        o.http_client,
        o.http_transport,
        o.http_config,
        o.api_key,
        o.base_url,
        o.reasoning_enabled,
        o.reasoning_budget,
        o.provider_build_overrides,
        o.retry_options,
        o.max_cache_entries,
        o.client_ttl,
        o.auto_middleware,
    );

    // Create LRU cache with specified capacity (default: 100 entries)
    let cache_capacity = max_cache_entries.unwrap_or(100);
    let cache =
        LruCache::new(NonZeroUsize::new(cache_capacity).expect("Cache capacity must be > 0"));
    let completion_cache =
        LruCache::new(NonZeroUsize::new(cache_capacity).expect("Cache capacity must be > 0"));
    let speech_cache =
        LruCache::new(NonZeroUsize::new(cache_capacity).expect("Cache capacity must be > 0"));
    let transcription_cache =
        LruCache::new(NonZeroUsize::new(cache_capacity).expect("Cache capacity must be > 0"));
    let video_cache =
        LruCache::new(NonZeroUsize::new(cache_capacity).expect("Cache capacity must be > 0"));

    ProviderRegistryHandle {
        providers,
        separator,
        middlewares,
        http_interceptors,
        http_client,
        http_transport,
        language_model_cache: Arc::new(TokioMutex::new(cache)),
        completion_model_cache: Arc::new(TokioMutex::new(completion_cache)),
        speech_model_cache: Arc::new(TokioMutex::new(speech_cache)),
        transcription_model_cache: Arc::new(TokioMutex::new(transcription_cache)),
        video_model_cache: Arc::new(TokioMutex::new(video_cache)),
        client_ttl,
        auto_middleware,
        http_config,
        api_key,
        base_url,
        reasoning_enabled,
        reasoning_budget,
        provider_build_overrides,
        retry_options,
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod completion_tests;
#[cfg(test)]
mod embedding_tests;
#[cfg(test)]
mod file_tests;
#[cfg(test)]
mod image_tests;
#[cfg(test)]
mod music_tests;
#[cfg(test)]
mod rerank_tests;
#[cfg(test)]
mod skills_tests;
#[cfg(test)]
mod speech_tests;
#[cfg(test)]
mod transcription_tests;
#[cfg(test)]
mod video_tests;
