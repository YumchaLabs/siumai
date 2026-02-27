//! Provider Registry - Vercel AI SDK Aligned Architecture
//!
//! This module provides a provider registry system that aligns with Vercel AI SDK's design:
//! - ProviderFactory trait for creating provider clients
//! - Registry stores factory instances (not hardcoded logic)
//! - Handles delegate to factories for client creation
//! - Easy to extend with new providers

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::retry_api::RetryOptions;
use crate::streaming::{ChatStream, ChatStreamHandle};
use crate::traits::{
    AudioCapability, ChatCapability, EmbeddingCapability, ImageExtras, ImageGenerationCapability,
    ProviderCapabilities, RerankCapability,
};
use crate::types::{
    AudioFeature, ChatMessage, ChatRequest, ChatResponse, EmbeddingResponse, ImageEditRequest,
    ImageGenerationRequest, ImageGenerationResponse, ImageVariationRequest, RerankRequest,
    RerankResponse, SttRequest, SttResponse, Tool, TtsRequest, TtsResponse,
};

use lru::LruCache;
use std::num::NonZeroUsize;
use tokio::sync::Mutex as TokioMutex;

/// Provider factory trait - similar to Vercel AI SDK's ProviderV3
///
/// Each provider implements this trait to create clients for different model types.
/// This allows the registry to be provider-agnostic and easily extensible.
///
/// Note: Middlewares are applied by the Handle after client creation, not by the factory.
/// This keeps the factory simple and aligns with Vercel AI SDK's design where
/// middleware wrapping happens at the registry level.
#[async_trait::async_trait]
pub trait ProviderFactory: Send + Sync {
    /// Create a language model client for the given model ID
    ///
    /// The returned client should NOT have middlewares applied - the Handle will apply them.
    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError>;

    /// Create a language model client with build context (interceptors, retry, etc.)
    /// Default implementation falls back to `language_model` for backward compatibility.
    async fn language_model_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.language_model(model_id).await
    }

    /// Create an embedding model client for the given model ID
    async fn embedding_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Default: delegate to language_model (many providers use same client)
        self.language_model(model_id).await
    }

    /// Create an embedding model client with build context (default delegates to language_model_with_ctx)
    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Default: keep behavior of embedding_model() to allow custom embedding clients
        self.embedding_model(model_id).await
    }

    /// Create an image model client for the given model ID
    async fn image_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.language_model(model_id).await
    }

    /// Create an image model client with build context (default delegates to language_model_with_ctx)
    async fn image_model_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.image_model(model_id).await
    }

    /// Create a speech model client for the given model ID
    async fn speech_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.language_model(model_id).await
    }

    /// Create a speech model client with build context (default delegates to language_model_with_ctx)
    async fn speech_model_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.speech_model(model_id).await
    }

    /// Create a transcription model client for the given model ID
    async fn transcription_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.speech_model(model_id).await
    }

    /// Create a transcription model client with build context (default delegates to speech_model_with_ctx)
    async fn transcription_model_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.transcription_model(model_id).await
    }

    /// Create a reranking model client for the given model ID
    async fn reranking_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.language_model(model_id).await
    }

    /// Create a reranking model client with build context (default delegates to reranking_model)
    async fn reranking_model_with_ctx(
        &self,
        model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.reranking_model(model_id).await
    }

    /// Get the provider name
    fn provider_id(&self) -> std::borrow::Cow<'static, str>;

    /// Declared provider-level capabilities (metadata only).
    ///
    /// This is used by registry handles to expose capability hints without
    /// requiring runtime lookups into the global provider registry.
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
    }
}

/// Cache entry with TTL support
struct CacheEntry {
    client: Arc<dyn LlmClient>,
    created_at: Instant,
}

impl CacheEntry {
    fn new(client: Arc<dyn LlmClient>) -> Self {
        Self {
            client,
            created_at: Instant::now(),
        }
    }

    fn is_expired(&self, ttl: Option<Duration>) -> bool {
        if let Some(ttl) = ttl {
            self.created_at.elapsed() > ttl
        } else {
            false
        }
    }
}

/// Options for creating a provider registry handle.
pub struct RegistryOptions {
    pub separator: char,
    pub language_model_middleware: Vec<Arc<dyn LanguageModelMiddleware>>,
    /// HTTP interceptors applied to all clients created via the registry
    pub http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional HTTP configuration applied to all clients created via the registry.
    /// When set, this configuration is passed through BuildContext to provider
    /// factories so they can build HTTP clients consistently with SiumaiBuilder.
    pub http_config: Option<crate::types::HttpConfig>,
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
    /// Registry-level HTTP configuration propagated via BuildContext.
    http_config: Option<crate::types::HttpConfig>,
    /// LRU cache for language model clients (key: "provider:model")
    /// Uses async Mutex for concurrent access and per-key build de-duplication
    language_model_cache: Arc<TokioMutex<LruCache<String, CacheEntry>>>,
    /// TTL for cached clients
    client_ttl: Option<Duration>,
    /// Whether to automatically add model-specific middlewares
    auto_middleware: bool,
    /// Registry-level retry options applied during client build (optional)
    retry_options: Option<RetryOptions>,
}

/// Build-time context for ProviderFactory client construction.
///
/// This struct carries all cross-cutting configuration needed to build
/// concrete provider clients in a unified way (HTTP config, auth, tracing,
/// interceptors, retry options, etc.).
#[derive(Default, Clone)]
pub struct BuildContext {
    /// HTTP interceptors applied at the registry / builder level.
    pub http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Unified retry options (optional).
    pub retry_options: Option<RetryOptions>,
    /// Optional model-level middlewares (applied before provider mapping).
    pub model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    /// Optional pre-built HTTP client. When present, factories should prefer
    /// this client over constructing a new one from `http_config`.
    pub http_client: Option<reqwest::Client>,
    /// Optional custom HTTP transport (Vercel-style "custom fetch" parity).
    pub http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    /// Optional HTTP configuration (timeouts, headers, proxy, user-agent, etc.).
    /// When no custom client is supplied, factories may use this to build one.
    pub http_config: Option<crate::types::HttpConfig>,
    /// Optional API key override. When `None`, factories may fall back to
    /// environment variables or other defaults.
    pub api_key: Option<String>,
    /// Optional base URL override for the provider.
    pub base_url: Option<String>,
    /// Optional organization identifier (e.g., OpenAI `organization`).
    pub organization: Option<String>,
    /// Optional project identifier (e.g., OpenAI `project`).
    pub project: Option<String>,
    /// Optional tracing configuration for providers that support it.
    pub tracing_config: Option<crate::observability::tracing::TracingConfig>,
    /// Optional Google token provider (e.g., for Vertex AI auth).
    #[cfg(any(feature = "google", feature = "google-vertex"))]
    pub gemini_token_provider: Option<std::sync::Arc<dyn crate::auth::TokenProvider>>,
    /// Optional common parameters (model id, temperature, max_tokens, etc.).
    /// When `None`, factories may construct minimal defaults based on `model_id`.
    pub common_params: Option<crate::types::CommonParams>,
    /// Optional canonical provider id override (for adapter-style providers).
    pub provider_id: Option<String>,
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

        Ok(LanguageModelHandle {
            factory: factory.clone(),
            provider_id,
            model_id,
            middlewares,
            cache: self.language_model_cache.clone(),
            client_ttl: self.client_ttl,
            http_interceptors: self.http_interceptors.clone(),
            http_config: self.http_config.clone(),
            retry_options: self.retry_options.clone(),
            capabilities: factory.capabilities(),
        })
    }

    /// Resolve embedding model - returns a handle that delegates to the factory
    pub fn embedding_model(&self, id: &str) -> Result<EmbeddingModelHandle, LlmError> {
        let (provider_id, model_id) = self.split_id(id)?;
        let factory = self.get_provider(&provider_id)?;

        Ok(EmbeddingModelHandle {
            factory: factory.clone(),
            model_id,
            http_interceptors: self.http_interceptors.clone(),
            http_config: self.http_config.clone(),
            retry_options: self.retry_options.clone(),
        })
    }

    /// Resolve image model - returns a handle that delegates to the factory
    pub fn image_model(&self, id: &str) -> Result<ImageModelHandle, LlmError> {
        let (provider_id, model_id) = self.split_id(id)?;
        let factory = self.get_provider(&provider_id)?;

        Ok(ImageModelHandle {
            factory: factory.clone(),
            model_id,
            http_interceptors: self.http_interceptors.clone(),
            http_config: self.http_config.clone(),
            retry_options: self.retry_options.clone(),
        })
    }

    /// Resolve speech model - returns a handle that delegates to the factory
    pub fn speech_model(&self, id: &str) -> Result<SpeechModelHandle, LlmError> {
        let (provider_id, model_id) = self.split_id(id)?;
        let factory = self.get_provider(&provider_id)?;

        Ok(SpeechModelHandle {
            factory: factory.clone(),
            model_id,
            http_interceptors: self.http_interceptors.clone(),
            http_config: self.http_config.clone(),
            retry_options: self.retry_options.clone(),
        })
    }

    /// Resolve transcription model - returns a handle that delegates to the factory
    pub fn transcription_model(&self, id: &str) -> Result<TranscriptionModelHandle, LlmError> {
        let (provider_id, model_id) = self.split_id(id)?;
        let factory = self.get_provider(&provider_id)?;

        Ok(TranscriptionModelHandle {
            factory: factory.clone(),
            model_id,
            http_interceptors: self.http_interceptors.clone(),
            http_config: self.http_config.clone(),
            retry_options: self.retry_options.clone(),
        })
    }

    /// Resolve reranking model - returns a handle that delegates to the factory
    pub fn reranking_model(&self, id: &str) -> Result<RerankingModelHandle, LlmError> {
        let (provider_id, model_id) = self.split_id(id)?;
        let factory = self.get_provider(&provider_id)?;

        Ok(RerankingModelHandle {
            factory: factory.clone(),
            model_id,
            http_interceptors: self.http_interceptors.clone(),
            http_config: self.http_config.clone(),
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
    let (
        separator,
        middlewares,
        http_interceptors,
        http_config,
        retry_options,
        max_cache_entries,
        client_ttl,
        auto_middleware,
    ) = if let Some(o) = opts {
        (
            o.separator,
            o.language_model_middleware,
            o.http_interceptors,
            o.http_config,
            o.retry_options,
            o.max_cache_entries,
            o.client_ttl,
            o.auto_middleware,
        )
    } else {
        // Defaults: no middlewares, no interceptors, auto middleware enabled
        (':', Vec::new(), Vec::new(), None, None, None, None, true)
    };

    // Create LRU cache with specified capacity (default: 100 entries)
    let cache_capacity = max_cache_entries.unwrap_or(100);
    let cache =
        LruCache::new(NonZeroUsize::new(cache_capacity).expect("Cache capacity must be > 0"));

    ProviderRegistryHandle {
        providers,
        separator,
        middlewares,
        http_interceptors,
        language_model_cache: Arc::new(TokioMutex::new(cache)),
        client_ttl,
        auto_middleware,
        http_config,
        retry_options,
    }
}

/// Language model handle - delegates to provider factory
///
/// This handle stores a reference to the provider factory and delegates
/// client creation to it. This aligns with Vercel AI SDK's design where
/// the registry returns model instances that know how to create themselves.
///
/// Features LRU cache with TTL to avoid rebuilding clients on every call.
#[derive(Clone)]
pub struct LanguageModelHandle {
    /// Provider factory for creating clients
    factory: Arc<dyn ProviderFactory>,
    /// Provider ID (e.g., "openai")
    pub provider_id: String,
    /// Model ID to pass to the factory (e.g., "gpt-4")
    pub model_id: String,
    /// Middlewares to apply to the client
    pub middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    /// Registry-level HTTP interceptors to attempt injecting into clients
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Registry-level HTTP configuration copied into the handle
    http_config: Option<crate::types::HttpConfig>,
    /// Shared LRU cache for clients
    cache: Arc<TokioMutex<LruCache<String, CacheEntry>>>,
    /// TTL for cached clients
    client_ttl: Option<Duration>,
    /// Registry-level retry options copied into the handle
    retry_options: Option<RetryOptions>,
    /// Provider-level capability hints captured at construction time
    capabilities: ProviderCapabilities,
}

impl LanguageModelHandle {
    /// Get or create a cached client
    ///
    /// This method implements LRU cache with TTL:
    /// 1. Check cache for existing client
    /// 2. If found and not expired, return it
    /// 3. If not found or expired, build new client and cache it
    /// 4. LRU eviction happens automatically when cache is full
    ///
    /// Note: Cache key includes the potentially overridden model_id to ensure
    /// correct caching when middleware overrides the model.
    async fn get_or_create_client(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Use provider_id + actual model_id as cache key
        // This ensures middleware overrides are cached correctly
        let cache_key = format!("{}:{}", self.provider_id, model_id);

        let mut cache = self.cache.lock().await;

        // Check if we have a cached client
        if let Some(entry) = cache.get(&cache_key) {
            if !entry.is_expired(self.client_ttl) {
                // Cache hit - return cached client
                return Ok(entry.client.clone());
            }
            // Expired - remove it
            cache.pop(&cache_key);
        }

        // Cache miss or expired - build new client
        drop(cache); // Release lock before async factory call
        // Construct build context from registry-level settings
        let ctx = BuildContext {
            http_interceptors: self.http_interceptors.clone(),
            retry_options: self.retry_options.clone(),
            http_config: self.http_config.clone(),
            ..Default::default()
        };
        // Note: model_middlewares are applied at the handle level; factories
        // should not re-apply them.
        let client_built = self.factory.language_model_with_ctx(model_id, &ctx).await?;

        // Client built via factory with BuildContext already contains interceptors/retry.
        let client = client_built;

        // Cache the new client
        let mut cache = self.cache.lock().await;
        cache.put(cache_key, CacheEntry::new(client.clone()));

        Ok(client)
    }
}

/// Implement unified client metadata trait for LanguageModelHandle.
///
/// This allows using a registry language model handle anywhere an `LlmClient`
/// is expected (e.g., inside the unified `Siumai` wrapper), while keeping
/// execution logic delegated to the underlying provider clients.
impl LlmClient for LanguageModelHandle {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Owned(self.provider_id.clone())
    }

    fn supported_models(&self) -> Vec<String> {
        // We only know the configured model id for this handle; return that.
        vec![self.model_id.clone()]
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.capabilities.clone()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }
}

/// Implementation of ChatCapability for LanguageModelHandle
///
/// This allows the handle to be used directly as a chat client, aligning with
/// Vercel AI SDK's design where registry.languageModel() returns a callable model.
#[async_trait::async_trait]
impl ChatCapability for LanguageModelHandle {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        // Apply middleware overrides (aligned with Vercel AI SDK)
        let model_id = if !self.middlewares.is_empty() {
            crate::execution::middleware::language_model::apply_model_id_override(
                &self.middlewares,
                &self.model_id,
            )
        } else {
            self.model_id.clone()
        };

        // Get or create cached client with potentially overridden model_id
        let client = self.get_or_create_client(&model_id).await?;

        // Apply middlewares if any
        if !self.middlewares.is_empty() {
            let mut req = ChatRequest::new(messages);
            if let Some(t) = tools {
                req = req.with_tools(t);
            }
            req = crate::execution::middleware::language_model::apply_transform_chain(
                &self.middlewares,
                req,
            );
            client.chat_with_tools(req.messages, req.tools).await
        } else {
            client.chat_with_tools(messages, tools).await
        }
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        // Apply middleware overrides (aligned with Vercel AI SDK)
        let model_id = if !self.middlewares.is_empty() {
            crate::execution::middleware::language_model::apply_model_id_override(
                &self.middlewares,
                &self.model_id,
            )
        } else {
            self.model_id.clone()
        };

        // Get or create cached client with potentially overridden model_id
        let client = self.get_or_create_client(&model_id).await?;

        // Apply middlewares if any
        if !self.middlewares.is_empty() {
            let mut req = ChatRequest::new(messages);
            if let Some(t) = tools {
                req = req.with_tools(t);
            }
            req = crate::execution::middleware::language_model::apply_transform_chain(
                &self.middlewares,
                req,
            );
            client.chat_stream(req.messages, req.tools).await
        } else {
            client.chat_stream(messages, tools).await
        }
    }

    async fn chat_stream_with_cancel(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStreamHandle, LlmError> {
        let this = self.clone();
        Ok(
            crate::utils::cancel::make_cancellable_stream_handle_from_handle_future(async move {
                // Align with chat_stream(...) middleware behavior, but preserve provider-specific cancellation.
                let model_id = if !this.middlewares.is_empty() {
                    crate::execution::middleware::language_model::apply_model_id_override(
                        &this.middlewares,
                        &this.model_id,
                    )
                } else {
                    this.model_id.clone()
                };

                let client = this.get_or_create_client(&model_id).await?;

                let mut req = ChatRequest::new(messages).with_streaming(true);
                if let Some(t) = tools {
                    req = req.with_tools(t);
                }

                if !this.middlewares.is_empty() {
                    req = crate::execution::middleware::language_model::apply_transform_chain(
                        &this.middlewares,
                        req,
                    );
                }

                client.chat_stream_request_with_cancel(req).await
            }),
        )
    }

    async fn chat_stream_request_with_cancel(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStreamHandle, LlmError> {
        let this = self.clone();
        Ok(
            crate::utils::cancel::make_cancellable_stream_handle_from_handle_future(async move {
                let model_id = if !this.middlewares.is_empty() {
                    crate::execution::middleware::language_model::apply_model_id_override(
                        &this.middlewares,
                        &this.model_id,
                    )
                } else {
                    this.model_id.clone()
                };

                let client = this.get_or_create_client(&model_id).await?;

                let mut req = request.with_streaming(true);
                if !this.middlewares.is_empty() {
                    req = crate::execution::middleware::language_model::apply_transform_chain(
                        &this.middlewares,
                        req,
                    );
                }

                client.chat_stream_request_with_cancel(req).await
            }),
        )
    }
}

/// Embedding model handle - delegates to factory for client creation
#[derive(Clone)]
pub struct EmbeddingModelHandle {
    factory: Arc<dyn ProviderFactory>,
    pub model_id: String,
    /// Registry-level HTTP interceptors to attempt injecting into clients
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Registry-level retry options copied into the handle
    retry_options: Option<RetryOptions>,
    /// Registry-level HTTP configuration copied into the handle
    http_config: Option<crate::types::HttpConfig>,
}

/// Implementation of EmbeddingCapability for EmbeddingModelHandle
///
/// This allows the handle to be used directly as an embedding client, aligning with
/// Vercel AI SDK's design where registry.textEmbeddingModel() returns a callable model.
#[async_trait::async_trait]
impl EmbeddingCapability for EmbeddingModelHandle {
    async fn embed(&self, input: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        // Build client from factory with context
        let ctx = BuildContext {
            http_interceptors: self.http_interceptors.clone(),
            retry_options: self.retry_options.clone(),
            http_config: self.http_config.clone(),
            ..Default::default()
        };
        let client_raw = self
            .factory
            .embedding_model_with_ctx(&self.model_id, &ctx)
            .await?;
        let client = client_raw;

        // Get embedding capability
        let embedding_client = client.as_embedding_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation("Provider does not support embeddings".to_string())
        })?;

        // Call embed
        embedding_client.embed(input).await
    }

    fn embedding_dimension(&self) -> usize {
        // Default dimension - providers should override this
        // We can't get this without building a client, so we return a default
        // In practice, users should check the provider's documentation
        1536 // OpenAI's default
    }
}

impl EmbeddingModelHandle {}

/// Image model handle - delegates to factory for client creation
#[derive(Clone)]
pub struct ImageModelHandle {
    factory: Arc<dyn ProviderFactory>,
    pub model_id: String,
    /// Registry-level HTTP interceptors to attempt injecting into clients
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Registry-level retry options copied into the handle
    retry_options: Option<RetryOptions>,
    /// Registry-level HTTP configuration copied into the handle
    http_config: Option<crate::types::HttpConfig>,
}

/// Implementation of ImageGenerationCapability for ImageModelHandle
///
/// This allows the handle to be used directly as an image generation client, aligning with
/// Vercel AI SDK's design where registry.imageModel() returns a callable model.
#[async_trait::async_trait]
impl ImageGenerationCapability for ImageModelHandle {
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        // Build client from factory with context
        let ctx = BuildContext {
            http_interceptors: self.http_interceptors.clone(),
            retry_options: self.retry_options.clone(),
            http_config: self.http_config.clone(),
            ..Default::default()
        };
        let client_raw = self
            .factory
            .image_model_with_ctx(&self.model_id, &ctx)
            .await?;
        let client = client_raw;

        // Get image generation capability
        let image_client = client.as_image_generation_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation("Provider does not support image generation".to_string())
        })?;

        // Call generate_images
        image_client.generate_images(request).await
    }
}

#[async_trait::async_trait]
impl ImageExtras for ImageModelHandle {
    async fn edit_image(
        &self,
        request: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let ctx = BuildContext {
            http_interceptors: self.http_interceptors.clone(),
            retry_options: self.retry_options.clone(),
            http_config: self.http_config.clone(),
            ..Default::default()
        };
        let client_raw = self
            .factory
            .image_model_with_ctx(&self.model_id, &ctx)
            .await?;
        let client = client_raw;

        let image_client = client.as_image_extras().ok_or_else(|| {
            LlmError::UnsupportedOperation("Provider does not support image extras".to_string())
        })?;

        image_client.edit_image(request).await
    }

    async fn create_variation(
        &self,
        request: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let ctx = BuildContext {
            http_interceptors: self.http_interceptors.clone(),
            retry_options: self.retry_options.clone(),
            http_config: self.http_config.clone(),
            ..Default::default()
        };
        let client_raw = self
            .factory
            .image_model_with_ctx(&self.model_id, &ctx)
            .await?;
        let client = client_raw;

        let image_client = client.as_image_extras().ok_or_else(|| {
            LlmError::UnsupportedOperation("Provider does not support image extras".to_string())
        })?;

        image_client.create_variation(request).await
    }

    fn get_supported_sizes(&self) -> Vec<String> {
        // Best-effort defaults (provider-specific metadata is optional).
        vec![
            "1024x1024".to_string(),
            "512x512".to_string(),
            "256x256".to_string(),
        ]
    }

    fn get_supported_formats(&self) -> Vec<String> {
        vec!["url".to_string(), "b64_json".to_string()]
    }

    fn supports_image_editing(&self) -> bool {
        false
    }

    fn supports_image_variations(&self) -> bool {
        false
    }
}

impl ImageModelHandle {}

/// Speech model handle (TTS) - delegates to factory for client creation
#[derive(Clone)]
pub struct SpeechModelHandle {
    factory: Arc<dyn ProviderFactory>,
    pub model_id: String,
    /// Registry-level HTTP interceptors to attempt injecting into clients
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Registry-level retry options copied into the handle
    retry_options: Option<RetryOptions>,
    /// Registry-level HTTP configuration copied into the handle
    http_config: Option<crate::types::HttpConfig>,
}

/// Implementation of AudioCapability for SpeechModelHandle
///
/// This allows the handle to be used directly as a TTS client, aligning with
/// Vercel AI SDK's design where registry.speechModel() returns a callable model.
#[async_trait::async_trait]
impl AudioCapability for SpeechModelHandle {
    fn supported_features(&self) -> &[AudioFeature] {
        // Default features - providers should override this
        &[AudioFeature::TextToSpeech]
    }

    async fn text_to_speech(&self, request: TtsRequest) -> Result<TtsResponse, LlmError> {
        // Build client from factory with context
        let ctx = BuildContext {
            http_interceptors: self.http_interceptors.clone(),
            retry_options: self.retry_options.clone(),
            http_config: self.http_config.clone(),
            ..Default::default()
        };
        let client_raw = self
            .factory
            .speech_model_with_ctx(&self.model_id, &ctx)
            .await?;
        let client = client_raw;

        // Get speech capability
        let speech_client = client.as_speech_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation("Provider does not support text-to-speech".to_string())
        })?;

        speech_client.tts(request).await
    }
}

impl SpeechModelHandle {
    /// Text to speech (deprecated - use trait method directly)
    #[deprecated(
        since = "0.10.3",
        note = "Use the AudioCapability trait method directly"
    )]
    pub async fn text_to_speech(
        &self,
        req: crate::types::TtsRequest,
    ) -> Result<crate::types::TtsResponse, LlmError> {
        // Delegate to trait implementation
        AudioCapability::text_to_speech(self, req).await
    }
}

/// Transcription model handle (STT) - delegates to factory for client creation
#[derive(Clone)]
pub struct TranscriptionModelHandle {
    factory: Arc<dyn ProviderFactory>,
    pub model_id: String,
    /// Registry-level HTTP interceptors to attempt injecting into clients
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Registry-level retry options copied into the handle
    retry_options: Option<RetryOptions>,
    /// Registry-level HTTP configuration copied into the handle
    http_config: Option<crate::types::HttpConfig>,
}

/// Implementation of AudioCapability for TranscriptionModelHandle
///
/// This allows the handle to be used directly as an STT client, aligning with
/// Vercel AI SDK's design where registry.transcriptionModel() returns a callable model.
#[async_trait::async_trait]
impl AudioCapability for TranscriptionModelHandle {
    fn supported_features(&self) -> &[AudioFeature] {
        // Default features - providers should override this
        &[AudioFeature::SpeechToText]
    }

    async fn speech_to_text(&self, request: SttRequest) -> Result<SttResponse, LlmError> {
        // Build client from factory with context
        let ctx = BuildContext {
            http_interceptors: self.http_interceptors.clone(),
            retry_options: self.retry_options.clone(),
            http_config: self.http_config.clone(),
            ..Default::default()
        };
        let client_raw = self
            .factory
            .transcription_model_with_ctx(&self.model_id, &ctx)
            .await?;
        let client = client_raw;

        // Get transcription capability
        let transcription_client = client.as_transcription_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation("Provider does not support speech-to-text".to_string())
        })?;

        transcription_client.stt(request).await
    }
}

impl TranscriptionModelHandle {}

/// Reranking model handle - delegates to factory for client creation
#[derive(Clone)]
pub struct RerankingModelHandle {
    factory: Arc<dyn ProviderFactory>,
    pub model_id: String,
    /// Registry-level HTTP interceptors to attempt injecting into clients
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Registry-level retry options copied into the handle
    retry_options: Option<RetryOptions>,
    /// Registry-level HTTP configuration copied into the handle
    http_config: Option<crate::types::HttpConfig>,
}

/// Implementation of RerankCapability for RerankingModelHandle
///
/// This allows the handle to be used directly as a reranking client, aligning with
/// Vercel AI SDK's design where registry.rerankingModel() returns a callable model.
#[async_trait::async_trait]
impl RerankCapability for RerankingModelHandle {
    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, LlmError> {
        // Build client from factory with context
        let ctx = BuildContext {
            http_interceptors: self.http_interceptors.clone(),
            retry_options: self.retry_options.clone(),
            http_config: self.http_config.clone(),
            ..Default::default()
        };
        let client_raw = self
            .factory
            .reranking_model_with_ctx(&self.model_id, &ctx)
            .await?;
        let client = client_raw;

        // Get rerank capability
        let rerank_client = client.as_rerank_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation("Provider does not support reranking".to_string())
        })?;

        rerank_client.rerank(request).await
    }

    fn max_documents(&self) -> Option<u32> {
        None
    }

    fn supported_models(&self) -> Vec<String> {
        vec![self.model_id.clone()]
    }
}

#[cfg(test)]
use std::sync::atomic::AtomicUsize;
#[cfg(test)]
use std::sync::atomic::Ordering;
#[cfg(test)]
pub static TEST_BUILD_COUNT: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
pub struct TestProvClient;
#[cfg(test)]
#[async_trait::async_trait]
impl ChatCapability for TestProvClient {
    async fn chat_with_tools(
        &self,
        _messages: Vec<crate::types::ChatMessage>,
        _tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<crate::types::ChatResponse, LlmError> {
        Ok(crate::types::ChatResponse::new(
            crate::types::MessageContent::Text("ok".to_string()),
        ))
    }

    async fn chat_stream(
        &self,
        _messages: Vec<crate::types::ChatMessage>,
        _tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<crate::streaming::ChatStream, LlmError> {
        Err(LlmError::UnsupportedOperation("mock stream".into()))
    }
}
#[cfg(test)]
impl LlmClient for TestProvClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov")
    }
    fn supported_models(&self) -> Vec<String> {
        vec!["model".into()]
    }
    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new().with_chat()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(TestProvClient)
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub struct TestProvEmbedClient;
#[cfg(test)]
#[async_trait::async_trait]
impl crate::traits::EmbeddingCapability for TestProvEmbedClient {
    async fn embed(&self, input: Vec<String>) -> Result<crate::types::EmbeddingResponse, LlmError> {
        Ok(crate::types::EmbeddingResponse::new(
            vec![vec![input.len() as f32]],
            "test-embed-model".to_string(),
        ))
    }
    fn embedding_dimension(&self) -> usize {
        1
    }
}
#[cfg(test)]
impl LlmClient for TestProvEmbedClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov_embed")
    }
    fn supported_models(&self) -> Vec<String> {
        vec!["model".into()]
    }
    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(TestProvEmbedClient)
    }
    fn as_embedding_capability(&self) -> Option<&dyn crate::traits::EmbeddingCapability> {
        Some(self)
    }
}

#[cfg(test)]
pub struct TestProviderFactory {
    id: &'static str,
}

#[cfg(test)]
impl TestProviderFactory {
    pub const fn new(id: &'static str) -> Self {
        Self { id }
    }
}

#[cfg(test)]
#[async_trait::async_trait]
impl ProviderFactory for TestProviderFactory {
    async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        TEST_BUILD_COUNT.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(TestProvClient))
    }

    async fn embedding_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        Ok(Arc::new(TestProvEmbedClient))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(self.id)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        match self.id {
            "testprov_embed" => ProviderCapabilities::new().with_embedding(),
            _ => ProviderCapabilities::new().with_chat(),
        }
    }
}

#[cfg(test)]
mod embedding_tests;

#[cfg(test)]
#[async_trait::async_trait]
impl crate::traits::ChatCapability for TestProvEmbedClient {
    async fn chat_with_tools(
        &self,
        _messages: Vec<crate::types::ChatMessage>,
        _tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<crate::types::ChatResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "chat not supported in TestProvEmbedClient".into(),
        ))
    }

    async fn chat_stream(
        &self,
        _messages: Vec<crate::types::ChatMessage>,
        _tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<crate::streaming::ChatStream, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "chat stream not supported in TestProvEmbedClient".into(),
        ))
    }
}
