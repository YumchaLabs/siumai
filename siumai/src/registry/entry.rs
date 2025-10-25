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
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::streaming::ChatStream;
use crate::traits::{
    AudioCapability, ChatCapability, EmbeddingCapability, ImageGenerationCapability,
};
use crate::types::{
    AudioFeature, ChatMessage, ChatRequest, ChatResponse, EmbeddingResponse,
    ImageGenerationRequest, ImageGenerationResponse, SttRequest, SttResponse, Tool, TtsRequest,
    TtsResponse,
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

    /// Create an embedding model client for the given model ID
    async fn embedding_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Default: delegate to language_model (many providers use same client)
        self.language_model(model_id).await
    }

    /// Create an image model client for the given model ID
    async fn image_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.language_model(model_id).await
    }

    /// Create a speech model client for the given model ID
    async fn speech_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.language_model(model_id).await
    }

    /// Create a transcription model client for the given model ID
    async fn transcription_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.speech_model(model_id).await
    }

    /// Get the provider name
    fn provider_name(&self) -> &'static str;
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
    /// LRU cache for language model clients (key: "provider:model")
    /// Uses async Mutex for concurrent access and per-key build de-duplication
    language_model_cache: Arc<TokioMutex<LruCache<String, CacheEntry>>>,
    /// TTL for cached clients
    client_ttl: Option<Duration>,
    /// Whether to automatically add model-specific middlewares
    auto_middleware: bool,
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
    /// # use siumai::registry::create_provider_registry;
    /// # use std::collections::HashMap;
    /// let registry = create_provider_registry(HashMap::new(), None);
    /// let handle = registry.language_model("openai:gpt-4")?;
    /// # Ok::<(), siumai::error::LlmError>(())
    /// ```
    pub fn language_model(&self, id: &str) -> Result<LanguageModelHandle, LlmError> {
        let (mut provider_id, model_id) = self.split_id(id)?;

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
            registry: None, // Will be set if we need to support provider override
            cache: self.language_model_cache.clone(),
            cache_key: id.to_string(),
            client_ttl: self.client_ttl,
        })
    }

    /// Resolve embedding model - returns a handle that delegates to the factory
    pub fn embedding_model(&self, id: &str) -> Result<EmbeddingModelHandle, LlmError> {
        let (provider_id, model_id) = self.split_id(id)?;
        let factory = self.get_provider(&provider_id)?;

        Ok(EmbeddingModelHandle {
            factory: factory.clone(),
            model_id,
        })
    }

    /// Resolve image model - returns a handle that delegates to the factory
    pub fn image_model(&self, id: &str) -> Result<ImageModelHandle, LlmError> {
        let (provider_id, model_id) = self.split_id(id)?;
        let factory = self.get_provider(&provider_id)?;

        Ok(ImageModelHandle {
            factory: factory.clone(),
            model_id,
        })
    }

    /// Resolve speech model - returns a handle that delegates to the factory
    pub fn speech_model(&self, id: &str) -> Result<SpeechModelHandle, LlmError> {
        let (provider_id, model_id) = self.split_id(id)?;
        let factory = self.get_provider(&provider_id)?;

        Ok(SpeechModelHandle {
            factory: factory.clone(),
            model_id,
        })
    }

    /// Resolve transcription model - returns a handle that delegates to the factory
    pub fn transcription_model(&self, id: &str) -> Result<TranscriptionModelHandle, LlmError> {
        let (provider_id, model_id) = self.split_id(id)?;
        let factory = self.get_provider(&provider_id)?;

        Ok(TranscriptionModelHandle {
            factory: factory.clone(),
            model_id,
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
/// use siumai::registry::{create_provider_registry, ProviderFactory};
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
    let (separator, middlewares, max_cache_entries, client_ttl, auto_middleware) =
        if let Some(o) = opts {
            (
                o.separator,
                o.language_model_middleware,
                o.max_cache_entries,
                o.client_ttl,
                o.auto_middleware,
            )
        } else {
            (':', Vec::new(), None, None, true) // auto_middleware defaults to true
        };

    // Create LRU cache with specified capacity (default: 100 entries)
    let cache_capacity = max_cache_entries.unwrap_or(100);
    let cache =
        LruCache::new(NonZeroUsize::new(cache_capacity).expect("Cache capacity must be > 0"));

    ProviderRegistryHandle {
        providers,
        separator,
        middlewares,
        language_model_cache: Arc::new(TokioMutex::new(cache)),
        client_ttl,
        auto_middleware,
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
    /// Registry handle for resolving overridden providers
    #[allow(dead_code)] // Reserved for future registry-based resolution overrides
    registry: Option<Arc<ProviderRegistryHandle>>,
    /// Shared LRU cache for clients
    cache: Arc<TokioMutex<LruCache<String, CacheEntry>>>,
    /// Cache key for this handle ("provider:model")
    #[allow(dead_code)] // Internally derived; kept for debugging/inspection
    cache_key: String,
    /// TTL for cached clients
    client_ttl: Option<Duration>,
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
        let client = self.factory.language_model(model_id).await?;

        // Cache the new client
        let mut cache = self.cache.lock().await;
        cache.put(cache_key, CacheEntry::new(client.clone()));

        Ok(client)
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
}

/// Embedding model handle - delegates to factory for client creation
#[derive(Clone)]
pub struct EmbeddingModelHandle {
    factory: Arc<dyn ProviderFactory>,
    pub model_id: String,
}

/// Implementation of EmbeddingCapability for EmbeddingModelHandle
///
/// This allows the handle to be used directly as an embedding client, aligning with
/// Vercel AI SDK's design where registry.textEmbeddingModel() returns a callable model.
#[async_trait::async_trait]
impl EmbeddingCapability for EmbeddingModelHandle {
    async fn embed(&self, input: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        // Build client from factory
        let client = self.factory.embedding_model(&self.model_id).await?;

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
        // Build client from factory
        let client = self.factory.image_model(&self.model_id).await?;

        // Get image generation capability
        let image_client = client.as_image_generation_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation("Provider does not support image generation".to_string())
        })?;

        // Call generate_images
        image_client.generate_images(request).await
    }

    fn get_supported_sizes(&self) -> Vec<String> {
        // Default sizes - providers should override this
        vec![
            "1024x1024".to_string(),
            "512x512".to_string(),
            "256x256".to_string(),
        ]
    }

    fn get_supported_formats(&self) -> Vec<String> {
        // Default formats
        vec!["url".to_string(), "b64_json".to_string()]
    }
}

impl ImageModelHandle {}

/// Speech model handle (TTS) - delegates to factory for client creation
#[derive(Clone)]
pub struct SpeechModelHandle {
    factory: Arc<dyn ProviderFactory>,
    pub model_id: String,
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
        // Build client from factory
        let client = self.factory.speech_model(&self.model_id).await?;

        // Get audio capability
        let audio_client = client.as_audio_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation("Provider does not support text-to-speech".to_string())
        })?;

        // Call text_to_speech
        audio_client.text_to_speech(request).await
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
        // Build client from factory
        let client = self.factory.transcription_model(&self.model_id).await?;

        // Get audio capability
        let audio_client = client.as_audio_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation("Provider does not support speech-to-text".to_string())
        })?;

        // Call speech_to_text
        audio_client.speech_to_text(request).await
    }
}

impl TranscriptionModelHandle {}
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
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
    fn provider_name(&self) -> &'static str {
        "testprov"
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
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    // Serialize registry tests that mutate global TEST_BUILD_COUNT to avoid flakiness under parallel runs
    static REG_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    fn reg_test_guard() -> std::sync::MutexGuard<'static, ()> {
        REG_TEST_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    #[derive(Clone)]
    #[allow(dead_code)]
    struct MockClient(std::sync::Arc<std::sync::Mutex<usize>>);

    #[async_trait::async_trait]
    impl ChatCapability for MockClient {
        async fn chat_with_tools(
            &self,
            _messages: Vec<crate::types::ChatMessage>,
            _tools: Option<Vec<crate::types::Tool>>,
        ) -> Result<crate::types::ChatResponse, LlmError> {
            *self.0.lock().unwrap() += 1;
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

    impl LlmClient for MockClient {
        fn provider_name(&self) -> &'static str {
            "mock"
        }
        fn supported_models(&self) -> Vec<String> {
            vec!["mock-model".into()]
        }
        fn capabilities(&self) -> crate::traits::ProviderCapabilities {
            crate::traits::ProviderCapabilities::new().with_chat()
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn clone_box(&self) -> Box<dyn LlmClient> {
            Box::new(self.clone())
        }
    }

    #[tokio::test]
    async fn language_model_handle_builds_client() {
        let _g = reg_test_guard();
        // Create registry with test provider factory
        let mut providers = HashMap::new();
        providers.insert(
            "testprov".to_string(),
            Arc::new(crate::registry::factories::TestProviderFactory) as Arc<dyn ProviderFactory>,
        );
        let reg = create_provider_registry(providers, None);
        let handle = reg.language_model("testprov:model").unwrap();

        // First call builds a new client
        TEST_BUILD_COUNT.store(0, std::sync::atomic::Ordering::SeqCst);
        let resp = handle.chat(vec![]).await.unwrap();
        assert_eq!(resp.content_text().unwrap_or_default(), "ok");
        assert_eq!(
            TEST_BUILD_COUNT.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "First call should build a new client"
        );

        // Second call uses cached client (LRU cache)
        let resp = handle.chat(vec![]).await.unwrap();
        assert_eq!(resp.content_text().unwrap_or_default(), "ok");
        assert_eq!(
            TEST_BUILD_COUNT.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "Second call should use cached client"
        );
    }

    #[tokio::test]
    async fn lru_cache_eviction() {
        let _g = reg_test_guard();
        // Create registry with small cache (2 entries)
        let mut providers = HashMap::new();
        providers.insert(
            "testprov".to_string(),
            Arc::new(crate::registry::factories::TestProviderFactory) as Arc<dyn ProviderFactory>,
        );
        let reg = create_provider_registry(
            providers,
            Some(RegistryOptions {
                separator: ':',
                language_model_middleware: Vec::new(),
                max_cache_entries: Some(2),
                client_ttl: None,
                auto_middleware: false, // Disable for testing
            }),
        );

        TEST_BUILD_COUNT.store(0, std::sync::atomic::Ordering::SeqCst);

        // Create 3 different handles
        let handle1 = reg.language_model("testprov:model1").unwrap();
        let handle2 = reg.language_model("testprov:model2").unwrap();
        let handle3 = reg.language_model("testprov:model3").unwrap();

        // Use handle1 and handle2 (cache: [model1, model2])
        handle1.chat(vec![]).await.unwrap();
        handle2.chat(vec![]).await.unwrap();
        assert_eq!(
            TEST_BUILD_COUNT.load(std::sync::atomic::Ordering::SeqCst),
            2
        );

        // Use handle3 (cache: [model2, model3], model1 evicted)
        handle3.chat(vec![]).await.unwrap();
        assert_eq!(
            TEST_BUILD_COUNT.load(std::sync::atomic::Ordering::SeqCst),
            3
        );

        // Use handle2 again (cache hit)
        handle2.chat(vec![]).await.unwrap();
        assert_eq!(
            TEST_BUILD_COUNT.load(std::sync::atomic::Ordering::SeqCst),
            3
        );

        // Use handle1 again (cache miss, model1 was evicted)
        handle1.chat(vec![]).await.unwrap();
        assert_eq!(
            TEST_BUILD_COUNT.load(std::sync::atomic::Ordering::SeqCst),
            4
        );
    }

    #[tokio::test]
    async fn ttl_expiration() {
        let _g = reg_test_guard();
        use std::time::Duration;

        // Create registry with TTL of 100ms
        let mut providers = HashMap::new();
        providers.insert(
            "testprov".to_string(),
            Arc::new(crate::registry::factories::TestProviderFactory) as Arc<dyn ProviderFactory>,
        );
        let reg = create_provider_registry(
            providers,
            Some(RegistryOptions {
                separator: ':',
                language_model_middleware: Vec::new(),
                max_cache_entries: None,
                client_ttl: Some(Duration::from_millis(100)),
                auto_middleware: false, // Disable for testing
            }),
        );

        TEST_BUILD_COUNT.store(0, std::sync::atomic::Ordering::SeqCst);

        let handle = reg.language_model("testprov:model").unwrap();

        // First call builds client
        handle.chat(vec![]).await.unwrap();
        assert_eq!(
            TEST_BUILD_COUNT.load(std::sync::atomic::Ordering::SeqCst),
            1
        );

        // Second call uses cached client (within TTL)
        handle.chat(vec![]).await.unwrap();
        assert_eq!(
            TEST_BUILD_COUNT.load(std::sync::atomic::Ordering::SeqCst),
            1
        );

        // Wait for TTL to expire
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Third call rebuilds client (TTL expired)
        handle.chat(vec![]).await.unwrap();
        assert_eq!(
            TEST_BUILD_COUNT.load(std::sync::atomic::Ordering::SeqCst),
            2
        );
    }
}

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
    fn provider_name(&self) -> &'static str {
        "testprov_embed"
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
mod embedding_tests {
    use super::*;
    #[tokio::test]
    async fn embedding_model_handle_builds_client() {
        // Create registry with test provider factory
        let mut providers = HashMap::new();
        providers.insert(
            "testprov_embed".to_string(),
            Arc::new(crate::registry::factories::TestProviderFactory) as Arc<dyn ProviderFactory>,
        );
        let reg = create_provider_registry(providers, None);
        let handle = reg.embedding_model("testprov_embed:model").unwrap();

        // Client is built on each call (no caching)
        let out = handle.embed(vec!["a".into(), "b".into()]).await.unwrap();
        assert_eq!(out.embeddings[0][0], 2.0);
    }
}

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
