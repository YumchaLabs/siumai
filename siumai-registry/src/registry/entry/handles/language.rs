use std::sync::Arc;
use std::time::Duration;

use lru::LruCache;
use tokio::sync::Mutex as TokioMutex;

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::retry_api::RetryOptions;
use crate::streaming::{ChatStream, ChatStreamHandle};
use crate::text::LanguageModel as FamilyLanguageModel;
use crate::traits::{
    ChatCapability, FileManagementCapability, MusicGenerationCapability, ProviderCapabilities,
    SkillsCapability, VideoGenerationCapability,
};
use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, FileDeleteResponse, FileListQuery, FileListResponse,
    FileObject, FileUploadRequest, MusicGenerationRequest, MusicGenerationResponse,
    SkillUploadRequest, SkillUploadResult, Tool, VideoGenerationRequest, VideoGenerationResponse,
    VideoTaskStatusResponse,
};
use siumai_core::video::VideoModel as FamilyVideoModel;

use super::super::ProviderFactory;
use super::super::build_context::build_registry_context;
use super::super::cache::CacheEntry;
use super::video_support::{
    apply_video_handle_default_model, video_model_handle_max_videos_per_call,
};

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
    pub(in crate::registry::entry) factory: Arc<dyn ProviderFactory>,
    /// Provider ID (e.g., "openai")
    pub provider_id: String,
    /// Model ID to pass to the factory (e.g., "gpt-4")
    pub model_id: String,
    /// Middlewares to apply to the client
    pub middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    /// Registry-level HTTP interceptors to attempt injecting into clients
    pub(in crate::registry::entry) http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Registry-level pre-built HTTP client copied into the handle
    pub(in crate::registry::entry) http_client: Option<reqwest::Client>,
    /// Registry-level custom HTTP transport copied into the handle
    pub(in crate::registry::entry) http_transport:
        Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    /// Registry-level HTTP configuration copied into the handle
    pub(in crate::registry::entry) http_config: Option<crate::types::HttpConfig>,
    /// Registry-level API key copied into the handle
    pub(in crate::registry::entry) api_key: Option<String>,
    /// Registry-level base URL copied into the handle
    pub(in crate::registry::entry) base_url: Option<String>,
    /// Registry-level unified reasoning flag copied into the handle
    pub(in crate::registry::entry) reasoning_enabled: Option<bool>,
    /// Registry-level unified reasoning budget copied into the handle
    pub(in crate::registry::entry) reasoning_budget: Option<i32>,
    /// Shared LRU cache for clients
    pub(in crate::registry::entry) cache: Arc<TokioMutex<LruCache<String, CacheEntry>>>,
    /// TTL for cached clients
    pub(in crate::registry::entry) client_ttl: Option<Duration>,
    /// Registry-level retry options copied into the handle
    pub(in crate::registry::entry) retry_options: Option<RetryOptions>,
    /// Provider-level capability hints captured at construction time
    pub(in crate::registry::entry) capabilities: ProviderCapabilities,
}

impl LanguageModelHandle {
    fn ensure_chat_capability(&self, stream: bool) -> Result<(), LlmError> {
        if !self.capabilities.supports("chat") {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not expose chat on the language_model handle",
                self.provider_id
            )));
        }

        if stream && !self.capabilities.supports("streaming") {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not expose chat streaming on the language_model handle",
                self.provider_id
            )));
        }

        Ok(())
    }

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
    async fn get_or_create_language_model(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn FamilyLanguageModel>, LlmError> {
        let cache_key = format!("{}:{}", self.provider_id, model_id);

        let mut cache = self.cache.lock().await;
        if let Some(entry) = cache.get(&cache_key) {
            if !entry.is_expired(self.client_ttl) {
                return Ok(entry.model.clone());
            }
            cache.pop(&cache_key);
        }

        drop(cache);
        let ctx = build_registry_context(
            &self.provider_id,
            &self.http_interceptors,
            &self.retry_options,
            &self.http_client,
            &self.http_transport,
            &self.http_config,
            &self.api_key,
            &self.base_url,
            self.reasoning_enabled,
            self.reasoning_budget,
        );
        let model = self
            .factory
            .language_model_text_with_ctx(model_id, &ctx)
            .await?;

        let mut cache = self.cache.lock().await;
        cache.put(cache_key, CacheEntry::new(model.clone()));

        Ok(model)
    }

    async fn build_language_client(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        let ctx = build_registry_context(
            &self.provider_id,
            &self.http_interceptors,
            &self.retry_options,
            &self.http_client,
            &self.http_transport,
            &self.http_config,
            &self.api_key,
            &self.base_url,
            self.reasoning_enabled,
            self.reasoning_budget,
        );

        self.factory
            .compat_language_client_with_ctx(model_id, &ctx)
            .await
    }

    async fn build_video_model(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn FamilyVideoModel>, LlmError> {
        let ctx = build_registry_context(
            &self.provider_id,
            &self.http_interceptors,
            &self.retry_options,
            &self.http_client,
            &self.http_transport,
            &self.http_config,
            &self.api_key,
            &self.base_url,
            self.reasoning_enabled,
            self.reasoning_budget,
        );

        self.factory
            .video_model_family_with_ctx(model_id, &ctx)
            .await
    }
}

impl crate::traits::ModelMetadata for LanguageModelHandle {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn model_id(&self) -> &str {
        &self.model_id
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

    fn as_chat_capability(&self) -> Option<&dyn ChatCapability> {
        self.capabilities.supports("chat").then_some(self)
    }

    fn as_file_management_capability(&self) -> Option<&dyn FileManagementCapability> {
        self.capabilities
            .supports("file_management")
            .then_some(self)
    }

    fn as_skills_capability(&self) -> Option<&dyn SkillsCapability> {
        self.capabilities.supports("skills").then_some(self)
    }

    fn as_video_generation_capability(&self) -> Option<&dyn VideoGenerationCapability> {
        self.capabilities.supports("video").then_some(self)
    }

    fn as_music_generation_capability(&self) -> Option<&dyn MusicGenerationCapability> {
        self.capabilities.supports("music").then_some(self)
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
        self.ensure_chat_capability(false)?;
        // Apply middleware overrides (aligned with Vercel AI SDK)
        let model_id = if !self.middlewares.is_empty() {
            crate::execution::middleware::language_model::apply_model_id_override(
                &self.middlewares,
                &self.model_id,
            )
        } else {
            self.model_id.clone()
        };

        // Get or create cached family model with potentially overridden model_id
        let model = self.get_or_create_language_model(&model_id).await?;

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
            model.generate(req).await
        } else {
            let mut req = ChatRequest::new(messages);
            if let Some(t) = tools {
                req = req.with_tools(t);
            }
            model.generate(req).await
        }
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        self.ensure_chat_capability(true)?;
        // Apply middleware overrides (aligned with Vercel AI SDK)
        let model_id = if !self.middlewares.is_empty() {
            crate::execution::middleware::language_model::apply_model_id_override(
                &self.middlewares,
                &self.model_id,
            )
        } else {
            self.model_id.clone()
        };

        // Get or create cached family model with potentially overridden model_id
        let model = self.get_or_create_language_model(&model_id).await?;

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
            model.stream(req.with_streaming(true)).await
        } else {
            let mut req = ChatRequest::new(messages).with_streaming(true);
            if let Some(t) = tools {
                req = req.with_tools(t);
            }
            model.stream(req).await
        }
    }

    async fn chat_stream_with_cancel(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStreamHandle, LlmError> {
        self.ensure_chat_capability(true)?;
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

                let model = this.get_or_create_language_model(&model_id).await?;

                let mut req = ChatRequest::new(messages).with_streaming(true);
                if let Some(t) = tools {
                    req = req.with_tools(t);
                }
                req.common_params.model = model_id.clone();

                if !this.middlewares.is_empty() {
                    req = crate::execution::middleware::language_model::apply_transform_chain(
                        &this.middlewares,
                        req,
                    );
                }

                model.stream_with_cancel(req).await
            }),
        )
    }

    async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.ensure_chat_capability(false)?;
        let model_id = if !self.middlewares.is_empty() {
            crate::execution::middleware::language_model::apply_model_id_override(
                &self.middlewares,
                &self.model_id,
            )
        } else {
            self.model_id.clone()
        };

        let model = self.get_or_create_language_model(&model_id).await?;

        let mut req = request.with_streaming(false);
        if req.common_params.model.trim().is_empty() {
            req.common_params.model = model_id.clone();
        }
        if !self.middlewares.is_empty() {
            req = crate::execution::middleware::language_model::apply_transform_chain(
                &self.middlewares,
                req,
            );
        }

        model.generate(req).await
    }

    async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        self.ensure_chat_capability(true)?;
        let model_id = if !self.middlewares.is_empty() {
            crate::execution::middleware::language_model::apply_model_id_override(
                &self.middlewares,
                &self.model_id,
            )
        } else {
            self.model_id.clone()
        };

        let model = self.get_or_create_language_model(&model_id).await?;

        let mut req = request.with_streaming(true);
        if req.common_params.model.trim().is_empty() {
            req.common_params.model = model_id.clone();
        }
        if !self.middlewares.is_empty() {
            req = crate::execution::middleware::language_model::apply_transform_chain(
                &self.middlewares,
                req,
            );
        }

        model.stream(req).await
    }

    async fn chat_stream_request_with_cancel(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStreamHandle, LlmError> {
        self.ensure_chat_capability(true)?;
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

                let model = this.get_or_create_language_model(&model_id).await?;

                let mut req = request.with_streaming(true);
                if req.common_params.model.trim().is_empty() {
                    req.common_params.model = model_id.clone();
                }
                if !this.middlewares.is_empty() {
                    req = crate::execution::middleware::language_model::apply_transform_chain(
                        &this.middlewares,
                        req,
                    );
                }

                model.stream_with_cancel(req).await
            }),
        )
    }
}

#[async_trait::async_trait]
impl FileManagementCapability for LanguageModelHandle {
    async fn upload_file(&self, request: FileUploadRequest) -> Result<FileObject, LlmError> {
        let client = self.build_language_client(&self.model_id).await?;
        let files = client.as_file_management_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation(format!(
                "Provider {} does not support file management.",
                self.provider_id
            ))
        })?;
        files.upload_file(request).await
    }

    async fn list_files(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError> {
        let client = self.build_language_client(&self.model_id).await?;
        let files = client.as_file_management_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation(format!(
                "Provider {} does not support file management.",
                self.provider_id
            ))
        })?;
        files.list_files(query).await
    }

    async fn retrieve_file(&self, file_id: String) -> Result<FileObject, LlmError> {
        let client = self.build_language_client(&self.model_id).await?;
        let files = client.as_file_management_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation(format!(
                "Provider {} does not support file management.",
                self.provider_id
            ))
        })?;
        files.retrieve_file(file_id).await
    }

    async fn delete_file(&self, file_id: String) -> Result<FileDeleteResponse, LlmError> {
        let client = self.build_language_client(&self.model_id).await?;
        let files = client.as_file_management_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation(format!(
                "Provider {} does not support file management.",
                self.provider_id
            ))
        })?;
        files.delete_file(file_id).await
    }

    async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        let client = self.build_language_client(&self.model_id).await?;
        let files = client.as_file_management_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation(format!(
                "Provider {} does not support file management.",
                self.provider_id
            ))
        })?;
        files.get_file_content(file_id).await
    }
}

#[async_trait::async_trait]
impl SkillsCapability for LanguageModelHandle {
    async fn upload_skill(
        &self,
        request: SkillUploadRequest,
    ) -> Result<SkillUploadResult, LlmError> {
        let client = self.build_language_client(&self.model_id).await?;
        let skills = client.as_skills_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation(format!(
                "Provider {} does not support skills.",
                self.provider_id
            ))
        })?;
        skills.upload_skill(request).await
    }
}

#[async_trait::async_trait]
impl VideoGenerationCapability for LanguageModelHandle {
    async fn create_video_task(
        &self,
        request: VideoGenerationRequest,
    ) -> Result<VideoGenerationResponse, LlmError> {
        let model = self.build_video_model(&self.model_id).await?;
        model
            .create_task(apply_video_handle_default_model(request, &self.model_id))
            .await
    }

    async fn query_video_task(&self, task_id: &str) -> Result<VideoTaskStatusResponse, LlmError> {
        let model = self.build_video_model(&self.model_id).await?;
        model.query_task(task_id).await
    }

    fn max_videos_per_call(&self) -> Option<u32> {
        video_model_handle_max_videos_per_call(&self.provider_id, &self.model_id)
    }

    fn get_supported_models(&self) -> Vec<String> {
        if self.capabilities.supports("video") {
            vec![self.model_id.clone()]
        } else {
            vec![]
        }
    }

    fn get_supported_resolutions(&self, _model: &str) -> Vec<String> {
        vec![]
    }

    fn get_supported_durations(&self, _model: &str) -> Vec<u32> {
        vec![]
    }
}

#[async_trait::async_trait]
impl MusicGenerationCapability for LanguageModelHandle {
    async fn generate_music(
        &self,
        request: MusicGenerationRequest,
    ) -> Result<MusicGenerationResponse, LlmError> {
        let client = self.build_language_client(&self.model_id).await?;
        let music = client.as_music_generation_capability().ok_or_else(|| {
            LlmError::UnsupportedOperation(format!(
                "Provider {} does not support music generation.",
                self.provider_id
            ))
        })?;
        music.generate_music(request).await
    }

    fn get_supported_music_models(&self) -> Vec<String> {
        if self.capabilities.supports("music") {
            vec![self.model_id.clone()]
        } else {
            vec![]
        }
    }

    fn get_supported_audio_formats(&self) -> Vec<String> {
        vec![]
    }

    fn supports_lyrics(&self) -> bool {
        self.capabilities.supports("music")
    }
}
