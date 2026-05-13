use std::sync::Arc;
use std::time::Duration;

use lru::LruCache;
use tokio::sync::Mutex as TokioMutex;

use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::retry_api::RetryOptions;
use crate::traits::VideoGenerationCapability;
use crate::types::{
    MaterializedVideoAsset, ProviderReference, VideoGenerationRequest, VideoGenerationResponse,
    VideoTaskStatusResponse,
};
use siumai_core::video::VideoModel as FamilyVideoModel;

use super::super::ProviderFactory;
use super::super::build_context::build_registry_context;
use super::super::cache::VideoCacheEntry;
use super::video_support::{
    apply_video_handle_default_model, video_model_handle_max_videos_per_call,
};

/// Video model handle - delegates to factory for client creation
#[derive(Clone)]
pub struct VideoModelHandle {
    pub(in crate::registry::entry) factory: Arc<dyn ProviderFactory>,
    pub(in crate::registry::entry) provider_id: String,
    pub model_id: String,
    /// Registry-level HTTP interceptors to attempt injecting into clients
    pub(in crate::registry::entry) http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Registry-level pre-built HTTP client copied into the handle
    pub(in crate::registry::entry) http_client: Option<reqwest::Client>,
    /// Registry-level custom HTTP transport copied into the handle
    pub(in crate::registry::entry) http_transport:
        Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    /// Registry-level retry options copied into the handle
    pub(in crate::registry::entry) retry_options: Option<RetryOptions>,
    /// Registry-level HTTP configuration copied into the handle
    pub(in crate::registry::entry) http_config: Option<crate::types::HttpConfig>,
    /// Registry-level API key copied into the handle
    pub(in crate::registry::entry) api_key: Option<String>,
    /// Registry-level base URL copied into the handle
    pub(in crate::registry::entry) base_url: Option<String>,
    /// Shared LRU cache for video-family models
    pub(in crate::registry::entry) cache: Arc<TokioMutex<LruCache<String, VideoCacheEntry>>>,
    /// TTL for cached video-family models
    pub(in crate::registry::entry) client_ttl: Option<Duration>,
}

/// Implementation of VideoGenerationCapability for VideoModelHandle
///
/// This allows the handle to be used directly as a task-oriented video client, aligned with the
/// provider registry `video_model(...)` entry point.
#[async_trait::async_trait]
impl VideoGenerationCapability for VideoModelHandle {
    async fn create_video_task(
        &self,
        request: VideoGenerationRequest,
    ) -> Result<VideoGenerationResponse, LlmError> {
        let model = self.get_or_create_video_model(&self.model_id).await?;
        model
            .create_task(apply_video_handle_default_model(request, &self.model_id))
            .await
    }

    async fn query_video_task(&self, task_id: &str) -> Result<VideoTaskStatusResponse, LlmError> {
        let model = self.get_or_create_video_model(&self.model_id).await?;
        model.query_task(task_id).await
    }

    async fn materialize_video_reference(
        &self,
        provider_reference: &ProviderReference,
    ) -> Result<MaterializedVideoAsset, LlmError> {
        let model = self.get_or_create_video_model(&self.model_id).await?;
        model.materialize_video_reference(provider_reference).await
    }

    fn max_videos_per_call(&self) -> Option<u32> {
        video_model_handle_max_videos_per_call(&self.provider_id, &self.model_id)
    }

    fn get_supported_models(&self) -> Vec<String> {
        vec![self.model_id.clone()]
    }

    fn get_supported_resolutions(&self, _model: &str) -> Vec<String> {
        Vec::new()
    }

    fn get_supported_durations(&self, _model: &str) -> Vec<u32> {
        Vec::new()
    }
}

impl crate::traits::ModelMetadata for VideoModelHandle {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

impl VideoModelHandle {
    async fn get_or_create_video_model(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn FamilyVideoModel>, LlmError> {
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
            None,
            None,
        );
        let model = self
            .factory
            .video_model_family_with_ctx(model_id, &ctx)
            .await?;

        let mut cache = self.cache.lock().await;
        cache.put(cache_key, VideoCacheEntry::new(model.clone()));

        Ok(model)
    }
}
