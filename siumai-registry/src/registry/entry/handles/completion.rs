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
use crate::traits::{CompletionCapability, ProviderCapabilities};
use crate::types::{CompletionRequest, CompletionResponse};
use siumai_core::completion::CompletionModel as FamilyCompletionModel;

use super::super::ProviderFactory;
use super::super::build_context::build_registry_context;
use super::super::cache::CompletionCacheEntry;

/// Completion model handle - delegates to factory for client creation.
#[derive(Clone)]
pub struct CompletionModelHandle {
    pub(in crate::registry::entry) factory: Arc<dyn ProviderFactory>,
    pub(in crate::registry::entry) provider_id: String,
    pub model_id: String,
    pub(in crate::registry::entry) middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    /// Registry-level HTTP interceptors to attempt injecting into clients.
    pub(in crate::registry::entry) http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Registry-level pre-built HTTP client copied into the handle.
    pub(in crate::registry::entry) http_client: Option<reqwest::Client>,
    /// Registry-level custom HTTP transport copied into the handle.
    pub(in crate::registry::entry) http_transport:
        Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    /// Registry-level retry options copied into the handle.
    pub(in crate::registry::entry) retry_options: Option<RetryOptions>,
    /// Registry-level HTTP configuration copied into the handle.
    pub(in crate::registry::entry) http_config: Option<crate::types::HttpConfig>,
    /// Registry-level API key copied into the handle.
    pub(in crate::registry::entry) api_key: Option<String>,
    /// Registry-level base URL copied into the handle.
    pub(in crate::registry::entry) base_url: Option<String>,
    /// Registry-level unified reasoning flag copied into the handle.
    pub(in crate::registry::entry) reasoning_enabled: Option<bool>,
    /// Registry-level unified reasoning budget copied into the handle.
    pub(in crate::registry::entry) reasoning_budget: Option<i32>,
    /// Shared LRU cache for completion-family models.
    pub(in crate::registry::entry) cache: Arc<TokioMutex<LruCache<String, CompletionCacheEntry>>>,
    /// TTL for cached completion-family models.
    pub(in crate::registry::entry) client_ttl: Option<Duration>,
    /// Provider-level capability hints captured at construction time.
    pub(in crate::registry::entry) capabilities: ProviderCapabilities,
}

impl CompletionModelHandle {
    fn ensure_completion_capability(&self, stream: bool) -> Result<(), LlmError> {
        if !self.capabilities.supports("completion") {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not expose completion on the completion_model handle",
                self.provider_id
            )));
        }

        if stream && !self.capabilities.supports("streaming") {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not expose completion streaming on the completion_model handle",
                self.provider_id
            )));
        }

        Ok(())
    }

    async fn get_or_create_completion_model(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn FamilyCompletionModel>, LlmError> {
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
            .completion_model_family_with_ctx(model_id, &ctx)
            .await?;

        let mut cache = self.cache.lock().await;
        cache.put(cache_key, CompletionCacheEntry::new(model.clone()));

        Ok(model)
    }
}

impl crate::traits::ModelMetadata for CompletionModelHandle {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

impl LlmClient for CompletionModelHandle {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Owned(self.provider_id.clone())
    }

    fn supported_models(&self) -> Vec<String> {
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

    fn as_completion_capability(&self) -> Option<&dyn CompletionCapability> {
        self.capabilities.supports("completion").then_some(self)
    }
}

#[async_trait::async_trait]
impl CompletionCapability for CompletionModelHandle {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        self.ensure_completion_capability(false)?;
        let model_id = if !self.middlewares.is_empty() {
            crate::execution::middleware::language_model::apply_model_id_override(
                &self.middlewares,
                &self.model_id,
            )
        } else {
            self.model_id.clone()
        };

        let model = self.get_or_create_completion_model(&model_id).await?;
        model
            .complete(request.with_model_if_missing(model_id))
            .await
    }

    async fn complete_stream(&self, request: CompletionRequest) -> Result<ChatStream, LlmError> {
        self.ensure_completion_capability(true)?;
        let model_id = if !self.middlewares.is_empty() {
            crate::execution::middleware::language_model::apply_model_id_override(
                &self.middlewares,
                &self.model_id,
            )
        } else {
            self.model_id.clone()
        };

        let model = self.get_or_create_completion_model(&model_id).await?;
        model.stream(request.with_model_if_missing(model_id)).await
    }

    async fn complete_stream_with_cancel(
        &self,
        request: CompletionRequest,
    ) -> Result<ChatStreamHandle, LlmError> {
        self.ensure_completion_capability(true)?;
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

                let model = this.get_or_create_completion_model(&model_id).await?;
                model
                    .stream_with_cancel(request.with_model_if_missing(model_id))
                    .await
            }),
        )
    }
}
