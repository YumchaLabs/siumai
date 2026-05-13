use std::sync::Arc;

use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::retry_api::RetryOptions;
use crate::traits::{EmbeddingCapability, EmbeddingExtensions};
use crate::types::{
    BatchEmbeddingRequest, BatchEmbeddingResponse, EmbeddingRequest, EmbeddingResponse,
};

use super::super::ProviderFactory;
use super::super::build_context::build_registry_context;

/// Embedding model handle - delegates to factory for client creation
#[derive(Clone)]
pub struct EmbeddingModelHandle {
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
}

/// Implementation of EmbeddingCapability for EmbeddingModelHandle
///
/// This allows the handle to be used directly as an embedding client, aligning with
/// Vercel AI SDK's design where registry.textEmbeddingModel() returns a callable model.
#[async_trait::async_trait]
impl EmbeddingCapability for EmbeddingModelHandle {
    async fn embed(&self, input: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
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
            .embedding_model_family_with_ctx(&self.model_id, &ctx)
            .await?;

        model
            .embed(crate::types::EmbeddingRequest {
                input,
                ..Default::default()
            })
            .await
    }

    fn as_embedding_extensions(&self) -> Option<&dyn EmbeddingExtensions> {
        Some(self)
    }

    fn embedding_dimension(&self) -> usize {
        1536
    }
}

impl crate::traits::ModelMetadata for EmbeddingModelHandle {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[async_trait::async_trait]
impl EmbeddingExtensions for EmbeddingModelHandle {
    async fn embed_with_config(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
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
        if let Ok(client) = self
            .factory
            .compat_embedding_client_with_ctx(&self.model_id, &ctx)
            .await
            && let Some(extensions) = client.as_embedding_extensions()
        {
            return extensions.embed_with_config(request).await;
        }

        let model = self
            .factory
            .embedding_model_family_with_ctx(&self.model_id, &ctx)
            .await?;
        model.embed(request).await
    }

    async fn embed_batch(
        &self,
        requests: BatchEmbeddingRequest,
    ) -> Result<BatchEmbeddingResponse, LlmError> {
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
            .embedding_model_family_with_ctx(&self.model_id, &ctx)
            .await?;
        model.embed_many(requests).await
    }
}
