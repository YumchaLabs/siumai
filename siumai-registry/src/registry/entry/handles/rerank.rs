use std::sync::Arc;

use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::retry_api::RetryOptions;
use crate::traits::RerankCapability;
use crate::types::{RerankRequest, RerankResponse};

use super::super::ProviderFactory;
use super::super::build_context::build_registry_context;

/// Reranking model handle - delegates to factory for client creation
#[derive(Clone)]
pub struct RerankingModelHandle {
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

/// Implementation of RerankCapability for RerankingModelHandle
///
/// This allows the handle to be used directly as a reranking client, aligning with
/// Vercel AI SDK's design where registry.rerankingModel() returns a callable model.
#[async_trait::async_trait]
impl RerankCapability for RerankingModelHandle {
    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, LlmError> {
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
            .reranking_model_family_with_ctx(&self.model_id, &ctx)
            .await?;

        model.rerank(request).await
    }

    fn max_documents(&self) -> Option<u32> {
        None
    }

    fn supported_models(&self) -> Vec<String> {
        vec![self.model_id.clone()]
    }
}

impl crate::traits::ModelMetadata for RerankingModelHandle {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}
