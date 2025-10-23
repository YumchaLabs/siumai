//! OpenAI Embedding Model Implementation

use crate::executors::embedding::HttpEmbeddingExecutor;
use crate::provider_core::{ProviderContext, ProviderSpec};
use crate::provider_model::EmbeddingModel;
use crate::retry_api::RetryOptions;
use crate::standards::openai::embedding::OpenAiEmbeddingAdapter;
use crate::types::EmbeddingRequest;
use crate::utils::http_interceptor::HttpInterceptor;
use secrecy::ExposeSecret;
use std::sync::Arc;

use super::super::config::OpenAiConfig;
use super::super::provider_impl::ModelConfig;
use super::super::spec::OpenAiSpec;

/// OpenAI Embedding Model
///
/// Encapsulates configuration for OpenAI embedding endpoint and creates HttpEmbeddingExecutor.
pub struct OpenAiEmbeddingModel {
    config: ModelConfig,
    model: String,
    adapter: Option<Arc<dyn OpenAiEmbeddingAdapter>>,
    openai_config: OpenAiConfig,
}

impl OpenAiEmbeddingModel {
    /// Create a new OpenAI Embedding Model
    pub fn new(
        config: OpenAiConfig,
        model: String,
        adapter: Option<Arc<dyn OpenAiEmbeddingAdapter>>,
    ) -> Self {
        let model_config = ModelConfig::from(&config);
        Self {
            config: model_config,
            model,
            adapter,
            openai_config: config,
        }
    }

    /// Create with adapter (for OpenAI-compatible providers)
    pub fn new_with_adapter(
        config: OpenAiConfig,
        model: String,
        adapter: Arc<dyn OpenAiEmbeddingAdapter>,
    ) -> Self {
        Self::new(config, model, Some(adapter))
    }

    /// Get the model name
    pub fn model(&self) -> &str {
        &self.model
    }
}

impl EmbeddingModel for OpenAiEmbeddingModel {
    fn create_executor(
        &self,
        http_client: reqwest::Client,
        interceptors: Vec<Arc<dyn HttpInterceptor>>,
        retry_options: Option<RetryOptions>,
    ) -> HttpEmbeddingExecutor {
        // Create OpenAI Standard (with optional adapter)
        let _embedding_standard = if let Some(adapter) = &self.adapter {
            crate::standards::openai::embedding::OpenAiEmbeddingStandard::with_adapter(
                adapter.clone(),
            )
        } else {
            crate::standards::openai::embedding::OpenAiEmbeddingStandard::new()
        };

        // Create spec
        let spec = Arc::new(OpenAiSpec::new());

        // Create provider context
        let ctx = ProviderContext::new(
            &self.config.provider_id,
            self.config.base_url.clone(),
            Some(self.config.api_key.expose_secret().to_string()),
            self.config.headers.clone(),
        )
        .with_org_project(
            self.config.organization.clone(),
            self.config.project.clone(),
        );

        // Get transformers from spec
        let dummy_request = EmbeddingRequest {
            model: Some(self.model.clone()),
            ..Default::default()
        };
        let bundle = spec.choose_embedding_transformers(&dummy_request, &ctx);

        // Create executor
        HttpEmbeddingExecutor {
            provider_id: self.config.provider_id.clone(),
            http_client,
            request_transformer: bundle.request,
            response_transformer: bundle.response,
            provider_spec: spec,
            provider_context: ctx,
            interceptors,
            before_send: None, // TODO: Implement before_send for embedding
            retry_options,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_openai_embedding_model() {
        let config = OpenAiConfig::new("test-key");
        let model = OpenAiEmbeddingModel::new(config, "text-embedding-3-small".to_string(), None);

        assert_eq!(model.model(), "text-embedding-3-small");
    }

    #[test]
    fn test_create_executor() {
        let config = OpenAiConfig::new("test-key");
        let model = OpenAiEmbeddingModel::new(config, "text-embedding-3-small".to_string(), None);

        let http_client = reqwest::Client::new();
        let executor = model.create_executor(http_client, vec![], None);

        assert_eq!(executor.provider_id, "openai");
    }
}
