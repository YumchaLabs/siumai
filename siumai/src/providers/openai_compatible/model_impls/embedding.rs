//! OpenAI-Compatible Embedding Model Implementation

use crate::executors::embedding::HttpEmbeddingExecutor;
use crate::provider_core::{ProviderContext, ProviderSpec};
use crate::provider_model::EmbeddingModel;
use crate::providers::openai_compatible::openai_config::OpenAiCompatibleConfig;
use crate::providers::openai_compatible::spec::OpenAiCompatibleSpec;
use crate::retry_api::RetryOptions;
use crate::types::EmbeddingRequest;
use crate::utils::http_interceptor::HttpInterceptor;
use std::collections::HashMap;
use std::sync::Arc;

/// Convert HeaderMap to HashMap<String, String>
fn headermap_to_hashmap(headers: &reqwest::header::HeaderMap) -> HashMap<String, String> {
    headers
        .iter()
        .filter_map(|(k, v)| {
            v.to_str()
                .ok()
                .map(|v_str| (k.as_str().to_string(), v_str.to_string()))
        })
        .collect()
}

/// OpenAI-Compatible Embedding Model
///
/// Represents an embedding model from an OpenAI-compatible provider.
/// Uses the OpenAI Standard Layer with provider-specific adapter.
pub struct OpenAiCompatibleEmbeddingModel {
    config: OpenAiCompatibleConfig,
    model: String,
}

impl OpenAiCompatibleEmbeddingModel {
    /// Create a new OpenAI-Compatible Embedding Model
    ///
    /// # Arguments
    /// * `config` - Provider configuration
    /// * `model` - Model name
    pub fn new(config: OpenAiCompatibleConfig, model: String) -> Self {
        Self { config, model }
    }
}

impl EmbeddingModel for OpenAiCompatibleEmbeddingModel {
    fn create_executor(
        &self,
        http_client: reqwest::Client,
        _interceptors: Vec<Arc<dyn HttpInterceptor>>,
        retry_options: Option<RetryOptions>,
    ) -> HttpEmbeddingExecutor {
        // Create spec (OpenAiCompatibleSpec handles adapter resolution)
        let spec = Arc::new(OpenAiCompatibleSpec);

        // Create provider context
        let ctx = ProviderContext::new(
            &self.config.provider_id,
            self.config.base_url.clone(),
            Some(self.config.api_key.clone()),
            headermap_to_hashmap(&self.config.custom_headers),
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
            before_send: None, // TODO: Implement before_send for embedding if needed
            retry_options,
        }
    }
}
