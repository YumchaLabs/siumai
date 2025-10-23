//! OpenAI-Compatible Image Model Implementation

use crate::executors::image::HttpImageExecutor;
use crate::provider_core::{ProviderContext, ProviderSpec};
use crate::provider_model::ImageModel;
use crate::providers::openai_compatible::openai_config::OpenAiCompatibleConfig;
use crate::providers::openai_compatible::spec::OpenAiCompatibleSpec;
use crate::retry_api::RetryOptions;
use crate::utils::http_headers::headermap_to_hashmap;
use crate::utils::http_interceptor::HttpInterceptor;
use std::sync::Arc;

/// OpenAI-Compatible Image Model
///
/// Represents an image generation model from an OpenAI-compatible provider.
/// Uses the OpenAI Standard Layer with provider-specific adapter.
pub struct OpenAiCompatibleImageModel {
    config: OpenAiCompatibleConfig,
    model: String,
}

impl OpenAiCompatibleImageModel {
    /// Create a new OpenAI-Compatible Image Model
    ///
    /// # Arguments
    /// * `config` - Provider configuration
    /// * `model` - Model name
    pub fn new(config: OpenAiCompatibleConfig, model: String) -> Self {
        Self { config, model }
    }
}

impl ImageModel for OpenAiCompatibleImageModel {
    fn create_executor(
        &self,
        http_client: reqwest::Client,
        interceptors: Vec<Arc<dyn HttpInterceptor>>,
        retry_options: Option<RetryOptions>,
    ) -> HttpImageExecutor {
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
        let dummy_request = crate::types::ImageGenerationRequest {
            model: Some(self.model.clone()),
            ..Default::default()
        };
        let bundle = spec.choose_image_transformers(&dummy_request, &ctx);

        // Create executor
        HttpImageExecutor {
            provider_id: self.config.provider_id.clone(),
            http_client,
            request_transformer: bundle.request,
            response_transformer: bundle.response,
            provider_spec: spec,
            provider_context: ctx,
            interceptors,
            before_send: None, // TODO: Implement before_send for image if needed
            retry_options,
        }
    }
}
