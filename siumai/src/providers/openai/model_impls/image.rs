//! OpenAI Image Model Implementation

use crate::executors::image::HttpImageExecutor;
use crate::provider_core::{ProviderContext, ProviderSpec};
use crate::provider_model::ImageModel;
use crate::retry_api::RetryOptions;
use crate::standards::openai::image::OpenAiImageAdapter;
use crate::types::ImageGenerationRequest;
use crate::utils::http_interceptor::HttpInterceptor;
use secrecy::ExposeSecret;
use std::sync::Arc;

use super::super::config::OpenAiConfig;
use super::super::provider_impl::ModelConfig;
use super::super::spec::OpenAiSpec;

/// OpenAI Image Model
///
/// Encapsulates configuration for OpenAI image generation endpoint and creates HttpImageExecutor.
pub struct OpenAiImageModel {
    config: ModelConfig,
    model: String,
    adapter: Option<Arc<dyn OpenAiImageAdapter>>,
    openai_config: OpenAiConfig,
}

impl OpenAiImageModel {
    /// Create a new OpenAI Image Model
    pub fn new(
        config: OpenAiConfig,
        model: String,
        adapter: Option<Arc<dyn OpenAiImageAdapter>>,
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
        adapter: Arc<dyn OpenAiImageAdapter>,
    ) -> Self {
        Self::new(config, model, Some(adapter))
    }

    /// Get the model name
    pub fn model(&self) -> &str {
        &self.model
    }
}

impl ImageModel for OpenAiImageModel {
    fn create_executor(
        &self,
        http_client: reqwest::Client,
        interceptors: Vec<Arc<dyn HttpInterceptor>>,
        retry_options: Option<RetryOptions>,
    ) -> HttpImageExecutor {
        // Create OpenAI Standard (with optional adapter)
        let _image_standard = if let Some(adapter) = &self.adapter {
            crate::standards::openai::image::OpenAiImageStandard::with_adapter(adapter.clone())
        } else {
            crate::standards::openai::image::OpenAiImageStandard::new()
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
        let dummy_request = ImageGenerationRequest {
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
            before_send: None,
            retry_options,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_openai_image_model() {
        let config = OpenAiConfig::new("test-key");
        let model = OpenAiImageModel::new(config, "dall-e-3".to_string(), None);

        assert_eq!(model.model(), "dall-e-3");
    }

    #[test]
    fn test_create_executor() {
        let config = OpenAiConfig::new("test-key");
        let model = OpenAiImageModel::new(config, "dall-e-3".to_string(), None);

        let http_client = reqwest::Client::new();
        let executor = model.create_executor(http_client, vec![], None);

        assert_eq!(executor.provider_id, "openai");
    }
}
