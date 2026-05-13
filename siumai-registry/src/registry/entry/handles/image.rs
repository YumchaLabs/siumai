use std::sync::Arc;

use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::retry_api::RetryOptions;
use crate::traits::{ImageExtras, ImageGenerationCapability};
use crate::types::{
    ImageEditRequest, ImageGenerationRequest, ImageGenerationResponse, ImageVariationRequest,
};

use super::super::ProviderFactory;
use super::super::build_context::build_registry_context;

/// Image model handle - delegates to factory for client creation
#[derive(Clone)]
pub struct ImageModelHandle {
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

pub(in crate::registry::entry) fn image_model_handle_max_images_per_call(
    provider_id: &str,
    model_id: &str,
) -> Option<u32> {
    match provider_id {
        "openai" => Some(10),
        "deepinfra" | "fireworks" | "together" | "togetherai" => Some(1),
        "xai" => Some(3),
        "bedrock" => Some(if model_id == "amazon.nova-canvas-v1:0" {
            5
        } else {
            1
        }),
        "gemini" => Some(if model_id.starts_with("gemini-") {
            10
        } else {
            4
        }),
        "google-vertex" | "vertex" => Some(if model_id.starts_with("gemini-") {
            10
        } else {
            4
        }),
        other => match other {
            "openrouter" | "siliconflow" | "moonshot" | "moonshotai" | "openai-compatible" => {
                Some(10)
            }
            _ => None,
        },
    }
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
            .image_model_family_with_ctx(&self.model_id, &ctx)
            .await?;

        model.generate(request).await
    }

    fn max_images_per_call(&self) -> Option<u32> {
        image_model_handle_max_images_per_call(&self.provider_id, &self.model_id)
    }
}

#[async_trait::async_trait]
impl ImageExtras for ImageModelHandle {
    async fn edit_image(
        &self,
        request: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
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
        let client = self
            .factory
            .compat_image_client_with_ctx(&self.model_id, &ctx)
            .await?;

        let image_client = client.as_image_extras().ok_or_else(|| {
            LlmError::UnsupportedOperation("Provider does not support image extras".to_string())
        })?;

        image_client.edit_image(request).await
    }

    async fn create_variation(
        &self,
        request: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
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
        let client = self
            .factory
            .compat_image_client_with_ctx(&self.model_id, &ctx)
            .await?;

        let image_client = client.as_image_extras().ok_or_else(|| {
            LlmError::UnsupportedOperation("Provider does not support image extras".to_string())
        })?;

        image_client.create_variation(request).await
    }

    fn get_supported_sizes(&self) -> Vec<String> {
        vec![
            "1024x1024".to_string(),
            "512x512".to_string(),
            "256x256".to_string(),
        ]
    }

    fn get_supported_formats(&self) -> Vec<String> {
        vec!["url".to_string(), "b64_json".to_string()]
    }

    fn supports_image_editing(&self) -> bool {
        false
    }

    fn supports_image_variations(&self) -> bool {
        false
    }
}

impl crate::traits::ModelMetadata for ImageModelHandle {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}
