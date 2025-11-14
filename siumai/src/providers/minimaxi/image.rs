//! MiniMaxi Image Generation Helper Functions
//!
//! Internal helper functions for image generation capability implementation.

use crate::core::ProviderContext;
use crate::error::LlmError;
use crate::execution::executors::image::{HttpImageExecutor, ImageExecutorBuilder};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::transformers::request::ImageHttpBody;
use crate::providers::minimaxi::spec::MinimaxiSpec;
use crate::retry_api::RetryOptions;
use std::sync::Arc;

/// Build image executor for MiniMaxi
pub(super) fn build_image_executor(
    api_key: &str,
    base_url: &str,
    http_client: &reqwest::Client,
    retry_options: Option<&RetryOptions>,
    http_interceptors: &[Arc<dyn HttpInterceptor>],
) -> Arc<HttpImageExecutor> {
    // MiniMaxi image API uses OpenAI-style Bearer token authentication
    // We need to inject the Authorization header into http_extra_headers
    let extra_headers = super::utils::create_openai_auth_headers(api_key);

    let ctx = ProviderContext {
        provider_id: "minimaxi".to_string(),
        api_key: Some(api_key.to_string()),
        base_url: base_url.to_string(),
        http_extra_headers: extra_headers,
        organization: None,
        project: None,
        extras: Default::default(),
    };

    let spec = Arc::new(MinimaxiSpec::new());

    let mut builder = ImageExecutorBuilder::new("minimaxi", http_client.clone())
        .with_spec(spec)
        .with_context(ctx);

    // Use OpenAI image standard with MiniMaxi adapter. When the external
    // std-openai crate is enabled, we need to bridge from core image
    // transformers into the aggregator Request/ResponseTransformer traits.
    let image_standard = super::transformers::image::create_minimaxi_image_standard();

    #[cfg(feature = "std-openai-external")]
    {
        // Core-image -> aggregator bridge
        use siumai_core::execution::image::{
            ImageRequestTransformer as CoreImageRequestTransformer,
            ImageResponseTransformer as CoreImageResponseTransformer,
        };

        struct ImageRequestBridge(Arc<dyn CoreImageRequestTransformer>);
        impl crate::execution::transformers::request::RequestTransformer for ImageRequestBridge {
            fn provider_id(&self) -> &str {
                self.0.provider_id()
            }

            fn transform_chat(
                &self,
                _req: &crate::types::ChatRequest,
            ) -> Result<serde_json::Value, LlmError> {
                Err(LlmError::UnsupportedOperation(
                    "MiniMaxi image transformer does not handle chat".to_string(),
                ))
            }

            fn transform_image(
                &self,
                req: &crate::types::ImageGenerationRequest,
            ) -> Result<serde_json::Value, LlmError> {
                self.0.transform_image(req)
            }

            fn transform_image_edit(
                &self,
                req: &crate::types::ImageEditRequest,
            ) -> Result<ImageHttpBody, LlmError> {
                let body = self.0.transform_image_edit(req)?;
                Ok(match body {
                    siumai_core::execution::image::ImageHttpBody::Json(v) => ImageHttpBody::Json(v),
                    siumai_core::execution::image::ImageHttpBody::Multipart(f) => {
                        ImageHttpBody::Multipart(f)
                    }
                })
            }

            fn transform_image_variation(
                &self,
                req: &crate::types::ImageVariationRequest,
            ) -> Result<ImageHttpBody, LlmError> {
                let body = self.0.transform_image_variation(req)?;
                Ok(match body {
                    siumai_core::execution::image::ImageHttpBody::Json(v) => ImageHttpBody::Json(v),
                    siumai_core::execution::image::ImageHttpBody::Multipart(f) => {
                        ImageHttpBody::Multipart(f)
                    }
                })
            }
        }

        struct ImageResponseBridge(Arc<dyn CoreImageResponseTransformer>);
        impl crate::execution::transformers::response::ResponseTransformer for ImageResponseBridge {
            fn provider_id(&self) -> &str {
                self.0.provider_id()
            }

            fn transform_image_response(
                &self,
                raw: &serde_json::Value,
            ) -> Result<crate::types::ImageGenerationResponse, LlmError> {
                self.0.transform_image_response(raw)
            }
        }

        let transformers = image_standard.create_transformers("minimaxi");
        builder = builder.with_transformers(
            Arc::new(ImageRequestBridge(transformers.request)),
            Arc::new(ImageResponseBridge(transformers.response)),
        );
    }

    #[cfg(not(feature = "std-openai-external"))]
    {
        // Aggregator OpenAI image standard already returns Request/ResponseTransformer,
        // so we can plug them in directly without additional bridging.
        let transformers = image_standard.create_transformers("minimaxi");
        builder = builder.with_transformers(transformers.request, transformers.response);
    }

    if !http_interceptors.is_empty() {
        builder = builder.with_interceptors(http_interceptors.to_vec());
    }

    if let Some(retry) = retry_options {
        builder = builder.with_retry_options(retry.clone());
    }

    builder.build()
}
