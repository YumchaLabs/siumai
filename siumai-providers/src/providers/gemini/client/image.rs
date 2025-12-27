use super::GeminiClient;
use crate::error::LlmError;
use crate::traits::{ImageExtras, ImageGenerationCapability};
use crate::types::{ImageGenerationRequest, ImageGenerationResponse};
use async_trait::async_trait;
use std::sync::Arc;

#[async_trait]
impl ImageGenerationCapability for GeminiClient {
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        use crate::execution::executors::image::{ImageExecutor, ImageExecutorBuilder};
        let mut request = request;
        if request.model.is_none() {
            request.model = Some(self.config.model.clone());
        }
        let ctx = super::super::context::build_context(&self.config).await;
        let spec = Arc::new(crate::providers::gemini::spec::GeminiSpecWithConfig::new(
            self.config.clone(),
        ));

        let exec = ImageExecutorBuilder::new("gemini", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone())
            .build_for_request(&request);

        if let Some(opts) = &self.retry_options {
            let mut opts = opts.clone();
            if opts.provider.is_none() {
                opts.provider = Some(crate::types::ProviderType::Gemini);
            }
            crate::retry_api::retry_with(
                || {
                    let rq = request.clone();
                    let exec = exec.clone();
                    async move { ImageExecutor::execute(&*exec, rq).await }
                },
                opts,
            )
            .await
        } else {
            ImageExecutor::execute(&*exec, request).await
        }
    }
}

impl ImageExtras for GeminiClient {
    fn get_supported_sizes(&self) -> Vec<String> {
        vec![
            "1024x1024".to_string(),
            "768x768".to_string(),
            "512x512".to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ProviderSpec;

    #[test]
    fn spec_with_config_uses_request_model_for_image_url_and_body() {
        let base = crate::providers::gemini::GeminiConfig::default()
            .with_model("chat-model".to_string())
            .with_common_params(crate::types::CommonParams {
                model: "chat-model".to_string(),
                ..Default::default()
            });
        let spec = crate::providers::gemini::spec::GeminiSpecWithConfig::new(base);
        let ctx = crate::core::ProviderContext::new(
            "gemini",
            "https://example/v1beta".to_string(),
            Some("KEY".to_string()),
            std::collections::HashMap::new(),
        );

        let req = ImageGenerationRequest {
            prompt: "hi".to_string(),
            model: Some("image-model".to_string()),
            ..Default::default()
        };

        let url = spec.image_url(&req, &ctx);
        assert!(url.ends_with("models/image-model:generateContent"));

        let bundle = spec.choose_image_transformers(&req, &ctx);
        let body = bundle.request.transform_image(&req).unwrap();
        // Image requests should carry the model into the body (protocol typed request uses cfg.model).
        assert_eq!(body["model"], serde_json::json!("image-model"));
    }
}
