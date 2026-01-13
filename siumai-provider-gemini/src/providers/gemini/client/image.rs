use super::GeminiClient;
use crate::error::LlmError;
use crate::traits::{ImageExtras, ImageGenerationCapability};
use crate::types::{ImageEditRequest, ImageVariationRequest};
use crate::types::{ImageGenerationRequest, ImageGenerationResponse, Warning};
use async_trait::async_trait;
use std::sync::Arc;

fn imagen_warning_parity(request: &ImageGenerationRequest) -> Vec<Warning> {
    let model = request.model.as_deref().unwrap_or_default().trim();
    if !model.starts_with("imagen-") {
        return Vec::new();
    }

    let mut warnings = Vec::new();

    if request.size.is_some() {
        warnings.push(Warning::unsupported_setting(
            "size",
            Some("This model does not support the `size` option. Use `aspectRatio` instead."),
        ));
    }

    if request.seed.is_some() {
        warnings.push(Warning::unsupported_setting(
            "seed",
            Some("This model does not support the `seed` option through this provider."),
        ));
    }

    warnings
}

fn merge_warnings(
    mut resp: ImageGenerationResponse,
    extra: Vec<Warning>,
) -> ImageGenerationResponse {
    if extra.is_empty() {
        return resp;
    }

    match resp.warnings.as_mut() {
        Some(existing) => existing.extend(extra),
        None => resp.warnings = Some(extra),
    }

    resp
}

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

        let parity_warnings = imagen_warning_parity(&request);

        let ctx = super::super::context::build_context(&self.config).await;
        let spec = Arc::new(crate::providers::gemini::spec::GeminiSpecWithConfig::new(
            self.config.clone(),
        ));

        let builder = ImageExecutorBuilder::new("gemini", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone());
        let builder = if let Some(transport) = self.config.http_transport.clone() {
            builder.with_transport(transport)
        } else {
            builder
        };

        let exec = if let Some(retry) = self.retry_options.clone() {
            builder
                .with_retry_options(retry)
                .build_for_request(&request)
        } else {
            builder.build_for_request(&request)
        };

        let resp = ImageExecutor::execute(&*exec, request).await?;
        Ok(merge_warnings(resp, parity_warnings))
    }
}

#[async_trait]
impl ImageExtras for GeminiClient {
    async fn edit_image(
        &self,
        request: ImageEditRequest,
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

        let builder = ImageExecutorBuilder::new("gemini", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone());
        let builder = if let Some(transport) = self.config.http_transport.clone() {
            builder.with_transport(transport)
        } else {
            builder
        };

        let selector = ImageGenerationRequest {
            model: request.model.clone(),
            ..Default::default()
        };

        let exec = if let Some(retry) = self.retry_options.clone() {
            builder
                .with_retry_options(retry)
                .build_for_request(&selector)
        } else {
            builder.build_for_request(&selector)
        };

        ImageExecutor::execute_edit(&*exec, request).await
    }

    async fn create_variation(
        &self,
        _request: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Gemini provider does not support image variations".to_string(),
        ))
    }

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
        // Gemini generateContent does not support image editing.
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
    fn imagen_warnings_match_vercel_parity() {
        let req = ImageGenerationRequest {
            model: Some("imagen-3.0-generate-002".to_string()),
            size: Some("1024x1024".to_string()),
            seed: Some(123),
            ..Default::default()
        };

        let warnings = imagen_warning_parity(&req);
        assert_eq!(
            warnings,
            vec![
                Warning::unsupported_setting(
                    "size",
                    Some(
                        "This model does not support the `size` option. Use `aspectRatio` instead."
                    )
                ),
                Warning::unsupported_setting(
                    "seed",
                    Some("This model does not support the `seed` option through this provider.")
                )
            ]
        );
    }

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
