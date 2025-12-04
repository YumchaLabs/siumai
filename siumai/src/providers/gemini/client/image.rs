use super::GeminiClient;
use crate::error::LlmError;
use crate::traits::ImageGenerationCapability;
use crate::types::{ImageGenerationRequest, ImageGenerationResponse};
use async_trait::async_trait;
use std::sync::Arc;

#[async_trait]
impl ImageGenerationCapability for GeminiClient {
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        use crate::core::ProviderContext;
        use crate::execution::executors::image::{HttpImageExecutor, ImageExecutor};
        use secrecy::ExposeSecret;

        // Merge Authorization from token_provider if present
        let mut extra_headers = self.config.http_config.headers.clone();
        if let Some(ref tp) = self.config.token_provider
            && let Ok(tok) = tp.token().await
        {
            extra_headers.insert("Authorization".to_string(), format!("Bearer {tok}"));
        }

        // Use lightweight wrapper spec with explicit URL behavior
        let spec = crate::providers::gemini::spec::create_image_wrapper(
            self.config.base_url.clone(),
            self.config.model.clone(),
        );

        let ctx = ProviderContext::new(
            "gemini",
            self.config.base_url.clone(),
            Some(self.config.api_key.expose_secret().to_string()),
            extra_headers.clone(),
        );

        let req_tx = super::super::transformers::GeminiRequestTransformer {
            config: self.config.clone(),
        };
        let resp_tx = super::super::transformers::GeminiResponseTransformer {
            config: self.config.clone(),
        };

        if let Some(opts) = &self.retry_options {
            let mut opts = opts.clone();
            if opts.provider.is_none() {
                opts.provider = Some(crate::types::ProviderType::Gemini);
            }
            let http = self.http_client.clone();
            let spec_clone = spec.clone();
            let ctx_clone = ctx.clone();
            let config = self.config.clone();
            let interceptors = self.http_interceptors.clone();
            crate::retry_api::retry_with(
                || {
                    let rq = request.clone();
                    let http = http.clone();
                    let spec = spec_clone.clone();
                    let ctx = ctx_clone.clone();
                    let interceptors = interceptors.clone();
                    let req_tx = super::super::transformers::GeminiRequestTransformer {
                        config: config.clone(),
                    };
                    let resp_tx = super::super::transformers::GeminiResponseTransformer {
                        config: config.clone(),
                    };
                    async move {
                        let exec = HttpImageExecutor {
                            provider_id: "gemini".to_string(),
                            http_client: http,
                            request_transformer: Arc::new(req_tx),
                            response_transformer: Arc::new(resp_tx),
                            provider_spec: spec,
                            provider_context: ctx,
                            policy: crate::execution::ExecutionPolicy::new()
                                .with_interceptors(interceptors),
                        };
                        ImageExecutor::execute(&exec, rq).await
                    }
                },
                opts,
            )
            .await
        } else {
            let exec = HttpImageExecutor {
                provider_id: "gemini".to_string(),
                http_client: self.http_client.clone(),
                request_transformer: Arc::new(req_tx),
                response_transformer: Arc::new(resp_tx),
                provider_spec: spec,
                provider_context: ctx,
                policy: crate::execution::ExecutionPolicy::new()
                    .with_interceptors(self.http_interceptors.clone())
                    .with_retry_options(self.retry_options.clone()),
            };
            ImageExecutor::execute(&exec, request).await
        }
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
        false
    }
    fn supports_image_variations(&self) -> bool {
        false
    }
}
