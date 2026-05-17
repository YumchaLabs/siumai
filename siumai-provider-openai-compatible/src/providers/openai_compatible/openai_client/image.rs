use super::{OpenAiCompatibleClient, model_slot_is_missing};
use crate::error::LlmError;
use crate::execution::executors::image::{HttpImageExecutor, ImageExecutor, ImageExecutorBuilder};
use crate::traits::{ImageExtras, ImageGenerationCapability};
use crate::types::{
    ImageEditRequest, ImageGenerationRequest, ImageGenerationResponse, ImageVariationRequest,
};
use async_trait::async_trait;
use std::sync::Arc;

impl OpenAiCompatibleClient {
    fn resolve_image_model_default(&self) -> Option<String> {
        self.resolve_family_model_or_config(super::super::config::get_default_image_model(
            &self.config.provider_id,
        ))
    }

    async fn build_image_executor(
        &self,
        request: &ImageGenerationRequest,
    ) -> Result<Arc<HttpImageExecutor>, LlmError> {
        let ctx = self.build_context().await?;
        let spec = Arc::new(self.compat_spec());
        let mut builder =
            ImageExecutorBuilder::new(self.config.provider_id.clone(), self.http_client.clone())
                .with_spec(spec)
                .with_context(ctx)
                .with_interceptors(self.http_interceptors.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        Ok(builder.build_for_request(request))
    }
}

#[async_trait]
impl ImageGenerationCapability for OpenAiCompatibleClient {
    async fn generate_images(
        &self,
        mut request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        if !self.config.adapter.supports_image_generation() {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not support image generation",
                self.config.provider_id
            )));
        }
        if model_slot_is_missing(request.model.as_deref()) {
            request.model = self.resolve_image_model_default();
        }
        let exec = self.build_image_executor(&request).await?;
        ImageExecutor::execute(&*exec, request).await
    }

    fn max_images_per_call(&self) -> Option<u32> {
        match self.config.provider_id.as_str() {
            "deepinfra" | "fireworks" | "together" | "togetherai" => Some(1),
            _ => Some(10),
        }
    }
}

#[async_trait]
impl ImageExtras for OpenAiCompatibleClient {
    async fn edit_image(
        &self,
        request: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        if !self.config.adapter.supports_image_editing() {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not support image editing",
                self.config.provider_id
            )));
        }

        let exec = self
            .build_image_executor(&ImageGenerationRequest::default())
            .await?;
        ImageExecutor::execute_edit(&*exec, request).await
    }

    async fn create_variation(
        &self,
        request: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        if !self.config.adapter.supports_image_variations() {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not support image variations",
                self.config.provider_id
            )));
        }

        let exec = self
            .build_image_executor(&ImageGenerationRequest::default())
            .await?;
        ImageExecutor::execute_variation(&*exec, request).await
    }

    fn get_supported_sizes(&self) -> Vec<String> {
        self.config.adapter.get_supported_image_sizes()
    }

    fn get_supported_formats(&self) -> Vec<String> {
        self.config.adapter.get_supported_image_formats()
    }

    fn supports_image_editing(&self) -> bool {
        self.config.adapter.supports_image_editing()
    }

    fn supports_image_variations(&self) -> bool {
        self.config.adapter.supports_image_variations()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::transport::{
        HttpTransport, HttpTransportMultipartRequest, HttpTransportRequest, HttpTransportResponse,
    };
    use crate::providers::openai_compatible::OpenAiCompatibleConfig;
    use crate::standards::openai::compat::provider_registry::{
        ConfigurableAdapter, ProviderConfig, ProviderFieldMappings,
    };
    use crate::types::{ImageEditInput, ProviderOptionsMap};
    use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
    use std::sync::{Arc, Mutex};

    fn normalize_multipart_body(req: &HttpTransportMultipartRequest) -> String {
        let mut body = String::from_utf8_lossy(&req.body).into_owned();
        if let Some(content_type) = req.headers.get(CONTENT_TYPE).and_then(|v| v.to_str().ok())
            && let Some(boundary) = content_type.split("boundary=").nth(1)
        {
            body = body.replace(boundary.trim(), "<BOUNDARY>");
        }
        body
    }

    #[derive(Clone, Default)]
    struct CaptureTransport {
        last: Arc<Mutex<Option<HttpTransportRequest>>>,
    }

    impl CaptureTransport {
        fn take(&self) -> Option<HttpTransportRequest> {
            self.last.lock().unwrap().take()
        }
    }

    #[async_trait]
    impl HttpTransport for CaptureTransport {
        async fn execute_json(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last.lock().unwrap() = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportResponse {
                status: 401,
                headers,
                body: br#"{"error":{"message":"unauthorized","type":"auth_error","code":"unauthorized"}}"#
                    .to_vec(),
            })
        }
    }

    #[derive(Clone)]
    struct JsonResponseTransport {
        response_body: Arc<Vec<u8>>,
        last: Arc<Mutex<Option<HttpTransportRequest>>>,
    }

    impl JsonResponseTransport {
        fn new(response: serde_json::Value) -> Self {
            Self {
                response_body: Arc::new(serde_json::to_vec(&response).expect("response json")),
                last: Arc::new(Mutex::new(None)),
            }
        }

        fn take(&self) -> Option<HttpTransportRequest> {
            self.last.lock().unwrap().take()
        }
    }

    #[async_trait]
    impl HttpTransport for JsonResponseTransport {
        async fn execute_json(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last.lock().unwrap() = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportResponse {
                status: 200,
                headers,
                body: self.response_body.as_ref().clone(),
            })
        }
    }

    #[derive(Clone)]
    struct MultipartResponseTransport {
        response_body: Arc<Vec<u8>>,
        last: Arc<Mutex<Option<HttpTransportMultipartRequest>>>,
    }

    impl MultipartResponseTransport {
        fn new(response: serde_json::Value) -> Self {
            Self {
                response_body: Arc::new(serde_json::to_vec(&response).expect("response json")),
                last: Arc::new(Mutex::new(None)),
            }
        }

        fn take(&self) -> Option<HttpTransportMultipartRequest> {
            self.last.lock().unwrap().take()
        }
    }

    #[async_trait]
    impl HttpTransport for MultipartResponseTransport {
        async fn execute_json(
            &self,
            _request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportResponse {
                status: 501,
                headers,
                body: br#"{"error":{"message":"json unsupported in test","type":"test_error","code":"unsupported"}}"#
                    .to_vec(),
            })
        }

        async fn execute_multipart(
            &self,
            request: HttpTransportMultipartRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last.lock().unwrap() = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportResponse {
                status: 200,
                headers,
                body: self.response_body.as_ref().clone(),
            })
        }
    }

    fn make_together_image_adapter() -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "together".to_string(),
            name: "Together AI".to_string(),
            base_url: "https://api.together.xyz/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["image_generation".to_string()],
            default_model: Some("black-forest-labs/FLUX.1-schnell".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }))
    }

    fn make_siliconflow_image_adapter() -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "siliconflow".to_string(),
            name: "SiliconFlow".to_string(),
            base_url: "https://api.siliconflow.cn/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["image_generation".to_string()],
            default_model: Some("stability-ai/sdxl".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }))
    }

    #[tokio::test]
    async fn generate_images_runtime_together_preserves_request_shape_at_transport_boundary() {
        let transport = CaptureTransport::default();
        let cfg = OpenAiCompatibleConfig::new(
            "together",
            "test-key",
            "https://api.together.xyz/v1",
            make_together_image_adapter(),
        )
        .with_model("black-forest-labs/FLUX.1-schnell")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ImageGenerationRequest {
            prompt: "a tiny purple robot".to_string(),
            negative_prompt: Some("blurry".to_string()),
            size: Some("1024x1024".to_string()),
            aspect_ratio: None,
            count: 1,
            model: Some("black-forest-labs/FLUX.1-schnell".to_string()),
            quality: None,
            style: None,
            seed: Some(7),
            steps: None,
            guidance_scale: None,
            enhance_prompt: None,
            response_format: Some("url".to_string()),
            extra_params: Default::default(),
            provider_options_map: Default::default(),
            http_config: None,
        };

        let _ = client.generate_images(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(
            captured.url,
            "https://api.together.xyz/v1/images/generations"
        );
        assert_eq!(
            captured.body["model"],
            serde_json::json!("black-forest-labs/FLUX.1-schnell")
        );
        assert_eq!(
            captured.body["prompt"],
            serde_json::json!("a tiny purple robot")
        );
        assert_eq!(captured.body["width"], serde_json::json!(1024));
        assert_eq!(captured.body["height"], serde_json::json!(1024));
        assert_eq!(captured.body["seed"], serde_json::json!(7));
        assert_eq!(
            captured.body["response_format"],
            serde_json::json!("base64")
        );
        assert!(captured.body.get("size").is_none());
        assert!(captured.body.get("n").is_none());
    }

    #[tokio::test]
    async fn generate_images_runtime_together_missing_model_uses_image_family_default() {
        let transport = CaptureTransport::default();
        let cfg = OpenAiCompatibleConfig::new(
            "together",
            "test-key",
            "https://api.together.xyz/v1",
            make_together_image_adapter(),
        )
        .with_model("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ImageGenerationRequest {
            prompt: "a tiny purple robot".to_string(),
            negative_prompt: Some("blurry".to_string()),
            size: Some("1024x1024".to_string()),
            aspect_ratio: None,
            count: 1,
            model: None,
            quality: None,
            style: None,
            seed: None,
            steps: None,
            guidance_scale: None,
            enhance_prompt: None,
            response_format: Some("url".to_string()),
            extra_params: Default::default(),
            provider_options_map: Default::default(),
            http_config: None,
        };

        let _ = client.generate_images(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(
            captured.body["model"],
            serde_json::json!("black-forest-labs/FLUX.1-schnell")
        );
    }

    #[tokio::test]
    async fn generate_images_runtime_compat_image_provider_options_map_to_request_body() {
        let transport = JsonResponseTransport::new(serde_json::json!({
            "data": [{ "b64_json": "image-1" }]
        }));
        let cfg = OpenAiCompatibleConfig::new(
            "together",
            "test-key",
            "https://api.together.xyz/v1",
            make_together_image_adapter(),
        )
        .with_model("black-forest-labs/FLUX.1-schnell")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ImageGenerationRequest {
            prompt: "a tiny teal robot".to_string(),
            model: Some("black-forest-labs/FLUX.1-schnell".to_string()),
            provider_options_map: {
                let mut map = ProviderOptionsMap::default();
                map.insert(
                    "openaiCompatible",
                    serde_json::json!({ "user": "compat-user" }),
                );
                map.insert(
                    "together",
                    serde_json::json!({ "quality": "hd", "user": "provider-user" }),
                );
                map
            },
            ..Default::default()
        };

        let response = client
            .generate_images(request)
            .await
            .expect("image response");
        let captured = transport.take().expect("captured request");

        assert_eq!(response.images.len(), 1);
        assert_eq!(captured.body["user"], serde_json::json!("provider-user"));
        assert_eq!(captured.body["quality"], serde_json::json!("hd"));
        assert_eq!(
            captured.body["response_format"],
            serde_json::json!("base64")
        );
    }

    #[tokio::test]
    async fn generate_images_runtime_together_seed_is_sent_without_warning() {
        let transport = JsonResponseTransport::new(serde_json::json!({
            "data": [{ "b64_json": "image-1" }]
        }));
        let cfg = OpenAiCompatibleConfig::new(
            "together",
            "test-key",
            "https://api.together.xyz/v1",
            make_together_image_adapter(),
        )
        .with_model("black-forest-labs/FLUX.1-schnell")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ImageGenerationRequest {
            prompt: "a tiny green robot".to_string(),
            model: Some("black-forest-labs/FLUX.1-schnell".to_string()),
            seed: Some(7),
            ..Default::default()
        };

        let response = client
            .generate_images(request)
            .await
            .expect("image response");
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.body["seed"], serde_json::json!(7));
        assert!(response.warnings.is_none());
    }

    #[tokio::test]
    async fn edit_image_runtime_together_materializes_data_url_inputs_before_multipart_transform() {
        let transport = MultipartResponseTransport::new(serde_json::json!({
            "created": 123,
            "data": [{ "b64_json": "image-1" }]
        }));
        let cfg = OpenAiCompatibleConfig::new(
            "together",
            "test-key",
            "https://api.together.xyz/v1",
            make_together_image_adapter(),
        )
        .with_model("black-forest-labs/FLUX.1-schnell")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ImageEditRequest {
            images: vec![ImageEditInput::url("data:image/png;base64,aW1hZ2Utb25l")],
            mask: Some(ImageEditInput::url("data:image/png;base64,bWFzay1vbmU=")),
            prompt: "replace the background with a neon skyline".to_string(),
            model: Some("black-forest-labs/FLUX.1-schnell".to_string()),
            count: Some(1),
            size: Some("1024x1024".to_string()),
            aspect_ratio: None,
            seed: None,
            response_format: Some("b64_json".to_string()),
            extra_params: Default::default(),
            provider_options_map: Default::default(),
            http_config: None,
        };

        let response = client
            .edit_image(request)
            .await
            .expect("image edit response");
        assert_eq!(response.images[0].b64_json.as_deref(), Some("image-1"));

        let captured = transport.take().expect("captured multipart request");
        assert_eq!(captured.url, "https://api.together.xyz/v1/images/edits");
        assert!(
            captured
                .headers
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("multipart/form-data; boundary="))
        );

        let body_text = normalize_multipart_body(&captured);
        assert!(body_text.contains("name=\"model\""));
        assert!(body_text.contains("black-forest-labs/FLUX.1-schnell"));
        assert!(body_text.contains("name=\"prompt\""));
        assert!(body_text.contains("replace the background with a neon skyline"));
        assert!(body_text.contains("name=\"response_format\""));
        assert!(body_text.contains("b64_json"));
        assert!(body_text.contains("name=\"image\""));
        assert!(body_text.contains("filename=\"image-0\""));
        assert!(body_text.contains("name=\"mask\""));
        assert!(body_text.contains("filename=\"mask\""));
        assert!(body_text.contains("Content-Type: image/png"));
        assert!(body_text.contains("image-one"));
        assert!(body_text.contains("mask-one"));
    }

    #[tokio::test]
    async fn generate_images_runtime_siliconflow_preserves_request_shape_at_transport_boundary() {
        let transport = CaptureTransport::default();
        let cfg = OpenAiCompatibleConfig::new(
            "siliconflow",
            "test-key",
            "https://api.siliconflow.cn/v1",
            make_siliconflow_image_adapter(),
        )
        .with_model("stability-ai/sdxl")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ImageGenerationRequest {
            prompt: "a tiny orange robot".to_string(),
            negative_prompt: Some("blurry".to_string()),
            size: Some("1024x1024".to_string()),
            aspect_ratio: None,
            count: 1,
            model: Some("stability-ai/sdxl".to_string()),
            quality: None,
            style: None,
            seed: None,
            steps: None,
            guidance_scale: None,
            enhance_prompt: None,
            response_format: Some("url".to_string()),
            extra_params: Default::default(),
            provider_options_map: Default::default(),
            http_config: None,
        };

        let _ = client.generate_images(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(
            captured.url,
            "https://api.siliconflow.cn/v1/images/generations"
        );
        assert_eq!(
            captured.body["model"],
            serde_json::json!("stability-ai/sdxl")
        );
        assert_eq!(
            captured.body["prompt"],
            serde_json::json!("a tiny orange robot")
        );
        assert_eq!(captured.body["size"], serde_json::json!("1024x1024"));
        assert_eq!(captured.body["n"], serde_json::json!(1));
        assert_eq!(captured.body["response_format"], serde_json::json!("url"));
    }

    #[test]
    fn image_logic_stays_out_of_monolithic_client_module() {
        let source = include_str!("../openai_client.rs");
        for forbidden in [
            "fn resolve_image_model_default(",
            "fn build_image_executor(",
            "impl ImageGenerationCapability for OpenAiCompatibleClient",
            "impl ImageExtras for OpenAiCompatibleClient",
        ] {
            assert!(
                !source.contains(forbidden),
                "OpenAI-compatible image logic should live in openai_client/image.rs"
            );
        }
    }
}
