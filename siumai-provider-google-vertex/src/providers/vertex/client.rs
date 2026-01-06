//! Google Vertex AI client (minimal).
//!
//! This client currently focuses on Vertex Imagen via the `:predict` endpoint.

use async_trait::async_trait;
use reqwest::Client as HttpClient;
use std::borrow::Cow;
use std::sync::Arc;

use crate::auth::TokenProvider;
use crate::client::LlmClient;
use crate::error::LlmError;
use crate::execution::executors::image::{ImageExecutor, ImageExecutorBuilder};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::retry_api::RetryOptions;
use crate::streaming::ChatStream;
use crate::traits::{ChatCapability, ImageExtras, ImageGenerationCapability, ProviderCapabilities};
use crate::types::{
    ChatMessage, ChatResponse, ImageEditRequest, ImageGenerationRequest, ImageGenerationResponse,
    Tool,
};

/// Minimal config for Google Vertex client (delegate auth to HttpConfig headers / token providers).
#[derive(Clone)]
pub struct GoogleVertexConfig {
    /// Vertex base URL, typically:
    /// `https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google`
    pub base_url: String,
    /// Default model id (e.g., `imagen-3.0-generate-002`).
    pub model: String,
    /// Per-request HTTP config (headers, timeouts, etc.).
    pub http_config: crate::types::HttpConfig,
    /// Optional Bearer token provider (e.g., ADC). When present, an `Authorization` header
    /// will be injected automatically if one is not already set.
    pub token_provider: Option<Arc<dyn TokenProvider>>,
}

impl std::fmt::Debug for GoogleVertexConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("GoogleVertexConfig");
        ds.field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("http_config", &self.http_config);

        if self.token_provider.is_some() {
            ds.field("has_token_provider", &true);
        }

        ds.finish()
    }
}

#[derive(Clone)]
pub struct GoogleVertexClient {
    http_client: HttpClient,
    config: GoogleVertexConfig,
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    retry_options: Option<RetryOptions>,
}

impl GoogleVertexClient {
    pub fn new(config: GoogleVertexConfig, http_client: HttpClient) -> Self {
        Self {
            http_client,
            config,
            http_interceptors: Vec::new(),
            retry_options: None,
        }
    }

    pub fn with_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors;
        self
    }

    pub fn with_retry_options(mut self, retry_options: RetryOptions) -> Self {
        self.retry_options = Some(retry_options);
        self
    }

    fn build_context(&self) -> crate::core::ProviderContext {
        crate::core::ProviderContext::new(
            "vertex",
            self.config.base_url.clone(),
            None,
            self.config.http_config.headers.clone(),
        )
    }

    fn has_auth_header(headers: &std::collections::HashMap<String, String>) -> bool {
        headers
            .keys()
            .any(|k| k.eq_ignore_ascii_case("authorization"))
    }

    async fn inject_auth_header(
        &self,
        req_http: &mut crate::types::HttpConfig,
    ) -> Result<(), LlmError> {
        if Self::has_auth_header(&req_http.headers) {
            return Ok(());
        }
        let Some(tp) = &self.config.token_provider else {
            return Ok(());
        };
        let token = tp.token().await?;
        req_http
            .headers
            .insert("authorization".to_string(), format!("Bearer {token}"));
        Ok(())
    }
}

#[async_trait]
impl ImageGenerationCapability for GoogleVertexClient {
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let mut request = request;
        if request.model.is_none() {
            request.model = Some(self.config.model.clone());
        }
        let mut http_config = self.config.http_config.clone();
        self.inject_auth_header(&mut http_config).await?;
        request.http_config = Some(http_config);

        let ctx = self.build_context();
        let spec = Arc::new(
            crate::standards::vertex_imagen::VertexImagenStandard::new().create_spec("vertex"),
        );

        let builder = ImageExecutorBuilder::new("vertex", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone());

        let exec = if let Some(retry) = self.retry_options.clone() {
            builder
                .with_retry_options(retry)
                .build_for_request(&request)
        } else {
            builder.build_for_request(&request)
        };

        ImageExecutor::execute(&*exec, request).await
    }
}

#[async_trait]
impl ImageExtras for GoogleVertexClient {
    async fn edit_image(
        &self,
        request: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let mut request = request;
        if request.model.is_none() {
            request.model = Some(self.config.model.clone());
        }
        let mut http_config = self.config.http_config.clone();
        self.inject_auth_header(&mut http_config).await?;
        request.http_config = Some(http_config);

        let ctx = self.build_context();
        let spec = Arc::new(
            crate::standards::vertex_imagen::VertexImagenStandard::new().create_spec("vertex"),
        );

        let builder = ImageExecutorBuilder::new("vertex", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone());

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
        _request: crate::types::ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Vertex provider does not support image variations".to_string(),
        ))
    }

    fn get_supported_sizes(&self) -> Vec<String> {
        vec![]
    }

    fn get_supported_formats(&self) -> Vec<String> {
        vec!["b64_json".to_string()]
    }

    fn supports_image_editing(&self) -> bool {
        true
    }

    fn supports_image_variations(&self) -> bool {
        false
    }
}

#[async_trait]
impl ChatCapability for GoogleVertexClient {
    async fn chat_with_tools(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Vertex provider does not support chat".to_string(),
        ))
    }

    async fn chat_stream(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Vertex provider does not support streaming chat".to_string(),
        ))
    }
}

impl LlmClient for GoogleVertexClient {
    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed("vertex")
    }

    fn supported_models(&self) -> Vec<String> {
        vec![self.config.model.clone()]
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_image_generation()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_image_generation_capability(&self) -> Option<&dyn ImageGenerationCapability> {
        Some(self)
    }

    fn as_image_extras(&self) -> Option<&dyn ImageExtras> {
        Some(self)
    }
}
