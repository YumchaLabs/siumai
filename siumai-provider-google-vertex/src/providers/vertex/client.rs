//! Google Vertex AI client.
//!
//! This client owns the provider-specific Vertex runtime for:
//! - Gemini chat via `generateContent`
//! - Vertex text embeddings via `:predict`
//! - Gemini image generation/edit/variation via `generateContent`
//! - Imagen image generation/editing via `:predict`
//! - Veo video task creation/status via `:predictLongRunning` and `:fetchPredictOperation`

use async_trait::async_trait;
use reqwest::Client as HttpClient;
use siumai_protocol_gemini::standards::gemini::types::{GeminiConfig, SharedIdGenerator};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use crate::auth::TokenProvider;
use crate::client::LlmClient;
use crate::core::ProviderSpec;
use crate::error::LlmError;
use crate::execution::executors::image::{ImageExecutor, ImageExecutorBuilder};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::retry_api::RetryOptions;
use crate::streaming::ChatStream;
use crate::traits::{
    ChatCapability, EmbeddingCapability, EmbeddingExtensions, ImageExtras,
    ImageGenerationCapability, ProviderCapabilities, VideoGenerationCapability,
};
use crate::types::{
    BatchEmbeddingRequest, BatchEmbeddingResponse, ChatMessage, ChatRequest, ChatResponse,
    EmbeddingRequest, EmbeddingResponse, HttpConfig, ImageEditRequest, ImageGenerationRequest,
    ImageGenerationResponse, Tool,
    video::{VideoGenerationRequest, VideoGenerationResponse, VideoTaskStatusResponse},
};

fn effective_embedding_model(request: &EmbeddingRequest, default_model: &str) -> String {
    request
        .model
        .clone()
        .filter(|model| !model.trim().is_empty())
        .unwrap_or_else(|| default_model.to_string())
}

fn http_configs_match(left: Option<&HttpConfig>, right: Option<&HttpConfig>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => {
            left.timeout == right.timeout
                && left.connect_timeout == right.connect_timeout
                && left.headers == right.headers
                && left.proxy == right.proxy
                && left.user_agent == right.user_agent
                && left.stream_disable_compression == right.stream_disable_compression
        }
        _ => false,
    }
}

fn requests_can_be_coalesced(
    baseline: &EmbeddingRequest,
    candidate: &EmbeddingRequest,
    default_model: &str,
) -> bool {
    !baseline.input.is_empty()
        && !candidate.input.is_empty()
        && effective_embedding_model(baseline, default_model)
            == effective_embedding_model(candidate, default_model)
        && baseline.dimensions == candidate.dimensions
        && baseline.encoding_format == candidate.encoding_format
        && baseline.user == candidate.user
        && baseline.task_type == candidate.task_type
        && baseline.title == candidate.title
        && baseline.provider_options_map.0 == candidate.provider_options_map.0
        && http_configs_match(
            baseline.http_config.as_ref(),
            candidate.http_config.as_ref(),
        )
}

fn vertex_image_max_images_per_call(model_id: &str, base_url: &str) -> u32 {
    let normalized = normalize_vertex_model_id(model_id);
    if normalized.starts_with("gemini-") {
        10
    } else if crate::standards::vertex_imagen::is_vertex_imagen_model(&normalized, base_url) {
        4
    } else {
        4
    }
}

fn normalize_vertex_model_id(model: &str) -> String {
    let trimmed = model.trim().trim_matches('/');
    if trimmed.is_empty() {
        return String::new();
    }
    if let Some(pos) = trimmed.rfind("/models/") {
        return trimmed[(pos + "/models/".len())..].to_string();
    }
    if let Some(rest) = trimmed.strip_prefix("models/") {
        return rest.to_string();
    }
    trimmed.to_string()
}

fn is_vertex_gemini_image_model(model_id: &str) -> bool {
    normalize_vertex_model_id(model_id).starts_with("gemini-")
}

fn coalesce_batch_requests(
    requests: &[EmbeddingRequest],
    default_model: &str,
    max_values_per_call: usize,
) -> Option<(EmbeddingRequest, Vec<usize>)> {
    let baseline = requests.first()?;
    if requests
        .iter()
        .skip(1)
        .any(|request| !requests_can_be_coalesced(baseline, request, default_model))
    {
        return None;
    }

    let lengths: Vec<usize> = requests.iter().map(|request| request.input.len()).collect();
    let total_inputs: usize = lengths.iter().sum();
    if total_inputs > max_values_per_call {
        return None;
    }

    let mut merged = baseline.clone();
    merged.model = Some(effective_embedding_model(baseline, default_model));
    merged.input = requests
        .iter()
        .flat_map(|request| request.input.iter().cloned())
        .collect();
    Some((merged, lengths))
}

fn split_coalesced_response(
    response: EmbeddingResponse,
    lengths: &[usize],
) -> Result<BatchEmbeddingResponse, LlmError> {
    let total_inputs: usize = lengths.iter().sum();
    if total_inputs != response.embeddings.len() {
        return Err(LlmError::ParseError(format!(
            "Vertex batch embedding returned {} vectors for {} flattened inputs",
            response.embeddings.len(),
            total_inputs
        )));
    }

    let mut index = 0usize;
    let mut responses = Vec::with_capacity(lengths.len());
    for len in lengths {
        let next = index + len;
        responses.push(Ok(EmbeddingResponse {
            embeddings: response.embeddings[index..next].to_vec(),
            model: response.model.clone(),
            usage: None,
            metadata: response.metadata.clone(),
            response: None,
        }));
        index = next;
    }

    let mut metadata = HashMap::new();
    metadata.insert("coalesced".to_string(), serde_json::Value::Bool(true));
    if let Some(usage) = response.usage {
        metadata.insert(
            "aggregated_usage".to_string(),
            serde_json::json!({
                "prompt_tokens": usage.prompt_tokens,
                "total_tokens": usage.total_tokens,
            }),
        );
    }
    if let Some(http_response) = response.response {
        metadata.insert(
            "response".to_string(),
            serde_json::to_value(http_response).map_err(|error| {
                LlmError::ParseError(format!(
                    "Serialize Vertex batch response envelope failed: {error}"
                ))
            })?,
        );
    }

    Ok(BatchEmbeddingResponse {
        responses,
        metadata,
    })
}

/// Minimal config for Google Vertex client (delegate auth to HttpConfig headers / token providers).
#[derive(Clone)]
pub struct GoogleVertexConfig {
    /// Vertex base URL, typically:
    /// `https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google`
    pub base_url: String,
    /// Default model id (e.g., `imagen-3.0-generate-002`).
    pub model: String,
    /// Shared request defaults used by chat/embedding entry points.
    pub common_params: crate::types::CommonParams,
    /// Optional API key (express mode). When set and no `Authorization` header is present,
    /// it will be passed as the `key` query parameter.
    pub api_key: Option<String>,
    /// Per-request HTTP config (headers, timeouts, etc.).
    pub http_config: crate::types::HttpConfig,
    /// Optional custom HTTP transport (Vercel-style "custom fetch" parity).
    pub http_transport:
        Option<std::sync::Arc<dyn crate::execution::http::transport::HttpTransport>>,
    /// Optional custom stable ID generator aligned with AI SDK `generateId`.
    pub generate_id: Option<SharedIdGenerator>,
    /// Optional Bearer token provider (e.g., ADC). When present, an `Authorization` header
    /// will be injected automatically if one is not already set.
    pub token_provider: Option<Arc<dyn TokenProvider>>,
    /// Optional HTTP interceptors applied to all requests built from this config.
    pub http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional model-level middlewares applied before provider mapping (chat only).
    pub model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
}

impl std::fmt::Debug for GoogleVertexConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("GoogleVertexConfig");
        ds.field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("common_params", &self.common_params)
            .field("http_config", &self.http_config);

        if self.api_key.is_some() {
            ds.field("has_api_key", &true);
        }

        if self.token_provider.is_some() {
            ds.field("has_token_provider", &true);
        }

        ds.finish()
    }
}

impl GoogleVertexConfig {
    /// Create a config from an explicit Vertex base URL and model id.
    pub fn new<B: Into<String>, M: Into<String>>(base_url: B, model: M) -> Self {
        let model = model.into();
        Self {
            base_url: base_url.into(),
            model: model.clone(),
            common_params: crate::types::CommonParams {
                model,
                ..Default::default()
            },
            api_key: None,
            http_config: crate::types::HttpConfig::default(),
            http_transport: None,
            generate_id: None,
            token_provider: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
        }
    }

    /// Create an express-mode config that uses the shared Google Vertex express base URL.
    pub fn express<S: Into<String>, M: Into<String>>(api_key: S, model: M) -> Self {
        Self::new(crate::auth::vertex::GOOGLE_VERTEX_EXPRESS_BASE_URL, model).with_api_key(api_key)
    }

    /// Create an enterprise-mode config from project + location.
    pub fn enterprise<P: AsRef<str>, L: AsRef<str>, M: Into<String>>(
        project: P,
        location: L,
        model: M,
    ) -> Self {
        Self::new(
            crate::auth::vertex::google_vertex_base_url(
                project.as_ref().trim(),
                location.as_ref().trim(),
            ),
            model,
        )
    }

    pub fn with_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        let model = model.into();
        self.common_params.model = model.clone();
        self.model = model;
        self
    }

    pub fn with_common_params(mut self, common_params: crate::types::CommonParams) -> Self {
        self.model = common_params.model.clone();
        self.common_params = common_params;
        self
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.common_params.temperature = Some(temperature);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.common_params.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.common_params.top_p = Some(top_p);
        self
    }

    pub fn with_stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.common_params.stop_sequences = Some(stop_sequences);
        self
    }

    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        let api_key = api_key.into();
        let trimmed = api_key.trim().to_string();
        self.api_key = (!trimmed.is_empty()).then_some(trimmed);
        self
    }

    pub fn with_http_config(mut self, http_config: crate::types::HttpConfig) -> Self {
        self.http_config = http_config;
        self
    }

    pub fn with_http_transport(
        mut self,
        transport: std::sync::Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.http_transport = Some(transport);
        self
    }

    pub fn with_generate_id<F>(mut self, generate_id: F) -> Self
    where
        F: Fn() -> String + Send + Sync + 'static,
    {
        self.generate_id = Some(Arc::new(generate_id));
        self
    }

    pub fn with_shared_generate_id(mut self, generate_id: SharedIdGenerator) -> Self {
        self.generate_id = Some(generate_id);
        self
    }

    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http_config.timeout = Some(timeout);
        self
    }

    pub fn with_connect_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http_config.connect_timeout = Some(timeout);
        self
    }

    pub fn with_http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.http_config.stream_disable_compression = disable;
        self
    }

    pub fn with_token_provider(mut self, token_provider: Arc<dyn TokenProvider>) -> Self {
        self.token_provider = Some(token_provider);
        self
    }

    pub fn with_http_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors;
        self
    }

    pub fn with_http_interceptor(mut self, interceptor: Arc<dyn HttpInterceptor>) -> Self {
        self.http_interceptors.push(interceptor);
        self
    }

    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.model_middlewares = middlewares;
        self
    }

    pub fn validate(&self) -> Result<(), LlmError> {
        if self.base_url.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Google Vertex requires a non-empty base_url".to_string(),
            ));
        }
        if self.model.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Google Vertex requires a non-empty model id".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct GoogleVertexClient {
    http_client: HttpClient,
    config: GoogleVertexConfig,
    common_params: crate::types::CommonParams,
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    retry_options: Option<RetryOptions>,
}

impl GoogleVertexClient {
    /// Construct a `GoogleVertexClient` from a config-first `GoogleVertexConfig`.
    pub fn from_config(config: GoogleVertexConfig) -> Result<Self, LlmError> {
        let http_client =
            crate::execution::http::client::build_http_client_from_config(&config.http_config)?;
        Self::with_http_client(config, http_client)
    }

    /// Construct a `GoogleVertexClient` from a `GoogleVertexConfig` with a caller-supplied HTTP client.
    pub fn with_http_client(
        config: GoogleVertexConfig,
        http_client: HttpClient,
    ) -> Result<Self, LlmError> {
        config.validate()?;

        let http_interceptors = config.http_interceptors.clone();
        let model_middlewares = config.model_middlewares.clone();

        Ok(Self::new(config, http_client)
            .with_interceptors(http_interceptors)
            .with_model_middlewares(model_middlewares))
    }

    pub fn new(config: GoogleVertexConfig, http_client: HttpClient) -> Self {
        let mut common_params = config.common_params.clone();
        if common_params.model.trim().is_empty() {
            common_params.model = config.model.clone();
        }
        Self {
            http_client,
            config,
            common_params,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
            retry_options: None,
        }
    }

    pub fn with_common_params(mut self, common_params: crate::types::CommonParams) -> Self {
        self.common_params = common_params;
        self
    }

    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    pub fn common_params(&self) -> &crate::types::CommonParams {
        &self.common_params
    }

    pub fn with_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors;
        self
    }

    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.model_middlewares = middlewares;
        self
    }

    pub fn with_retry_options(mut self, retry_options: RetryOptions) -> Self {
        self.retry_options = Some(retry_options);
        self
    }

    async fn build_context(&self) -> crate::core::ProviderContext {
        super::context::build_context(&self.config).await
    }

    fn gemini_chat_config(&self) -> GeminiConfig {
        let model = if self.common_params.model.trim().is_empty() {
            self.config.model.clone()
        } else {
            self.common_params.model.clone()
        };
        let mut common_params = self.common_params.clone();
        common_params.model = model.clone();

        let mut gemini_config = GeminiConfig::default()
            .with_base_url(self.config.base_url.clone())
            .with_model(model)
            .with_common_params(common_params);
        gemini_config.provider_metadata_key = Some("vertex".to_string());
        if let Some(generate_id) = self.config.generate_id.clone() {
            gemini_config = gemini_config.with_shared_generate_id(generate_id);
        }
        gemini_config
    }

    fn gemini_image_config(&self) -> GeminiConfig {
        self.gemini_chat_config()
    }

    fn image_spec_for_model(&self, model: &str) -> Arc<dyn ProviderSpec> {
        if is_vertex_gemini_image_model(model) {
            Arc::new(
                crate::standards::vertex_gemini_image::VertexGeminiImageStandard::new()
                    .with_gemini_config(self.gemini_image_config())
                    .create_spec("vertex"),
            )
        } else {
            Arc::new(
                crate::standards::vertex_imagen::VertexImagenStandard::new().create_spec("vertex"),
            )
        }
    }

    fn prepare_chat_request(
        &self,
        mut request: ChatRequest,
        stream: bool,
    ) -> Result<ChatRequest, LlmError> {
        if request.common_params.model.trim().is_empty() {
            request.common_params.model = self.common_params.model.clone();
        }
        if request.common_params.model.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Google Vertex chat request requires a non-empty model id".to_string(),
            ));
        }
        if request.http_config.is_none() {
            request.http_config = Some(self.config.http_config.clone());
        }
        request.stream = stream;
        Ok(request)
    }

    async fn chat_request_via_spec(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        use crate::execution::executors::chat::{ChatExecutor, ChatExecutorBuilder};

        let request = self.prepare_chat_request(request, false)?;
        let ctx = self.build_context().await;
        let spec = Arc::new(
            crate::standards::vertex_generative_ai::VertexGenerativeAiStandard::new()
                .with_gemini_config(self.gemini_chat_config())
                .create_spec("vertex"),
        );
        let bundle = spec.choose_chat_transformers(&request, &ctx);

        let mut exec = ChatExecutorBuilder::new("vertex", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_transformer_bundle(bundle)
            .with_stream_disable_compression(self.config.http_config.stream_disable_compression)
            .with_interceptors(self.http_interceptors.clone())
            .with_middlewares(self.model_middlewares.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            exec = exec.with_transport(transport);
        }
        if let Some(retry) = self.retry_options.clone() {
            exec = exec.with_retry_options(retry);
        }

        let exec = exec.build();
        ChatExecutor::execute(&*exec, request).await
    }

    async fn chat_stream_request_via_spec(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStream, LlmError> {
        use crate::execution::executors::chat::{ChatExecutor, ChatExecutorBuilder};

        let request = self.prepare_chat_request(request, true)?;
        let ctx = self.build_context().await;
        let spec = Arc::new(
            crate::standards::vertex_generative_ai::VertexGenerativeAiStandard::new()
                .with_gemini_config(self.gemini_chat_config())
                .create_spec("vertex"),
        );
        let bundle = spec.choose_chat_transformers(&request, &ctx);

        let mut exec = ChatExecutorBuilder::new("vertex", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_transformer_bundle(bundle)
            .with_stream_disable_compression(self.config.http_config.stream_disable_compression)
            .with_interceptors(self.http_interceptors.clone())
            .with_middlewares(self.model_middlewares.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            exec = exec.with_transport(transport);
        }
        if let Some(retry) = self.retry_options.clone() {
            exec = exec.with_retry_options(retry);
        }

        let exec = exec.build();
        ChatExecutor::execute_stream(&*exec, request).await
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

    #[cfg(test)]
    pub(crate) fn _debug_has_token_provider(&self) -> bool {
        self.config.token_provider.is_some()
    }

    #[cfg(test)]
    pub(crate) fn _debug_has_api_key(&self) -> bool {
        self.config.api_key.is_some()
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

        let ctx = self.build_context().await;
        let spec = self.image_spec_for_model(request.model.as_deref().unwrap_or(""));

        let mut builder = ImageExecutorBuilder::new("vertex", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

        let exec = if let Some(retry) = self.retry_options.clone() {
            builder
                .with_retry_options(retry)
                .build_for_request(&request)
        } else {
            builder.build_for_request(&request)
        };

        ImageExecutor::execute(&*exec, request).await
    }

    fn max_images_per_call(&self) -> Option<u32> {
        Some(vertex_image_max_images_per_call(
            self.common_params.model.as_str(),
            self.config.base_url.as_str(),
        ))
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

        let ctx = self.build_context().await;
        let spec = self.image_spec_for_model(request.model.as_deref().unwrap_or(""));

        let mut builder = ImageExecutorBuilder::new("vertex", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

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
        request: crate::types::ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let mut request = request;
        if request.model.is_none() {
            request.model = Some(self.config.model.clone());
        }
        let mut http_config = self.config.http_config.clone();
        self.inject_auth_header(&mut http_config).await?;
        request.http_config = Some(http_config);

        let ctx = self.build_context().await;
        let spec = self.image_spec_for_model(request.model.as_deref().unwrap_or(""));

        let mut builder = ImageExecutorBuilder::new("vertex", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

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

        ImageExecutor::execute_variation(&*exec, request).await
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
        true
    }
}

#[cfg(feature = "google-vertex")]
#[async_trait]
impl ChatCapability for GoogleVertexClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(self.common_params.clone())
            .http_config(self.config.http_config.clone());
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        self.chat_request_via_spec(builder.build()).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(self.common_params.clone())
            .http_config(self.config.http_config.clone())
            .stream(true);
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        self.chat_stream_request_via_spec(builder.build()).await
    }

    async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.chat_request_via_spec(request).await
    }

    async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        self.chat_stream_request_via_spec(request).await
    }
}

#[async_trait]
impl EmbeddingCapability for GoogleVertexClient {
    async fn embed(&self, input: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        use crate::execution::executors::embedding::{EmbeddingExecutor, EmbeddingExecutorBuilder};

        let req = EmbeddingRequest::new(input).with_model(self.common_params.model.clone());

        let ctx = self.build_context().await;
        let spec = Arc::new(
            crate::standards::vertex_embedding::VertexEmbeddingStandard::new()
                .create_spec("vertex"),
        );

        let mut exec = EmbeddingExecutorBuilder::new("vertex", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            exec = exec.with_transport(transport);
        }

        let exec = if let Some(retry) = self.retry_options.clone() {
            exec.with_retry_options(retry).build_for_request(&req)
        } else {
            exec.build_for_request(&req)
        };

        EmbeddingExecutor::execute(&*exec, req).await
    }

    fn as_embedding_extensions(&self) -> Option<&dyn EmbeddingExtensions> {
        Some(self)
    }

    fn embedding_dimension(&self) -> usize {
        768
    }

    fn max_tokens_per_embedding(&self) -> usize {
        2048
    }

    fn supported_embedding_models(&self) -> Vec<String> {
        vec!["text-embedding-004".to_string()]
    }
}

#[async_trait]
impl EmbeddingExtensions for GoogleVertexClient {
    async fn embed_with_config(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        use crate::execution::executors::embedding::{EmbeddingExecutor, EmbeddingExecutorBuilder};

        let mut request = request;
        if request.model.as_deref().is_none_or(str::is_empty) {
            request.model = Some(self.common_params.model.clone());
        }

        let ctx = self.build_context().await;
        let spec = Arc::new(
            crate::standards::vertex_embedding::VertexEmbeddingStandard::new()
                .create_spec("vertex"),
        );

        let mut exec = EmbeddingExecutorBuilder::new("vertex", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            exec = exec.with_transport(transport);
        }

        let exec = if let Some(retry) = self.retry_options.clone() {
            exec.with_retry_options(retry).build_for_request(&request)
        } else {
            exec.build_for_request(&request)
        };

        EmbeddingExecutor::execute(&*exec, request).await
    }

    async fn embed_batch(
        &self,
        requests: BatchEmbeddingRequest,
    ) -> Result<BatchEmbeddingResponse, LlmError> {
        if requests.requests.is_empty() {
            return Ok(BatchEmbeddingResponse {
                responses: Vec::new(),
                metadata: HashMap::new(),
            });
        }

        if let Some((merged_request, lengths)) =
            coalesce_batch_requests(&requests.requests, &self.common_params.model, 2048)
            && let Ok(response) = self.embed_with_config(merged_request).await
        {
            return split_coalesced_response(response, &lengths);
        }

        let mut responses = Vec::new();
        for request in requests.requests {
            let result = self
                .embed_with_config(request)
                .await
                .map_err(|error| error.to_string());
            responses.push(result);
            if requests.batch_options.fail_fast && responses.last().is_some_and(|r| r.is_err()) {
                break;
            }
        }

        Ok(BatchEmbeddingResponse {
            responses,
            metadata: HashMap::new(),
        })
    }
}

impl LlmClient for GoogleVertexClient {
    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed("vertex")
    }

    fn supported_models(&self) -> Vec<String> {
        super::models::get_default_models()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_embedding()
            .with_image_generation()
            .with_custom_feature("video", true)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_chat_capability(&self) -> Option<&dyn ChatCapability> {
        Some(self)
    }

    fn as_image_generation_capability(&self) -> Option<&dyn ImageGenerationCapability> {
        Some(self)
    }

    fn as_image_extras(&self) -> Option<&dyn ImageExtras> {
        Some(self)
    }

    fn as_embedding_capability(&self) -> Option<&dyn EmbeddingCapability> {
        Some(self)
    }

    fn as_embedding_extensions(&self) -> Option<&dyn EmbeddingExtensions> {
        Some(self)
    }

    fn as_video_generation_capability(&self) -> Option<&dyn VideoGenerationCapability> {
        Some(self)
    }
}

#[async_trait]
impl VideoGenerationCapability for GoogleVertexClient {
    async fn create_video_task(
        &self,
        request: VideoGenerationRequest,
    ) -> Result<VideoGenerationResponse, LlmError> {
        super::video::create_video_task(
            &self.config,
            &self.common_params.model,
            &self.http_client,
            self.retry_options.as_ref(),
            &self.http_interceptors,
            request,
        )
        .await
    }

    async fn query_video_task(&self, task_id: &str) -> Result<VideoTaskStatusResponse, LlmError> {
        super::video::query_video_task(
            &self.config,
            &self.common_params.model,
            &self.http_client,
            self.retry_options.as_ref(),
            &self.http_interceptors,
            task_id,
        )
        .await
    }

    fn polling_options(
        &self,
        request: &VideoGenerationRequest,
    ) -> Result<siumai_core::video::VideoPollingOptions, LlmError> {
        super::video::polling_options(request)
    }

    fn max_videos_per_call(&self) -> Option<u32> {
        Some(4)
    }

    fn get_supported_models(&self) -> Vec<String> {
        super::video::get_supported_video_models()
    }

    fn get_supported_resolutions(&self, model: &str) -> Vec<String> {
        super::video::get_supported_resolutions(model)
    }

    fn get_supported_durations(&self, model: &str) -> Vec<u32> {
        super::video::get_supported_durations(model)
    }
}

impl crate::traits::ModelMetadata for GoogleVertexClient {
    fn provider_id(&self) -> &str {
        "vertex"
    }

    fn model_id(&self) -> &str {
        &self.common_params.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::transport::{
        HttpTransport, HttpTransportRequest, HttpTransportResponse, HttpTransportStreamBody,
        HttpTransportStreamResponse,
    };
    use crate::streaming::LanguageModelV3StreamPart;
    use crate::types::{ChatStreamEvent, ChatStreamPart, FinishReason, ResponseFormat};
    use async_trait::async_trait;
    use futures_util::StreamExt;
    use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
    use std::{
        sync::{Arc, Mutex},
        time::Duration,
    };

    #[derive(Clone, Default)]
    struct CaptureTransport {
        last: Arc<Mutex<Option<HttpTransportRequest>>>,
        last_stream: Arc<Mutex<Option<HttpTransportRequest>>>,
    }

    impl CaptureTransport {
        fn take(&self) -> Option<HttpTransportRequest> {
            self.last.lock().expect("lock").take()
        }

        fn take_stream(&self) -> Option<HttpTransportRequest> {
            self.last_stream.lock().expect("lock").take()
        }
    }

    #[async_trait]
    impl HttpTransport for CaptureTransport {
        async fn execute_json(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last.lock().expect("lock") = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportResponse {
                status: 401,
                headers,
                body: br#"{"error":{"type":"auth_error","message":"unauthorized"}}"#.to_vec(),
            })
        }

        async fn execute_stream(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportStreamResponse, LlmError> {
            *self.last_stream.lock().expect("lock") = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportStreamResponse {
                status: 401,
                headers,
                body: HttpTransportStreamBody::from_bytes(
                    br#"{"error":{"type":"auth_error","message":"unauthorized"}}"#.to_vec(),
                ),
            })
        }
    }

    #[derive(Clone)]
    struct FixtureStreamTransport {
        last_stream: Arc<Mutex<Option<HttpTransportRequest>>>,
        body: Arc<Vec<u8>>,
    }

    impl FixtureStreamTransport {
        fn new(body: Vec<u8>) -> Self {
            Self {
                last_stream: Arc::new(Mutex::new(None)),
                body: Arc::new(body),
            }
        }

        fn take_stream(&self) -> Option<HttpTransportRequest> {
            self.last_stream.lock().expect("lock").take()
        }
    }

    #[async_trait]
    impl HttpTransport for FixtureStreamTransport {
        async fn execute_json(
            &self,
            _request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportResponse {
                status: 501,
                headers,
                body: br#"{"error":{"type":"test_error","message":"json unsupported in test"}}"#
                    .to_vec(),
            })
        }

        async fn execute_stream(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportStreamResponse, LlmError> {
            *self.last_stream.lock().expect("lock") = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(
                CONTENT_TYPE,
                HeaderValue::from_static("text/event-stream; charset=utf-8"),
            );

            Ok(HttpTransportStreamResponse {
                status: 200,
                headers,
                body: HttpTransportStreamBody::from_bytes((*self.body).clone()),
            })
        }
    }

    #[derive(Clone)]
    struct FixtureJsonTransport {
        last: Arc<Mutex<Option<HttpTransportRequest>>>,
        body: Arc<Vec<u8>>,
    }

    impl FixtureJsonTransport {
        fn new(body: Vec<u8>) -> Self {
            Self {
                last: Arc::new(Mutex::new(None)),
                body: Arc::new(body),
            }
        }

        fn take(&self) -> Option<HttpTransportRequest> {
            self.last.lock().expect("lock").take()
        }
    }

    #[async_trait]
    impl HttpTransport for FixtureJsonTransport {
        async fn execute_json(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last.lock().expect("lock") = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportResponse {
                status: 200,
                headers,
                body: (*self.body).clone(),
            })
        }

        async fn execute_stream(
            &self,
            _request: HttpTransportRequest,
        ) -> Result<HttpTransportStreamResponse, LlmError> {
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportStreamResponse {
                status: 501,
                headers,
                body: HttpTransportStreamBody::from_bytes(
                    br#"{"error":{"type":"test_error","message":"stream unsupported in test"}}"#
                        .to_vec(),
                ),
            })
        }
    }

    fn vertex_json_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "value": { "type": "string" }
            },
            "required": ["value"],
            "additionalProperties": false
        })
    }

    fn make_vertex_structured_output_request(model: &str) -> ChatRequest {
        ChatRequest::builder()
            .model(model)
            .messages(vec![ChatMessage::user("hi").build()])
            .response_format(ResponseFormat::json_schema(vertex_json_schema()))
            .build()
            .with_provider_option(
                "vertex",
                serde_json::json!({
                    "structuredOutputs": true
                }),
            )
    }

    fn vertex_sse_body(frames: &[serde_json::Value], include_done: bool) -> Vec<u8> {
        let mut sse = String::new();
        for frame in frames {
            sse.push_str("data: ");
            sse.push_str(
                &serde_json::to_string(frame).expect("serialize vertex stream frame for test"),
            );
            sse.push_str("\n\n");
        }
        if include_done {
            sse.push_str("data: [DONE]\n\n");
        }
        sse.into_bytes()
    }

    fn vertex_structured_output_success_stream_body() -> Vec<u8> {
        vertex_sse_body(
            &[
                serde_json::json!({
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    { "text": "{\"value\":\"te" }
                                ]
                            }
                        }
                    ]
                }),
                serde_json::json!({
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    { "text": "st\"}" }
                                ]
                            }
                        }
                    ]
                }),
                serde_json::json!({
                    "candidates": [
                        {
                            "finishReason": "STOP",
                            "safetyRatings": [
                                {
                                    "category": "HARM_CATEGORY_DEROGATORY",
                                    "probability": "NEGLIGIBLE"
                                }
                            ]
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 7,
                        "candidatesTokenCount": 4,
                        "totalTokenCount": 11
                    }
                }),
            ],
            true,
        )
    }

    fn vertex_structured_output_interrupted_stream_body() -> Vec<u8> {
        vertex_sse_body(
            &[serde_json::json!({
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                { "text": "{\"value\":" }
                            ]
                        }
                    }
                ]
            })],
            false,
        )
    }

    fn vertex_tool_call_and_source_response_body() -> Vec<u8> {
        serde_json::json!({
            "candidates": [
                {
                    "content": {
                        "parts": [
                            { "functionCall": { "name": "weather", "args": { "city": "Tokyo" } } }
                        ]
                    },
                    "groundingMetadata": {
                        "groundingChunks": [
                            { "web": { "uri": "https://example.com", "title": "Example" } }
                        ]
                    },
                    "finishReason": "STOP"
                }
            ],
            "modelVersion": "gemini-2.5-flash"
        })
        .to_string()
        .into_bytes()
    }

    fn vertex_tool_call_and_source_stream_body() -> Vec<u8> {
        vertex_sse_body(
            &[
                serde_json::json!({
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    { "functionCall": { "name": "weather", "args": { "city": "Tokyo" } } }
                                ]
                            },
                            "groundingMetadata": {
                                "groundingChunks": [
                                    { "web": { "uri": "https://example.com", "title": "Example" } }
                                ]
                            }
                        }
                    ]
                }),
                serde_json::json!({
                    "candidates": [
                        {
                            "finishReason": "STOP"
                        }
                    ]
                }),
            ],
            true,
        )
    }

    fn header_value(req: &HttpTransportRequest, name: &str) -> Option<String> {
        req.headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(ToString::to_string)
    }

    fn vertex_gemini_image_response_body() -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": "base64-generated-image"
                                }
                            }
                        ],
                        "role": "model"
                    },
                    "finishReason": "STOP"
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 100,
                "totalTokenCount": 110
            }
        }))
        .expect("serialize vertex gemini image response")
    }

    fn assert_vertex_structured_output_stream_request(req: &HttpTransportRequest) {
        assert!(
            req.url
                .contains("/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key=test-key"),
            "unexpected url: {}",
            req.url
        );
        assert_eq!(
            header_value(req, "accept"),
            Some("text/event-stream".to_string())
        );
        assert_eq!(
            req.body["generationConfig"]["responseMimeType"],
            serde_json::json!("application/json")
        );
        assert!(req.body["generationConfig"].get("responseSchema").is_some());
        assert!(
            req.body["generationConfig"]
                .get("responseJsonSchema")
                .is_none()
        );
    }

    async fn collect_stream_events(mut stream: ChatStream) -> Vec<ChatStreamEvent> {
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            match item {
                Ok(event) => events.push(event),
                Err(err) => panic!("collect vertex client stream event failed: {err:?}"),
            }
        }
        events
    }

    #[test]
    fn google_vertex_llmclient_exposes_expected_capabilities() {
        let cfg = GoogleVertexConfig::new("https://example.invalid", "imagen-3.0-generate-002");
        let client = GoogleVertexClient::from_config(cfg).expect("from_config ok");
        let llm: &dyn crate::client::LlmClient = &client;
        assert_eq!(llm.provider_id(), std::borrow::Cow::Borrowed("vertex"));
        assert!(llm.as_chat_capability().is_some());
        assert!(llm.as_image_generation_capability().is_some());
        assert!(llm.as_embedding_capability().is_some());
        assert_eq!(crate::traits::ModelMetadata::provider_id(&client), "vertex");
        assert_eq!(
            crate::traits::ModelMetadata::model_id(&client),
            "imagen-3.0-generate-002"
        );
    }

    #[test]
    fn google_vertex_config_convenience_constructors_set_expected_defaults() {
        let express = GoogleVertexConfig::express("test-key", "gemini-2.5-pro");
        assert_eq!(
            express.base_url,
            crate::auth::vertex::GOOGLE_VERTEX_EXPRESS_BASE_URL
        );
        assert_eq!(express.api_key.as_deref(), Some("test-key"));

        let enterprise = GoogleVertexConfig::enterprise("project-1", "us-central1", "gemini");
        assert_eq!(
            enterprise.base_url,
            crate::auth::vertex::google_vertex_base_url("project-1", "us-central1")
        );
        assert_eq!(enterprise.api_key, None);
    }

    #[test]
    fn google_vertex_config_http_convenience_helpers() {
        let config = GoogleVertexConfig::new("https://example.invalid", "gemini-2.5-flash")
            .with_timeout(Duration::from_secs(11))
            .with_connect_timeout(Duration::from_secs(3))
            .with_http_stream_disable_compression(true)
            .with_http_interceptor(Arc::new(
                crate::execution::http::interceptor::LoggingInterceptor,
            ));

        assert_eq!(config.http_config.timeout, Some(Duration::from_secs(11)));
        assert_eq!(
            config.http_config.connect_timeout,
            Some(Duration::from_secs(3))
        );
        assert!(config.http_config.stream_disable_compression);
        assert_eq!(config.http_interceptors.len(), 1);
    }

    #[test]
    fn google_vertex_config_preserves_custom_generate_id() {
        let config = GoogleVertexConfig::new("https://example.invalid", "gemini-2.5-flash")
            .with_generate_id(|| "vertex-config-id".to_string());

        assert_eq!(
            config.generate_id.as_ref().map(|generate_id| generate_id()),
            Some("vertex-config-id".to_string())
        );
    }

    #[tokio::test]
    async fn google_vertex_generate_images_routes_gemini_image_models_through_generate_content() {
        let transport = FixtureJsonTransport::new(vertex_gemini_image_response_body());
        let client = GoogleVertexClient::from_config(
            GoogleVertexConfig::new("https://example.invalid", "gemini-2.5-flash-image")
                .with_api_key("test-key")
                .with_http_transport(Arc::new(transport.clone())),
        )
        .expect("from_config ok");

        let response = client
            .generate_images(ImageGenerationRequest {
                prompt: "A beautiful sunset".to_string(),
                aspect_ratio: Some("16:9".to_string()),
                seed: Some(12_345),
                size: Some("1024x1024".to_string()),
                ..Default::default()
            })
            .await
            .expect("vertex gemini image response");

        assert_eq!(response.images.len(), 1);
        assert_eq!(
            response.images[0].b64_json.as_deref(),
            Some("base64-generated-image")
        );
        assert_eq!(
            response.warnings,
            Some(vec![crate::types::Warning::unsupported(
                "size",
                Some("This model does not support the `size` option. Use `aspectRatio` instead."),
            )])
        );

        let request = transport.take().expect("captured image request");
        assert!(
            request
                .url
                .contains("/models/gemini-2.5-flash-image:generateContent?key=test-key"),
            "unexpected url: {}",
            request.url
        );
        assert!(header_value(&request, "x-goog-api-key").is_none());
        assert_eq!(
            request.body["generationConfig"]["responseModalities"],
            serde_json::json!(["IMAGE"])
        );
        assert_eq!(
            request.body["generationConfig"]["imageConfig"]["aspectRatio"],
            serde_json::json!("16:9")
        );
        assert_eq!(
            request.body["generationConfig"]["seed"],
            serde_json::json!(12_345)
        );
        assert_eq!(
            request.body["contents"][0]["parts"],
            serde_json::json!([{ "text": "A beautiful sunset" }])
        );
    }

    #[tokio::test]
    async fn google_vertex_generate_images_for_gemini_models_forward_vertex_provider_options() {
        let transport = FixtureJsonTransport::new(vertex_gemini_image_response_body());
        let client = GoogleVertexClient::from_config(
            GoogleVertexConfig::new("https://example.invalid", "gemini-2.5-flash-image")
                .with_api_key("test-key")
                .with_http_transport(Arc::new(transport.clone())),
        )
        .expect("from_config ok");

        let response = client
            .generate_images(
                ImageGenerationRequest {
                    prompt: "A watercolor otter".to_string(),
                    ..Default::default()
                }
                .with_provider_option(
                    "vertex",
                    serde_json::json!({
                        "mediaResolution": "MEDIA_RESOLUTION_MEDIUM",
                        "imageConfig": {
                            "imageSize": "1536x1024"
                        }
                    }),
                ),
            )
            .await
            .expect("vertex gemini image response");

        assert_eq!(response.images.len(), 1);

        let request = transport.take().expect("captured image request");
        assert_eq!(
            request.body["generationConfig"]["responseModalities"],
            serde_json::json!(["IMAGE"])
        );
        assert_eq!(
            request.body["generationConfig"]["mediaResolution"],
            serde_json::json!("MEDIA_RESOLUTION_MEDIUM")
        );
        assert_eq!(
            request.body["generationConfig"]["imageConfig"]["imageSize"],
            serde_json::json!("1536x1024")
        );
    }

    #[tokio::test]
    async fn google_vertex_generate_images_for_gemini_models_ignore_google_alias_provider_options()
    {
        let transport = FixtureJsonTransport::new(vertex_gemini_image_response_body());
        let client = GoogleVertexClient::from_config(
            GoogleVertexConfig::new("https://example.invalid", "gemini-2.5-flash-image")
                .with_api_key("test-key")
                .with_http_transport(Arc::new(transport.clone())),
        )
        .expect("from_config ok");

        let response = client
            .generate_images(
                ImageGenerationRequest {
                    prompt: "A watercolor otter".to_string(),
                    ..Default::default()
                }
                .with_provider_option(
                    "google",
                    serde_json::json!({
                        "mediaResolution": "MEDIA_RESOLUTION_MEDIUM",
                        "imageConfig": {
                            "imageSize": "1536x1024"
                        }
                    }),
                ),
            )
            .await
            .expect("vertex gemini image response");

        assert_eq!(response.images.len(), 1);

        let request = transport.take().expect("captured image request");
        assert_eq!(
            request.body["generationConfig"]["responseModalities"],
            serde_json::json!(["IMAGE"])
        );
        assert!(
            request.body["generationConfig"]
                .get("mediaResolution")
                .is_none()
        );
        assert!(
            request.body["generationConfig"]
                .get("imageConfig")
                .is_none(),
            "unexpected imageConfig: {}",
            request.body["generationConfig"]
        );
    }

    #[tokio::test]
    async fn google_vertex_edit_image_routes_gemini_image_models_through_generate_content() {
        let transport = FixtureJsonTransport::new(vertex_gemini_image_response_body());
        let client = GoogleVertexClient::from_config(
            GoogleVertexConfig::new("https://example.invalid", "gemini-2.5-flash-image")
                .with_api_key("test-key")
                .with_http_transport(Arc::new(transport.clone())),
        )
        .expect("from_config ok");

        let response = client
            .edit_image(ImageEditRequest {
                prompt: "Add a hat to this cat".to_string(),
                images: vec![crate::types::ImageEditInput::base64_with_media_type(
                    "base64-source-image",
                    "image/png",
                )],
                ..Default::default()
            })
            .await
            .expect("vertex gemini image edit response");

        assert_eq!(response.images.len(), 1);

        let request = transport.take().expect("captured edit request");
        assert!(
            request
                .url
                .contains("/models/gemini-2.5-flash-image:generateContent?key=test-key"),
            "unexpected url: {}",
            request.url
        );
        assert_eq!(
            request.body["contents"][0]["parts"],
            serde_json::json!([
                { "text": "Add a hat to this cat" },
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": "base64-source-image"
                    }
                }
            ])
        );
        assert_eq!(
            request.body["generationConfig"]["responseModalities"],
            serde_json::json!(["IMAGE"])
        );
    }

    #[tokio::test]
    async fn google_vertex_edit_image_for_gemini_models_preserves_url_inputs_as_file_data() {
        let transport = FixtureJsonTransport::new(vertex_gemini_image_response_body());
        let client = GoogleVertexClient::from_config(
            GoogleVertexConfig::new("https://example.invalid", "gemini-2.5-flash-image")
                .with_api_key("test-key")
                .with_http_transport(Arc::new(transport.clone())),
        )
        .expect("from_config ok");

        let response = client
            .edit_image(ImageEditRequest {
                prompt: "Add a hat to this cat".to_string(),
                images: vec![crate::types::ImageEditInput::url(
                    "https://example.com/cat.png",
                )],
                ..Default::default()
            })
            .await
            .expect("vertex gemini image edit response");

        assert_eq!(response.images.len(), 1);

        let request = transport.take().expect("captured edit request");
        assert_eq!(
            request.body["contents"][0]["parts"][1],
            serde_json::json!({
                "fileData": {
                    "fileUri": "https://example.com/cat.png",
                    "mimeType": "image/jpeg"
                }
            })
        );
    }

    #[tokio::test]
    async fn google_vertex_create_variation_routes_gemini_image_models_through_generate_content() {
        let transport = FixtureJsonTransport::new(vertex_gemini_image_response_body());
        let client = GoogleVertexClient::from_config(
            GoogleVertexConfig::new("https://example.invalid", "gemini-2.5-flash-image")
                .with_api_key("test-key")
                .with_http_transport(Arc::new(transport.clone())),
        )
        .expect("from_config ok");

        let response = client
            .create_variation(crate::types::ImageVariationRequest {
                image: crate::types::ImageEditInput::base64_with_media_type(
                    "base64-source-image",
                    "image/png",
                ),
                aspect_ratio: Some("1:1".to_string()),
                seed: Some(7),
                extra_params: std::collections::HashMap::from([(
                    "prompt".to_string(),
                    serde_json::json!("Make it watercolor"),
                )]),
                ..Default::default()
            })
            .await
            .expect("vertex gemini image variation response");

        assert_eq!(response.images.len(), 1);

        let request = transport.take().expect("captured variation request");
        assert!(
            request
                .url
                .contains("/models/gemini-2.5-flash-image:generateContent?key=test-key"),
            "unexpected url: {}",
            request.url
        );
        assert_eq!(
            request.body["contents"][0]["parts"],
            serde_json::json!([
                { "text": "Make it watercolor" },
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": "base64-source-image"
                    }
                }
            ])
        );
        assert_eq!(
            request.body["generationConfig"]["imageConfig"]["aspectRatio"],
            serde_json::json!("1:1")
        );
        assert_eq!(
            request.body["generationConfig"]["seed"],
            serde_json::json!(7)
        );
    }

    #[tokio::test]
    async fn google_vertex_gemini_image_requests_reject_mask_and_n_greater_than_one() {
        let transport = CaptureTransport::default();
        let client = GoogleVertexClient::from_config(
            GoogleVertexConfig::new("https://example.invalid", "gemini-2.5-flash-image")
                .with_api_key("test-key")
                .with_http_transport(Arc::new(transport.clone())),
        )
        .expect("from_config ok");

        let count_error = client
            .generate_images(ImageGenerationRequest {
                prompt: "A beautiful sunset".to_string(),
                count: 2,
                ..Default::default()
            })
            .await
            .expect_err("expected n>1 error");
        assert!(
            count_error.to_string().contains(
                "Gemini image models do not support generating a set number of images per call"
            ),
            "unexpected error: {count_error}"
        );

        let mask_error = client
            .edit_image(ImageEditRequest {
                prompt: "Edit this image".to_string(),
                images: vec![crate::types::ImageEditInput::base64_with_media_type(
                    "base64-source-image",
                    "image/png",
                )],
                mask: Some(crate::types::ImageEditInput::base64_with_media_type(
                    "base64-mask-image",
                    "image/png",
                )),
                ..Default::default()
            })
            .await
            .expect_err("expected mask error");
        assert!(
            mask_error
                .to_string()
                .contains("Gemini image models do not support mask-based image editing."),
            "unexpected error: {mask_error}"
        );
        assert!(transport.take().is_none());
    }

    #[test]
    fn prepare_chat_request_for_stream_sets_stream_and_fills_defaults() {
        let cfg = GoogleVertexConfig::new("https://example.invalid", "gemini-2.5-flash")
            .with_http_config(crate::types::HttpConfig::default());
        let client = GoogleVertexClient::from_config(cfg).expect("from_config ok");

        let request = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .build();

        let prepared = client
            .prepare_chat_request(request, true)
            .expect("prepare stream request");

        assert!(prepared.stream);
        assert_eq!(prepared.common_params.model, "gemini-2.5-flash");
        assert!(prepared.http_config.is_some());
    }

    #[test]
    fn prepare_chat_request_for_non_stream_clears_stream_and_preserves_explicit_model() {
        let cfg = GoogleVertexConfig::new("https://example.invalid", "gemini-2.5-flash")
            .with_http_config(crate::types::HttpConfig::default());
        let client = GoogleVertexClient::from_config(cfg).expect("from_config ok");

        let request = ChatRequest::builder()
            .model("gemini-1.5-pro")
            .messages(vec![ChatMessage::user("hi").build()])
            .stream(true)
            .build();

        let prepared = client
            .prepare_chat_request(request, false)
            .expect("prepare non-stream request");

        assert!(!prepared.stream);
        assert_eq!(prepared.common_params.model, "gemini-1.5-pro");
        assert!(prepared.http_config.is_some());
    }

    #[tokio::test]
    async fn google_vertex_chat_stream_request_preserves_stable_options_at_transport_boundary() {
        let transport = CaptureTransport::default();
        let client = GoogleVertexClient::from_config(
            GoogleVertexConfig::new("https://example.invalid", "gemini-2.5-flash")
                .with_api_key("test-key")
                .with_http_transport(Arc::new(transport.clone())),
        )
        .expect("from_config ok");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model("gemini-2.5-flash")
            .messages(vec![ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "lookup_weather",
                "Look up the weather",
                serde_json::json!({
                    "type": "object",
                    "properties": { "location": { "type": "string" } },
                    "required": ["location"],
                    "additionalProperties": false
                }),
            )])
            .tool_choice(crate::types::ToolChoice::None)
            .response_format(crate::types::ResponseFormat::json_schema(schema.clone()))
            .build()
            .with_provider_option(
                "vertex",
                serde_json::json!({
                    "thinkingConfig": {
                        "thinkingBudget": 2048,
                        "includeThoughts": true
                    },
                    "structuredOutputs": true
                }),
            );

        let _ = client.chat_stream_request(request).await;

        let captured = transport.take_stream().expect("captured stream request");
        assert!(
            captured
                .url
                .contains("/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key=test-key"),
            "unexpected url: {}",
            captured.url
        );
        assert_eq!(
            captured.headers.get("accept").and_then(|v| v.to_str().ok()),
            Some("text/event-stream")
        );
        assert!(transport.take().is_none());
        assert_eq!(
            captured.body["generationConfig"]["thinkingConfig"],
            serde_json::json!({
                "thinkingBudget": 2048,
                "includeThoughts": true
            })
        );
        assert_eq!(
            captured.body["generationConfig"]["responseMimeType"],
            serde_json::json!("application/json")
        );
        assert!(
            captured.body["generationConfig"]
                .get("responseSchema")
                .is_some()
        );
        assert!(
            captured.body["generationConfig"]
                .get("responseJsonSchema")
                .is_none()
        );
        assert_eq!(
            captured.body["toolConfig"],
            serde_json::json!({
                "functionCallingConfig": { "mode": "NONE" }
            })
        );
        assert_eq!(
            captured.body["tools"][0]["functionDeclarations"][0]["name"],
            serde_json::json!("lookup_weather")
        );
        assert_eq!(
            captured.body["contents"][0]["parts"][0]["text"],
            serde_json::json!("hi")
        );
    }

    #[tokio::test]
    async fn google_vertex_chat_response_uses_custom_generate_id_for_tool_calls_and_sources() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let transport = FixtureJsonTransport::new(vertex_tool_call_and_source_response_body());
        let counter = Arc::new(AtomicUsize::new(0));
        let client = GoogleVertexClient::from_config(
            GoogleVertexConfig::new("https://example.invalid", "gemini-2.5-flash")
                .with_api_key("test-key")
                .with_generate_id({
                    let counter = Arc::clone(&counter);
                    move || format!("vertex-chat-id-{}", counter.fetch_add(1, Ordering::Relaxed))
                })
                .with_http_transport(Arc::new(transport.clone())),
        )
        .expect("from_config ok");

        let response = client
            .chat_request(
                ChatRequest::builder()
                    .model("gemini-2.5-flash")
                    .messages(vec![ChatMessage::user("hi").build()])
                    .build(),
            )
            .await
            .expect("vertex chat response");

        let tool_call = response.tool_calls()[0].as_tool_call().expect("tool call");
        assert_eq!(tool_call.tool_call_id, "vertex-chat-id-0");
        assert_eq!(
            response
                .provider_metadata
                .as_ref()
                .and_then(|meta| meta.get("vertex"))
                .and_then(|meta| meta.get("sources"))
                .and_then(|sources| sources.as_array())
                .and_then(|sources| sources.first())
                .and_then(|source| source.get("id"))
                .and_then(|value| value.as_str()),
            Some("vertex-chat-id-1")
        );

        let request = transport.take().expect("captured request");
        assert!(
            request
                .url
                .contains("/models/gemini-2.5-flash:generateContent?key=test-key"),
            "unexpected url: {}",
            request.url
        );
    }

    #[tokio::test]
    async fn google_vertex_chat_stream_uses_custom_generate_id_and_include_raw_chunks() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let transport = FixtureStreamTransport::new(vertex_tool_call_and_source_stream_body());
        let counter = Arc::new(AtomicUsize::new(0));
        let client = GoogleVertexClient::from_config(
            GoogleVertexConfig::new("https://example.invalid", "gemini-2.5-flash")
                .with_api_key("test-key")
                .with_generate_id({
                    let counter = Arc::clone(&counter);
                    move || {
                        format!(
                            "vertex-stream-id-{}",
                            counter.fetch_add(1, Ordering::Relaxed)
                        )
                    }
                })
                .with_http_transport(Arc::new(transport.clone())),
        )
        .expect("from_config ok");

        let events = collect_stream_events(
            client
                .chat_stream_request(
                    ChatRequest::builder()
                        .model("gemini-2.5-flash")
                        .messages(vec![ChatMessage::user("hi").build()])
                        .include_raw_chunks(true)
                        .stream(true)
                        .build(),
                )
                .await
                .expect("vertex stream ok"),
        )
        .await;

        let source_id = events.iter().find_map(|event| match event {
            ChatStreamEvent::Part {
                part: ChatStreamPart::Source { id, .. },
            } => Some(id.as_str()),
            _ => None,
        });
        let tool_call_id = events.iter().find_map(|event| match event {
            ChatStreamEvent::ToolCallDelta { id, .. } => Some(id.as_str()),
            _ => None,
        });
        let has_raw = events.iter().any(|event| {
            matches!(
                LanguageModelV3StreamPart::try_from_chat_event(event),
                Some(LanguageModelV3StreamPart::Raw { .. })
            )
        });

        assert_eq!(source_id, Some("vertex-stream-id-0"));
        assert_eq!(tool_call_id, Some("vertex-stream-id-1"));
        assert!(has_raw, "expected raw stream part in {events:?}");

        let request = transport.take_stream().expect("captured stream request");
        assert_eq!(
            request
                .headers
                .get("accept")
                .and_then(|value| value.to_str().ok()),
            Some("text/event-stream")
        );
    }

    #[tokio::test]
    async fn google_vertex_client_structured_output_stream_end_preserves_metadata_and_extracts_json()
     {
        let model = "gemini-2.5-flash";
        let transport = FixtureStreamTransport::new(vertex_structured_output_success_stream_body());
        let client = GoogleVertexClient::from_config(
            GoogleVertexConfig::new("https://example.invalid", model)
                .with_api_key("test-key")
                .with_http_transport(Arc::new(transport.clone())),
        )
        .expect("from_config ok");

        let events = collect_stream_events(
            client
                .chat_stream_request(make_vertex_structured_output_request(model))
                .await
                .expect("vertex stream ok"),
        )
        .await;

        let content = events
            .iter()
            .filter_map(|event| match event {
                ChatStreamEvent::ContentDelta { delta, .. } => Some(delta.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(content, "{\"value\":\"test\"}");

        let end = events
            .iter()
            .find_map(|event| match event {
                ChatStreamEvent::StreamEnd { response } => Some(response),
                _ => None,
            })
            .expect("expected stream end");
        assert_eq!(end.finish_reason, Some(FinishReason::Stop));

        let provider_metadata = end
            .provider_metadata
            .as_ref()
            .expect("expected provider metadata");
        let vertex_meta = provider_metadata
            .get("vertex")
            .expect("expected provider_metadata.vertex");
        assert!(
            !provider_metadata.contains_key("google"),
            "did not expect provider_metadata.google on vertex path"
        );
        assert_eq!(
            vertex_meta
                .get("usageMetadata")
                .and_then(|usage| usage.get("totalTokenCount"))
                .and_then(|value| value.as_u64()),
            Some(11)
        );
        assert_eq!(
            vertex_meta
                .get("safetyRatings")
                .and_then(|ratings| ratings.as_array())
                .and_then(|ratings| ratings.first())
                .and_then(|rating| rating.get("category"))
                .and_then(|value| value.as_str()),
            Some("HARM_CATEGORY_DEROGATORY")
        );

        let value = siumai_core::structured_output::extract_json_value_from_stream(Box::pin(
            futures::stream::iter(events.into_iter().map(Ok::<_, LlmError>)),
        ))
        .await
        .expect("structured output value");
        assert_eq!(value["value"], "test");

        let req = transport.take_stream().expect("captured stream request");
        assert_vertex_structured_output_stream_request(&req);
    }

    #[tokio::test]
    async fn google_vertex_client_structured_output_stream_returns_incomplete_json_error() {
        let model = "gemini-2.5-flash";
        let transport =
            FixtureStreamTransport::new(vertex_structured_output_interrupted_stream_body());
        let client = GoogleVertexClient::from_config(
            GoogleVertexConfig::new("https://example.invalid", model)
                .with_api_key("test-key")
                .with_http_transport(Arc::new(transport.clone())),
        )
        .expect("from_config ok");

        let err = siumai_core::structured_output::extract_json_value_from_stream(
            client
                .chat_stream_request(make_vertex_structured_output_request(model))
                .await
                .expect("vertex stream ok"),
        )
        .await
        .expect_err("interrupted stream should fail");

        match err {
            LlmError::ParseError(message) => {
                assert!(message.contains("stream ended before a complete JSON value was produced"))
            }
            other => panic!("expected ParseError, got {other:?}"),
        }

        let req = transport.take_stream().expect("captured stream request");
        assert_vertex_structured_output_stream_request(&req);
    }
}
