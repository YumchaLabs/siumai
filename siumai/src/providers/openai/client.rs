//! `OpenAI` Client Implementation
//!
//! Main client structure that aggregates all `OpenAI` capabilities.

use async_trait::async_trait;
use secrecy::ExposeSecret;
use std::sync::Arc;

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::params::OpenAiParams;
use crate::streaming::ChatStream;
use crate::traits::*;
use crate::types::*;

use super::models::OpenAiModels;
use super::rerank::OpenAiRerank;
use super::types::OpenAiSpecificParams;
use super::utils::get_default_models;
use crate::executors::chat::ChatExecutor;
use crate::middleware::language_model::LanguageModelMiddleware;
use crate::retry_api::RetryOptions;

/// `OpenAI` Client
pub struct OpenAiClient {
    /// API key and endpoint configuration
    api_key: secrecy::SecretString,
    base_url: String,
    organization: Option<String>,
    project: Option<String>,
    http_config: HttpConfig,
    /// Models capability implementation
    models_capability: OpenAiModels,
    /// Rerank capability implementation
    rerank_capability: OpenAiRerank,
    /// Common parameters
    common_params: CommonParams,
    /// OpenAI-specific parameters
    openai_params: OpenAiParams,
    /// OpenAI-specific configuration
    specific_params: OpenAiSpecificParams,
    /// HTTP client for making requests
    http_client: reqwest::Client,
    /// Tracing configuration
    tracing_config: Option<crate::tracing::TracingConfig>,
    /// Tracing guard to keep tracing system active (not cloned)
    /// NOTE: Tracing subscriber functionality has been moved to siumai-extras
    #[allow(dead_code)]
    _tracing_guard: Option<()>,
    /// Web search config
    web_search_config: crate::types::WebSearchConfig,
    /// Unified retry options for chat
    retry_options: Option<RetryOptions>,
    /// Optional HTTP interceptors applied to all chat requests
    http_interceptors: Vec<std::sync::Arc<dyn crate::utils::http_interceptor::HttpInterceptor>>,
    /// Optional model-level middlewares applied before provider mapping
    model_middlewares: Vec<std::sync::Arc<dyn LanguageModelMiddleware>>,
}

impl Clone for OpenAiClient {
    fn clone(&self) -> Self {
        Self {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            organization: self.organization.clone(),
            project: self.project.clone(),
            http_config: self.http_config.clone(),
            models_capability: self.models_capability.clone(),
            rerank_capability: self.rerank_capability.clone(),
            common_params: self.common_params.clone(),
            openai_params: self.openai_params.clone(),
            specific_params: self.specific_params.clone(),
            http_client: self.http_client.clone(),
            tracing_config: self.tracing_config.clone(),
            _tracing_guard: None, // Don't clone the tracing guard
            web_search_config: self.web_search_config.clone(),
            retry_options: self.retry_options.clone(),
            http_interceptors: self.http_interceptors.clone(),
            model_middlewares: Vec::new(),
        }
    }
}

impl std::fmt::Debug for OpenAiClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("OpenAiClient");

        debug_struct
            .field("provider_name", &"openai")
            .field("model", &self.common_params.model)
            .field("base_url", &self.base_url)
            .field("temperature", &self.common_params.temperature)
            .field("max_tokens", &self.common_params.max_tokens)
            .field("top_p", &self.common_params.top_p)
            .field("seed", &self.common_params.seed)
            .field("has_tracing", &self.tracing_config.is_some());

        // Only show organization/project if they exist (but don't show the actual values)
        if self.specific_params.organization.is_some() {
            debug_struct.field("has_organization", &true);
        }
        if self.specific_params.project.is_some() {
            debug_struct.field("has_project", &true);
        }

        debug_struct.finish()
    }
}

impl OpenAiClient {
    /// Convenience: configure structured outputs using a JSON object schema.
    /// Only applied to Responses API requests. For chat/completions, this is ignored.
    pub fn with_json_object_schema(mut self, schema: serde_json::Value, strict: bool) -> Self {
        let fmt = serde_json::json!({
            "type": "json_object",
            "json_schema": {
                "schema": schema,
                "strict": strict
            }
        });
        self.specific_params.response_format = Some(fmt);
        self
    }

    /// Convenience: configure structured outputs using a named JSON schema.
    /// Only applied to Responses API requests.
    pub fn with_json_named_schema<S: Into<String>>(
        mut self,
        name: S,
        schema: serde_json::Value,
        strict: bool,
    ) -> Self {
        let fmt = serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": name.into(),
                "schema": schema,
                "strict": strict
            }
        });
        self.specific_params.response_format = Some(fmt);
        self
    }

    /// Creates a new `OpenAI` client with configuration and HTTP client
    pub fn new(config: super::OpenAiConfig, http_client: reqwest::Client) -> Self {
        let specific_params = OpenAiSpecificParams {
            organization: config.organization.clone(),
            project: config.project.clone(),
            ..Default::default()
        };

        let models_capability = OpenAiModels::new(
            config.api_key.clone(),
            config.base_url.clone(),
            http_client.clone(),
            config.organization.clone(),
            config.project.clone(),
            config.http_config.clone(),
        );

        let rerank_capability = OpenAiRerank::new(
            config.api_key.clone(),
            config.base_url.clone(),
            http_client.clone(),
            config.organization.clone(),
            config.project.clone(),
            config.http_config.clone(),
        );

        Self {
            api_key: config.api_key,
            base_url: config.base_url,
            organization: config.organization,
            project: config.project,
            http_config: config.http_config,
            models_capability,
            rerank_capability,
            common_params: config.common_params,
            openai_params: config.openai_params,
            specific_params,
            http_client,
            tracing_config: None,
            _tracing_guard: None,
            web_search_config: config.web_search_config,
            retry_options: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
        }
    }

    /// Set the tracing configuration
    pub(crate) fn set_tracing_config(&mut self, config: Option<crate::tracing::TracingConfig>) {
        self.tracing_config = config;
    }

    /// Set unified retry options
    pub fn set_retry_options(&mut self, options: Option<RetryOptions>) {
        self.retry_options = options;
    }

    /// Creates a new `OpenAI` client with configuration (for OpenAI-compatible providers)
    pub fn new_with_config(config: super::OpenAiConfig) -> Self {
        let http_client = reqwest::Client::new();
        Self::new(config, http_client)
    }

    /// Helper: Build ProviderContext with OpenAI-specific extras
    fn build_context(&self) -> crate::provider_core::ProviderContext {
        let mut extras = std::collections::HashMap::new();
        if let Some(fmt) = &self.specific_params.response_format {
            extras.insert("openai.response_format".to_string(), fmt.clone());
        }

        crate::provider_core::ProviderContext::new(
            "openai",
            self.base_url.clone(),
            Some(self.api_key.expose_secret().to_string()),
            self.http_config.headers.clone(),
        )
        .with_org_project(self.organization.clone(), self.project.clone())
        .with_extras(extras)
    }

    /// Helper: Build ChatExecutor with common configuration
    fn build_chat_executor(
        &self,
        request: &ChatRequest,
    ) -> Arc<crate::executors::chat::HttpChatExecutor> {
        use crate::executors::chat::ChatExecutorBuilder;
        use crate::provider_core::ProviderSpec;

        let ctx = self.build_context();
        let spec = Arc::new(crate::providers::openai::spec::OpenAiSpec::new());
        let bundle = spec.choose_chat_transformers(request, &ctx);
        let before_send_hook = spec.chat_before_send(request, &ctx);

        let mut builder = ChatExecutorBuilder::new("openai", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_transformer_bundle(bundle)
            .with_stream_disable_compression(self.http_config.stream_disable_compression)
            .with_interceptors(self.http_interceptors.clone())
            .with_middlewares(self.model_middlewares.clone());

        if let Some(hook) = before_send_hook {
            builder = builder.with_before_send(hook);
        }

        builder.build()
    }

    /// Helper: Build EmbeddingExecutor with common configuration
    fn build_embedding_executor(&self) -> Arc<crate::executors::embedding::HttpEmbeddingExecutor> {
        use crate::provider_core::ProviderSpec;

        let ctx = self.build_context();
        let spec = Arc::new(crate::providers::openai::spec::OpenAiSpec::new());
        let req = EmbeddingRequest::new(vec![]).with_model(self.common_params.model.clone());
        let bundle = spec.choose_embedding_transformers(&req, &ctx);

        Arc::new(crate::executors::embedding::HttpEmbeddingExecutor {
            provider_id: "openai".to_string(),
            http_client: self.http_client.clone(),
            request_transformer: bundle.request,
            response_transformer: bundle.response,
            provider_spec: spec,
            provider_context: ctx,
            interceptors: vec![],
            before_send: None,
            retry_options: self.retry_options.clone(),
        })
    }

    /// Helper: Build ImageExecutor with common configuration
    fn build_image_executor(
        &self,
        request: &ImageGenerationRequest,
    ) -> crate::executors::image::HttpImageExecutor {
        use crate::provider_core::ProviderSpec;

        let ctx = self.build_context();
        let spec = Arc::new(crate::providers::openai::spec::OpenAiSpec::new());
        let bundle = spec.choose_image_transformers(request, &ctx);

        crate::executors::image::HttpImageExecutor {
            provider_id: "openai".to_string(),
            http_client: self.http_client.clone(),
            request_transformer: bundle.request,
            response_transformer: bundle.response,
            provider_spec: spec,
            provider_context: ctx,
            interceptors: vec![],
            before_send: None,
            retry_options: self.retry_options.clone(),
        }
    }

    /// Helper: Build AudioExecutor with common configuration
    fn build_audio_executor(&self) -> crate::executors::audio::HttpAudioExecutor {
        let ctx = self.build_context();
        let spec = Arc::new(crate::providers::openai::spec::OpenAiSpec::new());

        crate::executors::audio::HttpAudioExecutor {
            provider_id: "openai".to_string(),
            http_client: self.http_client.clone(),
            transformer: Arc::new(super::transformers::OpenAiAudioTransformer),
            provider_spec: spec,
            provider_context: ctx,
        }
    }

    /// Stream chat via ProviderSpec (unified path)
    async fn chat_stream_via_spec(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        use crate::executors::chat::ChatExecutor;

        let request = ChatRequest {
            messages,
            tools,
            common_params: self.common_params.clone(),
            stream: true,
            ..Default::default()
        };

        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute_stream(&*exec, request).await
    }

    /// Execute chat (non-stream) via ProviderSpec with a fully-formed ChatRequest
    async fn chat_request_via_spec(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        use crate::executors::chat::ChatExecutor;
        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute(&*exec, request).await
    }

    /// Execute chat (stream) via ProviderSpec with a fully-formed ChatRequest
    async fn chat_stream_request_via_spec(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStream, LlmError> {
        use crate::executors::chat::ChatExecutor;
        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute_stream(&*exec, request).await
    }

    /// Get OpenAI-specific parameters
    pub const fn specific_params(&self) -> &OpenAiSpecificParams {
        &self.specific_params
    }

    /// Get common parameters (for testing and debugging)
    pub const fn common_params(&self) -> &CommonParams {
        &self.common_params
    }

    /// Install HTTP interceptors for all chat requests.
    pub fn with_http_interceptors(
        mut self,
        interceptors: Vec<std::sync::Arc<dyn crate::utils::http_interceptor::HttpInterceptor>>,
    ) -> Self {
        self.http_interceptors = interceptors;
        self
    }

    /// Install model-level middlewares for chat requests.
    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<std::sync::Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.model_middlewares = middlewares;
        self
    }

    // chat_capability removed after executors migration

    /// Update OpenAI-specific parameters
    pub fn with_specific_params(mut self, params: OpenAiSpecificParams) -> Self {
        self.specific_params = params;
        self
    }

    /// Set organization
    pub fn with_organization(mut self, organization: String) -> Self {
        self.specific_params.organization = Some(organization);
        self
    }

    /// Set project
    pub fn with_project(mut self, project: String) -> Self {
        self.specific_params.project = Some(project);
        self
    }

    /// Set response format for structured output
    pub fn with_response_format(mut self, format: serde_json::Value) -> Self {
        self.specific_params.response_format = Some(format);
        self
    }

    /// Set logit bias
    pub fn with_logit_bias(mut self, bias: serde_json::Value) -> Self {
        self.specific_params.logit_bias = Some(bias);
        self
    }

    /// Enable logprobs
    pub const fn with_logprobs(mut self, enabled: bool, top_logprobs: Option<u32>) -> Self {
        self.specific_params.logprobs = Some(enabled);
        self.specific_params.top_logprobs = top_logprobs;
        self
    }

    /// Set presence penalty
    pub const fn with_presence_penalty(mut self, penalty: f32) -> Self {
        self.specific_params.presence_penalty = Some(penalty);
        self
    }

    /// Set frequency penalty
    pub const fn with_frequency_penalty(mut self, penalty: f32) -> Self {
        self.specific_params.frequency_penalty = Some(penalty);
        self
    }

    /// Set user identifier
    pub fn with_user(mut self, user: String) -> Self {
        self.specific_params.user = Some(user);
        self
    }
}

impl OpenAiClient {
    async fn chat_with_tools_inner(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        use crate::executors::chat::ChatExecutor;

        let request = ChatRequest {
            messages,
            tools,
            common_params: self.common_params.clone(),
            ..Default::default()
        };

        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute(&*exec, request).await
    }
}

#[async_trait]
impl ChatCapability for OpenAiClient {
    /// Chat with tools implementation
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        if let Some(opts) = &self.retry_options {
            crate::retry_api::retry_with(
                || {
                    let m = messages.clone();
                    let t = tools.clone();
                    async move { self.chat_with_tools_inner(m, t).await }
                },
                opts.clone(),
            )
            .await
        } else {
            self.chat_with_tools_inner(messages, tools).await
        }
    }

    /// Streaming chat with tools
    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        self.chat_stream_via_spec(messages, tools).await
    }

    // (removed: nested impl block)

    async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.chat_request_via_spec(request).await
    }

    async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        self.chat_stream_request_via_spec(request).await
    }
}

#[async_trait]
impl ModelListingCapability for OpenAiClient {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.models_capability.list_models().await
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        self.models_capability.get_model(model_id).await
    }
}

#[async_trait]
impl EmbeddingCapability for OpenAiClient {
    async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        use crate::executors::embedding::EmbeddingExecutor;

        let request = EmbeddingRequest::new(texts).with_model(self.common_params.model.clone());
        let exec = self.build_embedding_executor();

        EmbeddingExecutor::execute(&*exec, request).await
    }

    fn embedding_dimension(&self) -> usize {
        // Return dimension based on model
        let model = if !self.common_params.model.is_empty() {
            &self.common_params.model
        } else {
            "text-embedding-3-small"
        };

        match model {
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "text-embedding-ada-002" => 1536,
            _ => 1536, // Default fallback
        }
    }

    fn max_tokens_per_embedding(&self) -> usize {
        8192 // OpenAI's current limit
    }

    fn supported_embedding_models(&self) -> Vec<String> {
        vec![
            "text-embedding-3-small".to_string(),
            "text-embedding-3-large".to_string(),
            "text-embedding-ada-002".to_string(),
        ]
    }
}

// Provide extended embedding APIs that accept EmbeddingRequest directly
#[async_trait]
impl EmbeddingExtensions for OpenAiClient {
    async fn embed_with_config(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        use crate::executors::embedding::{EmbeddingExecutor, HttpEmbeddingExecutor};

        if let Some(opts) = &self.retry_options {
            let http0 = self.http_client.clone();
            let base0 = self.base_url.clone();
            let api_key0 = self.api_key.clone();
            let org0 = self.organization.clone();
            let proj0 = self.project.clone();
            crate::retry_api::retry_with(
                || {
                    let rq = request.clone();
                    let http = http0.clone();
                    let base = base0.clone();
                    let api_key = api_key0.clone();
                    let org = org0.clone();
                    let proj = proj0.clone();
                    async move {
                        use crate::provider_core::{ProviderContext, ProviderSpec};
                        use secrecy::ExposeSecret;

                        let spec = crate::providers::openai::spec::OpenAiSpec::new();
                        let ctx = ProviderContext::new(
                            "openai",
                            base,
                            Some(api_key.expose_secret().to_string()),
                            self.http_config.headers.clone(),
                        )
                        .with_org_project(org, proj);
                        let bundle = spec.choose_embedding_transformers(&rq, &ctx);
                        let spec_arc = Arc::new(spec);

                        let exec = HttpEmbeddingExecutor {
                            provider_id: "openai".to_string(),
                            http_client: http,
                            request_transformer: bundle.request,
                            response_transformer: bundle.response,
                            provider_spec: spec_arc,
                            provider_context: ctx,
                            interceptors: vec![],
                            before_send: None,
                            retry_options: None,
                        };
                        EmbeddingExecutor::execute(&exec, rq).await
                    }
                },
                opts.clone(),
            )
            .await
        } else {
            use crate::provider_core::{ProviderContext, ProviderSpec};
            use secrecy::ExposeSecret;

            let spec = crate::providers::openai::spec::OpenAiSpec::new();
            let ctx = ProviderContext::new(
                "openai",
                self.base_url.clone(),
                Some(self.api_key.expose_secret().to_string()),
                self.http_config.headers.clone(),
            )
            .with_org_project(self.organization.clone(), self.project.clone());
            let bundle = spec.choose_embedding_transformers(&request, &ctx);
            let spec_arc = Arc::new(spec);

            let exec = HttpEmbeddingExecutor {
                provider_id: "openai".to_string(),
                http_client: self.http_client.clone(),
                request_transformer: bundle.request,
                response_transformer: bundle.response,
                provider_spec: spec_arc,
                provider_context: ctx,
                interceptors: vec![],
                before_send: None,
                retry_options: None,
            };
            exec.execute(request).await
        }
    }
}

#[async_trait]
impl AudioCapability for OpenAiClient {
    fn supported_features(&self) -> &[crate::types::AudioFeature] {
        use crate::types::AudioFeature::*;
        const FEATURES: &[crate::types::AudioFeature] = &[TextToSpeech, SpeechToText];
        FEATURES
    }

    async fn text_to_speech(
        &self,
        request: crate::types::TtsRequest,
    ) -> Result<crate::types::TtsResponse, LlmError> {
        use crate::executors::audio::AudioExecutor;

        let exec = self.build_audio_executor();
        let result_bytes = AudioExecutor::tts(&exec, request.clone()).await?;

        Ok(crate::types::TtsResponse {
            audio_data: result_bytes,
            format: request.format.unwrap_or_else(|| "mp3".to_string()),
            duration: None,
            sample_rate: None,
            metadata: std::collections::HashMap::new(),
        })
    }

    async fn speech_to_text(
        &self,
        request: crate::types::SttRequest,
    ) -> Result<crate::types::SttResponse, LlmError> {
        use crate::executors::audio::AudioExecutor;

        let exec = self.build_audio_executor();
        let text = AudioExecutor::stt(&exec, request).await?;

        Ok(crate::types::SttResponse {
            text,
            language: None,
            confidence: None,
            words: None,
            duration: None,
            metadata: std::collections::HashMap::new(),
        })
    }
}

impl LlmProvider for OpenAiClient {
    fn provider_name(&self) -> &'static str {
        "openai"
    }

    fn supported_models(&self) -> Vec<String> {
        get_default_models()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_audio()
            .with_embedding()
            .with_custom_feature("structured_output", true)
            .with_custom_feature("batch_processing", true)
            .with_custom_feature("rerank", true)
    }

    fn http_client(&self) -> &reqwest::Client {
        &self.http_client
    }
}

impl LlmClient for OpenAiClient {
    fn provider_name(&self) -> &'static str {
        LlmProvider::provider_name(self)
    }

    fn supported_models(&self) -> Vec<String> {
        LlmProvider::supported_models(self)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        LlmProvider::capabilities(self)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_embedding_capability(&self) -> Option<&dyn EmbeddingCapability> {
        Some(self)
    }

    fn as_audio_capability(&self) -> Option<&dyn AudioCapability> {
        // Provide audio via executor-backed implementation
        Some(self)
    }

    fn as_image_generation_capability(&self) -> Option<&dyn ImageGenerationCapability> {
        // Return the image generation capability
        Some(self)
    }

    fn as_file_management_capability(
        &self,
    ) -> Option<&dyn crate::traits::FileManagementCapability> {
        Some(self)
    }
}

#[async_trait]
impl RerankCapability for OpenAiClient {
    /// Rerank documents based on their relevance to a query
    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, LlmError> {
        self.rerank_capability.rerank(request).await
    }

    /// Get the maximum number of documents that can be reranked
    fn max_documents(&self) -> Option<u32> {
        self.rerank_capability.max_documents()
    }

    /// Get supported rerank models for this provider
    fn supported_models(&self) -> Vec<String> {
        self.rerank_capability.supported_models()
    }
}

#[async_trait]
impl ImageGenerationCapability for OpenAiClient {
    /// Generate images from text prompts.
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        use crate::executors::image::ImageExecutor;

        let exec = self.build_image_executor(&request);
        exec.execute(request).await
    }

    /// Edit an existing image with a text prompt.
    async fn edit_image(
        &self,
        request: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        use crate::executors::image::ImageExecutor;

        let exec = self.build_image_executor(&ImageGenerationRequest::default());
        ImageExecutor::execute_edit(&exec, request).await
    }

    /// Create variations of an existing image.
    async fn create_variation(
        &self,
        request: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        use crate::executors::image::ImageExecutor;

        let exec = self.build_image_executor(&ImageGenerationRequest::default());
        ImageExecutor::execute_variation(&exec, request).await
    }

    /// Get supported image sizes for this provider.
    fn get_supported_sizes(&self) -> Vec<String> {
        // SiliconFlow supports additional sizes
        if self.base_url.contains("siliconflow.cn") {
            vec![
                "1024x1024".to_string(),
                "960x1280".to_string(),
                "768x1024".to_string(),
                "720x1440".to_string(),
                "720x1280".to_string(),
            ]
        } else {
            // OpenAI supported sizes
            vec![
                "256x256".to_string(),
                "512x512".to_string(),
                "1024x1024".to_string(),
                "1792x1024".to_string(),
                "1024x1792".to_string(),
                "2048x2048".to_string(),
            ]
        }
    }

    /// Get supported response formats for this provider.
    fn get_supported_formats(&self) -> Vec<String> {
        if self.base_url.contains("siliconflow.cn") {
            vec!["url".to_string()]
        } else {
            vec!["url".to_string(), "b64_json".to_string()]
        }
    }

    /// Check if the provider supports image editing.
    fn supports_image_editing(&self) -> bool {
        // SiliconFlow doesn't support editing/variations via OpenAI-compatible paths
        !self.base_url.contains("siliconflow.cn")
    }

    /// Check if the provider supports image variations.
    fn supports_image_variations(&self) -> bool {
        // SiliconFlow doesn't support editing/variations via OpenAI-compatible paths
        !self.base_url.contains("siliconflow.cn")
    }
}

#[async_trait::async_trait]
impl crate::traits::FileManagementCapability for OpenAiClient {
    async fn upload_file(
        &self,
        request: crate::types::FileUploadRequest,
    ) -> Result<crate::types::FileObject, LlmError> {
        let cfg = super::OpenAiConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            organization: self.organization.clone(),
            project: self.project.clone(),
            common_params: self.common_params.clone(),
            openai_params: self.openai_params.clone(),
            http_config: self.http_config.clone(),
            web_search_config: self.web_search_config.clone(),
        };
        let files = super::files::OpenAiFiles::new(cfg, self.http_client.clone());
        files.upload_file(request).await
    }

    async fn list_files(
        &self,
        query: Option<crate::types::FileListQuery>,
    ) -> Result<crate::types::FileListResponse, LlmError> {
        let cfg = super::OpenAiConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            organization: self.organization.clone(),
            project: self.project.clone(),
            common_params: self.common_params.clone(),
            openai_params: self.openai_params.clone(),
            http_config: self.http_config.clone(),
            web_search_config: self.web_search_config.clone(),
        };
        let files = super::files::OpenAiFiles::new(cfg, self.http_client.clone());
        files.list_files(query).await
    }

    async fn retrieve_file(&self, file_id: String) -> Result<crate::types::FileObject, LlmError> {
        let cfg = super::OpenAiConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            organization: self.organization.clone(),
            project: self.project.clone(),
            common_params: self.common_params.clone(),
            openai_params: self.openai_params.clone(),
            http_config: self.http_config.clone(),
            web_search_config: self.web_search_config.clone(),
        };
        let files = super::files::OpenAiFiles::new(cfg, self.http_client.clone());
        files.retrieve_file(file_id).await
    }

    async fn delete_file(
        &self,
        file_id: String,
    ) -> Result<crate::types::FileDeleteResponse, LlmError> {
        let cfg = super::OpenAiConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            organization: self.organization.clone(),
            project: self.project.clone(),
            common_params: self.common_params.clone(),
            openai_params: self.openai_params.clone(),
            http_config: self.http_config.clone(),
            web_search_config: self.web_search_config.clone(),
        };
        let files = super::files::OpenAiFiles::new(cfg, self.http_client.clone());
        files.delete_file(file_id).await
    }

    async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        let cfg = super::OpenAiConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            organization: self.organization.clone(),
            project: self.project.clone(),
            common_params: self.common_params.clone(),
            openai_params: self.openai_params.clone(),
            http_config: self.http_config.clone(),
            web_search_config: self.web_search_config.clone(),
        };
        let files = super::files::OpenAiFiles::new(cfg, self.http_client.clone());
        files.get_file_content(file_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::openai::OpenAiConfig;
    use crate::providers::openai::transformers;
    use crate::transformers::request::RequestTransformer;
    use crate::utils::http_interceptor::{HttpInterceptor, HttpRequestContext};
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_openai_client_creation() {
        let config = OpenAiConfig::new("test-key");
        let client = OpenAiClient::new(config, reqwest::Client::new());

        assert_eq!(LlmProvider::provider_name(&client), "openai");
        assert!(!LlmProvider::supported_models(&client).is_empty());
    }

    #[test]
    fn test_openai_client_with_specific_params() {
        let config = OpenAiConfig::new("test-key")
            .with_organization("org-123")
            .with_project("proj-456");
        let client = OpenAiClient::new(config, reqwest::Client::new())
            .with_presence_penalty(0.5)
            .with_frequency_penalty(0.3);

        assert_eq!(
            client.specific_params().organization,
            Some("org-123".to_string())
        );
        assert_eq!(
            client.specific_params().project,
            Some("proj-456".to_string())
        );
        assert_eq!(client.specific_params().presence_penalty, Some(0.5));
        assert_eq!(client.specific_params().frequency_penalty, Some(0.3));
    }

    #[test]
    fn test_openai_client_uses_builder_model() {
        let config = OpenAiConfig::new("test-key").with_model("gpt-4");
        let client = OpenAiClient::new(config, reqwest::Client::new());

        // Verify that the client stores the model from the builder
        assert_eq!(client.common_params.model, "gpt-4");
    }

    #[tokio::test]
    async fn test_openai_chat_request_uses_client_model() {
        use crate::types::{ChatMessage, MessageContent, MessageMetadata, MessageRole};

        let config = OpenAiConfig::new("test-key").with_model("gpt-4-test");
        let client = OpenAiClient::new(config, reqwest::Client::new());

        // Create a test message
        let message = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Text("Hello".to_string()),
            metadata: MessageMetadata::default(),
        };

        // Create a ChatRequest to test the legacy chat method
        let request = ChatRequest {
            messages: vec![message],
            tools: None,
            common_params: client.common_params.clone(),
            ..Default::default()
        };

        // Test that the request body includes the correct model (via transformers)
        let tx = transformers::OpenAiRequestTransformer;
        let body = tx.transform_chat(&request).unwrap();
        assert_eq!(body["model"], "gpt-4-test");
    }

    #[test]
    fn responses_builtins_and_previous_id_injected_non_stream() {
        // Build config with basic settings
        let cfg = OpenAiConfig::new("test-key").with_model("gpt-4.1-mini");
        let http = reqwest::Client::new();
        let client = OpenAiClient::new(cfg, http);

        // Interceptor to capture transformed JSON body and abort before HTTP send
        struct Capture(Arc<Mutex<Option<serde_json::Value>>>);
        impl HttpInterceptor for Capture {
            fn on_before_send(
                &self,
                _ctx: &HttpRequestContext,
                _rb: reqwest::RequestBuilder,
                body: &serde_json::Value,
                _headers: &reqwest::header::HeaderMap,
            ) -> Result<reqwest::RequestBuilder, LlmError> {
                *self.0.lock().unwrap() = Some(body.clone());
                Err(LlmError::InvalidParameter("stop".into()))
            }
        }

        let captured = Arc::new(Mutex::new(None));
        let cap = Capture(captured.clone());
        let client = client.with_http_interceptors(vec![Arc::new(cap)]);

        // Create request with provider_options for Responses API
        use crate::types::{OpenAiBuiltInTool, OpenAiOptions, ResponsesApiConfig};
        let request =
            crate::types::ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
                .with_openai_options(
                    OpenAiOptions::new()
                        .with_built_in_tool(OpenAiBuiltInTool::WebSearch)
                        .with_responses_api(
                            ResponsesApiConfig::new()
                                .with_previous_response("resp_123".to_string()),
                        ),
                );

        // Invoke non-stream chat, which should hit interceptor and abort
        let err = futures::executor::block_on(client.chat_request(request)).unwrap_err();
        match err {
            LlmError::InvalidParameter(s) => assert_eq!(s, "stop"),
            other => panic!("unexpected: {:?}", other),
        }

        // Assert captured body contains previous_response_id and built-in tool
        let body = captured.lock().unwrap().clone().expect("captured body");
        assert_eq!(
            body.get("previous_response_id")
                .and_then(|v| v.as_str())
                .unwrap_or(""),
            "resp_123"
        );
        let tools = body
            .get("tools")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        assert!(
            tools
                .iter()
                .any(|t| t.get("type").and_then(|s| s.as_str()) == Some("web_search_preview"))
        );
    }

    #[test]
    fn responses_builtins_dedup_non_stream() {
        // Duplicate built-ins should be deduplicated
        let cfg = OpenAiConfig::new("test-key").with_model("gpt-4.1-mini");
        let http = reqwest::Client::new();
        let client = OpenAiClient::new(cfg, http);

        struct Capture(std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>);
        impl HttpInterceptor for Capture {
            fn on_before_send(
                &self,
                _ctx: &HttpRequestContext,
                _rb: reqwest::RequestBuilder,
                body: &serde_json::Value,
                _headers: &reqwest::header::HeaderMap,
            ) -> Result<reqwest::RequestBuilder, LlmError> {
                *self.0.lock().unwrap() = Some(body.clone());
                Err(LlmError::InvalidParameter("stop".into()))
            }
        }
        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let cap = Capture(captured.clone());
        let client = client.with_http_interceptors(vec![std::sync::Arc::new(cap)]);

        // Create request with duplicate built-in tools
        use crate::types::{OpenAiBuiltInTool, OpenAiOptions, ResponsesApiConfig};
        let request =
            crate::types::ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
                .with_openai_options(
                    OpenAiOptions::new()
                        .with_built_in_tool(OpenAiBuiltInTool::WebSearch)
                        .with_built_in_tool(OpenAiBuiltInTool::WebSearch)
                        .with_responses_api(ResponsesApiConfig::new()),
                );

        let _ = futures::executor::block_on(client.chat_request(request));
        let body = captured.lock().unwrap().clone().expect("captured body");
        let tools = body
            .get("tools")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let web_count = tools
            .iter()
            .filter(|t| t.get("type").and_then(|s| s.as_str()) == Some("web_search_preview"))
            .count();
        assert_eq!(web_count, 1, "duplicate built-ins must be deduplicated");
    }

    #[test]
    fn responses_file_search_key_includes_max_num_results() {
        // Two file_search entries with same ids but different max_num_results should NOT be deduplicated
        let cfg = OpenAiConfig::new("test-key").with_model("gpt-4.1-mini");
        let http = reqwest::Client::new();
        let client = OpenAiClient::new(cfg, http);

        struct Capture(std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>);
        impl HttpInterceptor for Capture {
            fn on_before_send(
                &self,
                _ctx: &HttpRequestContext,
                _rb: reqwest::RequestBuilder,
                body: &serde_json::Value,
                _headers: &reqwest::header::HeaderMap,
            ) -> Result<reqwest::RequestBuilder, LlmError> {
                *self.0.lock().unwrap() = Some(body.clone());
                Err(LlmError::InvalidParameter("stop".into()))
            }
        }

        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let cap = Capture(captured.clone());
        let client = client.with_http_interceptors(vec![std::sync::Arc::new(cap)]);

        // Create request with two file_search tools with different max_num_results
        use crate::types::{OpenAiBuiltInTool, OpenAiOptions, ResponsesApiConfig};
        let request =
            crate::types::ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
                .with_openai_options(
                    OpenAiOptions::new()
                        .with_built_in_tool(OpenAiBuiltInTool::FileSearchOptions {
                            vector_store_ids: Some(vec!["vs1".into()]),
                            max_num_results: Some(10),
                            ranking_options: None,
                            filters: None,
                        })
                        .with_built_in_tool(OpenAiBuiltInTool::FileSearchOptions {
                            vector_store_ids: Some(vec!["vs1".into()]),
                            max_num_results: Some(20),
                            ranking_options: None,
                            filters: None,
                        })
                        .with_responses_api(ResponsesApiConfig::new()),
                );

        let _ = futures::executor::block_on(client.chat_request(request));
        let body = captured.lock().unwrap().clone().expect("captured body");
        let tools = body
            .get("tools")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let files: Vec<_> = tools
            .iter()
            .filter(|t| t.get("type").and_then(|s| s.as_str()) == Some("file_search"))
            .collect();
        assert_eq!(
            files.len(),
            2,
            "file_search entries with different max_num_results must both remain"
        );
        let mut maxes: Vec<u64> = files
            .iter()
            .filter_map(|t| t.get("max_num_results").and_then(|v| v.as_u64()))
            .collect();
        maxes.sort();
        assert_eq!(maxes, vec![10, 20]);
    }

    #[test]
    fn responses_file_search_dedup_respects_ranking_options() {
        // Two file_search entries with same ids and max_num_results but different ranking_options should NOT be deduplicated
        let cfg = OpenAiConfig::new("test-key").with_model("gpt-4.1-mini");
        let http = reqwest::Client::new();
        let client = OpenAiClient::new(cfg, http);

        struct Capture(std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>);
        impl HttpInterceptor for Capture {
            fn on_before_send(
                &self,
                _ctx: &HttpRequestContext,
                _rb: reqwest::RequestBuilder,
                body: &serde_json::Value,
                _headers: &reqwest::header::HeaderMap,
            ) -> Result<reqwest::RequestBuilder, LlmError> {
                *self.0.lock().unwrap() = Some(body.clone());
                Err(LlmError::InvalidParameter("stop".into()))
            }
        }

        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let cap = Capture(captured.clone());
        let client = client.with_http_interceptors(vec![std::sync::Arc::new(cap)]);

        // Create request with two file_search tools with different ranking_options
        use crate::types::{
            FileSearchRankingOptions, OpenAiBuiltInTool, OpenAiOptions, ResponsesApiConfig,
        };
        let request =
            crate::types::ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
                .with_openai_options(
                    OpenAiOptions::new()
                        .with_built_in_tool(OpenAiBuiltInTool::FileSearchOptions {
                            vector_store_ids: Some(vec!["vs1".into()]),
                            max_num_results: Some(10),
                            ranking_options: Some(FileSearchRankingOptions {
                                ranker: Some("semantic".into()),
                                score_threshold: Some(0.6),
                            }),
                            filters: None,
                        })
                        .with_built_in_tool(OpenAiBuiltInTool::FileSearchOptions {
                            vector_store_ids: Some(vec!["vs1".into()]),
                            max_num_results: Some(10),
                            ranking_options: Some(FileSearchRankingOptions {
                                ranker: Some("bm25".into()),
                                score_threshold: Some(0.2),
                            }),
                            filters: None,
                        })
                        .with_responses_api(ResponsesApiConfig::new()),
                );

        let _ = futures::executor::block_on(client.chat_request(request));
        let body = captured.lock().unwrap().clone().expect("captured body");
        let tools = body
            .get("tools")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let files: Vec<_> = tools
            .iter()
            .filter(|t| t.get("type").and_then(|s| s.as_str()) == Some("file_search"))
            .collect();
        assert_eq!(
            files.len(),
            2,
            "file_search with different ranking_options must both remain"
        );
    }

    #[test]
    fn responses_file_search_dedup_respects_filters() {
        // Two file_search entries with same ids/max_num_results but different filters should NOT be deduplicated
        let cfg = OpenAiConfig::new("test-key").with_model("gpt-4.1-mini");
        let http = reqwest::Client::new();
        let client = OpenAiClient::new(cfg, http);

        struct Capture(std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>);
        impl HttpInterceptor for Capture {
            fn on_before_send(
                &self,
                _ctx: &HttpRequestContext,
                _rb: reqwest::RequestBuilder,
                body: &serde_json::Value,
                _headers: &reqwest::header::HeaderMap,
            ) -> Result<reqwest::RequestBuilder, LlmError> {
                *self.0.lock().unwrap() = Some(body.clone());
                Err(LlmError::InvalidParameter("stop".into()))
            }
        }

        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let cap = Capture(captured.clone());
        let client = client.with_http_interceptors(vec![std::sync::Arc::new(cap)]);

        // Create request with two file_search tools with different filters
        use crate::types::{
            FileSearchFilter, OpenAiBuiltInTool, OpenAiOptions, ResponsesApiConfig,
        };
        let request =
            crate::types::ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
                .with_openai_options(
                    OpenAiOptions::new()
                        .with_built_in_tool(OpenAiBuiltInTool::FileSearchOptions {
                            vector_store_ids: Some(vec!["vs1".into()]),
                            max_num_results: Some(10),
                            ranking_options: None,
                            filters: Some(FileSearchFilter::Eq {
                                key: "doctype".into(),
                                value: serde_json::json!("pdf"),
                            }),
                        })
                        .with_built_in_tool(OpenAiBuiltInTool::FileSearchOptions {
                            vector_store_ids: Some(vec!["vs1".into()]),
                            max_num_results: Some(10),
                            ranking_options: None,
                            filters: Some(FileSearchFilter::Eq {
                                key: "doctype".into(),
                                value: serde_json::json!("md"),
                            }),
                        })
                        .with_responses_api(ResponsesApiConfig::new()),
                );

        let _ = futures::executor::block_on(client.chat_request(request));
        let body = captured.lock().unwrap().clone().expect("captured body");
        let tools = body
            .get("tools")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let files: Vec<_> = tools
            .iter()
            .filter(|t| t.get("type").and_then(|s| s.as_str()) == Some("file_search"))
            .collect();
        assert_eq!(
            files.len(),
            2,
            "file_search with different filters must both remain"
        );
    }

    #[test]
    fn responses_response_format_injected_non_stream() {
        let cfg = OpenAiConfig::new("test-key").with_model("gpt-4.1-mini");
        let http = reqwest::Client::new();
        let client = OpenAiClient::new(cfg, http);

        struct Capture(std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>);
        impl HttpInterceptor for Capture {
            fn on_before_send(
                &self,
                _ctx: &HttpRequestContext,
                _rb: reqwest::RequestBuilder,
                body: &serde_json::Value,
                _headers: &reqwest::header::HeaderMap,
            ) -> Result<reqwest::RequestBuilder, LlmError> {
                *self.0.lock().unwrap() = Some(body.clone());
                Err(LlmError::InvalidParameter("stop".into()))
            }
        }
        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let cap = Capture(captured.clone());
        let client = client.with_http_interceptors(vec![std::sync::Arc::new(cap)]);

        // Create request with response_format in provider_options
        use crate::types::{OpenAiOptions, ResponsesApiConfig};
        let response_format = serde_json::json!({
            "type": "json_object",
            "json_schema": {
                "name": "response",
                "strict": true,
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"]
                }
            }
        });
        let request =
            crate::types::ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
                .with_openai_options(OpenAiOptions::new().with_responses_api(
                    ResponsesApiConfig::new().with_response_format(response_format),
                ));

        let _ = futures::executor::block_on(client.chat_request(request));
        let body = captured.lock().unwrap().clone().expect("captured body");
        let rf = body
            .get("response_format")
            .cloned()
            .expect("has response_format");
        assert_eq!(
            rf.get("type").and_then(|v| v.as_str()).unwrap_or(""),
            "json_object"
        );
        let sch = rf
            .get("json_schema")
            .and_then(|v| v.get("schema"))
            .cloned()
            .expect("schema present");
        assert_eq!(
            sch.get("type").and_then(|v| v.as_str()).unwrap_or(""),
            "object"
        );
    }

    #[test]
    fn responses_response_format_injected_stream() {
        let cfg = OpenAiConfig::new("test-key").with_model("gpt-4.1-mini");
        let http = reqwest::Client::new();
        let client = OpenAiClient::new(cfg, http);

        struct Capture(std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>);
        impl HttpInterceptor for Capture {
            fn on_before_send(
                &self,
                _ctx: &HttpRequestContext,
                _rb: reqwest::RequestBuilder,
                body: &serde_json::Value,
                _headers: &reqwest::header::HeaderMap,
            ) -> Result<reqwest::RequestBuilder, LlmError> {
                *self.0.lock().unwrap() = Some(body.clone());
                Err(LlmError::InvalidParameter("stop".into()))
            }
        }
        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let cap = Capture(captured.clone());
        let client = client.with_http_interceptors(vec![std::sync::Arc::new(cap)]);

        // Create request with named schema response_format
        use crate::types::{OpenAiOptions, ResponsesApiConfig};
        let response_format = serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": "User",
                "strict": true,
                "schema": {
                    "type": "object",
                    "properties": {"age": {"type": "integer"}},
                    "required": ["age"]
                }
            }
        });
        let mut request =
            crate::types::ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
                .with_openai_options(OpenAiOptions::new().with_responses_api(
                    ResponsesApiConfig::new().with_response_format(response_format),
                ));
        request.stream = true;

        // trigger stream path
        let _ = futures::executor::block_on(client.chat_stream_request(request));
        let body = captured.lock().unwrap().clone().expect("captured body");
        let rf = body
            .get("response_format")
            .cloned()
            .expect("has response_format");
        assert_eq!(
            rf.get("type").and_then(|v| v.as_str()).unwrap_or(""),
            "json_schema"
        );
        let name = rf
            .get("json_schema")
            .and_then(|v| v.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert_eq!(name, "User");
    }

    #[test]
    fn test_responses_api_extended_params() {
        // Test that all new ResponsesApiConfig parameters are correctly injected
        let config = OpenAiConfig::new("test-key")
            .with_base_url("https://api.openai.com/v1")
            .with_model("gpt-4o");
        let client = OpenAiClient::new(config, reqwest::Client::new());

        struct Capture(Arc<Mutex<Option<serde_json::Value>>>);
        impl HttpInterceptor for Capture {
            fn on_before_send(
                &self,
                _ctx: &HttpRequestContext,
                builder: reqwest::RequestBuilder,
                body: &serde_json::Value,
                _headers: &reqwest::header::HeaderMap,
            ) -> Result<reqwest::RequestBuilder, LlmError> {
                *self.0.lock().unwrap() = Some(body.clone());
                Err(LlmError::InvalidParameter("stop".into()))
            }
        }
        let captured = Arc::new(Mutex::new(None));
        let cap = Capture(captured.clone());
        let client = client.with_http_interceptors(vec![Arc::new(cap)]);

        // Create request with all extended ResponsesApiConfig parameters
        use crate::types::{OpenAiOptions, ResponsesApiConfig, TextVerbosity, Truncation};
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("user_id".to_string(), "test_123".to_string());

        let request = crate::types::ChatRequest::new(vec![
            crate::types::ChatMessage::user("Test message").build(),
        ])
        .with_openai_options(
            OpenAiOptions::new().with_responses_api(
                ResponsesApiConfig::new()
                    .with_background(true)
                    .with_include(vec!["file_search_call.results".to_string()])
                    .with_instructions("You are a helpful assistant".to_string())
                    .with_max_tool_calls(10)
                    .with_store(false)
                    .with_truncation(Truncation::Auto)
                    .with_text_verbosity(TextVerbosity::Medium)
                    .with_metadata(metadata.clone())
                    .with_parallel_tool_calls(true),
            ),
        );

        // Trigger chat path
        let _ = futures::executor::block_on(client.chat_request(request));
        let body = captured.lock().unwrap().clone().expect("captured body");

        // Verify all parameters are injected
        assert_eq!(body.get("background").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(
            body.get("include")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.get(0))
                .and_then(|v| v.as_str()),
            Some("file_search_call.results")
        );
        assert_eq!(
            body.get("instructions").and_then(|v| v.as_str()),
            Some("You are a helpful assistant")
        );
        assert_eq!(
            body.get("max_tool_calls").and_then(|v| v.as_u64()),
            Some(10)
        );
        assert_eq!(body.get("store").and_then(|v| v.as_bool()), Some(false));
        assert_eq!(
            body.get("truncation").and_then(|v| v.as_str()),
            Some("auto")
        );

        // text_verbosity should be nested under "text.verbosity"
        assert_eq!(
            body.get("text")
                .and_then(|t| t.get("verbosity"))
                .and_then(|v| v.as_str()),
            Some("medium")
        );

        assert_eq!(
            body.get("parallel_tool_calls").and_then(|v| v.as_bool()),
            Some(true)
        );

        // Verify metadata
        let meta = body.get("metadata").expect("has metadata");
        assert_eq!(
            meta.get("user_id").and_then(|v| v.as_str()),
            Some("test_123")
        );
    }
}
