//! Azure OpenAI client (OpenAI-compatible endpoints).
//!
//! This client is built on top of `ProviderSpec` + executor builders in `siumai-core`.
//! It mirrors the approach used by other spec-driven providers (e.g. Anthropic Vertex).

use super::{AzureOpenAiConfig, AzureOpenAiSpec};
use crate::client::LlmClient;
use crate::core::ProviderContext;
use crate::core::ProviderSpec;
use crate::error::LlmError;
use crate::execution::executors::audio::{AudioExecutor, AudioExecutorBuilder};
use crate::execution::executors::chat::{ChatExecutor, ChatExecutorBuilder, HttpChatExecutor};
use crate::execution::executors::embedding::{
    EmbeddingExecutor, EmbeddingExecutorBuilder, HttpEmbeddingExecutor,
};
use crate::execution::executors::files::{FilesExecutor, FilesExecutorBuilder, HttpFilesExecutor};
use crate::execution::executors::image::{HttpImageExecutor, ImageExecutor, ImageExecutorBuilder};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::retry_api::RetryOptions;
use crate::streaming::ChatStream;
use crate::traits::{
    AudioCapability, ChatCapability, EmbeddingCapability, FileManagementCapability,
    ImageGenerationCapability, ProviderCapabilities, SpeechCapability, TranscriptionCapability,
};
use crate::types::{
    AudioFeature, ChatMessage, ChatRequest, ChatResponse, EmbeddingRequest, EmbeddingResponse,
    FileDeleteResponse, FileListQuery, FileListResponse, FileObject, FileUploadRequest,
    ImageGenerationRequest, ImageGenerationResponse, SttRequest, SttResponse, Tool, TtsRequest,
    TtsResponse,
};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct AzureOpenAiClient {
    config: AzureOpenAiConfig,
    http_client: reqwest::Client,
    retry_options: Option<RetryOptions>,
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
}

impl std::fmt::Debug for AzureOpenAiClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureOpenAiClient")
            .field("base_url", &self.config.base_url)
            .field("model", &self.config.common_params.model)
            .field("api_version", &self.config.url_config.api_version)
            .field(
                "deployment_based_urls",
                &self.config.url_config.use_deployment_based_urls,
            )
            .field("chat_mode", &self.config.chat_mode)
            .field("has_retry", &self.retry_options.is_some())
            .field("interceptors", &self.http_interceptors.len())
            .field("middlewares", &self.model_middlewares.len())
            .finish()
    }
}

impl AzureOpenAiClient {
    pub fn new(config: AzureOpenAiConfig, http_client: reqwest::Client) -> Result<Self, LlmError> {
        config.validate()?;
        Ok(Self {
            config,
            http_client,
            retry_options: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
        })
    }

    pub fn with_retry_options(mut self, opts: Option<RetryOptions>) -> Self {
        self.retry_options = opts;
        self
    }

    pub fn with_http_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
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

    fn build_context(&self) -> ProviderContext {
        let mut ctx = ProviderContext::new(
            "azure",
            self.config.base_url.clone(),
            Some(self.config.api_key.clone()),
            self.config.http_config.headers.clone(),
        );

        if self.config.url_config.use_deployment_based_urls
            && !self.config.common_params.model.is_empty()
        {
            ctx.extras.insert(
                "azureDeploymentId".to_string(),
                serde_json::Value::String(self.config.common_params.model.clone()),
            );
        }

        ctx
    }

    fn build_spec(&self) -> Arc<dyn crate::core::ProviderSpec> {
        Arc::new(
            AzureOpenAiSpec::new(self.config.url_config.clone())
                .with_chat_mode(self.config.chat_mode)
                .with_provider_metadata_key(self.config.provider_metadata_key),
        )
    }

    fn build_chat_executor(&self, request: &ChatRequest) -> Arc<HttpChatExecutor> {
        let ctx = self.build_context();
        let spec = self.build_spec();

        let bundle = spec.choose_chat_transformers(request, &ctx);
        let before_send_hook = spec.chat_before_send(request, &ctx);

        let mut builder = ChatExecutorBuilder::new("azure", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_transformer_bundle(bundle)
            .with_stream_disable_compression(self.config.http_config.stream_disable_compression)
            .with_interceptors(self.http_interceptors.clone())
            .with_middlewares(self.model_middlewares.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

        if let Some(hook) = before_send_hook {
            builder = builder.with_before_send(hook);
        }
        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        builder.build()
    }

    fn build_embedding_executor(&self, request: &EmbeddingRequest) -> Arc<HttpEmbeddingExecutor> {
        let ctx = self.build_context();
        let spec = self.build_spec();

        let mut builder = EmbeddingExecutorBuilder::new("azure", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        builder.build_for_request(request)
    }

    fn build_image_executor(&self, request: &ImageGenerationRequest) -> Arc<HttpImageExecutor> {
        let ctx = self.build_context();
        let spec = self.build_spec();

        let mut builder = ImageExecutorBuilder::new("azure", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        builder.build_for_request(request)
    }

    fn build_audio_executor(&self) -> Arc<crate::execution::executors::audio::HttpAudioExecutor> {
        let ctx = self.build_context();
        let spec = self.build_spec();

        let mut builder = AudioExecutorBuilder::new("azure", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        builder.build()
    }

    fn build_files_executor(&self) -> Arc<HttpFilesExecutor> {
        let ctx = self.build_context();
        let spec = self.build_spec();

        let mut builder = FilesExecutorBuilder::new("azure", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        builder.build()
    }

    async fn chat_request_via_spec(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute(&*exec, request).await
    }

    async fn chat_stream_request_via_spec(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStream, LlmError> {
        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute_stream(&*exec, request).await
    }
}

#[async_trait]
impl ChatCapability for AzureOpenAiClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        let mut req =
            ChatRequest::new(messages).with_common_params(self.config.common_params.clone());
        if let Some(t) = tools {
            req.tools = Some(t);
        }
        self.chat_request(req).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        let mut req =
            ChatRequest::new(messages).with_common_params(self.config.common_params.clone());
        req.stream = true;
        if let Some(t) = tools {
            req.tools = Some(t);
        }
        self.chat_stream_request(req).await
    }

    async fn chat_request(&self, mut request: ChatRequest) -> Result<ChatResponse, LlmError> {
        if request.common_params.model.trim().is_empty() {
            request.common_params.model = self.config.common_params.model.clone();
        }
        if request.common_params.model.trim().is_empty() {
            return Err(LlmError::InvalidParameter(
                "Azure OpenAI request requires a model (deployment id)".to_string(),
            ));
        }
        request.stream = false;
        self.chat_request_via_spec(request).await
    }

    async fn chat_stream_request(&self, mut request: ChatRequest) -> Result<ChatStream, LlmError> {
        if request.common_params.model.trim().is_empty() {
            request.common_params.model = self.config.common_params.model.clone();
        }
        if request.common_params.model.trim().is_empty() {
            return Err(LlmError::InvalidParameter(
                "Azure OpenAI request requires a model (deployment id)".to_string(),
            ));
        }
        request.stream = true;
        self.chat_stream_request_via_spec(request).await
    }
}

#[async_trait]
impl EmbeddingCapability for AzureOpenAiClient {
    async fn embed(&self, input: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        let model = if self.config.common_params.model.trim().is_empty() {
            "text-embedding-3-small".to_string()
        } else {
            self.config.common_params.model.clone()
        };
        let req = EmbeddingRequest::new(input).with_model(model);
        let exec = self.build_embedding_executor(&req);
        EmbeddingExecutor::execute(&*exec, req).await
    }

    fn embedding_dimension(&self) -> usize {
        1536
    }
}

#[async_trait]
impl ImageGenerationCapability for AzureOpenAiClient {
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let exec = self.build_image_executor(&request);
        ImageExecutor::execute(&*exec, request).await
    }
}

#[async_trait]
impl AudioCapability for AzureOpenAiClient {
    fn supported_features(&self) -> &[AudioFeature] {
        use AudioFeature::*;
        const FEATURES: &[AudioFeature] = &[TextToSpeech, SpeechToText];
        FEATURES
    }

    async fn text_to_speech(&self, request: TtsRequest) -> Result<TtsResponse, LlmError> {
        let exec = self.build_audio_executor();
        let result = AudioExecutor::tts(&*exec, request.clone()).await?;
        Ok(TtsResponse {
            audio_data: result.audio_data,
            format: request.format.unwrap_or_else(|| "mp3".to_string()),
            duration: result.duration,
            sample_rate: result.sample_rate,
            metadata: HashMap::new(),
        })
    }

    async fn speech_to_text(&self, request: SttRequest) -> Result<SttResponse, LlmError> {
        let exec = self.build_audio_executor();
        let result = AudioExecutor::stt(&*exec, request).await?;
        let raw = result.raw;

        let language = raw
            .get("language")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let duration = raw
            .get("duration")
            .and_then(|v| v.as_f64())
            .map(|d| d as f32);

        let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
        if let Some(usage) = raw.get("usage") {
            metadata.insert("usage".to_string(), usage.clone());
        }
        if let Some(segments) = raw.get("segments") {
            metadata.insert("segments".to_string(), segments.clone());
        }
        if let Some(logprobs) = raw.get("logprobs") {
            metadata.insert("logprobs".to_string(), logprobs.clone());
        }

        Ok(SttResponse {
            text: result.text,
            language,
            confidence: None,
            words: None,
            duration,
            metadata,
        })
    }
}

#[async_trait]
impl FileManagementCapability for AzureOpenAiClient {
    async fn upload_file(&self, request: FileUploadRequest) -> Result<FileObject, LlmError> {
        let exec = self.build_files_executor();
        FilesExecutor::upload(&*exec, request).await
    }

    async fn list_files(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError> {
        let exec = self.build_files_executor();
        FilesExecutor::list(&*exec, query).await
    }

    async fn retrieve_file(&self, file_id: String) -> Result<FileObject, LlmError> {
        let exec = self.build_files_executor();
        FilesExecutor::retrieve(&*exec, file_id).await
    }

    async fn delete_file(&self, file_id: String) -> Result<FileDeleteResponse, LlmError> {
        let exec = self.build_files_executor();
        FilesExecutor::delete(&*exec, file_id).await
    }

    async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        let exec = self.build_files_executor();
        FilesExecutor::get_content(&*exec, file_id).await
    }
}

impl LlmClient for AzureOpenAiClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("azure")
    }

    fn supported_models(&self) -> Vec<String> {
        vec![self.config.common_params.model.clone()]
    }

    fn capabilities(&self) -> ProviderCapabilities {
        AzureOpenAiSpec::new(self.config.url_config.clone())
            .with_chat_mode(self.config.chat_mode)
            .with_provider_metadata_key(self.config.provider_metadata_key)
            .capabilities()
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
        Some(self)
    }

    fn as_speech_capability(&self) -> Option<&dyn SpeechCapability> {
        Some(self)
    }

    fn as_transcription_capability(&self) -> Option<&dyn TranscriptionCapability> {
        Some(self)
    }

    fn as_image_generation_capability(&self) -> Option<&dyn ImageGenerationCapability> {
        Some(self)
    }

    fn as_file_management_capability(&self) -> Option<&dyn FileManagementCapability> {
        Some(self)
    }
}
