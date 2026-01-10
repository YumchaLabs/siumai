//! MiniMaxi Client Implementation
//!
//! Main client implementation that aggregates all MiniMaxi capabilities.

use async_trait::async_trait;

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::retry_api::RetryOptions;
use crate::streaming::ChatStream;
use crate::traits::{
    AudioCapability, ChatCapability, FileManagementCapability, ImageExtras,
    ImageGenerationCapability, MusicGenerationCapability, ProviderCapabilities,
    VideoGenerationCapability,
};
use crate::types::*;
use std::sync::Arc;

use super::config::MinimaxiConfig;
use super::files::MinimaxiFiles;

/// MiniMaxi client that implements all capabilities
pub struct MinimaxiClient {
    /// Configuration
    config: MinimaxiConfig,
    /// HTTP client
    http_client: reqwest::Client,
    /// HTTP configuration (headers, proxy, timeouts)
    http_config: HttpConfig,
    /// Tracing configuration
    tracing_config: Option<crate::observability::tracing::TracingConfig>,
    /// Tracing guard to keep tracing system active
    #[allow(dead_code)]
    _tracing_guard: Option<()>,
    /// Unified retry options
    retry_options: Option<RetryOptions>,
    /// Optional HTTP interceptors applied to all requests
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional model-level middlewares (applied to chat).
    model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    /// Optional custom HTTP transport (Vercel-style "custom fetch" parity).
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
}

impl Clone for MinimaxiClient {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            http_client: self.http_client.clone(),
            http_config: self.http_config.clone(),
            tracing_config: self.tracing_config.clone(),
            _tracing_guard: None, // Don't clone the tracing guard
            retry_options: self.retry_options.clone(),
            http_interceptors: self.http_interceptors.clone(),
            model_middlewares: self.model_middlewares.clone(),
            http_transport: self.http_transport.clone(),
        }
    }
}

impl std::fmt::Debug for MinimaxiClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MinimaxiClient")
            .field("provider_id", &"minimaxi")
            .field("model", &self.config.common_params.model)
            .field("base_url", &self.config.base_url)
            .field(
                "stream_disable_compression",
                &self.http_config.stream_disable_compression,
            )
            .field("has_tracing", &self.tracing_config.is_some())
            .finish()
    }
}

impl MinimaxiClient {
    /// Create a new MiniMaxi client
    pub fn new(config: MinimaxiConfig, http_client: reqwest::Client) -> Self {
        Self::new_with_http_config(config, http_client, HttpConfig::default())
    }

    pub(crate) fn new_with_http_config(
        config: MinimaxiConfig,
        http_client: reqwest::Client,
        http_config: HttpConfig,
    ) -> Self {
        Self {
            config,
            http_client,
            http_config,
            tracing_config: None,
            _tracing_guard: None,
            retry_options: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
            http_transport: None,
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &MinimaxiConfig {
        &self.config
    }

    /// Get the HTTP client
    pub fn http_client(&self) -> &reqwest::Client {
        &self.http_client
    }

    /// Set tracing configuration
    pub fn with_tracing(mut self, config: crate::observability::tracing::TracingConfig) -> Self {
        self.tracing_config = Some(config);
        self
    }

    /// Set retry options
    pub fn with_retry(mut self, retry_options: RetryOptions) -> Self {
        self.retry_options = Some(retry_options);
        self
    }

    /// Set HTTP interceptors
    pub fn with_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors;
        self
    }

    /// Install model-level middlewares for chat requests.
    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.model_middlewares = middlewares;
        self
    }

    /// Set a custom HTTP transport (Vercel-style "custom fetch" parity).
    pub fn with_http_transport(
        mut self,
        transport: Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.http_transport = Some(transport);
        self
    }

    fn build_context(&self) -> crate::core::ProviderContext {
        super::utils::build_context(
            &self.config.api_key,
            &self.config.base_url,
            &self.http_config,
        )
    }

    fn build_chat_executor(
        &self,
        request: &ChatRequest,
    ) -> Arc<crate::execution::executors::chat::HttpChatExecutor> {
        use crate::core::ProviderSpec;
        use crate::execution::executors::chat::ChatExecutorBuilder;

        let ctx = self.build_context();
        let spec = Arc::new(super::spec::MinimaxiSpec::new());
        let bundle = spec.choose_chat_transformers(request, &ctx);
        let before_send_hook = spec.chat_before_send(request, &ctx);

        let mut builder = ChatExecutorBuilder::new("minimaxi", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_transformer_bundle(bundle)
            .with_stream_disable_compression(self.http_config.stream_disable_compression)
            .with_interceptors(self.http_interceptors.clone())
            .with_middlewares(self.model_middlewares.clone());

        if let Some(transport) = self.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        if let Some(hook) = before_send_hook {
            builder = builder.with_before_send(hook);
        }

        builder.build()
    }
}

#[async_trait]
impl LlmClient for MinimaxiClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("minimaxi")
    }

    fn supported_models(&self) -> Vec<String> {
        vec![
            "MiniMax-M2".to_string(),
            "speech-2.6-hd".to_string(),
            "speech-2.6-turbo".to_string(),
            "hailuo-2.3".to_string(),
            "hailuo-2.3-fast".to_string(),
            "music-2.0".to_string(),
        ]
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_speech() // MiniMaxi only supports TTS (text-to-audio)
            .with_file_management()
            .with_custom_feature("video", true)
            .with_image_generation()
            .with_custom_feature("music", true)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_audio_capability(&self) -> Option<&dyn crate::traits::AudioCapability> {
        Some(self)
    }

    fn as_speech_capability(&self) -> Option<&dyn crate::traits::SpeechCapability> {
        Some(self)
    }

    fn as_transcription_capability(&self) -> Option<&dyn crate::traits::TranscriptionCapability> {
        None
    }

    fn as_image_generation_capability(
        &self,
    ) -> Option<&dyn crate::traits::ImageGenerationCapability> {
        Some(self)
    }

    fn as_image_extras(&self) -> Option<&dyn crate::traits::ImageExtras> {
        Some(self)
    }

    fn as_file_management_capability(
        &self,
    ) -> Option<&dyn crate::traits::FileManagementCapability> {
        Some(self)
    }

    fn as_video_generation_capability(
        &self,
    ) -> Option<&dyn crate::traits::VideoGenerationCapability> {
        Some(self)
    }

    fn as_music_generation_capability(
        &self,
    ) -> Option<&dyn crate::traits::MusicGenerationCapability> {
        Some(self)
    }
}

#[async_trait]
impl ChatCapability for MinimaxiClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        use crate::execution::executors::chat::ChatExecutor;

        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(self.config.common_params.clone())
            .http_config(self.http_config.clone());
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        let request = builder.build();

        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute(&*exec, request).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        use crate::execution::executors::chat::ChatExecutor;

        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(self.config.common_params.clone())
            .http_config(self.http_config.clone())
            .stream(true);
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        let request = builder.build();

        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute_stream(&*exec, request).await
    }
}

#[async_trait]
impl AudioCapability for MinimaxiClient {
    fn supported_features(&self) -> &[AudioFeature] {
        &[AudioFeature::TextToSpeech]
    }

    async fn text_to_speech(&self, request: TtsRequest) -> Result<TtsResponse, LlmError> {
        use crate::execution::executors::audio::AudioExecutor;

        let exec = super::audio::build_audio_executor(
            &self.config.api_key,
            &self.config.base_url,
            &self.http_config,
            &self.http_client,
            self.retry_options.as_ref(),
            &self.http_interceptors,
            self.http_transport.clone(),
        );

        // Execute TTS request - transformer will handle JSON parsing, hex decoding, and metadata extraction
        let result = exec.tts(request.clone()).await?;

        // Build response
        let format = request.format.unwrap_or_else(|| "mp3".to_string());

        Ok(TtsResponse {
            audio_data: result.audio_data,
            format,
            duration: result.duration,
            sample_rate: result.sample_rate,
            metadata: Default::default(),
        })
    }

    async fn speech_to_text(&self, _request: SttRequest) -> Result<SttResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Speech-to-text is not yet supported for MiniMaxi".to_string(),
        ))
    }
}

#[async_trait]
impl FileManagementCapability for MinimaxiClient {
    async fn upload_file(&self, request: FileUploadRequest) -> Result<FileObject, LlmError> {
        let files = MinimaxiFiles::new(
            self.config.clone(),
            self.http_client.clone(),
            self.http_config.clone(),
            self.http_interceptors.clone(),
            self.retry_options.clone(),
            self.http_transport.clone(),
        );
        files.upload_file(request).await
    }

    async fn list_files(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError> {
        let files = MinimaxiFiles::new(
            self.config.clone(),
            self.http_client.clone(),
            self.http_config.clone(),
            self.http_interceptors.clone(),
            self.retry_options.clone(),
            self.http_transport.clone(),
        );
        files.list_files(query).await
    }

    async fn retrieve_file(&self, file_id: String) -> Result<FileObject, LlmError> {
        let files = MinimaxiFiles::new(
            self.config.clone(),
            self.http_client.clone(),
            self.http_config.clone(),
            self.http_interceptors.clone(),
            self.retry_options.clone(),
            self.http_transport.clone(),
        );
        files.retrieve_file(file_id).await
    }

    async fn delete_file(&self, file_id: String) -> Result<FileDeleteResponse, LlmError> {
        let files = MinimaxiFiles::new(
            self.config.clone(),
            self.http_client.clone(),
            self.http_config.clone(),
            self.http_interceptors.clone(),
            self.retry_options.clone(),
            self.http_transport.clone(),
        );
        files.delete_file(file_id).await
    }

    async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        let files = MinimaxiFiles::new(
            self.config.clone(),
            self.http_client.clone(),
            self.http_config.clone(),
            self.http_interceptors.clone(),
            self.retry_options.clone(),
            self.http_transport.clone(),
        );
        files.get_file_content(file_id).await
    }
}

#[async_trait]
impl ImageGenerationCapability for MinimaxiClient {
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        use crate::execution::executors::image::ImageExecutor;

        let exec = super::image::build_image_executor(
            &request,
            &self.config.api_key,
            &self.config.base_url,
            &self.http_config,
            &self.http_client,
            self.retry_options.as_ref(),
            &self.http_interceptors,
            self.http_transport.clone(),
        );
        exec.execute(request).await
    }
}

impl ImageExtras for MinimaxiClient {
    fn get_supported_sizes(&self) -> Vec<String> {
        vec![
            "1024x1024".to_string(), // 1:1
            "1280x720".to_string(),  // 16:9
            "1152x864".to_string(),  // 4:3
            "1248x832".to_string(),  // 3:2
            "832x1248".to_string(),  // 2:3
            "864x1152".to_string(),  // 3:4
            "720x1280".to_string(),  // 9:16
            "1344x576".to_string(),  // 21:9
        ]
    }

    fn get_supported_formats(&self) -> Vec<String> {
        vec!["url".to_string(), "b64_json".to_string()]
    }
}

#[async_trait]
impl VideoGenerationCapability for MinimaxiClient {
    async fn create_video_task(
        &self,
        request: crate::types::video::VideoGenerationRequest,
    ) -> Result<crate::types::video::VideoGenerationResponse, LlmError> {
        super::video::create_video_task(
            &self.config.api_key,
            &self.config.base_url,
            &self.http_config,
            &self.http_client,
            self.retry_options.as_ref(),
            &self.http_interceptors,
            self.http_transport.clone(),
            request,
        )
        .await
    }

    async fn query_video_task(
        &self,
        task_id: &str,
    ) -> Result<crate::types::video::VideoTaskStatusResponse, LlmError> {
        super::video::query_video_task(
            &self.config.api_key,
            &self.config.base_url,
            &self.http_config,
            &self.http_client,
            self.retry_options.as_ref(),
            &self.http_interceptors,
            self.http_transport.clone(),
            task_id,
        )
        .await
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

#[async_trait]
impl MusicGenerationCapability for MinimaxiClient {
    async fn generate_music(
        &self,
        request: crate::types::music::MusicGenerationRequest,
    ) -> Result<crate::types::music::MusicGenerationResponse, LlmError> {
        super::music::generate_music(
            &self.config.api_key,
            &self.config.base_url,
            &self.http_config,
            &self.http_client,
            self.retry_options.as_ref(),
            &self.http_interceptors,
            self.http_transport.clone(),
            request,
        )
        .await
    }

    fn get_supported_music_models(&self) -> Vec<String> {
        super::music::get_supported_music_models()
    }

    fn supports_lyrics(&self) -> bool {
        true
    }

    fn get_supported_audio_formats(&self) -> Vec<String> {
        super::music::get_supported_audio_formats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Default)]
    struct NoopInterceptor;

    impl crate::execution::http::interceptor::HttpInterceptor for NoopInterceptor {}

    #[test]
    fn build_chat_executor_inherits_interceptors_and_retry() {
        let cfg = MinimaxiConfig::new("test-key");
        let client = MinimaxiClient::new(cfg, reqwest::Client::new())
            .with_interceptors(vec![Arc::new(NoopInterceptor)])
            .with_retry(RetryOptions::backoff());

        let req = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_common_params(client.config().common_params.clone())
            .with_http_config(client.http_config.clone());

        let exec = client.build_chat_executor(&req);
        assert_eq!(exec.policy.interceptors.len(), 1);
        assert!(exec.policy.retry_options.is_some());
    }
}
