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
    AudioCapability, ChatCapability, ImageGenerationCapability, MusicGenerationCapability,
    FileManagementCapability, ImageExtras, ProviderCapabilities, VideoGenerationCapability,
};
use crate::types::*;
use std::sync::Arc;

use super::chat::MinimaxiChatCapability;
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
    /// Chat capability
    chat_capability: MinimaxiChatCapability,
    /// Tracing configuration
    tracing_config: Option<crate::observability::tracing::TracingConfig>,
    /// Tracing guard to keep tracing system active
    #[allow(dead_code)]
    _tracing_guard: Option<()>,
    /// Unified retry options
    retry_options: Option<RetryOptions>,
    /// Optional HTTP interceptors applied to all requests
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
}

impl Clone for MinimaxiClient {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            http_client: self.http_client.clone(),
            http_config: self.http_config.clone(),
            chat_capability: self.chat_capability.clone(),
            tracing_config: self.tracing_config.clone(),
            _tracing_guard: None, // Don't clone the tracing guard
            retry_options: self.retry_options.clone(),
            http_interceptors: self.http_interceptors.clone(),
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
        let chat_capability = MinimaxiChatCapability::new(
            config.api_key.clone(),
            config.base_url.clone(),
            http_client.clone(),
            config.common_params.clone(),
            http_config.clone(),
        );

        Self {
            config,
            http_client,
            http_config,
            chat_capability,
            tracing_config: None,
            _tracing_guard: None,
            retry_options: None,
            http_interceptors: Vec::new(),
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

    /// Get chat capability
    pub fn chat_capability(&self) -> &MinimaxiChatCapability {
        &self.chat_capability
    }

    /// Set tracing configuration
    pub fn with_tracing(mut self, config: crate::observability::tracing::TracingConfig) -> Self {
        self.tracing_config = Some(config);
        self
    }

    /// Set retry options
    pub fn with_retry(mut self, retry_options: RetryOptions) -> Self {
        self.retry_options = Some(retry_options);
        self.chat_capability = self
            .chat_capability
            .clone()
            .with_retry_options(self.retry_options.clone());
        self
    }

    /// Set HTTP interceptors
    pub fn with_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors;
        self.chat_capability = self
            .chat_capability
            .clone()
            .with_interceptors(self.http_interceptors.clone());
        self
    }

    /// Install model-level middlewares for chat requests.
    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.chat_capability = self.chat_capability.clone().with_middlewares(middlewares);
        self
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

    fn as_file_management_capability(&self) -> Option<&dyn crate::traits::FileManagementCapability> {
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
        self.chat_capability.chat_with_tools(messages, tools).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        self.chat_capability.chat_stream(messages, tools).await
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
    fn with_interceptors_propagates_to_chat_capability() {
        let cfg = MinimaxiConfig::new("test-key");
        let client =
            MinimaxiClient::new(cfg, reqwest::Client::new()).with_interceptors(vec![Arc::new(
                NoopInterceptor,
            )]);

        assert_eq!(client.http_interceptors.len(), 1);
        assert_eq!(client.chat_capability().interceptors.len(), 1);
    }
}
