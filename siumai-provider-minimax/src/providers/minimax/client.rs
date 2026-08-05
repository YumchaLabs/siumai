//! MiniMax Client Implementation
//!
//! Main client implementation that aggregates all MiniMax capabilities.

use async_trait::async_trait;

use crate::core_compat::client::LlmClient;
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

use super::config::MinimaxConfig;
use super::files::MinimaxFiles;

/// MiniMax client that implements all capabilities
pub struct MinimaxClient {
    /// Configuration
    config: MinimaxConfig,
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

impl Clone for MinimaxClient {
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

impl std::fmt::Debug for MinimaxClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MinimaxClient")
            .field("provider_id", &"minimax")
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

impl MinimaxClient {
    /// Construct a `MinimaxClient` from a config-first `MinimaxConfig`.
    pub fn from_config(config: MinimaxConfig) -> Result<Self, LlmError> {
        let http_client =
            crate::execution::http::client::build_http_client_from_config(&config.http_config)?;
        Self::with_http_client(config, http_client)
    }

    /// Construct a `MinimaxClient` from a `MinimaxConfig` with a caller-supplied HTTP client.
    pub fn with_http_client(
        config: MinimaxConfig,
        http_client: reqwest::Client,
    ) -> Result<Self, LlmError> {
        config.validate()?;

        let http_interceptors = config.http_interceptors.clone();
        let model_middlewares = config.model_middlewares.clone();
        let http_config = config.http_config.clone();
        let mut client = Self::new_with_http_config(config, http_client, http_config);

        if let Some(transport) = client.config.http_transport.clone() {
            client = client.with_http_transport(transport);
        }

        Ok(client
            .with_interceptors(http_interceptors)
            .with_model_middlewares(model_middlewares))
    }

    /// Create a new MiniMax client
    pub fn new(config: MinimaxConfig, http_client: reqwest::Client) -> Self {
        let http_config = config.http_config.clone();
        Self::new_with_http_config(config, http_client, http_config)
    }

    pub(crate) fn new_with_http_config(
        config: MinimaxConfig,
        http_client: reqwest::Client,
        http_config: HttpConfig,
    ) -> Self {
        let http_transport = config.http_transport.clone();
        Self {
            config,
            http_client,
            http_config,
            tracing_config: None,
            _tracing_guard: None,
            retry_options: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
            http_transport,
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &MinimaxConfig {
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

    fn prepare_chat_request(
        &self,
        request: ChatRequest,
        stream: bool,
    ) -> Result<ChatRequest, LlmError> {
        let request = crate::provider_utils::chat_request::normalize_chat_request(
            request,
            crate::provider_utils::chat_request::ChatRequestDefaults::new(
                &self.config.common_params,
            )
            .with_provider_options_map(&self.config.default_provider_options_map),
            stream,
        );
        if request.common_params.model.trim().is_empty() {
            return Err(LlmError::InvalidParameter(
                "Model must be specified".to_string(),
            ));
        }
        Ok(request)
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
        let spec = Arc::new(super::spec::MinimaxSpec::for_endpoint(
            self.config.chat_endpoint,
        ));
        let bundle = spec.choose_chat_transformers(request, &ctx);

        let mut builder = ChatExecutorBuilder::new("minimax", self.http_client.clone())
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

        builder.build()
    }
}

impl crate::traits::ModelMetadata for MinimaxClient {
    fn provider_id(&self) -> &str {
        "minimax"
    }

    fn model_id(&self) -> &str {
        &self.config.common_params.model
    }
}

#[async_trait]
impl LlmClient for MinimaxClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("minimax")
    }

    fn supported_models(&self) -> Vec<String> {
        super::models::chat_models_for_endpoint(self.config.chat_endpoint)
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_speech() // MiniMax only supports TTS (text-to-audio)
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

    fn as_chat_capability(&self) -> Option<&dyn ChatCapability> {
        Some(self)
    }

    fn as_audio_capability(&self) -> Option<&dyn crate::traits::AudioCapability> {
        Some(self)
    }

    fn as_speech_capability(&self) -> Option<&dyn crate::traits::SpeechCapability> {
        Some(self)
    }

    fn as_speech_extras(&self) -> Option<&dyn crate::traits::SpeechExtras> {
        None
    }

    fn as_transcription_capability(&self) -> Option<&dyn crate::traits::TranscriptionCapability> {
        None
    }

    fn as_transcription_extras(&self) -> Option<&dyn crate::traits::TranscriptionExtras> {
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
impl ChatCapability for MinimaxClient {
    async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        use crate::execution::executors::chat::ChatExecutor;

        let request = self.prepare_chat_request(request, false)?;
        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute(&*exec, request).await
    }

    async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        use crate::execution::executors::chat::ChatExecutor;

        let request = self.prepare_chat_request(request, true)?;
        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute_stream(&*exec, request).await
    }

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
        let request = self.prepare_chat_request(builder.build(), false)?;
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
        let request = self.prepare_chat_request(builder.build(), true)?;
        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute_stream(&*exec, request).await
    }
}

#[async_trait]
impl AudioCapability for MinimaxClient {
    fn supported_features(&self) -> &[AudioFeature] {
        &[AudioFeature::TextToSpeech]
    }

    async fn text_to_speech(&self, request: TtsRequest) -> Result<TtsResponse, LlmError> {
        use crate::execution::executors::audio::AudioExecutor;
        let request = request.with_model_if_missing(self.config.common_params.model.clone());

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
            warnings: None,
            provider_metadata: None,
            request: result.request,
            response: result.response,
        })
    }

    async fn speech_to_text(&self, _request: SttRequest) -> Result<SttResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Speech-to-text is not yet supported for MiniMax".to_string(),
        ))
    }
}

#[async_trait]
impl FileManagementCapability for MinimaxClient {
    async fn upload_file(&self, request: FileUploadRequest) -> Result<FileObject, LlmError> {
        let files = MinimaxFiles::new(
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
        let files = MinimaxFiles::new(
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
        let files = MinimaxFiles::new(
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
        let files = MinimaxFiles::new(
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
        let files = MinimaxFiles::new(
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
impl ImageGenerationCapability for MinimaxClient {
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

impl ImageExtras for MinimaxClient {
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
impl VideoGenerationCapability for MinimaxClient {
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

    async fn materialize_video_reference(
        &self,
        provider_reference: &crate::types::ProviderReference,
    ) -> Result<crate::types::MaterializedVideoAsset, LlmError> {
        let file_id = provider_reference
            .preferred_value(&["minimax"])
            .ok_or_else(|| {
                LlmError::InvalidInput(format!(
                    "MiniMax video materialization expected a `minimax` provider reference, got providers {:?}",
                    provider_reference.available_providers()
                ))
            })?;

        let bytes =
            crate::traits::FileManagementCapability::get_file_content(self, file_id.to_string())
                .await?;

        Ok(crate::types::MaterializedVideoAsset::new(bytes))
    }

    fn max_videos_per_call(&self) -> Option<u32> {
        Some(1)
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
impl MusicGenerationCapability for MinimaxClient {
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
    use crate::execution::http::transport::{
        HttpTransport, HttpTransportRequest, HttpTransportResponse, HttpTransportStreamBody,
        HttpTransportStreamResponse,
    };
    use crate::provider_options::{MinimaxOptions, MinimaxServiceTier, MinimaxThinking};
    use crate::providers::minimax::ext::request_options::MinimaxChatRequestExt;
    use async_trait::async_trait;
    use reqwest::header::{ACCEPT, CONTENT_TYPE, HeaderMap, HeaderValue};
    use std::sync::Mutex;

    fn production_source() -> &'static str {
        include_str!("client.rs")
            .split_once("#[cfg(test)]")
            .expect("test marker should exist")
            .0
    }

    #[test]
    fn minimax_client_request_option_merging_does_not_read_response_metadata() {
        let source = production_source();

        assert!(
            source.contains("with_provider_options_map(&self.config.default_provider_options_map)"),
            "MiniMax client should merge config defaults through request provider options"
        );

        for forbidden in ["providerMetadata", "ProviderMetadata"] {
            assert!(
                !source.contains(forbidden),
                "MiniMax client request paths must not read response metadata"
            );
        }
    }

    #[test]
    fn minimax_client_advertises_only_models_supported_by_its_chat_route() {
        let client = MinimaxClient::new(MinimaxConfig::new("test-key"), reqwest::Client::new());

        let models = LlmClient::supported_models(&client);
        assert!(models.contains(&super::super::models::CHAT.to_string()));
        assert!(models.contains(&super::super::models::FLAGSHIP.to_string()));
    }

    #[derive(Clone, Default)]
    struct NoopInterceptor;

    impl crate::execution::http::interceptor::HttpInterceptor for NoopInterceptor {}

    #[derive(Clone, Default)]
    struct CaptureTransport {
        last: Arc<Mutex<Option<HttpTransportRequest>>>,
        last_stream: Arc<Mutex<Option<HttpTransportRequest>>>,
    }

    impl CaptureTransport {
        fn take_stream(&self) -> Option<HttpTransportRequest> {
            self.last_stream.lock().expect("lock stream request").take()
        }
    }

    #[async_trait]
    impl HttpTransport for CaptureTransport {
        async fn execute_json(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last.lock().expect("lock request") = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportResponse {
                status: 401,
                headers,
                body: br#"{"type":"error","error":{"type":"authentication_error","message":"unauthorized"}}"#
                    .to_vec(),
            })
        }

        async fn execute_stream(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportStreamResponse, LlmError> {
            *self.last_stream.lock().expect("lock stream request") = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportStreamResponse {
                status: 401,
                headers,
                body: HttpTransportStreamBody::from_bytes(
                    br#"{"type":"error","error":{"type":"authentication_error","message":"unauthorized"}}"#
                        .to_vec(),
                ),
            })
        }
    }

    #[test]
    fn build_chat_executor_inherits_interceptors_and_retry() {
        let cfg = MinimaxConfig::new("test-key");
        let client = MinimaxClient::new(cfg, reqwest::Client::new())
            .with_interceptors(vec![Arc::new(NoopInterceptor)])
            .with_retry(RetryOptions::backoff());

        let req = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_common_params(client.config().common_params.clone())
            .with_http_config(client.http_config.clone());

        let exec = client.build_chat_executor(&req);
        assert_eq!(exec.policy.interceptors.len(), 1);
        assert!(exec.policy.retry_options.is_some());
    }

    #[test]
    fn with_http_client_preserves_config_interceptors_and_model_metadata() {
        let cfg = MinimaxConfig::new("test-key")
            .with_model("MiniMax-M2")
            .with_http_interceptors(vec![Arc::new(NoopInterceptor)]);
        let client = MinimaxClient::with_http_client(cfg, reqwest::Client::new())
            .expect("with_http_client client");

        let req = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_common_params(client.config().common_params.clone())
            .with_http_config(client.http_config.clone());
        let exec = client.build_chat_executor(&req);

        assert_eq!(client.config().http_interceptors.len(), 1);
        assert_eq!(exec.policy.interceptors.len(), 1);
        assert_eq!(
            crate::traits::ModelMetadata::provider_id(&client),
            "minimax"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(&client),
            "MiniMax-M2"
        );
    }

    #[test]
    fn minimax_client_does_not_expose_speech_extras_without_provider_owned_support() {
        let cfg = MinimaxConfig::new("test-key").with_model("speech-2.5-hd");
        let client = MinimaxClient::with_http_client(cfg, reqwest::Client::new())
            .expect("with_http_client client");

        assert!(client.as_audio_capability().is_some());
        assert!(client.as_speech_capability().is_some());
        assert!(client.as_speech_extras().is_none());
        assert!(client.as_transcription_capability().is_none());
        assert!(client.as_transcription_extras().is_none());
    }

    #[tokio::test]
    async fn minimax_client_chat_stream_request_preserves_official_options_and_chat_features() {
        let transport = CaptureTransport::default();
        let client = MinimaxClient::from_config(
            MinimaxConfig::new("test-key")
                .with_base_url("https://example.com/custom")
                .with_model("MiniMax-M3")
                .with_http_transport(Arc::new(transport.clone())),
        )
        .expect("build minimax client");

        let request = ChatRequest::builder()
            .model("MiniMax-M3")
            .temperature(0.5)
            .max_tokens(256)
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
            .tool_choice(crate::types::ToolChoice::Required)
            .build()
            .with_minimax_options(
                MinimaxOptions::new()
                    .with_thinking(MinimaxThinking::Adaptive)
                    .with_service_tier(MinimaxServiceTier::Priority),
            );

        let _ = client.chat_stream_request(request).await;
        let captured = transport.take_stream().expect("captured stream request");

        assert_eq!(
            captured
                .headers
                .get(ACCEPT)
                .and_then(|value| value.to_str().ok()),
            Some("text/event-stream")
        );
        assert_eq!(captured.body["stream"], serde_json::json!(true));
        assert_eq!(
            captured.body["thinking"],
            serde_json::json!({ "type": "adaptive" })
        );
        assert_eq!(captured.body["service_tier"], serde_json::json!("priority"));
        assert_eq!(captured.body["max_tokens"], serde_json::json!(256));
        assert_eq!(captured.body["temperature"], serde_json::json!(0.5));
        assert!(captured.body["tools"].is_array());
        assert!(captured.body.get("tool_choice").is_some());
        assert!(captured.body.get("output_format").is_none());
    }

    #[tokio::test]
    async fn minimax_client_merges_config_default_provider_options_before_request_overrides() {
        let transport = CaptureTransport::default();
        let client = MinimaxClient::from_config(
            MinimaxConfig::new("test-key")
                .with_base_url("https://example.com/custom")
                .with_model("MiniMax-M3")
                .with_http_transport(Arc::new(transport.clone()))
                .with_thinking(MinimaxThinking::Adaptive)
                .with_service_tier(MinimaxServiceTier::Standard),
        )
        .expect("build minimax client");

        let request = ChatRequest::builder()
            .model("MiniMax-M3")
            .max_tokens(256)
            .messages(vec![ChatMessage::user("hi").build()])
            .build()
            .with_minimax_options(
                MinimaxOptions::new()
                    .with_thinking(MinimaxThinking::Disabled)
                    .with_service_tier(MinimaxServiceTier::Priority),
            );

        let _ = client.chat_request(request).await;
        let captured = transport
            .last
            .lock()
            .expect("lock request")
            .take()
            .expect("captured");

        assert_eq!(
            captured.body["thinking"],
            serde_json::json!({ "type": "disabled" })
        );
        assert_eq!(captured.body["service_tier"], serde_json::json!("priority"));
        assert_eq!(captured.body["max_tokens"], serde_json::json!(256));
    }

    #[tokio::test]
    async fn minimax_client_chat_request_fills_missing_common_params_from_config_defaults() {
        let transport = CaptureTransport::default();
        let mut config = MinimaxConfig::new("test-key")
            .with_base_url("https://example.com/custom")
            .with_http_transport(Arc::new(transport.clone()));
        config.common_params.temperature = Some(0.4);
        config.common_params.max_tokens = Some(256);
        let client = MinimaxClient::from_config(config).expect("build minimax client");

        let request = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .build();

        let _ = client.chat_request(request).await;
        let captured = transport
            .last
            .lock()
            .expect("lock request")
            .take()
            .expect("captured");

        assert_eq!(
            captured.body["model"],
            serde_json::json!(super::super::models::CHAT)
        );
        assert_eq!(captured.body["temperature"], serde_json::json!(0.4));
        assert_eq!(captured.body["max_tokens"], serde_json::json!(256));
    }

    #[tokio::test]
    async fn minimax_client_chat_stream_request_preserves_explicit_common_params_over_defaults() {
        let transport = CaptureTransport::default();
        let mut config = MinimaxConfig::new("test-key")
            .with_base_url("https://example.com/custom")
            .with_http_transport(Arc::new(transport.clone()));
        config.common_params.temperature = Some(0.4);
        config.common_params.max_tokens = Some(256);
        let client = MinimaxClient::from_config(config).expect("build minimax client");

        let request = ChatRequest::builder()
            .model("MiniMax-M2-Explicit")
            .temperature(0.7)
            .messages(vec![ChatMessage::user("hi").build()])
            .build();

        let _ = client.chat_stream_request(request).await;
        let captured = transport.take_stream().expect("captured stream request");

        assert_eq!(
            captured.body["model"],
            serde_json::json!("MiniMax-M2-Explicit")
        );
        assert_eq!(captured.body["temperature"], serde_json::json!(0.7));
        assert_eq!(captured.body["max_tokens"], serde_json::json!(256));
        assert_eq!(captured.body["stream"], serde_json::json!(true));
    }
}
