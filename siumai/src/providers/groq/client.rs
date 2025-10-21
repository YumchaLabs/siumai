//! `Groq` Client Implementation
//!
//! Main client implementation that aggregates all Groq capabilities.

use async_trait::async_trait;

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::streaming::ChatStream;
use crate::traits::{
    AudioCapability, ChatCapability, ModelListingCapability, ProviderCapabilities,
};
use crate::types::*;

use super::api::GroqModels;
use super::chat::GroqChatCapability;
use super::config::GroqConfig;
use crate::middleware::language_model::LanguageModelMiddleware;
use crate::retry_api::RetryOptions;
use crate::utils::http_interceptor::HttpInterceptor;
use std::sync::Arc;

/// `Groq` client that implements all capabilities
pub struct GroqClient {
    /// Configuration
    config: GroqConfig,
    /// HTTP client
    http_client: reqwest::Client,
    /// Chat capability
    chat_capability: GroqChatCapability,
    /// Models capability
    models_capability: GroqModels,
    /// Tracing configuration
    tracing_config: Option<crate::tracing::TracingConfig>,
    /// Tracing guard to keep tracing system active
    /// NOTE: Tracing subscriber functionality has been moved to siumai-extras
    #[allow(dead_code)]
    _tracing_guard: Option<()>,
    /// Unified retry options for chat
    retry_options: Option<RetryOptions>,
    /// Optional HTTP interceptors applied to all chat requests
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
}

impl Clone for GroqClient {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            http_client: self.http_client.clone(),
            chat_capability: self.chat_capability.clone(),
            models_capability: self.models_capability.clone(),
            tracing_config: self.tracing_config.clone(),
            _tracing_guard: None, // Don't clone the tracing guard
            retry_options: self.retry_options.clone(),
            http_interceptors: self.http_interceptors.clone(),
        }
    }
}

impl std::fmt::Debug for GroqClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroqClient")
            .field("provider_name", &"groq")
            .field("model", &self.config.common_params.model)
            .field("base_url", &self.config.base_url)
            .field("temperature", &self.config.common_params.temperature)
            .field("max_tokens", &self.config.common_params.max_tokens)
            .field("top_p", &self.config.common_params.top_p)
            .field("seed", &self.config.common_params.seed)
            .field("has_tracing", &self.tracing_config.is_some())
            .finish()
    }
}

impl GroqClient {
    /// Create a new `Groq` client
    pub fn new(config: GroqConfig, http_client: reqwest::Client) -> Self {
        let chat_capability = GroqChatCapability::new(
            config.api_key.clone(),
            config.base_url.clone(),
            http_client.clone(),
            config.http_config.clone(),
            config.common_params.clone(),
        );

        let models_capability = GroqModels::new(
            config.api_key.clone(),
            config.base_url.clone(),
            http_client.clone(),
            config.http_config.clone(),
        );

        Self {
            config,
            http_client,
            chat_capability,
            models_capability,
            tracing_config: None,
            _tracing_guard: None,
            retry_options: None,
            http_interceptors: Vec::new(),
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &GroqConfig {
        &self.config
    }

    /// Get the HTTP client
    pub fn http_client(&self) -> &reqwest::Client {
        &self.http_client
    }

    /// Get chat capability
    pub fn chat_capability(&self) -> &GroqChatCapability {
        &self.chat_capability
    }

    /// Install model-level middlewares for chat requests.
    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<std::sync::Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.chat_capability = self.chat_capability.clone().with_middlewares(middlewares);
        self
    }
}

#[async_trait]
impl LlmClient for GroqClient {
    fn provider_name(&self) -> &'static str {
        "groq"
    }

    fn supported_models(&self) -> Vec<String> {
        GroqConfig::supported_models()
            .iter()
            .map(|&s| s.to_string())
            .collect()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_custom_feature("audio", true)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }
}

#[async_trait]
impl ChatCapability for GroqClient {
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
                    async move { self.chat_capability.chat_with_tools(m, t).await }
                },
                opts.clone(),
            )
            .await
        } else {
            self.chat_capability.chat_with_tools(messages, tools).await
        }
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        self.chat_capability.chat_stream(messages, tools).await
    }

    async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        use crate::executors::chat::{ChatExecutor, HttpChatExecutor};
        let http = self.http_client.clone();
        let base = self.config.base_url.clone();
        let api_key = self.config.api_key.clone();
        let custom_headers = self.config.http_config.headers.clone();
        let req_tx = super::transformers::GroqRequestTransformer;
        let resp_tx = super::transformers::GroqResponseTransformer;
        let api_key_clone = api_key.clone();
        let custom_headers_clone = custom_headers.clone();
        let headers_builder = move || {
            use secrecy::ExposeSecret;
            let api_key = api_key_clone.clone();
            let custom_headers = custom_headers_clone.clone();
            Box::pin(async move {
                super::utils::build_headers(api_key.expose_secret(), &custom_headers)
            })
                as std::pin::Pin<
                    Box<
                        dyn std::future::Future<
                                Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>,
                            > + Send,
                    >,
                >
        };
        let exec = HttpChatExecutor {
            provider_id: "groq".to_string(),
            http_client: http,
            request_transformer: std::sync::Arc::new(req_tx),
            response_transformer: std::sync::Arc::new(resp_tx),
            stream_transformer: None,
            json_stream_converter: None,
            stream_disable_compression: self.config.http_config.stream_disable_compression,
            interceptors: self.http_interceptors.clone(),
            middlewares: self.chat_capability.middlewares.clone(),
            build_url: Box::new(move |_stream, _req| format!("{}/chat/completions", base)),
            build_headers: std::sync::Arc::new(headers_builder),
            before_send: None,
        };
        exec.execute(request).await
    }

    async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        use crate::executors::chat::{ChatExecutor, HttpChatExecutor};
        let http = self.http_client.clone();
        let base = self.config.base_url.clone();
        let api_key = self.config.api_key.clone();
        let custom_headers = self.config.http_config.headers.clone();
        let req_tx = super::transformers::GroqRequestTransformer;
        let resp_tx = super::transformers::GroqResponseTransformer;
        let inner = super::streaming::GroqEventConverter::new();
        let stream_tx = super::transformers::GroqStreamChunkTransformer {
            provider_id: "groq".to_string(),
            inner,
        };
        let api_key_clone = api_key.clone();
        let custom_headers_clone = custom_headers.clone();
        let headers_builder = move || {
            use secrecy::ExposeSecret;
            let api_key = api_key_clone.clone();
            let custom_headers = custom_headers_clone.clone();
            Box::pin(async move {
                super::utils::build_headers(api_key.expose_secret(), &custom_headers)
            })
                as std::pin::Pin<
                    Box<
                        dyn std::future::Future<
                                Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>,
                            > + Send,
                    >,
                >
        };
        let exec = HttpChatExecutor {
            provider_id: "groq".to_string(),
            http_client: http,
            request_transformer: std::sync::Arc::new(req_tx),
            response_transformer: std::sync::Arc::new(resp_tx),
            stream_transformer: Some(std::sync::Arc::new(stream_tx)),
            json_stream_converter: None,
            stream_disable_compression: self.config.http_config.stream_disable_compression,
            interceptors: self.http_interceptors.clone(),
            middlewares: self.chat_capability.middlewares.clone(),
            build_url: Box::new(move |_stream, _req| format!("{}/chat/completions", base)),
            build_headers: std::sync::Arc::new(headers_builder),
            before_send: None,
        };
        exec.execute_stream(request).await
    }
}

#[async_trait]
impl AudioCapability for GroqClient {
    fn supported_features(&self) -> &[crate::types::AudioFeature] {
        // Minimal list for Groq (whisper + tts)
        use crate::types::AudioFeature::*;
        const FEATURES: &[crate::types::AudioFeature] = &[TextToSpeech, SpeechToText];
        FEATURES
    }

    async fn text_to_speech(
        &self,
        request: crate::types::TtsRequest,
    ) -> Result<crate::types::TtsResponse, LlmError> {
        use crate::executors::audio::{AudioExecutor, HttpAudioExecutor};
        let http = self.http_client.clone();
        let base = self.config.base_url.clone();
        let api_key = self.config.api_key.clone();
        let custom_headers = self.config.http_config.headers.clone();
        let transformer = super::transformers::GroqAudioTransformer;
        let api_key_clone = api_key.clone();
        let custom_headers_clone = custom_headers.clone();
        let headers_builder = move || {
            use secrecy::ExposeSecret;
            let api_key = api_key_clone.clone();
            let custom_headers = custom_headers_clone.clone();
            Box::pin(async move {
                super::utils::build_headers(api_key.expose_secret(), &custom_headers)
            })
                as std::pin::Pin<
                    Box<
                        dyn std::future::Future<
                                Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>,
                            > + Send,
                    >,
                >
        };
        let exec = HttpAudioExecutor {
            provider_id: "groq".to_string(),
            http_client: http,
            transformer: std::sync::Arc::new(transformer),
            build_base_url: Box::new(move || base.clone()),
            build_headers: Box::new(headers_builder),
        };
        let audio_bytes = exec.tts(request).await?;
        Ok(crate::types::TtsResponse {
            audio_data: audio_bytes,
            format: "wav".to_string(),
            duration: None,
            sample_rate: None,
            metadata: std::collections::HashMap::new(),
        })
    }

    async fn speech_to_text(
        &self,
        request: crate::types::SttRequest,
    ) -> Result<crate::types::SttResponse, LlmError> {
        use crate::executors::audio::{AudioExecutor, HttpAudioExecutor};
        let http = self.http_client.clone();
        let base = self.config.base_url.clone();
        let api_key = self.config.api_key.clone();
        let custom_headers = self.config.http_config.headers.clone();
        let transformer = super::transformers::GroqAudioTransformer;
        let api_key_clone = api_key.clone();
        let custom_headers_clone = custom_headers.clone();
        let headers_builder = move || {
            use secrecy::ExposeSecret;
            let api_key = api_key_clone.clone();
            let custom_headers = custom_headers_clone.clone();
            Box::pin(async move {
                super::utils::build_headers(api_key.expose_secret(), &custom_headers)
            })
                as std::pin::Pin<
                    Box<
                        dyn std::future::Future<
                                Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>,
                            > + Send,
                    >,
                >
        };
        let exec = HttpAudioExecutor {
            provider_id: "groq".to_string(),
            http_client: http,
            transformer: std::sync::Arc::new(transformer),
            build_base_url: Box::new(move || base.clone()),
            build_headers: Box::new(headers_builder),
        };
        let text = exec.stt(request).await?;
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

impl GroqClient {
    /// Set the tracing guard to keep tracing system active
    /// NOTE: Tracing subscriber functionality has been moved to siumai-extras
    pub(crate) fn set_tracing_guard(&mut self, guard: Option<()>) {
        self._tracing_guard = guard;
    }

    /// Set the tracing configuration
    pub(crate) fn set_tracing_config(&mut self, config: Option<crate::tracing::TracingConfig>) {
        self.tracing_config = config;
    }

    /// Set unified retry options
    pub fn set_retry_options(&mut self, options: Option<RetryOptions>) {
        self.retry_options = options;
    }

    /// Install HTTP interceptors for all chat requests.
    pub fn with_http_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors.clone();
        let mut cap = self.chat_capability.clone();
        cap.interceptors = interceptors;
        self.chat_capability = cap;
        self
    }
}

#[async_trait]
impl ModelListingCapability for GroqClient {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.models_capability.list_models().await
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        self.models_capability.get_model(model_id).await
    }
}
