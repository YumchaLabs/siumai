//! OpenAI Compatible Client
//!
//! This module provides a client implementation for OpenAI-compatible providers.

use super::openai_config::OpenAiCompatibleConfig;
use crate::client::LlmClient;
use crate::core::{ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::executors::audio::{AudioExecutor, AudioExecutorBuilder, HttpAudioExecutor};
use crate::execution::executors::chat::{ChatExecutor, ChatExecutorBuilder};
use crate::execution::executors::embedding::{EmbeddingExecutor, HttpEmbeddingExecutor};
use crate::execution::executors::image::{HttpImageExecutor, ImageExecutor};
use crate::standards::openai::compat::provider_registry::ConfigurableAdapter;
// use crate::providers::openai_compatible::RequestType; // no longer needed here
use crate::retry_api::RetryOptions;
use crate::streaming::ChatStream;
use crate::traits::{
    AudioCapability, ChatCapability, EmbeddingCapability, ImageExtras, ImageGenerationCapability,
    ModelListingCapability, RerankCapability, SpeechCapability, SpeechExtras,
    TranscriptionCapability, TranscriptionExtras,
};
// use crate::execution::transformers::request::RequestTransformer; // unused
use crate::types::*;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use siumai_core::traits::ModelMetadata;
use std::sync::Arc;
// removed: HashMap import not needed after legacy removal
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;

/// OpenAI Compatible Chat Response with provider-specific fields
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiCompatibleChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAiCompatibleChoice>,
    pub usage: Option<OpenAiCompatibleUsage>,
}

/// OpenAI Compatible Choice with provider-specific fields
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiCompatibleChoice {
    pub index: u32,
    pub message: OpenAiCompatibleMessage,
    pub finish_reason: Option<String>,
}

/// OpenAI Compatible Message with provider-specific fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiCompatibleMessage {
    pub role: String,
    pub content: Option<serde_json::Value>,
    pub tool_calls: Option<Vec<OpenAiCompatibleToolCall>>,
    pub tool_call_id: Option<String>,

    // Provider-specific thinking/reasoning fields
    pub thinking: Option<String>,          // Standard thinking field
    pub reasoning_content: Option<String>, // DeepSeek reasoning field
    pub reasoning: Option<String>,         // Alternative reasoning field
}

/// OpenAI Compatible Tool Call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiCompatibleToolCall {
    pub id: String,
    pub r#type: String,
    pub function: Option<OpenAiCompatibleFunction>,
}

/// OpenAI Compatible Function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiCompatibleFunction {
    pub name: String,
    pub arguments: String,
}

/// OpenAI Compatible Usage
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiCompatibleUsage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

/// OpenAI compatible client
///
/// This is a separate client implementation that uses the adapter system
/// to handle provider-specific differences without modifying the core OpenAI client.
#[derive(Clone)]
pub struct OpenAiCompatibleClient {
    config: OpenAiCompatibleConfig,
    http_client: reqwest::Client,
    retry_options: Option<RetryOptions>,
    /// Optional HTTP interceptors applied to all chat requests
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional model-level middlewares
    model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
}

fn model_slot_is_missing(model: Option<&str>) -> bool {
    match model {
        Some(value) => value.trim().is_empty(),
        None => true,
    }
}

impl std::fmt::Debug for OpenAiCompatibleClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiCompatibleClient")
            .field("provider_id", &self.config.provider_id)
            .field("model", &self.config.model)
            .field("base_url", &self.config.base_url)
            .field("has_api_key", &(!self.config.api_key.is_empty()))
            .field("has_retry", &self.retry_options.is_some())
            .field("interceptors", &self.http_interceptors.len())
            .field("middlewares", &self.model_middlewares.len())
            .finish()
    }
}

impl OpenAiCompatibleClient {
    fn primary_default_model(&self) -> Option<&'static str> {
        super::default_models::get_default_chat_model(&self.config.provider_id)
    }

    fn resolve_family_model_or_config(
        &self,
        family_default: Option<&'static str>,
    ) -> Option<String> {
        let configured_model = self.config.model.trim();
        if configured_model.is_empty() {
            return family_default.map(str::to_string);
        }

        if self
            .primary_default_model()
            .is_some_and(|default_model| default_model == configured_model)
        {
            return family_default
                .map(str::to_string)
                .or_else(|| Some(self.config.model.clone()));
        }

        Some(self.config.model.clone())
    }

    fn resolve_embedding_model_default(&self) -> Option<String> {
        self.resolve_family_model_or_config(super::config::get_default_embedding_model(
            &self.config.provider_id,
        ))
    }

    fn resolve_rerank_model_default(&self) -> Option<String> {
        self.resolve_family_model_or_config(super::config::get_default_rerank_model(
            &self.config.provider_id,
        ))
    }

    fn resolve_image_model_default(&self) -> Option<String> {
        self.resolve_family_model_or_config(super::config::get_default_image_model(
            &self.config.provider_id,
        ))
    }

    fn resolve_speech_model_default(&self) -> Option<String> {
        self.resolve_family_model_or_config(super::config::get_default_speech_model(
            &self.config.provider_id,
        ))
    }

    fn resolve_transcription_model_default(&self) -> Option<String> {
        self.resolve_family_model_or_config(super::config::get_default_transcription_model(
            &self.config.provider_id,
        ))
    }

    fn build_context(&self) -> ProviderContext {
        // Merge custom headers from HttpConfig + config.custom_headers + adapter.custom_headers
        let mut extra_headers: std::collections::HashMap<String, String> =
            self.config.http_config.headers.clone();
        let cfg_map =
            crate::execution::http::headers::headermap_to_hashmap(&self.config.custom_headers);
        extra_headers.extend(cfg_map);
        let adapter_map = crate::execution::http::headers::headermap_to_hashmap(
            &self.config.adapter.custom_headers(),
        );
        extra_headers.extend(adapter_map);

        ProviderContext::new(
            self.config.provider_id.clone(),
            self.config.base_url.clone(),
            Some(self.config.api_key.clone()),
            extra_headers,
        )
    }

    fn ensure_chat_surface(&self, stream: bool) -> Result<(), LlmError> {
        let caps = self.config.adapter.capabilities();
        if !caps.chat {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not support chat",
                self.config.provider_id
            )));
        }
        if stream && !caps.streaming {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not support chat streaming",
                self.config.provider_id
            )));
        }

        Ok(())
    }

    fn ensure_embedding_surface(&self) -> Result<(), LlmError> {
        if !self.config.adapter.capabilities().embedding {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not support embeddings",
                self.config.provider_id
            )));
        }

        Ok(())
    }

    fn ensure_rerank_surface(&self) -> Result<(), LlmError> {
        if !self.config.adapter.capabilities().supports("rerank") {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not support rerank",
                self.config.provider_id
            )));
        }

        Ok(())
    }

    fn http_wiring(&self, ctx: ProviderContext) -> crate::execution::wiring::HttpExecutionWiring {
        let mut wiring = crate::execution::wiring::HttpExecutionWiring::new(
            self.config.provider_id.clone(),
            self.http_client.clone(),
            ctx,
        )
        .with_interceptors(self.http_interceptors.clone())
        .with_retry_options(self.retry_options.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            wiring = wiring.with_transport(transport);
        }

        wiring
    }

    /// Build the provider execution context (headers/base_url/api key + extra headers).
    ///
    /// This is primarily used by hybrid providers (e.g. Groq) that reuse the OpenAI-compatible
    /// client for chat/embedding/image but need to invoke non-chat executors with the same
    /// HTTP wiring (client, interceptors, retry).
    pub fn provider_context(&self) -> ProviderContext {
        self.build_context()
    }

    /// Clone the underlying `reqwest::Client`.
    pub fn http_client(&self) -> reqwest::Client {
        self.http_client.clone()
    }

    /// Clone the installed unified retry options.
    pub fn retry_options(&self) -> Option<RetryOptions> {
        self.retry_options.clone()
    }

    /// Clone the installed HTTP interceptors.
    pub fn http_interceptors(&self) -> Vec<Arc<dyn HttpInterceptor>> {
        self.http_interceptors.clone()
    }

    /// Clone the installed custom HTTP transport, if present.
    ///
    /// This is primarily used by hybrid providers (e.g. Groq) that reuse the OpenAI-compatible
    /// client for most capabilities but still need to invoke spec-driven executors with the
    /// same transport wiring.
    pub fn http_transport(
        &self,
    ) -> Option<Arc<dyn crate::execution::http::transport::HttpTransport>> {
        self.config.http_transport.clone()
    }

    /// Clone the config-level common params template.
    pub fn common_params(&self) -> CommonParams {
        self.config.common_params.clone()
    }

    /// Clone the config-level HTTP config template.
    pub fn http_config(&self) -> HttpConfig {
        self.config.http_config.clone()
    }

    /// Clone the provider adapter.
    pub fn adapter(&self) -> Arc<dyn crate::providers::openai_compatible::ProviderAdapter> {
        self.config.adapter.clone()
    }

    /// Build a chat executor with an explicit provider spec.
    pub fn build_chat_executor_with_spec(
        &self,
        request: &ChatRequest,
        spec: Arc<dyn ProviderSpec>,
    ) -> Arc<crate::execution::executors::chat::HttpChatExecutor> {
        let ctx = self.build_context();
        let bundle = spec.choose_chat_transformers(request, &ctx);

        let mut builder =
            ChatExecutorBuilder::new(self.config.provider_id.clone(), self.http_client.clone())
                .with_spec(spec)
                .with_context(ctx)
                .with_transformer_bundle(bundle)
                .with_runtime_transformer_selection()
                .with_stream_disable_compression(self.config.http_config.stream_disable_compression)
                .with_interceptors(self.http_interceptors.clone())
                .with_middlewares(self.model_middlewares.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        builder.build()
    }

    fn build_chat_executor(
        &self,
        request: &ChatRequest,
    ) -> Arc<crate::execution::executors::chat::HttpChatExecutor> {
        let spec = Arc::new(
            crate::providers::openai_compatible::spec::OpenAiCompatibleSpecWithAdapter::new(
                self.config.adapter.clone(),
            ),
        );
        self.build_chat_executor_with_spec(request, spec)
    }
    fn merge_common_params(&self, request_params: CommonParams) -> CommonParams {
        let defaults = &self.config.common_params;

        CommonParams {
            model: if request_params.model.trim().is_empty() {
                defaults.model.clone()
            } else {
                request_params.model
            },
            temperature: request_params.temperature.or(defaults.temperature),
            max_tokens: request_params.max_tokens.or(defaults.max_tokens),
            max_completion_tokens: request_params
                .max_completion_tokens
                .or(defaults.max_completion_tokens),
            top_p: request_params.top_p.or(defaults.top_p),
            top_k: request_params.top_k.or(defaults.top_k),
            stop_sequences: request_params
                .stop_sequences
                .or_else(|| defaults.stop_sequences.clone()),
            seed: request_params.seed.or(defaults.seed),
            frequency_penalty: request_params
                .frequency_penalty
                .or(defaults.frequency_penalty),
            presence_penalty: request_params
                .presence_penalty
                .or(defaults.presence_penalty),
        }
    }

    fn prepare_chat_request(
        &self,
        mut request: ChatRequest,
        stream: bool,
    ) -> Result<ChatRequest, LlmError> {
        self.ensure_chat_surface(stream)?;
        request.common_params = self.merge_common_params(request.common_params);
        if request.common_params.model.trim().is_empty() {
            return Err(LlmError::InvalidParameter(
                "OpenAI-compatible request requires a model".to_string(),
            ));
        }
        if request.http_config.is_none() {
            request.http_config = Some(self.config.http_config.clone());
        }
        request.stream = stream;
        Ok(request)
    }
    fn build_embedding_executor(&self, request: &EmbeddingRequest) -> Arc<HttpEmbeddingExecutor> {
        use crate::execution::executors::embedding::EmbeddingExecutorBuilder;

        let ctx = self.build_context();
        let spec = Arc::new(
            crate::providers::openai_compatible::spec::OpenAiCompatibleSpecWithAdapter::new(
                self.config.adapter.clone(),
            ),
        );
        let mut builder = EmbeddingExecutorBuilder::new(
            self.config.provider_id.clone(),
            self.http_client.clone(),
        )
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
        use crate::execution::executors::image::ImageExecutorBuilder;

        let ctx = self.build_context();
        let spec = Arc::new(
            crate::providers::openai_compatible::spec::OpenAiCompatibleSpecWithAdapter::new(
                self.config.adapter.clone(),
            ),
        );
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

        builder.build_for_request(request)
    }

    fn build_audio_executor(&self) -> Arc<HttpAudioExecutor> {
        let ctx = self.build_context();
        let spec = Arc::new(
            crate::providers::openai_compatible::spec::OpenAiCompatibleSpecWithAdapter::new(
                self.config.adapter.clone(),
            ),
        );

        let mut builder =
            AudioExecutorBuilder::new(self.config.provider_id.clone(), self.http_client.clone())
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

    /// Execute a non-stream chat via an explicit ProviderSpec.
    pub async fn chat_request_with_spec(
        &self,
        request: ChatRequest,
        spec: Arc<dyn ProviderSpec>,
    ) -> Result<ChatResponse, LlmError> {
        let request = self.prepare_chat_request(request, false)?;
        let exec = self.build_chat_executor_with_spec(&request, spec);
        ChatExecutor::execute(&*exec, request).await
    }

    /// Execute a stream chat via an explicit ProviderSpec.
    pub async fn chat_stream_request_with_spec(
        &self,
        request: ChatRequest,
        spec: Arc<dyn ProviderSpec>,
    ) -> Result<ChatStream, LlmError> {
        let request = self.prepare_chat_request(request, true)?;
        let exec = self.build_chat_executor_with_spec(&request, spec);
        ChatExecutor::execute_stream(&*exec, request).await
    }

    /// Execute a non-stream chat via ProviderSpec.
    async fn chat_request_via_spec(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute(&*exec, request).await
    }

    /// Execute a stream chat via ProviderSpec.
    async fn chat_stream_request_via_spec(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStream, LlmError> {
        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute_stream(&*exec, request).await
    }
    /// Create a new OpenAI compatible client
    pub async fn new(config: OpenAiCompatibleConfig) -> Result<Self, LlmError> {
        // Validate configuration
        config.validate()?;

        let http_interceptors = config.http_interceptors.clone();
        let model_middlewares = config.model_middlewares.clone();

        // Validate model with adapter
        if !config.model.is_empty() {
            config.adapter.validate_model(&config.model)?;
        }

        // Create HTTP client with configuration
        let http_client = Self::build_http_client(&config)?;

        Ok(Self {
            config,
            http_client,
            retry_options: None,
            http_interceptors,
            model_middlewares,
        })
    }

    /// Construct an `OpenAiCompatibleClient` from an `OpenAiCompatibleConfig` (config-first construction).
    ///
    /// This is a convenience alias for `OpenAiCompatibleClient::new(...)` to align naming with
    /// other provider clients (`*_Client::from_config`).
    pub async fn from_config(config: OpenAiCompatibleConfig) -> Result<Self, LlmError> {
        Self::new(config).await
    }

    /// Construct an `OpenAiCompatibleClient` from a built-in provider id + API key.
    ///
    /// This uses the bundled provider registry (base_url + field mappings) and a
    /// configuration-driven adapter (`ConfigurableAdapter`).
    ///
    /// If `model` is None, we fall back to the provider's `default_model` when available.
    pub async fn from_builtin(
        provider_id: &str,
        api_key: &str,
        model: Option<&str>,
    ) -> Result<Self, LlmError> {
        let provider = super::config::get_provider_config(provider_id).ok_or_else(|| {
            LlmError::ConfigurationError(format!(
                "Unknown OpenAI-compatible provider id: {provider_id}"
            ))
        })?;

        let model = model.or(provider.default_model.as_deref()).ok_or_else(|| {
            LlmError::ConfigurationError(format!(
                "Missing model for OpenAI-compatible provider: {provider_id}"
            ))
        })?;

        let adapter = std::sync::Arc::new(ConfigurableAdapter::new(provider.clone()));
        let cfg = OpenAiCompatibleConfig::new(&provider.id, api_key, &provider.base_url, adapter)
            .with_model(model);

        Self::from_config(cfg).await
    }

    /// Construct an `OpenAiCompatibleClient` from a built-in provider id, reading the API key from env.
    ///
    /// Env lookup precedence:
    /// 1) `ProviderConfig.api_key_env` (when provided)
    /// 2) `ProviderConfig.api_key_env_aliases` (fallbacks)
    /// 3) `${PROVIDER_ID}_API_KEY` (uppercased, `-` replaced with `_`)
    pub async fn from_builtin_env(
        provider_id: &str,
        model: Option<&str>,
    ) -> Result<Self, LlmError> {
        let provider = super::config::get_provider_config(provider_id).ok_or_else(|| {
            LlmError::ConfigurationError(format!(
                "Unknown OpenAI-compatible provider id: {provider_id}"
            ))
        })?;

        fn default_env_var(id: &str) -> String {
            format!("{}_API_KEY", id.to_ascii_uppercase().replace('-', "_"))
        }

        let mut candidates: Vec<String> = Vec::new();
        if let Some(name) = &provider.api_key_env {
            candidates.push(name.clone());
        }
        candidates.extend(provider.api_key_env_aliases.clone());
        candidates.push(default_env_var(&provider.id));

        let api_key = candidates
            .iter()
            .find_map(|k| std::env::var(k).ok())
            .ok_or_else(|| {
                LlmError::MissingApiKey(format!(
                    "API key not found for provider '{provider_id}'. Tried: {}",
                    candidates.join(", ")
                ))
            })?;

        Self::from_builtin(provider_id, &api_key, model).await
    }

    /// Create a new OpenAI compatible client with custom HTTP client
    pub async fn with_http_client(
        config: OpenAiCompatibleConfig,
        http_client: reqwest::Client,
    ) -> Result<Self, LlmError> {
        // Validate configuration
        config.validate()?;

        let http_interceptors = config.http_interceptors.clone();
        let model_middlewares = config.model_middlewares.clone();

        // Validate model with adapter
        if !config.model.is_empty() {
            config.adapter.validate_model(&config.model)?;
        }

        Ok(Self {
            config,
            http_client,
            retry_options: None,
            http_interceptors,
            model_middlewares,
        })
    }

    // Removed legacy build_headers; headers are created in executor closures

    /// Set unified retry options
    pub fn set_retry_options(&mut self, options: Option<RetryOptions>) {
        self.retry_options = options;
    }

    /// Install HTTP interceptors for all chat requests.
    pub fn with_http_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors;
        self
    }

    /// Install model-level middlewares
    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.model_middlewares = middlewares;
        self
    }

    /// Build HTTP client with configuration
    fn build_http_client(config: &OpenAiCompatibleConfig) -> Result<reqwest::Client, LlmError> {
        // Use unified HTTP client builder
        crate::execution::http::client::build_http_client_from_config(&config.http_config)
    }

    /// Get the provider ID
    pub fn provider_id(&self) -> &str {
        &self.config.provider_id
    }

    /// Get the current model
    pub fn model(&self) -> &str {
        &self.config.model
    }

    // Removed legacy build_chat_request; executors use transformers directly

    // Removed legacy parse_chat_response; response transformer handles mapping

    // removed legacy send_request; executors handle requests
}

impl ModelMetadata for OpenAiCompatibleClient {
    fn provider_id(&self) -> &str {
        self.config.provider_id.as_str()
    }

    fn model_id(&self) -> &str {
        self.config.model.as_str()
    }
}

// Removed legacy chat_with_tools_inner; ChatCapability now uses HttpChatExecutor

#[async_trait]
impl ChatCapability for OpenAiCompatibleClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        self.ensure_chat_surface(false)?;
        // Build unified ChatRequest
        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(self.config.common_params.clone())
            .http_config(self.config.http_config.clone());
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        let req = builder.build();

        // Execute via ProviderSpec
        self.chat_request_via_spec(req).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        self.ensure_chat_surface(true)?;
        // Unified ChatRequest
        let mut builder = crate::types::ChatRequest::builder()
            .messages(messages)
            .common_params(self.config.common_params.clone())
            .http_config(self.config.http_config.clone())
            .stream(true);
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        let request = builder.build();

        // Execute via ProviderSpec
        self.chat_stream_request_via_spec(request).await
    }

    async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        let request = self.prepare_chat_request(request, false)?;
        self.chat_request_via_spec(request).await
    }

    async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        let request = self.prepare_chat_request(request, true)?;
        self.chat_stream_request_via_spec(request).await
    }
}

#[async_trait]
impl EmbeddingCapability for OpenAiCompatibleClient {
    async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        self.ensure_embedding_surface()?;
        let mut req = crate::types::EmbeddingRequest::new(texts);
        if let Some(model) = self.resolve_embedding_model_default() {
            req.model = Some(model);
        }
        let exec = self.build_embedding_executor(&req);
        EmbeddingExecutor::execute(&*exec, req).await
    }

    fn embedding_dimension(&self) -> usize {
        // Default dimension, could be made configurable per model
        1536
    }
}

#[async_trait]
impl crate::traits::EmbeddingExtensions for OpenAiCompatibleClient {
    async fn embed_with_config(
        &self,
        mut request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        self.ensure_embedding_surface()?;
        if model_slot_is_missing(request.model.as_deref()) {
            request.model = self.resolve_embedding_model_default();
        }

        let exec = self.build_embedding_executor(&request);
        EmbeddingExecutor::execute(&*exec, request).await
    }
}

#[async_trait]
impl RerankCapability for OpenAiCompatibleClient {
    async fn rerank(&self, mut request: RerankRequest) -> Result<RerankResponse, LlmError> {
        use crate::execution::executors::rerank::{RerankExecutor, RerankExecutorBuilder};

        self.ensure_rerank_surface()?;

        if request.model.trim().is_empty()
            && let Some(model) = self.resolve_rerank_model_default()
        {
            request.model = model;
        }

        let ctx = self.build_context();
        let spec = std::sync::Arc::new(
            crate::providers::openai_compatible::spec::OpenAiCompatibleSpecWithAdapter::new(
                self.config.adapter.clone(),
            ),
        );

        let mut builder =
            RerankExecutorBuilder::new(self.config.provider_id.clone(), self.http_client.clone())
                .with_spec(spec)
                .with_context(ctx)
                .with_interceptors(self.http_interceptors.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        let exec = builder.build_for_request(&request);
        RerankExecutor::execute(&*exec, request).await
    }
}

impl OpenAiCompatibleClient {
    /// List available models from the provider
    async fn list_models_internal(&self) -> Result<Vec<ModelInfo>, LlmError> {
        let spec = std::sync::Arc::new(
            crate::providers::openai_compatible::spec::OpenAiCompatibleSpecWithAdapter::new(
                self.config.adapter.clone(),
            ),
        );
        let ctx = self.build_context();
        let url = spec.models_url(&ctx);
        let config = self.http_wiring(ctx).config(spec);

        let result =
            crate::execution::executors::common::execute_get_request(&config, &url, None).await?;
        let models_response: serde_json::Value = result.json;

        // Parse OpenAI-compatible models response
        let models = models_response
            .get("data")
            .and_then(|data| data.as_array())
            .ok_or_else(|| LlmError::ParseError("Invalid models response format".to_string()))?;

        let mut model_infos = Vec::new();
        for model in models {
            if let Some(model_id) = model.get("id").and_then(|id| id.as_str()) {
                let model_info = ModelInfo {
                    id: model_id.to_string(),
                    name: Some(model_id.to_string()),
                    description: model
                        .get("description")
                        .and_then(|d| d.as_str())
                        .map(|s| s.to_string()),
                    owned_by: model
                        .get("owned_by")
                        .and_then(|o| o.as_str())
                        .unwrap_or(&self.config.provider_id)
                        .to_string(),
                    created: model.get("created").and_then(|c| c.as_u64()),
                    capabilities: self.determine_model_capabilities(model_id),
                    context_window: None, // Not typically provided by OpenAI-compatible APIs
                    max_output_tokens: None, // Not typically provided by OpenAI-compatible APIs
                    input_cost_per_token: None, // Not typically provided by OpenAI-compatible APIs
                    output_cost_per_token: None, // Not typically provided by OpenAI-compatible APIs
                };
                model_infos.push(model_info);
            }
        }

        Ok(model_infos)
    }

    /// Determine model capabilities based on model ID
    fn determine_model_capabilities(&self, model_id: &str) -> Vec<String> {
        let mut capabilities = vec!["chat".to_string()];

        // Add capabilities based on model name patterns
        if model_id.contains("embed") || model_id.contains("embedding") {
            capabilities.push("embedding".to_string());
        }

        if model_id.contains("rerank") || model_id.contains("bge-reranker") {
            capabilities.push("rerank".to_string());
        }

        if model_id.contains("flux")
            || model_id.contains("stable-diffusion")
            || model_id.contains("kolors")
        {
            capabilities.push("image_generation".to_string());
        }

        // Add thinking capability for supported models
        if self
            .config
            .adapter
            .get_model_config(model_id)
            .supports_thinking
        {
            capabilities.push("thinking".to_string());
        }

        capabilities
    }

    /// Get detailed information about a specific model
    async fn get_model_internal(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        // Best-effort: prefer the dedicated retrieve endpoint when the provider supports it,
        // then fallback to the list endpoint, and finally a synthetic ModelInfo.
        let spec = std::sync::Arc::new(
            crate::providers::openai_compatible::spec::OpenAiCompatibleSpecWithAdapter::new(
                self.config.adapter.clone(),
            ),
        );
        let ctx = self.build_context();
        let url = spec.model_url(&model_id, &ctx);
        let config = self.http_wiring(ctx).config(spec);

        let direct =
            crate::execution::executors::common::execute_get_request(&config, &url, None).await;

        match direct {
            Ok(result) => {
                let json = result.json;

                // OpenAI model retrieve response is usually a single object.
                if let Some(model_id) = json.get("id").and_then(|id| id.as_str()) {
                    return Ok(ModelInfo {
                        id: model_id.to_string(),
                        name: Some(model_id.to_string()),
                        description: json
                            .get("description")
                            .and_then(|d| d.as_str())
                            .map(|s| s.to_string()),
                        owned_by: json
                            .get("owned_by")
                            .and_then(|o| o.as_str())
                            .unwrap_or(&self.config.provider_id)
                            .to_string(),
                        created: json.get("created").and_then(|c| c.as_u64()),
                        capabilities: self.determine_model_capabilities(model_id),
                        context_window: None,
                        max_output_tokens: None,
                        input_cost_per_token: None,
                        output_cost_per_token: None,
                    });
                }

                // Some vendors might wrap it as `data: [...]` (rare). Best-effort parse.
                if let Some(model) = json
                    .get("data")
                    .and_then(|d| d.as_array())
                    .and_then(|arr| arr.first())
                    && let Some(model_id) = model.get("id").and_then(|id| id.as_str())
                {
                    return Ok(ModelInfo {
                        id: model_id.to_string(),
                        name: Some(model_id.to_string()),
                        description: model
                            .get("description")
                            .and_then(|d| d.as_str())
                            .map(|s| s.to_string()),
                        owned_by: model
                            .get("owned_by")
                            .and_then(|o| o.as_str())
                            .unwrap_or(&self.config.provider_id)
                            .to_string(),
                        created: model.get("created").and_then(|c| c.as_u64()),
                        capabilities: self.determine_model_capabilities(model_id),
                        context_window: None,
                        max_output_tokens: None,
                        input_cost_per_token: None,
                        output_cost_per_token: None,
                    });
                }
            }
            Err(LlmError::ApiError { code: 404, .. }) => {
                // Fall through to list+basic.
            }
            Err(e) => {
                // If the provider advertises ModelListingCapability but doesn't support
                // the retrieve endpoint, it may still support listing.
                // For other errors (auth/rate limit/etc.), don't mask the failure.
                return Err(e);
            }
        }

        // Fallback: try to find from list.
        let models = self.list_models_internal().await?;
        if let Some(model) = models.into_iter().find(|m| m.id == model_id) {
            return Ok(model);
        }

        // Final fallback: create a basic model info.
        Ok(ModelInfo {
            id: model_id.clone(),
            name: Some(model_id.clone()),
            description: Some(format!("{} model: {}", self.config.provider_id, model_id)),
            owned_by: self.config.provider_id.clone(),
            created: None,
            capabilities: self.determine_model_capabilities(&model_id),
            context_window: None,
            max_output_tokens: None,
            input_cost_per_token: None,
            output_cost_per_token: None,
        })
    }
}

#[async_trait]
impl ModelListingCapability for OpenAiCompatibleClient {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.list_models_internal().await
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        self.get_model_internal(model_id).await
    }
}

#[async_trait]
impl AudioCapability for OpenAiCompatibleClient {
    fn supported_features(&self) -> &[crate::types::AudioFeature] {
        use crate::types::AudioFeature::{SpeechToText, TextToSpeech};

        const SPEECH_ONLY: &[crate::types::AudioFeature] = &[TextToSpeech];
        const TRANSCRIPTION_ONLY: &[crate::types::AudioFeature] = &[SpeechToText];
        const SPEECH_AND_TRANSCRIPTION: &[crate::types::AudioFeature] =
            &[TextToSpeech, SpeechToText];
        const NONE: &[crate::types::AudioFeature] = &[];

        let caps = self.capabilities();
        match (caps.supports("speech"), caps.supports("transcription")) {
            (true, true) => SPEECH_AND_TRANSCRIPTION,
            (true, false) => SPEECH_ONLY,
            (false, true) => TRANSCRIPTION_ONLY,
            (false, false) => NONE,
        }
    }

    async fn text_to_speech(
        &self,
        request: crate::types::TtsRequest,
    ) -> Result<TtsResponse, LlmError> {
        let request = if let Some(model) = self.resolve_speech_model_default() {
            request.with_model_if_missing(model)
        } else {
            request
        };
        let exec = self.build_audio_executor();
        let result = AudioExecutor::tts(&*exec, request.clone()).await?;

        Ok(TtsResponse {
            audio_data: result.audio_data,
            format: request.format.unwrap_or_else(|| "mp3".to_string()),
            duration: result.duration,
            sample_rate: result.sample_rate,
            metadata: std::collections::HashMap::new(),
        })
    }

    async fn speech_to_text(
        &self,
        request: crate::types::SttRequest,
    ) -> Result<SttResponse, LlmError> {
        let request = if let Some(model) = self.resolve_transcription_model_default() {
            request.with_model_if_missing(model)
        } else {
            request
        };
        let exec = self.build_audio_executor();
        let result = AudioExecutor::stt(&*exec, request).await?;
        let raw = result.raw;

        let language = raw
            .get("language")
            .and_then(|value| value.as_str())
            .map(|value| value.to_string());
        let duration = raw
            .get("duration")
            .and_then(|value| value.as_f64())
            .map(|value| value as f32);
        let words = raw
            .get("words")
            .and_then(|value| value.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| {
                        let object = item.as_object()?;
                        let word = object.get("word")?.as_str()?.to_string();
                        let start = object.get("start")?.as_f64()? as f32;
                        let end = object.get("end")?.as_f64()? as f32;
                        Some(crate::types::WordTimestamp {
                            word,
                            start,
                            end,
                            confidence: None,
                        })
                    })
                    .collect::<Vec<_>>()
            });

        let mut metadata = std::collections::HashMap::new();
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
            words,
            duration,
            metadata,
        })
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
        let exec = self.build_image_executor(&request);
        ImageExecutor::execute(&*exec, request).await
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

        let exec = self.build_image_executor(&ImageGenerationRequest::default());
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

        let exec = self.build_image_executor(&ImageGenerationRequest::default());
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

impl LlmClient for OpenAiCompatibleClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        self.config.adapter.provider_id()
    }

    fn supported_models(&self) -> Vec<String> {
        // Return a basic list - could be enhanced with adapter-specific models
        vec![self.config.model.clone()]
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        let adapter_caps = self.config.adapter.capabilities();
        let has_full_audio =
            adapter_caps.audio || (adapter_caps.speech && adapter_caps.transcription);

        // Convert adapter capabilities to library capabilities
        let mut caps = crate::traits::ProviderCapabilities::new();

        if adapter_caps.chat {
            caps = caps.with_chat();
        }
        if adapter_caps.streaming {
            caps = caps.with_streaming();
        }
        if has_full_audio {
            caps = caps.with_audio();
        } else {
            if adapter_caps.speech {
                caps = caps.with_speech();
            }
            if adapter_caps.transcription {
                caps = caps.with_transcription();
            }
        }
        if adapter_caps.embedding {
            caps = caps.with_embedding();
        }
        if adapter_caps.supports("rerank") {
            caps = caps.with_rerank();
        }
        if adapter_caps.tools {
            caps = caps.with_tools();
        }
        if adapter_caps.vision {
            caps = caps.with_vision();
        }
        if self.config.adapter.supports_image_generation() {
            caps = caps.with_image_generation();
        }
        for (name, enabled) in &adapter_caps.custom_features {
            caps = caps.with_custom_feature(name, *enabled);
        }

        caps
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new((*self).clone())
    }

    fn as_chat_capability(&self) -> Option<&dyn ChatCapability> {
        if self.config.adapter.capabilities().chat {
            Some(self)
        } else {
            None
        }
    }

    fn as_embedding_capability(&self) -> Option<&dyn EmbeddingCapability> {
        if self.config.adapter.capabilities().embedding {
            Some(self)
        } else {
            None
        }
    }

    fn as_audio_capability(&self) -> Option<&dyn AudioCapability> {
        if self.capabilities().supports("audio") {
            Some(self)
        } else {
            None
        }
    }

    fn as_speech_capability(&self) -> Option<&dyn SpeechCapability> {
        if self.capabilities().supports("speech") {
            Some(self)
        } else {
            None
        }
    }

    fn as_speech_extras(&self) -> Option<&dyn SpeechExtras> {
        if self.capabilities().supports("speech") {
            Some(self)
        } else {
            None
        }
    }

    fn as_transcription_capability(&self) -> Option<&dyn TranscriptionCapability> {
        if self.capabilities().supports("transcription") {
            Some(self)
        } else {
            None
        }
    }

    fn as_transcription_extras(&self) -> Option<&dyn TranscriptionExtras> {
        if self.capabilities().supports("transcription") {
            Some(self)
        } else {
            None
        }
    }

    fn as_image_generation_capability(&self) -> Option<&dyn ImageGenerationCapability> {
        if self.config.adapter.supports_image_generation() {
            Some(self)
        } else {
            None
        }
    }

    fn as_image_extras(&self) -> Option<&dyn crate::traits::ImageExtras> {
        if self.config.adapter.supports_image_generation() {
            Some(self)
        } else {
            None
        }
    }

    fn as_rerank_capability(&self) -> Option<&dyn RerankCapability> {
        // Keep capability gating consistent with executor-level guards:
        // rerank must be explicitly declared by the adapter/spec.
        if self.config.adapter.capabilities().supports("rerank") {
            Some(self)
        } else {
            None
        }
    }

    fn as_model_listing_capability(&self) -> Option<&dyn crate::traits::ModelListingCapability> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::transport::{
        HttpTransport, HttpTransportMultipartRequest, HttpTransportRequest, HttpTransportResponse,
        HttpTransportStreamBody, HttpTransportStreamResponse,
    };
    use crate::provider_options::{
        OpenRouterOptions, OpenRouterTransform, PerplexityOptions, PerplexitySearchContextSize,
        PerplexitySearchMode, PerplexitySearchRecencyFilter, PerplexityUserLocation,
    };
    use crate::providers::openai_compatible::ext::{
        OpenRouterChatRequestExt, PerplexityChatRequestExt, PerplexityChatResponseExt,
    };
    use crate::standards::openai::compat::provider_registry::{
        ConfigurableAdapter, ProviderConfig, ProviderFieldMappings,
    };
    use async_trait::async_trait;
    use futures_util::StreamExt;
    use reqwest::header::{ACCEPT, CONTENT_TYPE, HeaderMap, HeaderValue};
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Default)]
    struct NoopInterceptor;

    impl crate::execution::http::interceptor::HttpInterceptor for NoopInterceptor {}

    fn make_audio_adapter() -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "compat-audio".to_string(),
            name: "Compat Audio".to_string(),
            base_url: "https://api.test.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tts".to_string(), "stt".to_string()],
            default_model: None,
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }))
    }

    fn make_fireworks_transcription_adapter() -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "fireworks".to_string(),
            name: "Fireworks AI".to_string(),
            base_url: "https://api.fireworks.ai/inference/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["transcription".to_string()],
            default_model: Some("whisper-v3".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }))
    }

    fn make_fireworks_embedding_adapter() -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "fireworks".to_string(),
            name: "Fireworks AI".to_string(),
            base_url: "https://api.fireworks.ai/inference/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["embedding".to_string()],
            default_model: Some("nomic-ai/nomic-embed-text-v1.5".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }))
    }

    fn make_infini_embedding_adapter() -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "infini".to_string(),
            name: "Infini AI".to_string(),
            base_url: "https://cloud.infini-ai.com/maas/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["embedding".to_string()],
            default_model: Some("text-embedding-3-small".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }))
    }

    fn make_jina_rerank_adapter() -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "jina".to_string(),
            name: "Jina AI".to_string(),
            base_url: "https://api.jina.ai/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["rerank".to_string()],
            default_model: Some("jina-reranker-m0".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }))
    }

    fn make_voyageai_rerank_adapter() -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "voyageai".to_string(),
            name: "VoyageAI".to_string(),
            base_url: "https://api.voyageai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["rerank".to_string()],
            default_model: Some("rerank-2".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }))
    }

    fn make_together_audio_adapter() -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "together".to_string(),
            name: "Together AI".to_string(),
            base_url: "https://api.together.xyz/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["speech".to_string(), "transcription".to_string()],
            default_model: Some("cartesia/sonic-2".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }))
    }

    fn make_together_embedding_adapter() -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "together".to_string(),
            name: "Together AI".to_string(),
            base_url: "https://api.together.xyz/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["embedding".to_string()],
            default_model: Some("togethercomputer/m2-bert-80M-8k-retrieval".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }))
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

    fn make_siliconflow_audio_adapter() -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "siliconflow".to_string(),
            name: "SiliconFlow".to_string(),
            base_url: "https://api.siliconflow.cn/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["speech".to_string(), "transcription".to_string()],
            default_model: Some("FunAudioLLM/SenseVoiceSmall".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }))
    }

    fn make_siliconflow_rerank_adapter() -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "siliconflow".to_string(),
            name: "SiliconFlow".to_string(),
            base_url: "https://api.siliconflow.cn/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["rerank".to_string()],
            default_model: Some("BAAI/bge-reranker-v2-m3".to_string()),
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

    fn make_text_streaming_adapter() -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "compat-chat".to_string(),
            name: "Compat Chat".to_string(),
            base_url: "https://api.test.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: Some("compat-default-model".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }))
    }

    #[tokio::test]
    async fn prepare_chat_request_for_stream_sets_stream_and_fills_defaults() {
        let cfg = OpenAiCompatibleConfig::new(
            "compat-chat",
            "test-key",
            "https://api.test.com/v1",
            make_text_streaming_adapter(),
        )
        .with_model("compat-default-model")
        .with_http_config(crate::types::HttpConfig::default());
        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .build();

        let prepared = client
            .prepare_chat_request(request, true)
            .expect("prepare stream request");

        assert!(prepared.stream);
        assert_eq!(prepared.common_params.model, "compat-default-model");
        assert!(prepared.http_config.is_some());
    }

    #[tokio::test]
    async fn prepare_chat_request_for_non_stream_clears_stream_and_preserves_explicit_model() {
        let cfg = OpenAiCompatibleConfig::new(
            "compat-chat",
            "test-key",
            "https://api.test.com/v1",
            make_text_streaming_adapter(),
        )
        .with_model("compat-default-model")
        .with_http_config(crate::types::HttpConfig::default());
        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ChatRequest::builder()
            .model("compat-explicit-model")
            .messages(vec![ChatMessage::user("hi").build()])
            .stream(true)
            .build();

        let prepared = client
            .prepare_chat_request(request, false)
            .expect("prepare non-stream request");

        assert!(!prepared.stream);
        assert_eq!(prepared.common_params.model, "compat-explicit-model");
        assert!(prepared.http_config.is_some());
    }

    #[tokio::test]
    async fn openai_compatible_client_exposes_audio_capability_views() {
        let cfg = OpenAiCompatibleConfig::new(
            "compat-audio",
            "test-key",
            "https://api.test.com/v1",
            make_audio_adapter(),
        )
        .with_model("gpt-audio-mini");

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap();

        let caps = client.capabilities();
        assert!(caps.supports("audio"));
        assert!(caps.supports("speech"));
        assert!(caps.supports("transcription"));
        assert!(client.as_audio_capability().is_some());
        assert!(client.as_speech_capability().is_some());
        assert!(client.as_transcription_capability().is_some());

        let features = client.as_audio_capability().unwrap().supported_features();
        assert_eq!(features.len(), 2);
        assert!(features.contains(&crate::types::AudioFeature::TextToSpeech));
        assert!(features.contains(&crate::types::AudioFeature::SpeechToText));
    }

    #[tokio::test]
    async fn openai_compatible_client_exposes_fireworks_transcription_only_audio_views() {
        let cfg = OpenAiCompatibleConfig::new(
            "fireworks",
            "test-key",
            "https://api.fireworks.ai/inference/v1",
            make_fireworks_transcription_adapter(),
        )
        .with_model("whisper-v3");

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap();

        let caps = client.capabilities();
        assert!(caps.supports("transcription"));
        assert!(caps.supports("audio"));
        assert!(!caps.supports("speech"));
        assert!(client.as_audio_capability().is_some());
        assert!(client.as_transcription_capability().is_some());
        assert!(client.as_speech_capability().is_none());
        assert_eq!(
            client.as_audio_capability().unwrap().supported_features(),
            &[crate::types::AudioFeature::SpeechToText]
        );
    }

    #[tokio::test]
    async fn openai_compatible_client_fireworks_stt_uses_provider_audio_base_with_custom_transport()
    {
        let transport = MultipartResponseTransport::new(serde_json::json!({
            "text": "hello from fireworks",
            "language": "en"
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "fireworks",
            "test-key",
            "https://api.fireworks.ai/inference/v1",
            make_fireworks_transcription_adapter(),
        )
        .with_model("whisper-v3")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap();

        let mut request = crate::types::SttRequest::from_audio(b"abc".to_vec());
        request.model = Some("whisper-v3".to_string());
        request = request.with_media_type("audio/mpeg".to_string());

        let response = crate::traits::AudioCapability::speech_to_text(&client, request)
            .await
            .expect("fireworks stt should succeed through custom multipart transport");

        assert_eq!(response.text, "hello from fireworks");
        assert_eq!(response.language.as_deref(), Some("en"));

        let captured = transport.take().expect("captured multipart request");
        assert_eq!(
            captured.url,
            "https://audio.fireworks.ai/v1/audio/transcriptions"
        );
        assert!(
            captured
                .headers
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("multipart/form-data; boundary="))
        );

        let body_text = String::from_utf8_lossy(&captured.body);
        assert!(body_text.contains("name=\"model\""));
        assert!(body_text.contains("whisper-v3"));
        assert!(body_text.contains("name=\"response_format\""));
        assert!(body_text.contains("json"));
        assert!(body_text.contains("name=\"file\"; filename=\"audio.mp3\""));
        assert!(body_text.contains("Content-Type: audio/mpeg"));
        assert!(body_text.contains("abc"));
    }

    #[tokio::test]
    async fn openai_compatible_client_fireworks_stt_uses_explicit_base_url_override() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("POST", "/v1/audio/transcriptions")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"text":"hello from fireworks","language":"en"}"#)
            .create_async()
            .await;

        let cfg = OpenAiCompatibleConfig::new(
            "fireworks",
            "test-key",
            &format!("{}/v1", server.url()),
            make_fireworks_transcription_adapter(),
        )
        .with_model("whisper-v3");

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap();

        let mut request = crate::types::SttRequest::from_audio(b"abc".to_vec());
        request.model = Some("whisper-v3".to_string());
        request = request.with_media_type("audio/mpeg".to_string());

        let response = crate::traits::AudioCapability::speech_to_text(&client, request)
            .await
            .expect("fireworks stt should succeed against overridden base url");

        assert_eq!(response.text, "hello from fireworks");
        assert_eq!(response.language.as_deref(), Some("en"));
    }

    #[tokio::test]
    async fn embed_with_config_runtime_fireworks_uses_inference_boundary_and_preserves_request_shape()
     {
        let transport = CaptureTransport::default();
        let cfg = OpenAiCompatibleConfig::new(
            "fireworks",
            "test-key",
            "https://api.fireworks.ai/inference/v1",
            make_fireworks_embedding_adapter(),
        )
        .with_model("nomic-ai/nomic-embed-text-v1.5")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = EmbeddingRequest::single("hello fireworks embedding")
            .with_model("nomic-ai/nomic-embed-text-v1.5")
            .with_dimensions(256)
            .with_encoding_format(EmbeddingFormat::Base64)
            .with_user("compat-user-1");

        let _ = crate::traits::EmbeddingExtensions::embed_with_config(&client, request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(
            captured.url,
            "https://api.fireworks.ai/inference/v1/embeddings"
        );
        assert_eq!(
            captured.body["model"],
            serde_json::json!("nomic-ai/nomic-embed-text-v1.5")
        );
        assert_eq!(
            captured.body["input"],
            serde_json::json!(["hello fireworks embedding"])
        );
        assert_eq!(captured.body["dimensions"], serde_json::json!(256));
        assert_eq!(
            captured.body["encoding_format"],
            serde_json::json!("base64")
        );
        assert_eq!(captured.body["user"], serde_json::json!("compat-user-1"));
    }

    #[tokio::test]
    async fn embed_with_config_runtime_infini_preserves_request_shape_at_transport_boundary() {
        let transport = CaptureTransport::default();
        let cfg = OpenAiCompatibleConfig::new(
            "infini",
            "test-key",
            "https://cloud.infini-ai.com/maas/v1",
            make_infini_embedding_adapter(),
        )
        .with_model("text-embedding-3-small")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = EmbeddingRequest::single("hello infini embedding")
            .with_model("text-embedding-3-small")
            .with_dimensions(512)
            .with_encoding_format(EmbeddingFormat::Float)
            .with_user("compat-user-7");

        let _ = crate::traits::EmbeddingExtensions::embed_with_config(&client, request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(
            captured.url,
            "https://cloud.infini-ai.com/maas/v1/embeddings"
        );
        assert_eq!(
            captured.body["model"],
            serde_json::json!("text-embedding-3-small")
        );
        assert_eq!(
            captured.body["input"],
            serde_json::json!(["hello infini embedding"])
        );
        assert_eq!(captured.body["dimensions"], serde_json::json!(512));
        assert_eq!(captured.body["encoding_format"], serde_json::json!("float"));
        assert_eq!(captured.body["user"], serde_json::json!("compat-user-7"));
    }

    #[tokio::test]
    async fn embed_with_config_runtime_together_preserves_request_shape_at_transport_boundary() {
        let transport = CaptureTransport::default();
        let cfg = OpenAiCompatibleConfig::new(
            "together",
            "test-key",
            "https://api.together.xyz/v1",
            make_together_embedding_adapter(),
        )
        .with_model("togethercomputer/m2-bert-80M-8k-retrieval")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = EmbeddingRequest::single("hello together embedding")
            .with_model("togethercomputer/m2-bert-80M-8k-retrieval")
            .with_dimensions(384)
            .with_encoding_format(EmbeddingFormat::Float)
            .with_user("compat-user-4");

        let _ = crate::traits::EmbeddingExtensions::embed_with_config(&client, request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.url, "https://api.together.xyz/v1/embeddings");
        assert_eq!(
            captured.body["model"],
            serde_json::json!("togethercomputer/m2-bert-80M-8k-retrieval")
        );
        assert_eq!(
            captured.body["input"],
            serde_json::json!(["hello together embedding"])
        );
        assert_eq!(captured.body["dimensions"], serde_json::json!(384));
        assert_eq!(captured.body["encoding_format"], serde_json::json!("float"));
        assert_eq!(captured.body["user"], serde_json::json!("compat-user-4"));
    }

    #[tokio::test]
    async fn embed_with_config_runtime_together_missing_model_uses_embedding_family_default() {
        let transport = CaptureTransport::default();
        let cfg = OpenAiCompatibleConfig::new(
            "together",
            "test-key",
            "https://api.together.xyz/v1",
            make_together_embedding_adapter(),
        )
        .with_model("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = EmbeddingRequest::single("hello together embedding")
            .with_dimensions(384)
            .with_encoding_format(EmbeddingFormat::Float)
            .with_user("compat-user-4");

        let _ = crate::traits::EmbeddingExtensions::embed_with_config(&client, request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(
            captured.body["model"],
            serde_json::json!("togethercomputer/m2-bert-80M-8k-retrieval")
        );
    }

    #[tokio::test]
    async fn embed_with_config_runtime_together_missing_model_preserves_explicit_config_override() {
        let transport = CaptureTransport::default();
        let cfg = OpenAiCompatibleConfig::new(
            "together",
            "test-key",
            "https://api.together.xyz/v1",
            make_together_embedding_adapter(),
        )
        .with_model("custom-embedding-override")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = EmbeddingRequest::single("hello together embedding");

        let _ = crate::traits::EmbeddingExtensions::embed_with_config(&client, request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(
            captured.body["model"],
            serde_json::json!("custom-embedding-override")
        );
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
            count: 1,
            model: Some("black-forest-labs/FLUX.1-schnell".to_string()),
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
        assert_eq!(captured.body["size"], serde_json::json!("1024x1024"));
        assert_eq!(captured.body["n"], serde_json::json!(1));
        assert_eq!(captured.body["response_format"], serde_json::json!("url"));
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
    async fn openai_compatible_client_together_tts_uses_default_audio_base_with_custom_transport() {
        let transport = BytesResponseTransport::new(vec![1, 2, 3, 4], "audio/mpeg");

        let cfg = OpenAiCompatibleConfig::new(
            "together",
            "test-key",
            "https://api.together.xyz/v1",
            make_together_audio_adapter(),
        )
        .with_model("cartesia/sonic-2")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap();

        let request = crate::types::TtsRequest::new("hello from together".to_string())
            .with_voice("alloy".to_string())
            .with_format("mp3".to_string());

        let response = crate::traits::AudioCapability::text_to_speech(&client, request)
            .await
            .expect("together tts should succeed through custom transport");

        assert_eq!(response.audio_data, vec![1, 2, 3, 4]);
        assert_eq!(response.format, "mp3");

        let captured = transport.take().expect("captured json request");
        assert_eq!(captured.url, "https://api.together.xyz/v1/audio/speech");
        assert_eq!(
            captured.body["model"],
            serde_json::json!("cartesia/sonic-2")
        );
        assert_eq!(
            captured.body["input"],
            serde_json::json!("hello from together")
        );
        assert_eq!(captured.body["voice"], serde_json::json!("alloy"));
        assert_eq!(captured.body["response_format"], serde_json::json!("mp3"));
    }

    #[tokio::test]
    async fn openai_compatible_client_together_tts_missing_model_uses_speech_family_default() {
        let transport = BytesResponseTransport::new(vec![1, 2, 3, 4], "audio/mpeg");

        let cfg = OpenAiCompatibleConfig::new(
            "together",
            "test-key",
            "https://api.together.xyz/v1",
            make_together_audio_adapter(),
        )
        .with_model("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap();

        let request = crate::types::TtsRequest::new("hello from together".to_string())
            .with_voice("alloy".to_string())
            .with_format("mp3".to_string());

        let _ = crate::traits::AudioCapability::text_to_speech(&client, request)
            .await
            .expect("together tts should succeed through custom transport");

        let captured = transport.take().expect("captured json request");
        assert_eq!(
            captured.body["model"],
            serde_json::json!("cartesia/sonic-2")
        );
    }

    #[tokio::test]
    async fn openai_compatible_client_together_stt_uses_default_audio_base_with_custom_transport() {
        let transport = MultipartResponseTransport::new(serde_json::json!({
            "text": "hello from together",
            "language": "en"
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "together",
            "test-key",
            "https://api.together.xyz/v1",
            make_together_audio_adapter(),
        )
        .with_model("openai/whisper-large-v3")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap();

        let mut request = crate::types::SttRequest::from_audio(b"abc".to_vec());
        request = request.with_media_type("audio/mpeg".to_string());

        let response = crate::traits::AudioCapability::speech_to_text(&client, request)
            .await
            .expect("together stt should succeed through custom multipart transport");

        assert_eq!(response.text, "hello from together");
        assert_eq!(response.language.as_deref(), Some("en"));

        let captured = transport.take().expect("captured multipart request");
        assert_eq!(
            captured.url,
            "https://api.together.xyz/v1/audio/transcriptions"
        );
        assert!(
            captured
                .headers
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("multipart/form-data; boundary="))
        );

        let body_text = String::from_utf8_lossy(&captured.body);
        assert!(body_text.contains("name=\"model\""));
        assert!(body_text.contains("openai/whisper-large-v3"));
        assert!(body_text.contains("name=\"response_format\""));
        assert!(body_text.contains("json"));
        assert!(body_text.contains("name=\"file\"; filename=\"audio.mp3\""));
        assert!(body_text.contains("Content-Type: audio/mpeg"));
        assert!(body_text.contains("abc"));
    }

    #[tokio::test]
    async fn openai_compatible_client_fireworks_stt_missing_model_uses_transcription_family_default()
     {
        let transport = MultipartResponseTransport::new(serde_json::json!({
            "text": "hello from fireworks",
            "language": "en"
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "fireworks",
            "test-key",
            "https://api.fireworks.ai/inference/v1",
            make_fireworks_transcription_adapter(),
        )
        .with_model("accounts/fireworks/models/llama-v3p1-8b-instruct")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap();

        let request = crate::types::SttRequest::from_audio(b"abc".to_vec())
            .with_media_type("audio/mpeg".to_string());

        let _ = crate::traits::AudioCapability::speech_to_text(&client, request)
            .await
            .expect("fireworks stt should succeed through custom multipart transport");

        let captured = transport.take().expect("captured multipart request");
        let body_text = String::from_utf8_lossy(&captured.body);

        assert_eq!(
            captured.url,
            "https://audio.fireworks.ai/v1/audio/transcriptions"
        );
        assert!(body_text.contains("name=\"model\""));
        assert!(body_text.contains("whisper-v3"));
    }

    #[tokio::test]
    async fn openai_compatible_client_siliconflow_stt_uses_default_audio_base_with_custom_transport()
     {
        let transport = MultipartResponseTransport::new(serde_json::json!({
            "text": "hello from siliconflow",
            "language": "zh"
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "siliconflow",
            "test-key",
            "https://api.siliconflow.cn/v1",
            make_siliconflow_audio_adapter(),
        )
        .with_model("FunAudioLLM/SenseVoiceSmall")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap();

        let mut request = crate::types::SttRequest::from_audio(b"abc".to_vec());
        request.model = Some("FunAudioLLM/SenseVoiceSmall".to_string());
        request = request.with_media_type("audio/mpeg".to_string());

        let response = crate::traits::AudioCapability::speech_to_text(&client, request)
            .await
            .expect("siliconflow stt should succeed through custom multipart transport");

        assert_eq!(response.text, "hello from siliconflow");
        assert_eq!(response.language.as_deref(), Some("zh"));

        let captured = transport.take().expect("captured multipart request");
        assert_eq!(
            captured.url,
            "https://api.siliconflow.cn/v1/audio/transcriptions"
        );
        assert!(
            captured
                .headers
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("multipart/form-data; boundary="))
        );

        let body_text = String::from_utf8_lossy(&captured.body);
        assert!(body_text.contains("name=\"model\""));
        assert!(body_text.contains("FunAudioLLM/SenseVoiceSmall"));
        assert!(body_text.contains("name=\"response_format\""));
        assert!(body_text.contains("json"));
        assert!(body_text.contains("name=\"file\"; filename=\"audio.mp3\""));
        assert!(body_text.contains("Content-Type: audio/mpeg"));
        assert!(body_text.contains("abc"));
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

    #[tokio::test]
    async fn rerank_runtime_jina_missing_model_uses_rerank_family_default() {
        let transport = CaptureTransport::default();
        let cfg = OpenAiCompatibleConfig::new(
            "jina",
            "test-key",
            "https://api.jina.ai/v1",
            make_jina_rerank_adapter(),
        )
        .with_model("jina-embeddings-v2-base-en")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = RerankRequest::new(
            String::new(),
            "query".to_string(),
            vec!["doc-1".to_string(), "doc-2".to_string()],
        )
        .with_top_n(1);

        let _ = client.rerank(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(
            captured.body["model"],
            serde_json::json!("jina-reranker-m0")
        );
    }

    #[tokio::test]
    async fn rerank_runtime_siliconflow_preserves_request_shape_at_transport_boundary() {
        let transport = CaptureTransport::default();
        let cfg = OpenAiCompatibleConfig::new(
            "siliconflow",
            "test-key",
            "https://api.siliconflow.cn/v1",
            make_siliconflow_rerank_adapter(),
        )
        .with_model("BAAI/bge-reranker-v2-m3")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = RerankRequest::new(
            "BAAI/bge-reranker-v2-m3".to_string(),
            "query".to_string(),
            vec!["doc-1".to_string(), "doc-2".to_string()],
        )
        .with_top_n(1);

        let _ = client.rerank(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.url, "https://api.siliconflow.cn/v1/rerank");
        assert_eq!(
            captured.body["model"],
            serde_json::json!("BAAI/bge-reranker-v2-m3")
        );
        assert_eq!(captured.body["query"], serde_json::json!("query"));
        assert_eq!(
            captured.body["documents"],
            serde_json::json!(["doc-1", "doc-2"])
        );
        assert_eq!(captured.body["top_n"], serde_json::json!(1));
    }

    #[tokio::test]
    async fn rerank_runtime_jina_preserves_request_shape_at_transport_boundary() {
        let transport = CaptureTransport::default();
        let cfg = OpenAiCompatibleConfig::new(
            "jina",
            "test-key",
            "https://api.jina.ai/v1",
            make_jina_rerank_adapter(),
        )
        .with_model("jina-reranker-m0")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = RerankRequest::new(
            "jina-reranker-m0".to_string(),
            "query".to_string(),
            vec!["doc-1".to_string(), "doc-2".to_string()],
        )
        .with_top_n(1);

        let _ = client.rerank(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.url, "https://api.jina.ai/v1/rerank");
        assert_eq!(
            captured.body["model"],
            serde_json::json!("jina-reranker-m0")
        );
        assert_eq!(captured.body["query"], serde_json::json!("query"));
        assert_eq!(
            captured.body["documents"],
            serde_json::json!(["doc-1", "doc-2"])
        );
        assert_eq!(captured.body["top_n"], serde_json::json!(1));
    }

    #[tokio::test]
    async fn rerank_runtime_voyageai_preserves_request_shape_at_transport_boundary() {
        let transport = CaptureTransport::default();
        let cfg = OpenAiCompatibleConfig::new(
            "voyageai",
            "test-key",
            "https://api.voyageai.com/v1",
            make_voyageai_rerank_adapter(),
        )
        .with_model("rerank-2")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = RerankRequest::new(
            "rerank-2".to_string(),
            "query".to_string(),
            vec!["doc-1".to_string(), "doc-2".to_string()],
        )
        .with_top_n(1);

        let _ = client.rerank(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.url, "https://api.voyageai.com/v1/rerank");
        assert_eq!(captured.body["model"], serde_json::json!("rerank-2"));
        assert_eq!(captured.body["query"], serde_json::json!("query"));
        assert_eq!(
            captured.body["documents"],
            serde_json::json!(["doc-1", "doc-2"])
        );
        assert_eq!(captured.body["top_n"], serde_json::json!(1));
    }

    #[tokio::test]
    async fn openai_compatible_client_siliconflow_tts_uses_default_audio_base_with_custom_transport()
     {
        let transport = BytesResponseTransport::new(vec![9, 8, 7, 6], "audio/wav");

        let cfg = OpenAiCompatibleConfig::new(
            "siliconflow",
            "test-key",
            "https://api.siliconflow.cn/v1",
            make_siliconflow_audio_adapter(),
        )
        .with_model("FunAudioLLM/CosyVoice2-0.5B")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap();

        let request = crate::types::TtsRequest::new("hello from siliconflow".to_string())
            .with_model("FunAudioLLM/CosyVoice2-0.5B".to_string())
            .with_voice("alloy".to_string())
            .with_format("wav".to_string());

        let response = crate::traits::AudioCapability::text_to_speech(&client, request)
            .await
            .expect("siliconflow tts should succeed through custom transport");

        assert_eq!(response.audio_data, vec![9, 8, 7, 6]);
        assert_eq!(response.format, "wav");

        let captured = transport.take().expect("captured json request");
        assert_eq!(captured.url, "https://api.siliconflow.cn/v1/audio/speech");
        assert_eq!(
            captured.body["model"],
            serde_json::json!("FunAudioLLM/CosyVoice2-0.5B")
        );
        assert_eq!(
            captured.body["input"],
            serde_json::json!("hello from siliconflow")
        );
        assert_eq!(captured.body["voice"], serde_json::json!("alloy"));
        assert_eq!(captured.body["response_format"], serde_json::json!("wav"));
    }

    #[tokio::test]
    async fn build_chat_executor_exposes_runtime_provider_before_send_via_provider_spec() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.test.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg =
            OpenAiCompatibleConfig::new("deepseek", "test-key", "https://api.test.com/v1", adapter)
                .with_model("test-model");

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap();

        let req = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_common_params(CommonParams {
                model: "test-model".to_string(),
                ..Default::default()
            })
            .with_provider_option("deepseek", serde_json::json!({ "my_custom": 1 }));

        let exec = client.build_chat_executor(&req);
        assert!(exec.policy.before_send.is_none());
        assert!(
            exec.provider_spec
                .chat_before_send(&req, &exec.provider_context)
                .is_some()
        );
    }

    #[derive(Clone, Default)]
    struct CaptureTransport {
        last: Arc<Mutex<Option<HttpTransportRequest>>>,
        last_stream: Arc<Mutex<Option<HttpTransportRequest>>>,
    }

    impl CaptureTransport {
        fn take(&self) -> Option<HttpTransportRequest> {
            self.last.lock().unwrap().take()
        }

        fn take_stream(&self) -> Option<HttpTransportRequest> {
            self.last_stream.lock().unwrap().take()
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

        async fn execute_stream(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportStreamResponse, LlmError> {
            *self.last_stream.lock().unwrap() = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportStreamResponse {
                status: 401,
                headers,
                body: HttpTransportStreamBody::from_bytes(
                    br#"{"error":{"message":"unauthorized","type":"auth_error","code":"unauthorized"}}"#
                        .to_vec(),
                ),
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

        async fn execute_stream(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportStreamResponse, LlmError> {
            *self.last.lock().unwrap() = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportStreamResponse {
                status: 501,
                headers,
                body: HttpTransportStreamBody::from_bytes(
                    br#"{"error":{"message":"stream unsupported in test","type":"test_error","code":"unsupported"}}"#
                        .to_vec(),
                ),
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

    #[derive(Clone)]
    struct BytesResponseTransport {
        response_body: Arc<Vec<u8>>,
        response_content_type: &'static str,
        last: Arc<Mutex<Option<HttpTransportRequest>>>,
    }

    impl BytesResponseTransport {
        fn new(response_body: Vec<u8>, response_content_type: &'static str) -> Self {
            Self {
                response_body: Arc::new(response_body),
                response_content_type,
                last: Arc::new(Mutex::new(None)),
            }
        }

        fn take(&self) -> Option<HttpTransportRequest> {
            self.last.lock().unwrap().take()
        }
    }

    #[async_trait]
    impl HttpTransport for BytesResponseTransport {
        async fn execute_json(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last.lock().unwrap() = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(
                CONTENT_TYPE,
                HeaderValue::from_static(self.response_content_type),
            );

            Ok(HttpTransportResponse {
                status: 200,
                headers,
                body: self.response_body.as_ref().clone(),
            })
        }
    }

    #[derive(Clone)]
    struct SseResponseTransport {
        response_body: Arc<Vec<u8>>,
        last_stream: Arc<Mutex<Option<HttpTransportRequest>>>,
    }

    impl SseResponseTransport {
        fn new(body: impl Into<Vec<u8>>) -> Self {
            Self {
                response_body: Arc::new(body.into()),
                last_stream: Arc::new(Mutex::new(None)),
            }
        }

        fn take_stream(&self) -> Option<HttpTransportRequest> {
            self.last_stream.lock().unwrap().take()
        }
    }

    #[async_trait]
    impl HttpTransport for SseResponseTransport {
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

        async fn execute_stream(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportStreamResponse, LlmError> {
            *self.last_stream.lock().unwrap() = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));

            Ok(HttpTransportStreamResponse {
                status: 200,
                headers,
                body: HttpTransportStreamBody::from_bytes(self.response_body.as_ref().clone()),
            })
        }
    }

    #[tokio::test]
    async fn chat_request_preserves_request_level_provider_options() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.test.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = CaptureTransport::default();

        let cfg =
            OpenAiCompatibleConfig::new("deepseek", "test-key", "https://api.test.com/v1", adapter)
                .with_model("test-model")
                .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_common_params(CommonParams {
                model: "test-model".to_string(),
                ..Default::default()
            })
            .with_provider_option("deepseek", serde_json::json!({ "my_custom": 1 }));

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");
        assert_eq!(captured.body["my_custom"], serde_json::json!(1));
    }

    #[tokio::test]
    async fn chat_request_runtime_xai_preserves_stable_fields_at_transport_boundary() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "xai".to_string(),
            name: "xAI".to_string(),
            base_url: "https://api.x.ai/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = CaptureTransport::default();

        let cfg = OpenAiCompatibleConfig::new("xai", "test-key", "https://api.x.ai/v1", adapter)
            .with_model("grok-4")
            .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model("grok-4")
            .messages(vec![ChatMessage::user("hi").build()])
            .stop_sequences(vec!["END".to_string()])
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::None)
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                schema.clone(),
            ))
            .build()
            .with_provider_option(
                "xai",
                serde_json::json!({
                    "response_format": { "type": "json_object" },
                    "tool_choice": "auto",
                    "reasoningEffort": "high"
                }),
            );

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");

        assert!(captured.body.get("stop").is_none());
        assert_eq!(captured.body["tool_choice"], serde_json::json!("none"));
        assert_eq!(captured.body["reasoning_effort"], serde_json::json!("high"));
        assert_eq!(
            captured.body["response_format"],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            })
        );
    }

    #[tokio::test]
    async fn chat_request_runtime_deepseek_normalizes_reasoning_options_and_preserves_stable_fields_at_transport_boundary()
     {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.deepseek.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = CaptureTransport::default();

        let cfg = OpenAiCompatibleConfig::new(
            "deepseek",
            "test-key",
            "https://api.deepseek.com/v1",
            adapter,
        )
        .with_model("deepseek-chat")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::None)
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                schema.clone(),
            ))
            .build()
            .with_provider_option(
                "deepseek",
                serde_json::json!({
                    "enableReasoning": true,
                    "reasoningBudget": 2048,
                    "response_format": { "type": "json_object" },
                    "tool_choice": "auto"
                }),
            );

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.body["enable_reasoning"], serde_json::json!(true));
        assert_eq!(captured.body["reasoning_budget"], serde_json::json!(2048));
        assert!(captured.body.get("enableReasoning").is_none());
        assert!(captured.body.get("reasoningBudget").is_none());
        assert_eq!(captured.body["tool_choice"], serde_json::json!("none"));
        assert_eq!(
            captured.body["response_format"],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            })
        );
    }

    #[tokio::test]
    async fn chat_request_runtime_openrouter_preserves_stable_fields_and_vendor_params_at_transport_boundary()
     {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openrouter".to_string(),
            name: "OpenRouter".to_string(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
                "reasoning".to_string(),
            ],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = CaptureTransport::default();

        let cfg = OpenAiCompatibleConfig::new(
            "openrouter",
            "test-key",
            "https://openrouter.ai/api/v1",
            adapter,
        )
        .with_model("openai/gpt-4o")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model("openai/gpt-4o")
            .messages(vec![ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::None)
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                schema.clone(),
            ))
            .build()
            .with_provider_option(
                "openrouter",
                serde_json::json!({
                    "transforms": ["middle-out"],
                    "someVendorParam": true,
                    "response_format": { "type": "json_object" },
                    "tool_choice": "auto"
                }),
            );

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.body["tool_choice"], serde_json::json!("none"));
        assert_eq!(
            captured.body["transforms"],
            serde_json::json!(["middle-out"])
        );
        assert_eq!(captured.body["someVendorParam"], serde_json::json!(true));
        assert_eq!(
            captured.body["response_format"],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            })
        );
    }

    #[tokio::test]
    async fn chat_request_runtime_openrouter_typed_options_preserve_final_request_shape() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openrouter".to_string(),
            name: "OpenRouter".to_string(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
                "reasoning".to_string(),
            ],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = CaptureTransport::default();

        let cfg = OpenAiCompatibleConfig::new(
            "openrouter",
            "test-key",
            "https://openrouter.ai/api/v1",
            adapter,
        )
        .with_model("openai/gpt-4o")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model("openai/gpt-4o")
            .messages(vec![ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::None)
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                schema.clone(),
            ))
            .build()
            .with_openrouter_options(
                OpenRouterOptions::new()
                    .with_transform(OpenRouterTransform::MiddleOut)
                    .with_param("someVendorParam", serde_json::json!(true)),
            );

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.body["tool_choice"], serde_json::json!("none"));
        assert_eq!(
            captured.body["transforms"],
            serde_json::json!(["middle-out"])
        );
        assert_eq!(captured.body["someVendorParam"], serde_json::json!(true));
        assert_eq!(
            captured.body["response_format"],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            })
        );
    }

    #[tokio::test]
    async fn chat_stream_request_runtime_openrouter_preserves_stable_fields_reasoning_defaults_and_vendor_params_at_transport_boundary()
     {
        let transport = CaptureTransport::default();
        let client = crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(
            crate::builder::BuilderBase::default(),
            "openrouter",
        )
        .api_key("test")
        .model("openai/gpt-4o")
        .reasoning(true)
        .reasoning_budget(2048)
        .with_http_transport(Arc::new(transport.clone()))
        .build()
        .await
        .expect("builder should succeed");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model("openai/gpt-4o")
            .messages(vec![ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::None)
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                schema.clone(),
            ))
            .build()
            .with_openrouter_options(
                OpenRouterOptions::new()
                    .with_transform(OpenRouterTransform::MiddleOut)
                    .with_param("someVendorParam", serde_json::json!(true)),
            );

        let _ = crate::traits::ChatCapability::chat_stream_request(&client, request).await;
        let captured = transport.take_stream().expect("captured stream request");

        assert_eq!(
            captured
                .headers
                .get(ACCEPT)
                .and_then(|value| value.to_str().ok()),
            Some("text/event-stream")
        );
        assert_eq!(captured.body["stream"], serde_json::json!(true));
        assert_eq!(captured.body["enable_reasoning"], serde_json::json!(true));
        assert_eq!(captured.body["reasoning_budget"], serde_json::json!(2048));
        assert_eq!(captured.body["tool_choice"], serde_json::json!("none"));
        assert_eq!(
            captured.body["transforms"],
            serde_json::json!(["middle-out"])
        );
        assert_eq!(captured.body["someVendorParam"], serde_json::json!(true));
        assert_eq!(
            captured.body["response_format"],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            })
        );
    }

    #[tokio::test]
    async fn chat_request_runtime_perplexity_preserves_stable_fields_and_vendor_params_at_transport_boundary()
     {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "perplexity".to_string(),
            name: "Perplexity".to_string(),
            base_url: "https://api.perplexity.ai".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = CaptureTransport::default();

        let cfg = OpenAiCompatibleConfig::new(
            "perplexity",
            "test-key",
            "https://api.perplexity.ai",
            adapter,
        )
        .with_model("sonar")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model("sonar")
            .messages(vec![ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::None)
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                schema.clone(),
            ))
            .build()
            .with_provider_option(
                "perplexity",
                serde_json::json!({
                    "search_mode": "academic",
                    "someVendorParam": true,
                    "response_format": { "type": "json_object" },
                    "tool_choice": "auto"
                }),
            );

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.body["tool_choice"], serde_json::json!("none"));
        assert_eq!(captured.body["search_mode"], serde_json::json!("academic"));
        assert_eq!(captured.body["someVendorParam"], serde_json::json!(true));
        assert_eq!(
            captured.body["response_format"],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            })
        );
    }

    #[tokio::test]
    async fn chat_stream_request_runtime_perplexity_preserves_stable_fields_and_typed_vendor_params_at_transport_boundary()
     {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "perplexity".to_string(),
            name: "Perplexity".to_string(),
            base_url: "https://api.perplexity.ai".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = CaptureTransport::default();

        let cfg = OpenAiCompatibleConfig::new(
            "perplexity",
            "test-key",
            "https://api.perplexity.ai",
            adapter,
        )
        .with_model("sonar")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model("sonar")
            .messages(vec![ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::None)
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                schema.clone(),
            ))
            .build()
            .with_perplexity_options(
                PerplexityOptions::new()
                    .with_search_mode(PerplexitySearchMode::Academic)
                    .with_search_context_size(PerplexitySearchContextSize::High)
                    .with_return_images(true)
                    .with_user_location(PerplexityUserLocation::new().with_country("US"))
                    .with_param("someVendorParam", serde_json::json!(true)),
            );

        let _ = crate::traits::ChatCapability::chat_stream_request(&client, request).await;
        let captured = transport.take_stream().expect("captured stream request");

        assert_eq!(
            captured
                .headers
                .get(ACCEPT)
                .and_then(|value| value.to_str().ok()),
            Some("text/event-stream")
        );
        assert_eq!(captured.body["stream"], serde_json::json!(true));
        assert_eq!(captured.body["tool_choice"], serde_json::json!("none"));
        assert_eq!(captured.body["search_mode"], serde_json::json!("academic"));
        assert_eq!(captured.body["return_images"], serde_json::json!(true));
        assert_eq!(
            captured.body["web_search_options"]["search_context_size"],
            serde_json::json!("high")
        );
        assert_eq!(
            captured.body["web_search_options"]["user_location"]["country"],
            serde_json::json!("US")
        );
        assert_eq!(captured.body["someVendorParam"], serde_json::json!(true));
        assert_eq!(
            captured.body["response_format"],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            })
        );
    }

    #[tokio::test]
    async fn chat_request_runtime_perplexity_typed_options_preserve_final_request_shape() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "perplexity".to_string(),
            name: "Perplexity".to_string(),
            base_url: "https://api.perplexity.ai".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = CaptureTransport::default();

        let cfg = OpenAiCompatibleConfig::new(
            "perplexity",
            "test-key",
            "https://api.perplexity.ai",
            adapter,
        )
        .with_model("sonar")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model("sonar")
            .messages(vec![ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::None)
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                schema.clone(),
            ))
            .build()
            .with_perplexity_options(
                PerplexityOptions::new()
                    .with_search_mode(PerplexitySearchMode::Academic)
                    .with_search_recency_filter(PerplexitySearchRecencyFilter::Month)
                    .with_return_images(true)
                    .with_search_context_size(PerplexitySearchContextSize::High)
                    .with_user_location(PerplexityUserLocation::new().with_country("US"))
                    .with_param("someVendorParam", serde_json::json!(true)),
            );

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.body["tool_choice"], serde_json::json!("none"));
        assert_eq!(captured.body["search_mode"], serde_json::json!("academic"));
        assert_eq!(
            captured.body["search_recency_filter"],
            serde_json::json!("month")
        );
        assert_eq!(captured.body["return_images"], serde_json::json!(true));
        assert_eq!(
            captured.body["web_search_options"]["search_context_size"],
            serde_json::json!("high")
        );
        assert_eq!(
            captured.body["web_search_options"]["user_location"]["country"],
            serde_json::json!("US")
        );
        assert_eq!(captured.body["someVendorParam"], serde_json::json!(true));
        assert_eq!(
            captured.body["response_format"],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            })
        );
    }

    #[tokio::test]
    async fn chat_request_runtime_perplexity_exposes_typed_response_metadata() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "perplexity".to_string(),
            name: "Perplexity".to_string(),
            base_url: "https://api.perplexity.ai".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = JsonResponseTransport::new(serde_json::json!({
            "id": "chatcmpl-perplexity-test",
            "object": "chat.completion",
            "created": 1_741_392_000,
            "model": "sonar",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Rust async tooling kept improving across the ecosystem."
                    },
                    "finish_reason": "stop"
                }
            ],
            "citations": ["https://example.com/rust"],
            "images": [
                {
                    "image_url": "https://images.example.com/rust.png",
                    "origin_url": "https://example.com/rust",
                    "height": 900,
                    "width": 1600
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 17,
                "total_tokens": 28,
                "citation_tokens": 7,
                "num_search_queries": 2,
                "reasoning_tokens": 3
            }
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "perplexity",
            "test-key",
            "https://api.perplexity.ai",
            adapter,
        )
        .with_model("sonar")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ChatRequest::builder()
            .model("sonar")
            .messages(vec![ChatMessage::user("hi").build()])
            .build()
            .with_perplexity_options(PerplexityOptions::new().with_return_images(true));

        let response = client.chat_request(request).await.expect("response ok");
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.body["return_images"], serde_json::json!(true));
        assert_eq!(
            response.content_text(),
            Some("Rust async tooling kept improving across the ecosystem.")
        );

        let meta = response.perplexity_metadata().expect("perplexity metadata");
        assert_eq!(
            meta.citations.as_ref(),
            Some(&vec!["https://example.com/rust".to_string()])
        );
        assert_eq!(meta.images.as_ref().map(Vec::len), Some(1));
        assert_eq!(
            meta.images
                .as_ref()
                .and_then(|images| images.first())
                .map(|image| image.image_url.as_str()),
            Some("https://images.example.com/rust.png")
        );
        assert_eq!(
            meta.usage.as_ref().and_then(|usage| usage.citation_tokens),
            Some(7)
        );
        assert_eq!(
            meta.usage
                .as_ref()
                .and_then(|usage| usage.num_search_queries),
            Some(2)
        );
        assert_eq!(
            meta.usage.as_ref().and_then(|usage| usage.reasoning_tokens),
            Some(3)
        );
        assert_eq!(meta.extra.get("citations"), None);
    }

    #[tokio::test]
    async fn chat_stream_request_runtime_perplexity_exposes_typed_response_metadata() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "perplexity".to_string(),
            name: "Perplexity".to_string(),
            base_url: "https://api.perplexity.ai".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = SseResponseTransport::new(
            br#"data: {"id":"1","model":"sonar","created":1718345013,"citations":["https://example.com/rust"],"choices":[{"index":0,"delta":{"content":"Rust","role":"assistant"},"finish_reason":null}]}

data: {"id":"1","model":"sonar","created":1718345013,"choices":[{"index":0,"delta":{"content":" ecosystem","role":null},"finish_reason":"stop"}],"images":[{"image_url":"https://images.example.com/rust.png","origin_url":"https://example.com/rust","height":900,"width":1600}],"usage":{"prompt_tokens":11,"completion_tokens":17,"total_tokens":28,"citation_tokens":7,"num_search_queries":2,"reasoning_tokens":3}}

data: [DONE]

"#.to_vec(),
        );

        let cfg = OpenAiCompatibleConfig::new(
            "perplexity",
            "test-key",
            "https://api.perplexity.ai",
            adapter,
        )
        .with_model("sonar")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ChatRequest::builder()
            .model("sonar")
            .messages(vec![ChatMessage::user("hi").build()])
            .build()
            .with_perplexity_options(PerplexityOptions::new().with_return_images(true));

        let stream = crate::traits::ChatCapability::chat_stream_request(&client, request)
            .await
            .expect("stream ok");
        let events = stream.collect::<Vec<_>>().await;
        let captured = transport.take_stream().expect("captured stream request");

        assert_eq!(captured.body["return_images"], serde_json::json!(true));

        let end = events
            .iter()
            .find_map(|event| match event {
                Ok(crate::types::ChatStreamEvent::StreamEnd { response }) => Some(response),
                _ => None,
            })
            .expect("stream end event");
        assert_eq!(end.usage.as_ref().map(|usage| usage.total_tokens), Some(28));

        let meta = end.perplexity_metadata().expect("perplexity metadata");
        assert_eq!(
            meta.citations.as_ref(),
            Some(&vec!["https://example.com/rust".to_string()])
        );
        assert_eq!(meta.images.as_ref().map(Vec::len), Some(1));
        assert_eq!(
            meta.usage.as_ref().and_then(|usage| usage.citation_tokens),
            Some(7)
        );
        assert_eq!(
            meta.usage
                .as_ref()
                .and_then(|usage| usage.num_search_queries),
            Some(2)
        );
        assert_eq!(
            meta.usage.as_ref().and_then(|usage| usage.reasoning_tokens),
            Some(3)
        );
        assert_eq!(meta.extra.get("citations"), None);
    }

    #[tokio::test]
    async fn chat_stream_request_runtime_xai_strips_stream_only_fields_at_transport_boundary() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "xai".to_string(),
            name: "xAI".to_string(),
            base_url: "https://api.x.ai/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let transport = CaptureTransport::default();

        let cfg = OpenAiCompatibleConfig::new("xai", "test-key", "https://api.x.ai/v1", adapter)
            .with_model("grok-4")
            .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model("grok-4")
            .messages(vec![ChatMessage::user("hi").build()])
            .stop_sequences(vec!["END".to_string()])
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::None)
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                schema.clone(),
            ))
            .build()
            .with_provider_option(
                "xai",
                serde_json::json!({
                    "response_format": { "type": "json_object" },
                    "tool_choice": "auto",
                    "reasoningEffort": "high"
                }),
            );

        let _ = crate::traits::ChatCapability::chat_stream_request(&client, request).await;
        let captured = transport.take_stream().expect("captured stream request");

        assert!(captured.body.get("stop").is_none());
        assert!(captured.body.get("stream_options").is_none());
        assert_eq!(captured.body["tool_choice"], serde_json::json!("none"));
        assert_eq!(captured.body["reasoning_effort"], serde_json::json!("high"));
        assert_eq!(
            captured.body["response_format"],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            })
        );
    }
    #[tokio::test]
    async fn build_embedding_executor_wires_before_send_and_interceptors() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.test.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
                "embedding".to_string(),
            ],
            default_model: None,
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg =
            OpenAiCompatibleConfig::new("deepseek", "test-key", "https://api.test.com/v1", adapter)
                .with_model("text-embedding-3-small");

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap()
            .with_http_interceptors(vec![Arc::new(NoopInterceptor)]);

        let req = EmbeddingRequest::single("hi")
            .with_model("text-embedding-3-small")
            .with_provider_option("deepseek", serde_json::json!({ "my_custom": 1 }));

        let exec = client.build_embedding_executor(&req);
        assert_eq!(exec.policy.interceptors.len(), 1);
        assert!(exec.policy.before_send.is_none());
        assert!(
            exec.provider_spec
                .embedding_before_send(&req, &exec.provider_context)
                .is_some()
        );
    }

    #[tokio::test]
    async fn build_audio_executor_wires_openai_compatible_audio_spec() {
        let transport = CaptureTransport::default();
        let cfg = OpenAiCompatibleConfig::new(
            "compat-audio",
            "test-key",
            "https://api.test.com/v1",
            make_audio_adapter(),
        )
        .with_model("gpt-audio-mini")
        .with_http_transport(Arc::new(transport));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap()
            .with_http_interceptors(vec![Arc::new(NoopInterceptor)]);

        let exec = client.build_audio_executor();

        assert_eq!(exec.policy.interceptors.len(), 1);
        assert!(exec.policy.transport.is_some());
        assert!(exec.provider_spec.capabilities().supports("audio"));
        assert!(exec.provider_spec.capabilities().supports("speech"));
        assert!(exec.provider_spec.capabilities().supports("transcription"));
        assert_eq!(exec.provider_context.provider_id, "compat-audio");
        assert_eq!(exec.provider_context.base_url, "https://api.test.com/v1");
        assert_eq!(exec.transformer.provider_id(), "compat-audio");
        assert_eq!(exec.transformer.tts_endpoint(), "/audio/speech");
        assert_eq!(exec.transformer.stt_endpoint(), "/audio/transcriptions");
    }

    #[tokio::test]
    async fn builder_installs_provider_specific_params_adapter() {
        let client = crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(
            crate::builder::BuilderBase::default(),
            "deepseek",
        )
        .api_key("test")
        .model("deepseek-reasoner")
        .reasoning(true)
        .build()
        .await
        .expect("builder should succeed");

        let mut chat_body = serde_json::json!({});
        client
            .config
            .adapter
            .transform_request_params(
                &mut chat_body,
                "deepseek-reasoner",
                crate::providers::openai_compatible::types::RequestType::Chat,
            )
            .unwrap();
        assert_eq!(
            chat_body.get("enable_reasoning"),
            Some(&serde_json::Value::Bool(true))
        );

        let mut emb_body = serde_json::json!({});
        client
            .config
            .adapter
            .transform_request_params(
                &mut emb_body,
                "text-embedding-3-small",
                crate::providers::openai_compatible::types::RequestType::Embedding,
            )
            .unwrap();
        assert!(emb_body.get("enable_reasoning").is_none());
    }

    #[tokio::test]
    async fn builder_runtime_openrouter_reasoning_helpers_preserve_final_request_shape() {
        let transport = CaptureTransport::default();
        let client = crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(
            crate::builder::BuilderBase::default(),
            "openrouter",
        )
        .api_key("test")
        .model("openai/gpt-4o")
        .reasoning(true)
        .reasoning_budget(2048)
        .with_http_transport(Arc::new(transport.clone()))
        .build()
        .await
        .expect("builder should succeed");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model("openai/gpt-4o")
            .messages(vec![ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::None)
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                schema.clone(),
            ))
            .build();

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.body["enable_reasoning"], serde_json::json!(true));
        assert_eq!(captured.body["reasoning_budget"], serde_json::json!(2048));
        assert_eq!(captured.body["tool_choice"], serde_json::json!("none"));
        assert_eq!(
            captured.body["response_format"],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            })
        );
    }

    #[tokio::test]
    async fn test_client_creation() {
        let provider_config = crate::standards::openai::compat::provider_registry::ProviderConfig {
            id: "test".to_string(),
            name: "Test Provider".to_string(),
            base_url: "https://api.test.com/v1".to_string(),
            field_mappings:
                crate::standards::openai::compat::provider_registry::ProviderFieldMappings::default(
                ),
            capabilities: vec!["chat".to_string()],
            default_model: Some("test-model".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        };

        let config = OpenAiCompatibleConfig::new(
            "test",
            "test-key",
            "https://api.test.com/v1",
            Arc::new(ConfigurableAdapter::new(provider_config)),
        )
        .with_model("test-model");

        let client = OpenAiCompatibleClient::new(config).await.unwrap();
        assert_eq!(client.provider_id(), "test");
        assert_eq!(client.model(), "test-model");
    }

    #[tokio::test]
    async fn test_client_validation() {
        let provider_config = crate::standards::openai::compat::provider_registry::ProviderConfig {
            id: "test".to_string(),
            name: "Test Provider".to_string(),
            base_url: "https://api.test.com/v1".to_string(),
            field_mappings:
                crate::standards::openai::compat::provider_registry::ProviderFieldMappings::default(
                ),
            capabilities: vec!["chat".to_string()],
            default_model: Some("test-model".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        };

        // Invalid config should fail
        let config = OpenAiCompatibleConfig::new(
            "",
            "test-key",
            "https://api.test.com/v1",
            Arc::new(ConfigurableAdapter::new(provider_config)),
        );

        assert!(OpenAiCompatibleClient::new(config).await.is_err());
    }
}

// Removed legacy rerank parsing; rerank now routed through OpenAI Rerank Standard
