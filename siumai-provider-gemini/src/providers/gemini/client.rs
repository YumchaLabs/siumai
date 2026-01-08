//! Gemini Client Implementation
//!
//! Main client structure that aggregates all Gemini capabilities.

use async_trait::async_trait;
use backoff::ExponentialBackoffBuilder;
use reqwest::Client as HttpClient;
use std::sync::Arc;
use std::time::Duration;

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::streaming::ChatStream;
use crate::traits::*;
use crate::types::*;

use super::chat::GeminiChatCapability;
use super::file_search_stores::GeminiFileSearchStores;
use super::files::GeminiFiles;
use super::models::GeminiModels;
use super::types::{GeminiConfig, GenerationConfig, SafetySetting};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::retry_api::RetryOptions;

// Split capability implementations into submodules (no public API changes)
mod embedding;
mod image;
mod models;

/// Gemini client that implements the `LlmClient` trait
pub struct GeminiClient {
    /// HTTP client for making requests
    pub http_client: HttpClient,
    /// Gemini configuration
    pub config: GeminiConfig,
    /// Common parameters
    pub common_params: CommonParams,
    /// Gemini-specific parameters
    pub gemini_params: crate::params::gemini::GeminiParams,
    /// Chat capability implementation
    pub chat_capability: GeminiChatCapability,
    /// Models capability implementation
    pub models_capability: GeminiModels,
    /// Files capability implementation
    pub files_capability: GeminiFiles,
    /// Tracing configuration
    tracing_config: Option<crate::observability::tracing::TracingConfig>,
    /// Tracing guard to keep tracing system active
    /// NOTE: Tracing subscriber functionality has been moved to siumai-extras
    #[allow(dead_code)]
    _tracing_guard: Option<()>,
    /// Unified retry options for chat
    retry_options: Option<RetryOptions>,
    /// Optional HTTP interceptors applied to all chat requests
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional model-level middlewares for chat
    model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
}

impl Clone for GeminiClient {
    fn clone(&self) -> Self {
        Self {
            http_client: self.http_client.clone(),
            config: self.config.clone(),
            common_params: self.common_params.clone(),
            gemini_params: self.gemini_params.clone(),
            chat_capability: self.chat_capability.clone(),
            models_capability: self.models_capability.clone(),
            files_capability: self.files_capability.clone(),
            tracing_config: self.tracing_config.clone(),
            _tracing_guard: None, // Don't clone the tracing guard
            retry_options: self.retry_options.clone(),
            http_interceptors: self.http_interceptors.clone(),
            model_middlewares: self.model_middlewares.clone(),
        }
    }
}

impl GeminiClient {
    /// Create a new Gemini client with the given configuration
    pub fn new(config: GeminiConfig) -> Result<Self, LlmError> {
        let timeout = Duration::from_secs(config.timeout.unwrap_or(30));

        let http_client = HttpClient::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| {
                LlmError::ConfigurationError(format!("Failed to create HTTP client: {e}"))
            })?;

        Self::with_http_client(config, http_client)
    }

    /// Create a new Gemini client with a custom HTTP client
    pub fn with_http_client(
        config: GeminiConfig,
        http_client: HttpClient,
    ) -> Result<Self, LlmError> {
        // Build capability implementations with provided client
        let chat_capability =
            GeminiChatCapability::new(config.clone(), http_client.clone(), Vec::new());

        let models_capability = GeminiModels::new(config.clone(), http_client.clone());

        let files_capability =
            GeminiFiles::new(config.clone(), http_client.clone(), Vec::new(), None);

        // Use common parameters from config (already contains model, temperature, max_tokens, top_p, stop_sequences)
        let mut common_params = config.common_params.clone();
        // Ensure model is set
        if common_params.model.is_empty() {
            common_params.model = config.model.clone();
        }

        // Create Gemini-specific parameters (simplified - use defaults for now)
        let gemini_params = crate::params::gemini::GeminiParams {
            top_k: config
                .generation_config
                .as_ref()
                .and_then(|gc| gc.top_k)
                .map(|t| t as u32),
            candidate_count: config
                .generation_config
                .as_ref()
                .and_then(|gc| gc.candidate_count)
                .map(|t| t as u32),
            safety_settings: None, // Note: Conversion handled by Transformers layer
            generation_config: None, // Note: Populated by Transformers from Common/Provider params
            stream: None,
        };

        Ok(Self {
            http_client,
            config,
            common_params,
            gemini_params,
            chat_capability,
            models_capability,
            files_capability,
            tracing_config: None,
            _tracing_guard: None,
            retry_options: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
        })
    }

    /// Create a new Gemini client with API key
    pub fn with_api_key(api_key: String) -> Result<Self, LlmError> {
        let config = GeminiConfig::new(api_key);
        Self::new(config)
    }

    /// Get a provider-specific client for File Search Stores (Gemini-only)
    pub fn file_search_stores(&self) -> GeminiFileSearchStores {
        GeminiFileSearchStores::new(
            self.config.clone(),
            self.http_client.clone(),
            self.http_interceptors.clone(),
            self.retry_options.clone(),
        )
    }

    /// Set the model to use
    pub fn with_model(mut self, model: String) -> Self {
        // Update common params
        self.common_params.model = model.clone();

        // Update config
        self.config.model = model;
        self
    }

    /// Set the base URL
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.config.base_url = base_url;
        self
    }

    /// Set generation configuration
    pub fn with_generation_config(mut self, config: GenerationConfig) -> Self {
        self.config.generation_config = Some(config);
        self
    }

    /// Set safety settings
    pub fn with_safety_settings(mut self, settings: Vec<SafetySetting>) -> Self {
        self.config.safety_settings = Some(settings);
        self
    }

    /// Set HTTP timeout
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = Some(timeout.as_secs());
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        // Update common params
        self.common_params.temperature = Some(temperature);

        // Update generation config
        let mut generation_config = self.config.generation_config.unwrap_or_default();
        generation_config.temperature = Some(temperature);
        self.config.generation_config = Some(generation_config);
        self
    }

    /// Set max output tokens
    pub fn with_max_tokens(mut self, max_tokens: i32) -> Self {
        // Update common params
        self.common_params.max_tokens = Some(max_tokens as u32);

        // Update generation config
        let mut generation_config = self.config.generation_config.unwrap_or_default();
        generation_config.max_output_tokens = Some(max_tokens);
        self.config.generation_config = Some(generation_config);
        self
    }

    /// Set top-p
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        // Update common params
        self.common_params.top_p = Some(top_p);

        // Update generation config
        let mut generation_config = self.config.generation_config.unwrap_or_default();
        generation_config.top_p = Some(top_p);
        self.config.generation_config = Some(generation_config);
        self
    }

    /// Set top-k
    pub fn with_top_k(mut self, top_k: i32) -> Self {
        // Update Gemini params
        self.gemini_params.top_k = Some(top_k as u32);

        // Update generation config
        let mut generation_config = self.config.generation_config.unwrap_or_default();
        generation_config.top_k = Some(top_k);
        self.config.generation_config = Some(generation_config);
        self
    }

    /// Set stop sequences
    pub fn with_stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        // Update common params
        self.common_params.stop_sequences = Some(stop_sequences.clone());

        // Update generation config
        let mut generation_config = self.config.generation_config.unwrap_or_default();
        generation_config.stop_sequences = Some(stop_sequences);
        self.config.generation_config = Some(generation_config);
        self
    }

    /// Set candidate count
    pub fn with_candidate_count(mut self, count: i32) -> Self {
        let mut generation_config = self.config.generation_config.unwrap_or_default();
        generation_config.candidate_count = Some(count);
        self.config.generation_config = Some(generation_config);
        self
    }

    /// Enable structured output with JSON schema
    pub fn with_json_schema(mut self, schema: serde_json::Value) -> Self {
        let mut generation_config = self.config.generation_config.unwrap_or_default();
        generation_config.response_mime_type = Some("application/json".to_string());
        generation_config.response_schema = Some(schema);
        self.config.generation_config = Some(generation_config);
        self
    }

    /// Enable enum output with schema
    pub fn with_enum_schema(mut self, enum_values: Vec<String>) -> Self {
        let mut generation_config = self.config.generation_config.unwrap_or_default();
        generation_config.response_mime_type = Some("text/x.enum".to_string());

        // Create enum schema
        let schema = serde_json::json!({
            "type": "STRING",
            "enum": enum_values
        });
        generation_config.response_schema = Some(schema);
        self.config.generation_config = Some(generation_config);
        self
    }

    /// Set custom response MIME type and schema
    pub fn with_response_format(
        mut self,
        mime_type: String,
        schema: Option<serde_json::Value>,
    ) -> Self {
        let mut generation_config = self.config.generation_config.unwrap_or_default();
        generation_config.response_mime_type = Some(mime_type);
        if let Some(schema) = schema {
            generation_config.response_schema = Some(schema);
        }
        self.config.generation_config = Some(generation_config);
        self
    }

    /// Configure thinking behavior with specific budget
    pub fn with_thinking_budget(mut self, budget: i32) -> Self {
        let mut generation_config = self.config.generation_config.unwrap_or_default();
        let thinking_config = super::types::ThinkingConfig {
            thinking_budget: Some(budget),
            include_thoughts: Some(true),
        };
        generation_config.thinking_config = Some(thinking_config);
        self.config.generation_config = Some(generation_config);
        self
    }

    /// Enable dynamic thinking (model decides budget)
    pub fn with_dynamic_thinking(mut self) -> Self {
        let mut generation_config = self.config.generation_config.unwrap_or_default();
        let thinking_config = super::types::ThinkingConfig::dynamic();
        generation_config.thinking_config = Some(thinking_config);
        self.config.generation_config = Some(generation_config);
        self
    }

    /// Disable thinking functionality
    pub fn with_thinking_disabled(mut self) -> Self {
        let mut generation_config = self.config.generation_config.unwrap_or_default();
        let thinking_config = super::types::ThinkingConfig::disabled();
        generation_config.thinking_config = Some(thinking_config);
        self.config.generation_config = Some(generation_config);
        self
    }

    /// Configure thinking with custom settings
    pub fn with_thinking_config(mut self, config: super::types::ThinkingConfig) -> Self {
        let mut generation_config = self.config.generation_config.unwrap_or_default();
        generation_config.thinking_config = Some(config);
        self.config.generation_config = Some(generation_config);
        self
    }

    /// Set response format (alias for with_response_format for OpenAI compatibility)
    pub fn with_response_format_compat(self, format: serde_json::Value) -> Self {
        // For Gemini, we need to extract MIME type and schema from the format
        if let Some(mime_type) = format.get("type").and_then(|t| t.as_str()) {
            let gemini_mime_type = match mime_type {
                "json_object" => "application/json",
                "text" => "text/plain",
                _ => mime_type,
            };

            let schema = format
                .get("json_schema")
                .and_then(|s| s.get("schema"))
                .cloned();

            self.with_response_format(gemini_mime_type.to_string(), schema)
        } else {
            self
        }
    }

    /// Enable image generation capability
    pub fn with_image_generation(mut self) -> Self {
        let mut generation_config = self.config.generation_config.unwrap_or_default();
        generation_config.response_modalities = Some(vec!["TEXT".to_string(), "IMAGE".to_string()]);
        self.config.generation_config = Some(generation_config);
        self
    }

    /// Set custom response modalities
    pub fn with_response_modalities(mut self, modalities: Vec<String>) -> Self {
        let mut generation_config = self.config.generation_config.unwrap_or_default();
        generation_config.response_modalities = Some(modalities);
        self.config.generation_config = Some(generation_config);
        self
    }

    /// Get the API key
    pub fn api_key(&self) -> &str {
        use secrecy::ExposeSecret;
        self.config.api_key.expose_secret()
    }

    /// Get the base URL
    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    /// Control whether to disable compression for streaming (SSE) requests.
    pub fn http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.config.http_config.stream_disable_compression = disable;
        self
    }

    /// Get the model
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// Get the generation configuration
    pub const fn generation_config(&self) -> Option<&GenerationConfig> {
        self.config.generation_config.as_ref()
    }

    /// Get the safety settings
    pub const fn safety_settings(&self) -> Option<&Vec<SafetySetting>> {
        self.config.safety_settings.as_ref()
    }

    /// Get the configuration (for testing and debugging)
    pub const fn config(&self) -> &GeminiConfig {
        &self.config
    }

    /// Get chat capability (for testing and debugging)
    pub const fn chat_capability(&self) -> &GeminiChatCapability {
        &self.chat_capability
    }

    /// Get common parameters
    pub fn common_params(&self) -> &CommonParams {
        &self.common_params
    }

    /// Get Gemini-specific parameters
    pub fn gemini_params(&self) -> &crate::params::gemini::GeminiParams {
        &self.gemini_params
    }

    /// Get mutable common parameters
    pub fn common_params_mut(&mut self) -> &mut CommonParams {
        &mut self.common_params
    }

    /// Get mutable Gemini-specific parameters
    pub fn gemini_params_mut(&mut self) -> &mut crate::params::gemini::GeminiParams {
        &mut self.gemini_params
    }

    /// Set the tracing configuration
    #[doc(hidden)]
    pub fn set_tracing_config(
        &mut self,
        config: Option<crate::observability::tracing::TracingConfig>,
    ) {
        self.tracing_config = config;
    }

    /// Set unified retry options
    pub fn set_retry_options(&mut self, options: Option<RetryOptions>) {
        self.retry_options = options.map(|mut opts| {
            if matches!(opts.backend, crate::retry_api::RetryBackend::Backoff)
                && opts.backoff_executor.is_none()
            {
                opts.backoff_executor = Some(gemini_backoff_executor());
            }
            opts
        });
        // Rebuild files capability with updated retry options
        self.files_capability = GeminiFiles::new(
            self.config.clone(),
            self.http_client.clone(),
            self.http_interceptors.clone(),
            self.retry_options.clone(),
        );
    }

    /// Install HTTP interceptors for all chat requests.
    pub fn with_http_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors.clone();
        // Rebuild chat capability with interceptors and current middlewares
        let mws = self.model_middlewares.clone();
        self.chat_capability =
            GeminiChatCapability::new(self.config.clone(), self.http_client.clone(), interceptors)
                .with_middlewares(mws);
        // Rebuild files capability to apply interceptors
        self.files_capability = GeminiFiles::new(
            self.config.clone(),
            self.http_client.clone(),
            self.http_interceptors.clone(),
            self.retry_options.clone(),
        );
        self
    }

    /// Install model-level middlewares for chat requests.
    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.model_middlewares = middlewares.clone();
        // Rebuild chat capability preserving interceptors
        let interceptors = self.http_interceptors.clone();
        self.chat_capability =
            GeminiChatCapability::new(self.config.clone(), self.http_client.clone(), interceptors)
                .with_middlewares(middlewares);
        self
    }
}

impl GeminiClient {
    async fn chat_with_tools_inner(
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

    /// Create provider context for this client
    async fn build_context(&self) -> crate::core::ProviderContext {
        super::context::build_context(&self.config).await
    }

    /// Create chat executor using the builder pattern
    async fn build_chat_executor(
        &self,
        request: &ChatRequest,
    ) -> Arc<crate::execution::executors::chat::HttpChatExecutor> {
        use crate::core::ProviderSpec;
        use crate::execution::executors::chat::ChatExecutorBuilder;

        let ctx = self.build_context().await;
        let spec = Arc::new(crate::providers::gemini::spec::GeminiSpecWithConfig::new(
            self.config.clone(),
        ));
        let bundle = spec.choose_chat_transformers(request, &ctx);
        let before_send_hook = spec.chat_before_send(request, &ctx);

        let mut builder = ChatExecutorBuilder::new("gemini", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_transformer_bundle(bundle)
            .with_stream_disable_compression(self.config.http_config.stream_disable_compression)
            .with_interceptors(self.http_interceptors.clone())
            .with_middlewares(self.model_middlewares.clone());

        if let Some(hook) = before_send_hook {
            builder = builder.with_before_send(hook);
        }

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        builder.build()
    }

    /// Execute chat request via spec (unified implementation)
    async fn chat_request_via_spec(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        use crate::execution::executors::chat::ChatExecutor;

        let exec = self.build_chat_executor(&request).await;
        ChatExecutor::execute(&*exec, request).await
    }

    /// Execute streaming chat request via spec (unified implementation)
    async fn chat_stream_request_via_spec(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStream, LlmError> {
        use crate::execution::executors::chat::ChatExecutor;

        let exec = self.build_chat_executor(&request).await;
        ChatExecutor::execute_stream(&*exec, request).await
    }
}

#[async_trait]
impl ChatCapability for GeminiClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        self.chat_with_tools_inner(messages, tools).await
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

// NOTE: Embedding/Image implementations live under `client/embedding.rs` and `client/image.rs`.

#[async_trait]
impl FileManagementCapability for GeminiClient {
    async fn upload_file(&self, request: FileUploadRequest) -> Result<FileObject, LlmError> {
        self.files_capability.upload_file(request).await
    }

    async fn list_files(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError> {
        self.files_capability.list_files(query).await
    }

    async fn retrieve_file(&self, file_id: String) -> Result<FileObject, LlmError> {
        self.files_capability.retrieve_file(file_id).await
    }

    async fn delete_file(&self, file_id: String) -> Result<FileDeleteResponse, LlmError> {
        self.files_capability.delete_file(file_id).await
    }

    async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        self.files_capability.get_file_content(file_id).await
    }
}

impl LlmClient for GeminiClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("gemini")
    }

    fn supported_models(&self) -> Vec<String> {
        vec![
            "gemini-1.5-flash".to_string(),
            "gemini-1.5-flash-8b".to_string(),
            "gemini-1.5-pro".to_string(),
            "gemini-2.0-flash-exp".to_string(),
            "gemini-exp-1114".to_string(),
            "gemini-exp-1121".to_string(),
            "gemini-exp-1206".to_string(),
        ]
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_embedding()
            .with_file_management()
            .with_custom_feature("code_execution", true)
            .with_custom_feature("thinking_mode", true)
            .with_custom_feature("safety_settings", true)
            .with_custom_feature("cached_content", true)
            .with_custom_feature("json_schema", true)
            .with_image_generation()
            .with_custom_feature("enum_output", true)
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

    fn as_model_listing_capability(&self) -> Option<&dyn crate::traits::ModelListingCapability> {
        Some(self)
    }
}

pub(super) fn gemini_backoff_executor() -> crate::retry_api::BackoffRetryExecutor {
    let backoff = ExponentialBackoffBuilder::new()
        .with_initial_interval(Duration::from_millis(1000))
        .with_max_interval(Duration::from_secs(60))
        .with_multiplier(1.5)
        .with_max_elapsed_time(Some(Duration::from_secs(300)))
        .build();
    crate::retry_api::BackoffRetryExecutor::with_backoff(backoff)
}
