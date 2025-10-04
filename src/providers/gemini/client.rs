//! Gemini Client Implementation
//!
//! Main client structure that aggregates all Gemini capabilities.

use async_trait::async_trait;
use reqwest::Client as HttpClient;
use std::time::Duration;

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::stream::ChatStream;
use crate::traits::*;
use crate::types::*;

use super::chat::GeminiChatCapability;
use super::embeddings::GeminiEmbeddings;
use super::files::GeminiFiles;
use super::models::GeminiModels;
use super::types::{GeminiConfig, GenerationConfig, SafetySetting};

/// Gemini client that implements the `LlmClient` trait
#[derive(Debug)]
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
    /// Embedding capability implementation
    pub embedding_capability: GeminiEmbeddings,
    /// Models capability implementation
    pub models_capability: GeminiModels,
    /// Files capability implementation
    pub files_capability: GeminiFiles,
    /// Tracing configuration
    tracing_config: Option<crate::tracing::TracingConfig>,
    /// Tracing guard to keep tracing system active
    _tracing_guard: Option<Option<tracing_appender::non_blocking::WorkerGuard>>,
}

impl Clone for GeminiClient {
    fn clone(&self) -> Self {
        Self {
            http_client: self.http_client.clone(),
            config: self.config.clone(),
            common_params: self.common_params.clone(),
            gemini_params: self.gemini_params.clone(),
            chat_capability: self.chat_capability.clone(),
            embedding_capability: self.embedding_capability.clone(),
            models_capability: self.models_capability.clone(),
            files_capability: self.files_capability.clone(),
            tracing_config: self.tracing_config.clone(),
            _tracing_guard: None, // Don't clone the tracing guard
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
        let chat_capability = GeminiChatCapability::new(config.clone(), http_client.clone());

        let embedding_capability = GeminiEmbeddings::new(config.clone(), http_client.clone());

        let models_capability = GeminiModels::new(config.clone(), http_client.clone());

        let files_capability = GeminiFiles::new(config.clone(), http_client.clone());

        // Extract common parameters from config
        let common_params = CommonParams {
            model: config.model.clone(),
            temperature: config
                .generation_config
                .as_ref()
                .and_then(|gc| gc.temperature),
            max_tokens: config
                .generation_config
                .as_ref()
                .and_then(|gc| gc.max_output_tokens)
                .map(|t| t as u32),
            top_p: config.generation_config.as_ref().and_then(|gc| gc.top_p),
            stop_sequences: config
                .generation_config
                .as_ref()
                .and_then(|gc| gc.stop_sequences.clone()),
            seed: None, // Gemini doesn't support seed
        };

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
            safety_settings: None, // TODO: Convert from provider types to param types
            generation_config: None, // TODO: Convert from provider types to param types
            stream: None,
        };

        Ok(Self {
            http_client,
            config,
            common_params,
            gemini_params,
            chat_capability,
            embedding_capability,
            models_capability,
            files_capability,
            tracing_config: None,
            _tracing_guard: None,
        })
    }

    /// Create a new Gemini client with API key
    pub fn with_api_key(api_key: String) -> Result<Self, LlmError> {
        let config = GeminiConfig::new(api_key);
        Self::new(config)
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
    pub fn with_temperature(mut self, temperature: f32) -> Self {
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
    pub fn with_top_p(mut self, top_p: f32) -> Self {
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
        &self.config.api_key
    }

    /// Get the base URL
    pub fn base_url(&self) -> &str {
        &self.config.base_url
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

    /// Set the tracing guard to keep tracing system active
    pub(crate) fn set_tracing_guard(
        &mut self,
        guard: Option<Option<tracing_appender::non_blocking::WorkerGuard>>,
    ) {
        self._tracing_guard = guard;
    }

    /// Set the tracing configuration
    pub(crate) fn set_tracing_config(&mut self, config: Option<crate::tracing::TracingConfig>) {
        self.tracing_config = config;
    }
}

#[async_trait]
impl ChatCapability for GeminiClient {
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
impl EmbeddingCapability for GeminiClient {
    async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        self.embedding_capability.embed(texts).await
    }

    fn embedding_dimension(&self) -> usize {
        self.embedding_capability.embedding_dimension()
    }

    fn max_tokens_per_embedding(&self) -> usize {
        self.embedding_capability.max_tokens_per_embedding()
    }

    fn supported_embedding_models(&self) -> Vec<String> {
        self.embedding_capability.supported_embedding_models()
    }
}

#[async_trait]
impl ModelListingCapability for GeminiClient {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.models_capability.list_models().await
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        self.models_capability.get_model(model_id).await
    }
}

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
    fn provider_name(&self) -> &'static str {
        "gemini"
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
            .with_custom_feature("image_generation", true)
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
}
