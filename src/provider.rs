//! Siumai LLM Interface
//!
//! This module provides the main siumai interface for calling different provider functionality,
//! similar to `llm_dart`'s approach. It uses dynamic dispatch to route calls to the
//! appropriate provider implementation.

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::stream::ChatStream;
use crate::traits::*;
use crate::types::*;
use std::any::Any;
use std::collections::HashMap;
use std::time::Duration;

/// The main siumai LLM provider that can dynamically dispatch to different capabilities
///
/// This is inspired by `llm_dart`'s unified interface design, allowing you to
/// call different provider functionality through a single interface.
pub struct Siumai {
    /// The underlying provider client
    client: Box<dyn LlmClient>,
    /// Capability registry for dynamic dispatch
    #[allow(dead_code)]
    capabilities: HashMap<String, Box<dyn Any + Send + Sync>>,
    /// Provider-specific metadata
    metadata: ProviderMetadata,
}

impl Clone for Siumai {
    fn clone(&self) -> Self {
        // Clone the client using the ClientWrapper approach
        let client = self.client.clone_box();

        Self {
            client,
            capabilities: HashMap::new(), // Don't clone capabilities as they're not used
            metadata: self.metadata.clone(),
        }
    }
}

impl std::fmt::Debug for Siumai {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Siumai")
            .field("provider_type", &self.metadata.provider_type)
            .field("provider_name", &self.metadata.provider_name)
            .field(
                "supported_models_count",
                &self.metadata.supported_models.len(),
            )
            .field("capabilities", &self.metadata.capabilities)
            .finish()
    }
}

/// Metadata about the provider
#[derive(Debug, Clone)]
pub struct ProviderMetadata {
    pub provider_type: ProviderType,
    pub provider_name: String,
    pub supported_models: Vec<String>,
    pub capabilities: ProviderCapabilities,
}

impl Siumai {
    /// Create a new siumai provider
    pub fn new(client: Box<dyn LlmClient>) -> Self {
        let metadata = ProviderMetadata {
            provider_type: match client.provider_name() {
                "openai" => ProviderType::OpenAi,
                "anthropic" => ProviderType::Anthropic,
                "gemini" => ProviderType::Gemini,
                "ollama" => ProviderType::Ollama,
                "xai" => ProviderType::XAI,
                "groq" => ProviderType::Groq,
                name => ProviderType::Custom(name.to_string()),
            },
            provider_name: client.provider_name().to_string(),
            supported_models: client.supported_models(),
            capabilities: client.capabilities(),
        };

        Self {
            client,
            capabilities: HashMap::new(),
            metadata,
        }
    }

    /// Check if a capability is supported
    pub fn supports(&self, capability: &str) -> bool {
        self.metadata.capabilities.supports(capability)
    }

    /// Get provider metadata
    pub const fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }

    /// Get the underlying client
    pub fn client(&self) -> &dyn LlmClient {
        self.client.as_ref()
    }

    /// Type-safe audio capability access
    ///
    /// Note: This method provides access regardless of reported capability support.
    /// Actual support depends on the specific model being used.
    pub fn audio_capability(&self) -> AudioCapabilityProxy<'_> {
        AudioCapabilityProxy::new(self, self.supports("audio"))
    }

    /// Type-safe embedding capability access
    ///
    /// Note: This method provides access regardless of reported capability support.
    /// Actual support depends on the specific model being used.
    pub fn embedding_capability(&self) -> EmbeddingCapabilityProxy<'_> {
        EmbeddingCapabilityProxy::new(self, self.supports("embedding"))
    }

    /// Type-safe vision capability access
    ///
    /// Note: This method provides access regardless of reported capability support.
    /// Actual support depends on the specific model being used.
    pub fn vision_capability(&self) -> VisionCapabilityProxy<'_> {
        VisionCapabilityProxy::new(self, self.supports("vision"))
    }

    /// Generate embeddings for the given input texts
    ///
    /// This is a convenience method that directly calls the embedding functionality
    /// without requiring the user to go through the capability proxy.
    ///
    /// # Arguments
    /// * `texts` - List of strings to generate embeddings for
    ///
    /// # Returns
    /// List of embedding vectors (one per input text)
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::prelude::*;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Siumai::builder()
    ///     .openai()
    ///     .api_key("your-api-key")
    ///     .build()
    ///     .await?;
    ///
    /// let texts = vec!["Hello, world!".to_string()];
    /// let response = client.embed(texts).await?;
    /// println!("Got {} embeddings", response.embeddings.len());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        EmbeddingCapability::embed(self, texts).await
    }
}

#[async_trait::async_trait]
impl ChatCapability for Siumai {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        self.client.chat_with_tools(messages, tools).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        self.client.chat_stream(messages, tools).await
    }
}

#[async_trait::async_trait]
impl EmbeddingCapability for Siumai {
    async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        // Use the new capability method instead of downcasting
        if let Some(embedding_client) = self.client.as_embedding_capability() {
            embedding_client.embed(texts).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support embedding functionality. Consider using OpenAI, Gemini, or Ollama for embeddings.",
                self.client.provider_name()
            )))
        }
    }

    fn embedding_dimension(&self) -> usize {
        // Use the new capability method to get dimension
        if let Some(embedding_client) = self.client.as_embedding_capability() {
            embedding_client.embedding_dimension()
        } else {
            // Fallback to default dimension based on provider
            match self.client.provider_name() {
                "openai" => 1536,
                "ollama" => 384,
                "gemini" => 768,
                _ => 1536,
            }
        }
    }

    fn max_tokens_per_embedding(&self) -> usize {
        // Use the new capability method to get max tokens
        if let Some(embedding_client) = self.client.as_embedding_capability() {
            embedding_client.max_tokens_per_embedding()
        } else {
            // Fallback to default based on provider
            match self.client.provider_name() {
                "openai" => 8192,
                "ollama" => 8192,
                "gemini" => 2048,
                _ => 8192,
            }
        }
    }

    fn supported_embedding_models(&self) -> Vec<String> {
        // Use the new capability method to get supported models
        if let Some(embedding_client) = self.client.as_embedding_capability() {
            embedding_client.supported_embedding_models()
        } else {
            // Fallback to default models based on provider
            match self.client.provider_name() {
                "openai" => vec![
                    "text-embedding-3-small".to_string(),
                    "text-embedding-3-large".to_string(),
                    "text-embedding-ada-002".to_string(),
                ],
                "ollama" => vec![
                    "nomic-embed-text".to_string(),
                    "mxbai-embed-large".to_string(),
                ],
                "gemini" => vec![
                    "embedding-001".to_string(),
                    "text-embedding-004".to_string(),
                ],
                _ => vec![],
            }
        }
    }
}

#[async_trait::async_trait]
impl EmbeddingExtensions for Siumai {
    /// Generate embeddings with advanced configuration including task types.
    ///
    /// This method enables the unified interface to support provider-specific features
    /// like Gemini's task type optimization, OpenAI's custom dimensions, etc.
    ///
    /// # Arguments
    /// * `request` - Detailed embedding request with configuration
    ///
    /// # Returns
    /// Embedding response with vectors and metadata
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::prelude::*;
    /// use siumai::types::{EmbeddingRequest, EmbeddingTaskType};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = Siumai::builder()
    ///         .gemini()
    ///         .api_key("your-api-key")
    ///         .model("gemini-embedding-001")
    ///         .build()
    ///         .await?;
    ///
    ///     // Use task type for optimization
    ///     let request = EmbeddingRequest::query("What is machine learning?");
    ///     let response = client.embed_with_config(request).await?;
    ///     Ok(())
    /// }
    /// ```
    async fn embed_with_config(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        // For now, we'll provide a simplified implementation that works with the current architecture
        // The advanced features like task types will be supported through provider-specific interfaces

        if let Some(embedding_client) = self.client.as_embedding_capability() {
            // Use the basic embed method - the underlying implementations
            // can be enhanced to support more features in the future
            embedding_client.embed(request.input).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support embedding functionality. Consider using OpenAI, Gemini, or Ollama for embeddings.",
                self.client.provider_name()
            )))
        }
    }

    /// List available embedding models with their capabilities.
    ///
    /// # Returns
    /// Vector of embedding model information including supported task types
    async fn list_embedding_models(&self) -> Result<Vec<EmbeddingModelInfo>, LlmError> {
        // For now, provide basic model information based on the provider
        if let Some(_embedding_client) = self.client.as_embedding_capability() {
            let models = self.supported_embedding_models();
            let model_infos = models
                .into_iter()
                .map(|id| {
                    let mut model_info = EmbeddingModelInfo::new(
                        id.clone(),
                        id,
                        self.embedding_dimension(),
                        self.max_tokens_per_embedding(),
                    );

                    // Add task type support for Gemini models
                    if self.client.provider_name() == "gemini" {
                        model_info = model_info
                            .with_task(EmbeddingTaskType::RetrievalQuery)
                            .with_task(EmbeddingTaskType::RetrievalDocument)
                            .with_task(EmbeddingTaskType::SemanticSimilarity)
                            .with_task(EmbeddingTaskType::Classification)
                            .with_task(EmbeddingTaskType::Clustering)
                            .with_task(EmbeddingTaskType::QuestionAnswering)
                            .with_task(EmbeddingTaskType::FactVerification);
                    }

                    // Add custom dimensions support for OpenAI models
                    if self.client.provider_name() == "openai" {
                        model_info = model_info.with_custom_dimensions();
                    }

                    model_info
                })
                .collect();
            Ok(model_infos)
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support embedding functionality.",
                self.client.provider_name()
            )))
        }
    }
}

impl LlmClient for Siumai {
    fn provider_name(&self) -> &'static str {
        // We need to return a static str, so we'll use a match
        match self.metadata.provider_type {
            ProviderType::OpenAi => "openai",
            ProviderType::Anthropic => "anthropic",
            ProviderType::Gemini => "gemini",
            ProviderType::XAI => "xai",
            ProviderType::Ollama => "ollama",
            ProviderType::Custom(_) => "custom",
            ProviderType::Groq => "groq",
        }
    }

    fn supported_models(&self) -> Vec<String> {
        self.metadata.supported_models.clone()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.metadata.capabilities.clone()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }
}

/// Unified Interface Builder - Provider Abstraction Layer
///
/// ## 🎯 Core Responsibility: Unified Provider Interface
///
/// SiumaiBuilder provides a **unified interface** for creating LLM clients
/// across different providers while abstracting away provider-specific details.
///
/// ### ✅ What SiumaiBuilder Does:
/// - **Provider Abstraction**: Unified interface for all LLM providers
/// - **Parameter Unification**: Common parameter interface (temperature, max_tokens, etc.)
/// - **Reasoning Abstraction**: Unified reasoning interface across providers
/// - **Configuration Validation**: Validates configuration before client creation
/// - **Provider Selection**: Determines which provider to use based on configuration
/// - **Parameter Delegation**: Delegates to appropriate builders for actual construction
///
/// ### ❌ What SiumaiBuilder Does NOT Do:
/// - **Direct Client Creation**: Does not directly create HTTP clients
/// - **Parameter Mapping**: Does not handle provider-specific parameter mapping
/// - **HTTP Configuration**: Does not configure HTTP settings directly
///
/// ## 🏗️ Architecture Position
///
/// ```text
/// User Code
///     ↓
/// SiumaiBuilder (Unified Interface Layer) ← YOU ARE HERE
///     ↓
/// ┌─────────────────┬─────────────────────────────────────┐
/// ↓                 ↓                                     ↓
/// LlmBuilder        RequestBuilder                Provider Clients
/// (Client Config)   (Parameter Management)        (Implementation)
/// ```
///
/// ## 🔄 Delegation Pattern
///
/// SiumaiBuilder acts as a **coordinator** that delegates to specialized builders:
///
/// 1. **Parameter Validation**: Uses RequestBuilder for parameter validation
/// 2. **Client Construction**: Uses LlmBuilder or direct client constructors
/// 3. **Provider Selection**: Chooses appropriate implementation based on provider type
///
/// ### Example Flow:
/// ```rust,no_run
/// use siumai::prelude::*;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // 1. User configures through unified interface
///     let client = Siumai::builder()
///         .anthropic()                    // Provider selection
///         .api_key("your-api-key")        // Required API key
///         .model("claude-3-5-sonnet")     // Common parameter
///         .temperature(0.7)               // Common parameter
///         .reasoning(true)                // Unified reasoning
///         .build().await?;                // Delegation to appropriate builders
///     Ok(())
/// }
/// ```
///
/// This design allows users to switch providers with minimal code changes
/// while maintaining access to provider-specific features when needed.
pub struct SiumaiBuilder {
    pub(crate) provider_type: Option<ProviderType>,
    pub(crate) provider_name: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    capabilities: Vec<String>,
    common_params: CommonParams,
    http_config: HttpConfig,
    organization: Option<String>,
    project: Option<String>,
    tracing_config: Option<crate::tracing::TracingConfig>,
    // Unified reasoning configuration
    reasoning_enabled: Option<bool>,
    reasoning_budget: Option<i32>,
}

impl SiumaiBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            provider_type: None,
            provider_name: None,
            api_key: None,
            base_url: None,
            capabilities: Vec::new(),
            common_params: CommonParams::default(),
            http_config: HttpConfig::default(),
            organization: None,
            project: None,
            tracing_config: None,
            reasoning_enabled: None,
            reasoning_budget: None,
        }
    }

    /// Set the provider type
    pub fn provider(mut self, provider_type: ProviderType) -> Self {
        self.provider_type = Some(provider_type);
        self
    }

    /// Set the provider by name (dynamic dispatch)
    /// This provides the llm_dart-style ai().provider('name') interface
    pub fn provider_name<S: Into<String>>(mut self, name: S) -> Self {
        let name = name.into();
        self.provider_name = Some(name.clone());

        // Map provider name to type
        self.provider_type = Some(match name.as_str() {
            "openai" => ProviderType::OpenAi,
            "anthropic" => ProviderType::Anthropic,
            "gemini" => ProviderType::Gemini,
            "ollama" => ProviderType::Ollama,
            "xai" => ProviderType::XAI,
            "groq" => ProviderType::Groq,
            "siliconflow" => ProviderType::Custom("siliconflow".to_string()),
            "deepseek" => ProviderType::Custom("deepseek".to_string()),
            "openrouter" => ProviderType::Custom("openrouter".to_string()),
            _ => ProviderType::Custom(name),
        });
        self
    }

    // Provider convenience methods are now defined in src/provider_builders.rs

    /// Set the API key
    pub fn api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the base URL
    pub fn base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Set the model
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.common_params.model = model.into();
        self
    }

    /// Set temperature
    pub const fn temperature(mut self, temperature: f32) -> Self {
        self.common_params.temperature = Some(temperature);
        self
    }

    /// Set max tokens
    pub const fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.common_params.max_tokens = Some(max_tokens);
        self
    }

    /// Set top_p (nucleus sampling parameter)
    pub const fn top_p(mut self, top_p: f32) -> Self {
        self.common_params.top_p = Some(top_p);
        self
    }

    /// Set random seed for reproducible outputs
    pub const fn seed(mut self, seed: u64) -> Self {
        self.common_params.seed = Some(seed);
        self
    }

    /// Set stop sequences
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.common_params.stop_sequences = Some(sequences);
        self
    }

    /// Enable or disable reasoning mode (unified interface)
    ///
    /// This method provides a unified interface for enabling reasoning across all providers.
    /// It maps to provider-specific methods:
    /// - Anthropic: `thinking_budget` (10k tokens when enabled)
    /// - Gemini: `thinking` (dynamic when enabled)
    /// - Ollama: `reasoning` (enabled/disabled)
    /// - DeepSeek: `reasoning` (enabled/disabled)
    pub const fn reasoning(mut self, enabled: bool) -> Self {
        self.reasoning_enabled = Some(enabled);
        self
    }

    /// Set reasoning budget (unified interface)
    ///
    /// This method provides a unified interface for setting reasoning budgets.
    /// Different providers interpret this differently:
    /// - Anthropic: Direct token budget
    /// - Gemini: Token budget (-1 for dynamic, 0 for disabled)
    /// - Ollama: Ignored (uses boolean reasoning mode)
    /// - DeepSeek: Ignored (uses boolean reasoning mode)
    pub const fn reasoning_budget(mut self, budget: i32) -> Self {
        self.reasoning_budget = Some(budget);
        // If budget is set, automatically enable reasoning
        if budget > 0 {
            self.reasoning_enabled = Some(true);
        } else if budget == 0 {
            self.reasoning_enabled = Some(false);
        }
        self
    }

    /// Set organization (for `OpenAI`)
    pub fn organization<S: Into<String>>(mut self, organization: S) -> Self {
        self.organization = Some(organization.into());
        self
    }

    /// Set project (for `OpenAI`)
    pub fn project<S: Into<String>>(mut self, project: S) -> Self {
        self.project = Some(project.into());
        self
    }

    /// Enable a specific capability
    pub fn with_capability<S: Into<String>>(mut self, capability: S) -> Self {
        self.capabilities.push(capability.into());
        self
    }

    /// Enable audio capability
    pub fn with_audio(self) -> Self {
        self.with_capability("audio")
    }

    /// Enable vision capability
    pub fn with_vision(self) -> Self {
        self.with_capability("vision")
    }

    /// Enable embedding capability
    pub fn with_embedding(self) -> Self {
        self.with_capability("embedding")
    }

    /// Enable image generation capability
    pub fn with_image_generation(self) -> Self {
        self.with_capability("image_generation")
    }

    // === HTTP configuration (fine-grained) ===

    /// Set HTTP request timeout
    pub fn http_timeout(mut self, timeout: Duration) -> Self {
        self.http_config.timeout = Some(timeout);
        self
    }

    /// Set HTTP connect timeout
    pub fn http_connect_timeout(mut self, timeout: Duration) -> Self {
        self.http_config.connect_timeout = Some(timeout);
        self
    }

    /// Set HTTP user agent
    pub fn http_user_agent<S: Into<String>>(mut self, user_agent: S) -> Self {
        self.http_config.user_agent = Some(user_agent.into());
        self
    }

    /// Set HTTP proxy URL
    pub fn http_proxy<S: Into<String>>(mut self, proxy_url: S) -> Self {
        self.http_config.proxy = Some(proxy_url.into());
        self
    }

    /// Add a default HTTP header
    pub fn http_header<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.http_config.headers.insert(key.into(), value.into());
        self
    }

    /// Merge multiple default HTTP headers
    pub fn http_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.http_config.headers.extend(headers);
        self
    }

    // === Tracing Configuration ===

    /// Set custom tracing configuration
    ///
    /// This allows you to configure detailed tracing and monitoring for this client.
    /// The tracing configuration will override any global tracing settings.
    ///
    /// # Arguments
    /// * `config` - The tracing configuration to use
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::prelude::*;
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Siumai::builder()
    ///     .openai()
    ///     .api_key("your-key")
    ///     .model("gpt-4o-mini")
    ///     .tracing(TracingConfig::debug())
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn tracing(mut self, config: crate::tracing::TracingConfig) -> Self {
        self.tracing_config = Some(config);
        self
    }

    /// Enable debug tracing (development-friendly configuration)
    pub fn debug_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::development())
    }

    /// Enable minimal tracing (info level, LLM only)
    pub fn minimal_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::minimal())
    }

    /// Enable production-ready JSON tracing
    pub fn json_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::json_production())
    }

    /// Enable simple tracing (uses debug configuration)
    pub fn enable_tracing(self) -> Self {
        self.debug_tracing()
    }

    /// Disable tracing explicitly
    pub fn disable_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::disabled())
    }

    /// Build the siumai provider
    pub async fn build(self) -> Result<Siumai, LlmError> {
        // Helper: build an HTTP client from HttpConfig
        fn build_http_client_from_config(cfg: &HttpConfig) -> Result<reqwest::Client, LlmError> {
            let mut builder = reqwest::Client::builder();

            if let Some(timeout) = cfg.timeout {
                builder = builder.timeout(timeout);
            }
            if let Some(connect_timeout) = cfg.connect_timeout {
                builder = builder.connect_timeout(connect_timeout);
            }
            if let Some(proxy_url) = &cfg.proxy {
                let proxy = reqwest::Proxy::all(proxy_url)
                    .map_err(|e| LlmError::ConfigurationError(format!("Invalid proxy URL: {e}")))?;
                builder = builder.proxy(proxy);
            }
            if let Some(user_agent) = &cfg.user_agent {
                builder = builder.user_agent(user_agent);
            }

            // Default headers
            if !cfg.headers.is_empty() {
                let mut headers = reqwest::header::HeaderMap::new();
                for (k, v) in &cfg.headers {
                    let name =
                        reqwest::header::HeaderName::from_bytes(k.as_bytes()).map_err(|e| {
                            LlmError::ConfigurationError(format!("Invalid header name '{k}': {e}"))
                        })?;
                    let value = reqwest::header::HeaderValue::from_str(v).map_err(|e| {
                        LlmError::ConfigurationError(format!("Invalid header value for '{k}': {e}"))
                    })?;
                    headers.insert(name, value);
                }
                builder = builder.default_headers(headers);
            }

            builder.build().map_err(|e| {
                LlmError::ConfigurationError(format!("Failed to build HTTP client: {e}"))
            })
        }

        // Extract all needed values first to avoid borrow checker issues
        let provider_type = self.provider_type.clone().ok_or_else(|| {
            LlmError::ConfigurationError("Provider type not specified".to_string())
        })?;

        // Check if API key is required for this provider type
        let requires_api_key = match provider_type {
            ProviderType::Ollama => false, // Ollama doesn't require API key
            _ => true,                     // All other providers require API key
        };

        let api_key = if requires_api_key {
            self.api_key
                .clone()
                .ok_or_else(|| LlmError::ConfigurationError("API key not specified".to_string()))?
        } else {
            // For providers that don't require API key, use empty string or None
            self.api_key.clone().unwrap_or_default()
        };

        // Extract all needed values to avoid borrow checker issues
        let base_url = self.base_url.clone();
        let organization = self.organization.clone();
        let project = self.project.clone();
        let reasoning_enabled = self.reasoning_enabled;
        let reasoning_budget = self.reasoning_budget;
        let http_config = self.http_config.clone();
        // Build one HTTP client for this builder, reuse across providers when possible
        let built_http_client = build_http_client_from_config(&http_config)?;

        // Prepare common parameters with the correct model
        let mut common_params = self.common_params.clone();

        // Set default model if none provided
        if common_params.model.is_empty() {
            // Set default model based on provider type
            #[cfg(any(feature = "openai", feature = "anthropic", feature = "google"))]
            use crate::types::models::model_constants as models;

            common_params.model = match provider_type {
                #[cfg(feature = "openai")]
                ProviderType::OpenAi => models::openai::GPT_4O.to_string(),
                #[cfg(feature = "anthropic")]
                ProviderType::Anthropic => models::anthropic::CLAUDE_SONNET_3_5.to_string(),
                #[cfg(feature = "google")]
                ProviderType::Gemini => models::gemini::GEMINI_2_5_FLASH.to_string(),
                #[cfg(feature = "ollama")]
                ProviderType::Ollama => "llama3.2".to_string(),
                #[cfg(feature = "xai")]
                ProviderType::XAI => "grok-beta".to_string(),
                #[cfg(feature = "groq")]
                ProviderType::Groq => "llama-3.1-70b-versatile".to_string(),
                ProviderType::Custom(ref name) => match name.as_str() {
                    #[cfg(feature = "openai")]
                    "siliconflow" => {
                        models::openai_compatible::siliconflow::DEEPSEEK_V3_1.to_string()
                    }
                    #[cfg(feature = "openai")]
                    "deepseek" => models::openai_compatible::deepseek::CHAT.to_string(),
                    #[cfg(feature = "openai")]
                    "openrouter" => models::openai_compatible::openrouter::GPT_4O.to_string(),
                    _ => "default-model".to_string(),
                },

                // For disabled features, return error
                #[cfg(not(feature = "openai"))]
                ProviderType::OpenAi => {
                    return Err(LlmError::UnsupportedOperation(
                        "OpenAI feature not enabled".to_string(),
                    ));
                }
                #[cfg(not(feature = "anthropic"))]
                ProviderType::Anthropic => {
                    return Err(LlmError::UnsupportedOperation(
                        "Anthropic feature not enabled".to_string(),
                    ));
                }
                #[cfg(not(feature = "google"))]
                ProviderType::Gemini => {
                    return Err(LlmError::UnsupportedOperation(
                        "Google feature not enabled".to_string(),
                    ));
                }
                #[cfg(not(feature = "ollama"))]
                ProviderType::Ollama => {
                    return Err(LlmError::UnsupportedOperation(
                        "Ollama feature not enabled".to_string(),
                    ));
                }
                #[cfg(not(feature = "xai"))]
                ProviderType::XAI => {
                    return Err(LlmError::UnsupportedOperation(
                        "xAI feature not enabled".to_string(),
                    ));
                }
                #[cfg(not(feature = "groq"))]
                ProviderType::Groq => {
                    return Err(LlmError::UnsupportedOperation(
                        "Groq feature not enabled".to_string(),
                    ));
                }
            };
        }

        // Build provider-specific parameters
        let provider_params = match provider_type {
            ProviderType::Anthropic => {
                let mut params = ProviderParams::anthropic();

                // Map unified reasoning parameters to Anthropic-specific parameters
                if let Some(budget) = reasoning_budget {
                    params = params.with_param("thinking_budget", budget as u32);
                }

                Some(params)
            }
            ProviderType::Gemini => {
                let mut params = ProviderParams::gemini();

                // Map unified reasoning parameters to Gemini-specific parameters
                if let Some(budget) = reasoning_budget {
                    params = params.with_param("thinking_budget", budget as u32);
                }

                Some(params)
            }
            ProviderType::Ollama => {
                let mut params = ProviderParams::new();

                // Map unified reasoning to Ollama thinking
                if reasoning_enabled.unwrap_or(false) {
                    params = params.with_param("think", true);
                }

                Some(params)
            }
            _ => {
                // For other providers, no specific parameters for now
                None
            }
        };

        // Use RequestBuilder to validate parameters
        let _request_builder =
            crate::request_factory::RequestBuilderFactory::create_and_validate_builder(
                &provider_type,
                common_params.clone(),
                provider_params.clone(),
            )?;

        // Now create the appropriate client based on provider type
        // Parameters have already been validated by RequestBuilder
        let client: Box<dyn LlmClient> = match provider_type {
            #[cfg(feature = "openai")]
            ProviderType::OpenAi => {
                let mut config = crate::providers::openai::OpenAiConfig::new(api_key)
                    .with_base_url(
                        base_url.unwrap_or_else(|| "https://api.openai.com/v1".to_string()),
                    )
                    .with_model(common_params.model.clone());

                // Use validated common parameters
                if let Some(temp) = common_params.temperature {
                    config = config.with_temperature(temp);
                }
                if let Some(max_tokens) = common_params.max_tokens {
                    config = config.with_max_tokens(max_tokens);
                }

                // Set organization and project if provided
                if let Some(org) = organization {
                    config = config.with_organization(org);
                }
                if let Some(proj) = project {
                    config = config.with_project(proj);
                }

                let mut client =
                    crate::providers::openai::OpenAiClient::new(config, built_http_client.clone());
                // Initialize tracing if configured
                if let Some(tc) = self.tracing_config.clone() {
                    let guard = crate::tracing::init_tracing(tc).map_err(|e| {
                        LlmError::ConfigurationError(format!("Failed to init tracing: {e}"))
                    })?;
                    client.set_tracing_guard(Some(guard));
                }
                Box::new(client)
            }
            #[cfg(feature = "anthropic")]
            ProviderType::Anthropic => {
                let anthropic_base_url =
                    base_url.unwrap_or_else(|| "https://api.anthropic.com".to_string());

                // Extract Anthropic-specific parameters from validated provider_params
                let mut anthropic_params = crate::params::AnthropicParams::default();
                if let Some(ref params) = provider_params
                    && let Some(budget) = params.get::<u32>("thinking_budget")
                {
                    anthropic_params.thinking_budget = Some(budget);
                }

                let mut client = crate::providers::anthropic::AnthropicClient::new(
                    api_key,
                    anthropic_base_url,
                    built_http_client.clone(),
                    common_params.clone(),
                    anthropic_params,
                    http_config.clone(),
                );
                if let Some(tc) = self.tracing_config.clone() {
                    let guard = crate::tracing::init_tracing(tc.clone()).map_err(|e| {
                        LlmError::ConfigurationError(format!("Failed to init tracing: {e}"))
                    })?;
                    client.set_tracing_guard(Some(guard));
                    client.set_tracing_config(self.tracing_config.clone());
                }
                Box::new(client)
            }
            #[cfg(feature = "google")]
            ProviderType::Gemini => {
                // Create Gemini client using the provider-specific builder
                // Parameters have already been validated by RequestBuilder
                // Map HttpConfig into LlmBuilder to inherit timeouts/UA/proxy where supported
                let mut base_builder = crate::builder::LlmBuilder::new();
                if let Some(t) = http_config.timeout {
                    base_builder = base_builder.with_timeout(t);
                }
                if let Some(ct) = http_config.connect_timeout {
                    base_builder = base_builder.with_connect_timeout(ct);
                }
                if let Some(ref ua) = http_config.user_agent {
                    base_builder = base_builder.with_user_agent(ua);
                }
                if let Some(ref proxy) = http_config.proxy {
                    base_builder = base_builder.with_proxy(proxy.clone());
                }
                for (k, v) in &http_config.headers {
                    base_builder = base_builder.with_header(k.clone(), v.clone());
                }

                let mut builder = base_builder
                    .gemini()
                    .api_key(api_key)
                    .model(&common_params.model);

                // Apply validated common parameters
                if let Some(temp) = common_params.temperature {
                    builder = builder.temperature(temp);
                }
                if let Some(max_tokens) = common_params.max_tokens {
                    builder = builder.max_tokens(max_tokens as i32);
                }
                if let Some(top_p) = common_params.top_p {
                    builder = builder.top_p(top_p);
                }

                // Apply provider-specific parameters from validated provider_params
                if let Some(ref params) = provider_params
                    && let Some(budget) = params.get::<u32>("thinking_budget")
                {
                    builder = builder.thinking_budget(budget as i32);
                }

                // Apply tracing
                if let Some(tc) = self.tracing_config.clone() {
                    builder = builder.tracing(tc);
                }

                Box::new(builder.build().await.map_err(|e| {
                    LlmError::ConfigurationError(format!("Failed to build Gemini client: {e}"))
                })?)
            }
            #[cfg(feature = "xai")]
            ProviderType::XAI => {
                // Create xAI client using the provider-specific builder
                // Parameters have already been validated by RequestBuilder
                let mut base_builder = crate::builder::LlmBuilder::new();
                if let Some(t) = http_config.timeout {
                    base_builder = base_builder.with_timeout(t);
                }
                if let Some(ct) = http_config.connect_timeout {
                    base_builder = base_builder.with_connect_timeout(ct);
                }
                if let Some(ref ua) = http_config.user_agent {
                    base_builder = base_builder.with_user_agent(ua);
                }
                if let Some(ref proxy) = http_config.proxy {
                    base_builder = base_builder.with_proxy(proxy.clone());
                }
                for (k, v) in &http_config.headers {
                    base_builder = base_builder.with_header(k.clone(), v.clone());
                }

                let mut builder = base_builder
                    .xai()
                    .api_key(api_key)
                    .model(&common_params.model);

                // Apply validated common parameters
                if let Some(temp) = common_params.temperature {
                    builder = builder.temperature(temp);
                }
                if let Some(max_tokens) = common_params.max_tokens {
                    builder = builder.max_tokens(max_tokens);
                }
                if let Some(top_p) = common_params.top_p {
                    builder = builder.top_p(top_p);
                }

                if let Some(tc) = self.tracing_config.clone() {
                    builder = builder.tracing(tc);
                }

                Box::new(builder.build().await.map_err(|e| {
                    LlmError::ConfigurationError(format!("Failed to build xAI client: {e}"))
                })?)
            }
            #[cfg(feature = "ollama")]
            ProviderType::Ollama => {
                let ollama_base_url =
                    base_url.unwrap_or_else(|| "http://localhost:11434".to_string());

                // Extract Ollama-specific parameters from validated provider_params
                let mut ollama_params = crate::providers::ollama::config::OllamaParams::default();
                if let Some(ref params) = provider_params
                    && let Some(think) = params.get::<bool>("think")
                {
                    ollama_params.think = Some(think);
                }

                let config = crate::providers::ollama::config::OllamaConfig {
                    base_url: ollama_base_url,
                    model: Some(common_params.model.clone()),
                    common_params: common_params.clone(),
                    ollama_params,
                    http_config,
                };

                let mut client =
                    crate::providers::ollama::OllamaClient::new(config, built_http_client.clone());
                if let Some(tc) = self.tracing_config.clone() {
                    let guard = crate::tracing::init_tracing(tc.clone()).map_err(|e| {
                        LlmError::ConfigurationError(format!("Failed to init tracing: {e}"))
                    })?;
                    client.set_tracing_guard(Some(guard));
                    client.set_tracing_config(self.tracing_config.clone());
                }
                Box::new(client)
            }
            #[cfg(feature = "groq")]
            ProviderType::Groq => {
                let groq_base_url =
                    base_url.unwrap_or_else(|| "https://api.groq.com/openai/v1".to_string());

                let mut config = crate::providers::groq::GroqConfig::new(api_key)
                    .with_base_url(groq_base_url)
                    .with_model(common_params.model.clone());

                // Use validated common parameters
                if let Some(temp) = common_params.temperature {
                    config = config.with_temperature(temp);
                }
                if let Some(max_tokens) = common_params.max_tokens {
                    config = config.with_max_tokens(max_tokens);
                }

                let mut client =
                    crate::providers::groq::GroqClient::new(config, built_http_client.clone());
                if let Some(tc) = self.tracing_config.clone() {
                    let guard = crate::tracing::init_tracing(tc.clone()).map_err(|e| {
                        LlmError::ConfigurationError(format!("Failed to init tracing: {e}"))
                    })?;
                    client.set_tracing_guard(Some(guard));
                    client.set_tracing_config(self.tracing_config.clone());
                }
                Box::new(client)
            }
            ProviderType::Custom(name) => {
                match name.as_str() {
                    #[cfg(feature = "openai")]
                    "deepseek" => {
                        // Use OpenAI-compatible client for DeepSeek with registry adapter
                        let adapter =
                            crate::providers::openai_compatible::get_provider_adapter("deepseek")?;

                        let base_url =
                            base_url.unwrap_or_else(|| "https://api.deepseek.com/v1".to_string());

                        let mut config =
                            crate::providers::openai_compatible::OpenAiCompatibleConfig::new(
                                "deepseek", &api_key, &base_url, adapter,
                            );

                        // Set model
                        config = config.with_model(&common_params.model);

                        // Set common parameters through common_params field
                        if let Some(temp) = common_params.temperature {
                            config.common_params.temperature = Some(temp);
                        }
                        if let Some(max_tokens) = common_params.max_tokens {
                            config.common_params.max_tokens = Some(max_tokens);
                        }

                        // Propagate HTTP config
                        let config = config.with_http_config(http_config.clone());
                        let client = crate::providers::openai_compatible::OpenAiCompatibleClient::with_http_client(
                            config,
                            built_http_client.clone(),
                        )
                        .await?;
                        Box::new(client)
                    }
                    #[cfg(feature = "openai")]
                    "siliconflow" => {
                        // Use OpenAI-compatible client for SiliconFlow
                        let mut config = crate::providers::openai::OpenAiConfig::new(api_key)
                            .with_base_url(
                                base_url
                                    .unwrap_or_else(|| "https://api.siliconflow.cn/v1".to_string()),
                            )
                            .with_model(common_params.model.clone());

                        // Use validated common parameters
                        if let Some(temp) = common_params.temperature {
                            config = config.with_temperature(temp);
                        }
                        if let Some(max_tokens) = common_params.max_tokens {
                            config = config.with_max_tokens(max_tokens);
                        }

                        let mut client = crate::providers::openai::OpenAiClient::new(
                            config,
                            built_http_client.clone(),
                        );
                        if let Some(tc) = self.tracing_config.clone() {
                            let guard = crate::tracing::init_tracing(tc).map_err(|e| {
                                LlmError::ConfigurationError(format!("Failed to init tracing: {e}"))
                            })?;
                            client.set_tracing_guard(Some(guard));
                        }
                        Box::new(client)
                    }
                    #[cfg(feature = "openai")]
                    "openrouter" => {
                        // Use OpenAI-compatible client for OpenRouter
                        let mut config = crate::providers::openai::OpenAiConfig::new(api_key)
                            .with_base_url(
                                base_url
                                    .unwrap_or_else(|| "https://openrouter.ai/api/v1".to_string()),
                            )
                            .with_model(common_params.model.clone());

                        // Use validated common parameters
                        if let Some(temp) = common_params.temperature {
                            config = config.with_temperature(temp);
                        }
                        if let Some(max_tokens) = common_params.max_tokens {
                            config = config.with_max_tokens(max_tokens);
                        }

                        let mut client = crate::providers::openai::OpenAiClient::new(
                            config,
                            built_http_client.clone(),
                        );
                        if let Some(tc) = self.tracing_config.clone() {
                            let guard = crate::tracing::init_tracing(tc).map_err(|e| {
                                LlmError::ConfigurationError(format!("Failed to init tracing: {e}"))
                            })?;
                            client.set_tracing_guard(Some(guard));
                        }
                        Box::new(client)
                    }

                    _ => {
                        return Err(LlmError::UnsupportedOperation(format!(
                            "Custom provider '{name}' not yet implemented"
                        )));
                    }
                }
            }

            // Handle cases where required features are not enabled
            #[cfg(not(feature = "openai"))]
            ProviderType::OpenAi => {
                return Err(LlmError::UnsupportedOperation(
                    "OpenAI provider requires the 'openai' feature to be enabled".to_string(),
                ));
            }
            #[cfg(not(feature = "anthropic"))]
            ProviderType::Anthropic => {
                return Err(LlmError::UnsupportedOperation(
                    "Anthropic provider requires the 'anthropic' feature to be enabled".to_string(),
                ));
            }
            #[cfg(not(feature = "google"))]
            ProviderType::Gemini => {
                return Err(LlmError::UnsupportedOperation(
                    "Gemini provider requires the 'google' feature to be enabled".to_string(),
                ));
            }
            #[cfg(not(feature = "ollama"))]
            ProviderType::Ollama => {
                return Err(LlmError::UnsupportedOperation(
                    "Ollama provider requires the 'ollama' feature to be enabled".to_string(),
                ));
            }
            #[cfg(not(feature = "xai"))]
            ProviderType::XAI => {
                return Err(LlmError::UnsupportedOperation(
                    "xAI provider requires the 'xai' feature to be enabled".to_string(),
                ));
            }
            #[cfg(not(feature = "groq"))]
            ProviderType::Groq => {
                return Err(LlmError::UnsupportedOperation(
                    "Groq provider requires the 'groq' feature to be enabled".to_string(),
                ));
            }
        };

        Ok(Siumai::new(client))
    }
}

/// Type-safe proxy for audio capabilities
pub struct AudioCapabilityProxy<'a> {
    provider: &'a Siumai,
    reported_support: bool,
}

impl<'a> AudioCapabilityProxy<'a> {
    pub const fn new(provider: &'a Siumai, reported_support: bool) -> Self {
        Self {
            provider,
            reported_support,
        }
    }

    /// Check if the provider reports audio support (for reference only)
    ///
    /// Note: This is based on static capability information and may not reflect
    /// the actual capabilities of the current model. Use as a hint, not a restriction.
    /// The library will never block operations based on this information.
    pub const fn is_reported_as_supported(&self) -> bool {
        self.reported_support
    }

    /// Get provider name for debugging
    pub fn provider_name(&self) -> &'static str {
        self.provider.provider_name()
    }

    /// Get a support status message (optional, for user-controlled warnings)
    ///
    /// Returns a message about support status that you can choose to display or ignore.
    /// The library itself will not automatically warn or log anything.
    pub fn support_status_message(&self) -> String {
        if self.reported_support {
            format!("Provider {} reports audio support", self.provider_name())
        } else {
            format!(
                "Provider {} does not report audio support, but this may still work depending on the model",
                self.provider_name()
            )
        }
    }

    /// Placeholder for future audio operations
    ///
    /// This will attempt the operation regardless of reported support.
    /// Actual errors will come from the API if the model doesn't support it.
    pub async fn placeholder_operation(&self) -> Result<String, LlmError> {
        // No automatic warnings - let the user decide if they want to check support
        Err(LlmError::UnsupportedOperation(
            "Audio operations not yet implemented. Use provider-specific client.".to_string(),
        ))
    }
}

/// Type-safe proxy for embedding capabilities
pub struct EmbeddingCapabilityProxy<'a> {
    provider: &'a Siumai,
    reported_support: bool,
}

impl<'a> EmbeddingCapabilityProxy<'a> {
    pub const fn new(provider: &'a Siumai, reported_support: bool) -> Self {
        Self {
            provider,
            reported_support,
        }
    }

    /// Check if the provider reports embedding support (for reference only)
    pub const fn is_reported_as_supported(&self) -> bool {
        self.reported_support
    }

    /// Get provider name for debugging
    pub fn provider_name(&self) -> &'static str {
        self.provider.provider_name()
    }

    /// Get a support status message (optional, for user-controlled information)
    pub fn support_status_message(&self) -> String {
        if self.reported_support {
            format!(
                "Provider {} reports embedding support",
                self.provider_name()
            )
        } else {
            format!(
                "Provider {} does not report embedding support, but this may still work depending on the model",
                self.provider_name()
            )
        }
    }

    /// Generate embeddings for the given input texts
    pub async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        self.provider.embed(texts).await
    }

    /// Get the dimension of embeddings produced by this provider
    pub fn embedding_dimension(&self) -> usize {
        self.provider.embedding_dimension()
    }

    /// Get the maximum number of tokens that can be embedded at once
    pub fn max_tokens_per_embedding(&self) -> usize {
        self.provider.max_tokens_per_embedding()
    }

    /// Get supported embedding models for this provider
    pub fn supported_embedding_models(&self) -> Vec<String> {
        self.provider.supported_embedding_models()
    }

    /// Placeholder for future embedding operations (deprecated, use embed() instead)
    #[deprecated(note = "Use embed() method instead")]
    pub async fn placeholder_operation(&self) -> Result<String, LlmError> {
        // No automatic warnings - let the user decide if they want to check support
        Err(LlmError::UnsupportedOperation(
            "Use embed() method instead of placeholder_operation()".to_string(),
        ))
    }
}

/// Type-safe proxy for vision capabilities
pub struct VisionCapabilityProxy<'a> {
    provider: &'a Siumai,
    reported_support: bool,
}

impl<'a> VisionCapabilityProxy<'a> {
    pub const fn new(provider: &'a Siumai, reported_support: bool) -> Self {
        Self {
            provider,
            reported_support,
        }
    }

    /// Check if the provider reports vision support (for reference only)
    pub const fn is_reported_as_supported(&self) -> bool {
        self.reported_support
    }

    /// Get provider name for debugging
    pub fn provider_name(&self) -> &'static str {
        self.provider.provider_name()
    }

    /// Get a support status message (optional, for user-controlled information)
    pub fn support_status_message(&self) -> String {
        if self.reported_support {
            format!("Provider {} reports vision support", self.provider_name())
        } else {
            format!(
                "Provider {} does not report vision support, but this may still work depending on the model",
                self.provider_name()
            )
        }
    }

    /// Placeholder for future vision operations
    pub async fn placeholder_operation(&self) -> Result<String, LlmError> {
        // No automatic warnings - let the user decide if they want to check support
        Err(LlmError::UnsupportedOperation(
            "Vision operations not yet implemented. Use provider-specific client.".to_string(),
        ))
    }
}

impl Default for SiumaiBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SiumaiBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("SiumaiBuilder");

        debug_struct
            .field("provider_type", &self.provider_type)
            .field("provider_name", &self.provider_name)
            .field("base_url", &self.base_url)
            .field("model", &self.common_params.model)
            .field("temperature", &self.common_params.temperature)
            .field("max_tokens", &self.common_params.max_tokens)
            .field("top_p", &self.common_params.top_p)
            .field("seed", &self.common_params.seed)
            .field("capabilities_count", &self.capabilities.len())
            .field("reasoning_enabled", &self.reasoning_enabled)
            .field("reasoning_budget", &self.reasoning_budget)
            .field("has_tracing", &self.tracing_config.is_some())
            .field("timeout", &self.http_config.timeout);

        // Only show existence of sensitive fields, not their values
        if self.api_key.is_some() {
            debug_struct.field("has_api_key", &true);
        }
        if self.organization.is_some() {
            debug_struct.field("has_organization", &true);
        }
        if self.project.is_some() {
            debug_struct.field("has_project", &true);
        }

        debug_struct.finish()
    }
}

/// Provider registry for dynamic provider creation
pub struct ProviderRegistry {
    factories: HashMap<String, Box<dyn ProviderFactory>>,
}

/// Factory trait for creating providers
pub trait ProviderFactory: Send + Sync {
    fn create_provider(&self, config: ProviderConfig) -> Result<Box<dyn LlmClient>, LlmError>;
    fn supported_capabilities(&self) -> Vec<String>;
}

/// Configuration for provider creation
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub api_key: String,
    pub base_url: Option<String>,
    pub model: Option<String>,
    pub capabilities: Vec<String>,
}

impl ProviderRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Register a provider factory
    pub fn register<S: Into<String>>(&mut self, name: S, factory: Box<dyn ProviderFactory>) {
        self.factories.insert(name.into(), factory);
    }

    /// Create a provider by name
    pub fn create_provider(&self, name: &str, config: ProviderConfig) -> Result<Siumai, LlmError> {
        let factory = self
            .factories
            .get(name)
            .ok_or_else(|| LlmError::ConfigurationError(format!("Unknown provider: {name}")))?;

        let client = factory.create_provider(config)?;
        Ok(Siumai::new(client))
    }

    /// Get supported providers
    pub fn supported_providers(&self) -> Vec<String> {
        self.factories.keys().cloned().collect()
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    // Mock provider for testing that doesn't support embedding
    #[derive(Debug)]
    struct MockProvider;

    #[async_trait]
    impl ChatCapability for MockProvider {
        async fn chat_with_tools(
            &self,
            _messages: Vec<ChatMessage>,
            _tools: Option<Vec<Tool>>,
        ) -> Result<ChatResponse, LlmError> {
            Ok(ChatResponse {
                id: Some("mock-123".to_string()),
                content: MessageContent::Text("Mock response".to_string()),
                model: Some("mock-model".to_string()),
                usage: None,
                finish_reason: Some(FinishReason::Stop),
                tool_calls: None,
                thinking: None,
                metadata: std::collections::HashMap::new(),
            })
        }

        async fn chat_stream(
            &self,
            _messages: Vec<ChatMessage>,
            _tools: Option<Vec<Tool>>,
        ) -> Result<ChatStream, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "Streaming not supported in mock".to_string(),
            ))
        }
    }

    impl LlmClient for MockProvider {
        fn provider_name(&self) -> &'static str {
            "mock"
        }

        fn supported_models(&self) -> Vec<String> {
            vec!["mock-model".to_string()]
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_chat()
            // Note: not adding .with_embedding() to test unsupported case
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn clone_box(&self) -> Box<dyn LlmClient> {
            Box::new(MockProvider)
        }
    }

    #[tokio::test]
    async fn test_siumai_embedding_unsupported_provider() {
        // Create a mock provider that doesn't support embedding
        let mock_provider = MockProvider;
        let siumai = Siumai::new(Box::new(mock_provider));

        // Test that embedding returns an error for unsupported provider
        let result = siumai.embed(vec!["test".to_string()]).await;
        assert!(result.is_err());

        if let Err(LlmError::UnsupportedOperation(msg)) = result {
            assert!(msg.contains("does not support embedding functionality"));
        } else {
            panic!("Expected UnsupportedOperation error");
        }
    }

    #[test]
    fn test_embedding_capability_proxy() {
        let mock_provider = MockProvider;
        let siumai = Siumai::new(Box::new(mock_provider));

        let proxy = siumai.embedding_capability();
        assert_eq!(proxy.provider_name(), "custom"); // MockProvider gets mapped to "custom" type
        assert!(!proxy.is_reported_as_supported()); // Mock provider doesn't report embedding support
    }

    #[tokio::test]
    async fn test_embedding_capability_proxy_embed() {
        let mock_provider = MockProvider;
        let siumai = Siumai::new(Box::new(mock_provider));

        let proxy = siumai.embedding_capability();
        let result = proxy.embed(vec!["test".to_string()]).await;
        assert!(result.is_err());

        if let Err(LlmError::UnsupportedOperation(msg)) = result {
            assert!(msg.contains("does not support embedding functionality"));
        } else {
            panic!("Expected UnsupportedOperation error");
        }
    }

    #[tokio::test]
    async fn test_ollama_build_without_api_key() {
        // Test that Ollama can be built without API key
        let result = SiumaiBuilder::new()
            .ollama()
            .model("llama3.2")
            .build()
            .await;

        // This should not fail due to missing API key
        // Note: It might fail for other reasons (like Ollama not running), but not API key
        match result {
            Ok(_) => {
                // Success - Ollama client was created without API key
            }
            Err(LlmError::ConfigurationError(msg)) => {
                // Should not be an API key error
                assert!(
                    !msg.contains("API key not specified"),
                    "Ollama should not require API key, but got: {}",
                    msg
                );
            }
            Err(_) => {
                // Other errors are acceptable (e.g., network issues)
            }
        }
    }

    #[tokio::test]
    async fn test_openai_requires_api_key() {
        // Test that OpenAI still requires API key
        let result = SiumaiBuilder::new().openai().model("gpt-4o").build().await;

        // This should fail due to missing API key
        assert!(result.is_err());
        if let Err(LlmError::ConfigurationError(msg)) = result {
            assert!(msg.contains("API key not specified"));
        } else {
            panic!("Expected ConfigurationError for missing API key");
        }
    }
}
