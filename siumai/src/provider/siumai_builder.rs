use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::LanguageModelMiddleware;
use crate::retry_api::RetryOptions;
use crate::types::{CommonParams, HttpConfig, ProviderType};

/// Unified Interface Builder - Provider Abstraction Layer
///
/// Provides a unified interface for creating LLM clients across providers
/// while abstracting provider-specific details. Public API unchanged from
/// its original location in `provider.rs`.
pub struct SiumaiBuilder {
    pub(crate) provider_type: Option<ProviderType>,
    pub(crate) provider_id: Option<String>,
    pub(crate) api_key: Option<String>,
    pub(crate) base_url: Option<String>,
    pub(crate) capabilities: Vec<String>,
    pub(crate) common_params: CommonParams,
    pub(crate) http_config: HttpConfig,
    pub(crate) organization: Option<String>,
    pub(crate) project: Option<String>,
    pub(crate) tracing_config: Option<crate::observability::tracing::TracingConfig>,
    // Unified reasoning configuration
    pub(crate) reasoning_enabled: Option<bool>,
    pub(crate) reasoning_budget: Option<i32>,
    // Unified retry configuration
    pub(crate) retry_options: Option<RetryOptions>,
    /// Optional HTTP interceptors applied to chat requests (unified interface)
    pub(crate) http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Enable a built-in logging interceptor for HTTP debugging (no sensitive data)
    pub(crate) http_debug: bool,
    /// Custom HTTP client (takes precedence over all other HTTP settings)
    pub(crate) http_client: Option<reqwest::Client>,
    /// Advanced HTTP features are not handled here anymore; use HttpConfig only
    #[cfg(feature = "google")]
    /// Optional Bearer token provider for Gemini (e.g., Vertex AI).
    pub(crate) gemini_token_provider: Option<std::sync::Arc<dyn crate::auth::TokenProvider>>,
    /// Optional model-level middlewares applied before provider mapping
    pub(crate) model_middlewares: Vec<std::sync::Arc<dyn LanguageModelMiddleware>>,
}

impl SiumaiBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            provider_type: None,
            provider_id: None,
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
            retry_options: None,
            http_interceptors: Vec::new(),
            http_debug: false,
            http_client: None,

            #[cfg(feature = "google")]
            gemini_token_provider: None,
            model_middlewares: Vec::new(),
        }
    }

    /// Set the provider type
    pub fn provider(mut self, provider_type: ProviderType) -> Self {
        self.provider_type = Some(provider_type);
        self
    }

    /// Set the provider by canonical id (dynamic dispatch)
    /// Recommended to use canonical ids like "openai", "anthropic", "gemini".
    pub fn provider_id<S: Into<String>>(mut self, id: S) -> Self {
        let name = id.into();
        self.provider_id = Some(name.clone());

        // Map provider name to type
        self.provider_type = Some(match name.as_str() {
            "openai" => ProviderType::OpenAi,
            "openai-chat" => ProviderType::OpenAi,
            "openai-responses" => ProviderType::OpenAi,
            "anthropic" => ProviderType::Anthropic,
            "gemini" => ProviderType::Gemini,
            "ollama" => ProviderType::Ollama,
            "xai" => ProviderType::XAI,
            "groq" => ProviderType::Groq,
            "minimaxi" => ProviderType::MiniMaxi,
            "siliconflow" => ProviderType::Custom("siliconflow".to_string()),
            "deepseek" => ProviderType::Custom("deepseek".to_string()),
            "openrouter" => ProviderType::Custom("openrouter".to_string()),
            _ => ProviderType::Custom(name.clone()),
        });
        // Responses API routing is now handled via provider_options in ChatRequest
        self
    }

    // Provider convenience methods are defined in src/provider_builders.rs

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
    pub const fn reasoning(mut self, enabled: bool) -> Self {
        self.reasoning_enabled = Some(enabled);
        self
    }

    /// Set reasoning budget (unified interface)
    pub const fn reasoning_budget(mut self, budget: i32) -> Self {
        self.reasoning_budget = Some(budget);
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

    /// Install a custom HTTP interceptor at the unified interface level.
    pub fn with_http_interceptor(mut self, interceptor: Arc<dyn HttpInterceptor>) -> Self {
        self.http_interceptors.push(interceptor);
        self
    }

    /// Add a model-level middleware (applied before provider mapping).
    pub fn add_model_middleware(
        mut self,
        middleware: std::sync::Arc<dyn LanguageModelMiddleware>,
    ) -> Self {
        self.model_middlewares.push(middleware);
        self
    }

    /// Replace the model-level middleware list (applied before provider mapping).
    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<std::sync::Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.model_middlewares = middlewares;
        self
    }

    /// Enable a built-in logging interceptor for HTTP debugging (no sensitive data).
    pub fn http_debug(mut self, enabled: bool) -> Self {
        self.http_debug = enabled;
        self
    }

    /// Set custom HTTP client (takes precedence over all other HTTP settings)
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = Some(client);
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
    pub fn http_timeout(mut self, timeout: Duration) -> Self {
        self.http_config.timeout = Some(timeout);
        self
    }
    pub fn http_connect_timeout(mut self, timeout: Duration) -> Self {
        self.http_config.connect_timeout = Some(timeout);
        self
    }
    pub fn http_user_agent<S: Into<String>>(mut self, user_agent: S) -> Self {
        self.http_config.user_agent = Some(user_agent.into());
        self
    }
    pub fn http_proxy<S: Into<String>>(mut self, proxy_url: S) -> Self {
        self.http_config.proxy = Some(proxy_url.into());
        self
    }
    pub fn http_header<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.http_config.headers.insert(key.into(), value.into());
        self
    }
    pub fn http_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.http_config.headers.extend(headers);
        self
    }
    pub fn http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.http_config.stream_disable_compression = disable;
        self
    }

    // === API Consistency Aliases (matching LlmBuilder) ===
    /// Alias for `http_timeout` (API consistency with LlmBuilder)
    pub fn with_timeout(self, timeout: Duration) -> Self {
        self.http_timeout(timeout)
    }

    /// Alias for `http_connect_timeout` (API consistency with LlmBuilder)
    pub fn with_connect_timeout(self, timeout: Duration) -> Self {
        self.http_connect_timeout(timeout)
    }

    /// Alias for `http_user_agent` (API consistency with LlmBuilder)
    pub fn with_user_agent<S: Into<String>>(self, user_agent: S) -> Self {
        self.http_user_agent(user_agent)
    }

    /// Alias for `http_proxy` (API consistency with LlmBuilder)
    pub fn with_proxy<S: Into<String>>(self, proxy_url: S) -> Self {
        self.http_proxy(proxy_url)
    }

    /// Alias for `http_header` (API consistency with LlmBuilder)
    pub fn with_header<K: Into<String>, V: Into<String>>(self, key: K, value: V) -> Self {
        self.http_header(key, value)
    }

    /// Alias for `stop_sequences` (API consistency with LlmBuilder)
    pub fn stop<S: Into<String>>(self, sequences: Vec<S>) -> Self {
        self.stop_sequences(sequences.into_iter().map(|s| s.into()).collect())
    }
    pub fn with_x_goog_user_project(mut self, project_id: impl Into<String>) -> Self {
        self.http_config
            .headers
            .insert("x-goog-user-project".to_string(), project_id.into());
        self
    }
    pub fn base_url_for_vertex(mut self, project: &str, location: &str, publisher: &str) -> Self {
        let base = crate::utils::vertex_base_url(project, location, publisher);
        self.base_url = Some(base);
        self
    }
    #[cfg(feature = "google")]
    pub fn with_gemini_token_provider(
        mut self,
        provider: std::sync::Arc<dyn crate::auth::TokenProvider>,
    ) -> Self {
        self.gemini_token_provider = Some(provider);
        self
    }
    #[cfg(all(feature = "google", feature = "gcp"))]
    pub fn with_gemini_adc(mut self) -> Self {
        let adc = crate::auth::adc::AdcTokenProvider::default_client();
        self.gemini_token_provider = Some(std::sync::Arc::new(adc));
        self
    }

    // === Tracing Configuration ===
    pub fn tracing(mut self, config: crate::observability::tracing::TracingConfig) -> Self {
        self.tracing_config = Some(config);
        self
    }
    pub fn debug_tracing(self) -> Self {
        self.tracing(crate::observability::tracing::TracingConfig::development())
    }
    pub fn minimal_tracing(self) -> Self {
        self.tracing(crate::observability::tracing::TracingConfig::minimal())
    }
    pub fn json_tracing(self) -> Self {
        self.tracing(crate::observability::tracing::TracingConfig::json_production())
    }
    pub fn enable_tracing(self) -> Self {
        self.debug_tracing()
    }
    pub fn disable_tracing(self) -> Self {
        self.tracing(crate::observability::tracing::TracingConfig::disabled())
    }

    /// Set unified retry options for chat operations
    pub fn with_retry(mut self, options: RetryOptions) -> Self {
        self.retry_options = Some(options);
        self
    }

    /// Build the siumai provider (delegates to provider/build.rs)
    pub async fn build(self) -> Result<crate::provider::Siumai, LlmError> {
        crate::provider::build::build(self).await
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
            .field("provider_id", &self.provider_id)
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
