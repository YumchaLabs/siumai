use crate::types::{CommonParams, HttpConfig};
use secrecy::SecretString;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;

/// Gemini configuration parameters (protocol layer)
#[derive(Clone)]
pub struct GeminiConfig {
    /// API key for authentication (securely stored)
    pub api_key: SecretString,
    /// Base URL for the Gemini API
    pub base_url: String,
    /// Default model to use
    pub model: String,
    /// Common parameters shared across providers (temperature, max_tokens, top_p, stop_sequences)
    pub common_params: CommonParams,
    /// Default generation configuration (Gemini-specific: top_k, response_mime_type, etc.)
    pub generation_config: Option<super::GenerationConfig>,
    /// Default safety settings
    pub safety_settings: Option<Vec<super::SafetySetting>>,
    /// HTTP timeout in seconds
    pub timeout: Option<u64>,
    /// HTTP configuration (custom headers, proxy, user agent)
    pub http_config: HttpConfig,
    /// Optional Bearer token provider (e.g., Vertex AI enterprise auth)
    /// Not serialized/deserialized; runtime only.
    pub token_provider: Option<std::sync::Arc<dyn crate::auth::TokenProvider>>,

    /// Optional custom HTTP transport (Vercel-style "custom fetch" parity).
    pub http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,

    /// Override the Vercel-aligned providerMetadata namespace key (`google` vs `vertex`).
    ///
    /// When unset, we infer:
    /// - `vertex` when base_url looks like Vertex (`aiplatform.googleapis.com` / contains `vertex`)
    /// - otherwise `google`
    pub provider_metadata_key: Option<String>,

    /// Optional HTTP interceptors applied to all requests built from this config.
    pub http_interceptors: Vec<Arc<dyn HttpInterceptor>>,

    /// Optional model-level middlewares applied before provider mapping (chat only).
    pub model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
}

impl std::fmt::Debug for GeminiConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use secrecy::ExposeSecret;
        f.debug_struct("GeminiConfig")
            .field(
                "api_key_present",
                &(!self.api_key.expose_secret().is_empty()),
            )
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("common_params", &self.common_params)
            .field(
                "generation_config_present",
                &self.generation_config.is_some(),
            )
            .field("safety_settings_present", &self.safety_settings.is_some())
            .field("timeout", &self.timeout)
            .field("http_config", &self.http_config)
            .field("provider_metadata_key", &self.provider_metadata_key)
            .finish()
    }
}

impl Default for GeminiConfig {
    fn default() -> Self {
        Self {
            api_key: SecretString::from(String::new()),
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            model: "gemini-2.5-flash".to_string(),
            common_params: CommonParams::default(),
            generation_config: None,
            safety_settings: None,
            timeout: Some(30),
            http_config: HttpConfig::default(),
            token_provider: None,
            http_transport: None,
            provider_metadata_key: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
        }
    }
}

impl GeminiConfig {
    /// Create a new Gemini configuration with the given API key
    pub fn new<S: Into<String>>(api_key: S) -> Self {
        Self {
            api_key: SecretString::from(api_key.into()),
            ..Default::default()
        }
    }
    /// Set the model to use
    pub fn with_model(mut self, model: String) -> Self {
        self.common_params.model = model.clone();
        self.model = model;
        self
    }
    /// Set the base URL
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }
    /// Set common parameters (temperature, max_tokens, top_p, stop_sequences)
    pub fn with_common_params(mut self, params: CommonParams) -> Self {
        self.common_params = params;
        self
    }
    /// Set temperature
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.common_params.temperature = Some(temperature);
        self
    }
    /// Set max tokens
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.common_params.max_tokens = Some(max_tokens);
        self
    }
    /// Set top_p
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.common_params.top_p = Some(top_p);
        self
    }
    /// Set stop sequences
    pub fn with_stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.common_params.stop_sequences = Some(stop_sequences);
        self
    }

    /// Set top-k.
    pub fn with_top_k(mut self, top_k: i32) -> Self {
        let generation_config = self.generation_config.unwrap_or_default().with_top_k(top_k);
        self.generation_config = Some(generation_config);
        self
    }

    /// Set candidate count.
    pub fn with_candidate_count(mut self, count: i32) -> Self {
        let generation_config = self
            .generation_config
            .unwrap_or_default()
            .with_candidate_count(count);
        self.generation_config = Some(generation_config);
        self
    }

    /// Enable structured output with JSON schema.
    pub fn with_json_schema(mut self, schema: serde_json::Value) -> Self {
        let generation_config = self
            .generation_config
            .unwrap_or_default()
            .with_response_schema(schema)
            .with_response_mime_type("application/json".to_string());
        self.generation_config = Some(generation_config);
        self
    }

    /// Set thinking budget.
    pub fn with_thinking_budget(mut self, budget: i32) -> Self {
        let include_thoughts = budget != 0;
        let thinking = super::ThinkingConfig {
            thinking_budget: Some(budget),
            include_thoughts: Some(include_thoughts),
            thinking_level: None,
        };
        let generation_config = self
            .generation_config
            .unwrap_or_default()
            .with_thinking_config(thinking);
        self.generation_config = Some(generation_config);
        self
    }

    /// Control whether thought summaries are included.
    pub fn with_thought_summaries(mut self, include: bool) -> Self {
        let mut thinking = self
            .generation_config
            .as_ref()
            .and_then(|cfg| cfg.thinking_config.clone())
            .unwrap_or_default();
        thinking.include_thoughts = Some(include);
        let generation_config = self
            .generation_config
            .unwrap_or_default()
            .with_thinking_config(thinking);
        self.generation_config = Some(generation_config);
        self
    }

    /// Enable dynamic thinking.
    pub fn with_dynamic_thinking(mut self) -> Self {
        let generation_config = self
            .generation_config
            .unwrap_or_default()
            .with_thinking_config(super::ThinkingConfig::dynamic());
        self.generation_config = Some(generation_config);
        self
    }

    /// Unified reasoning toggle.
    pub fn with_reasoning(mut self, enable: bool) -> Self {
        let thinking = if enable {
            super::ThinkingConfig::dynamic()
        } else {
            super::ThinkingConfig::disabled()
        };
        let generation_config = self
            .generation_config
            .unwrap_or_default()
            .with_thinking_config(thinking);
        self.generation_config = Some(generation_config);
        self
    }

    /// Unified reasoning budget.
    pub fn with_reasoning_budget(self, budget: i32) -> Self {
        self.with_thinking_budget(budget)
    }

    /// Attempt to disable thinking.
    pub fn with_disable_thinking(mut self) -> Self {
        let generation_config = self
            .generation_config
            .unwrap_or_default()
            .with_thinking_config(super::ThinkingConfig::disabled());
        self.generation_config = Some(generation_config);
        self
    }

    /// Set generation configuration (Gemini-specific: top_k, response_mime_type, etc.)
    pub fn with_generation_config(mut self, config: super::GenerationConfig) -> Self {
        self.generation_config = Some(config);
        self
    }
    /// Set safety settings
    pub fn with_safety_settings(mut self, settings: Vec<super::SafetySetting>) -> Self {
        self.safety_settings = Some(settings);
        self
    }
    /// Set HTTP timeout
    pub const fn with_timeout(mut self, timeout: u64) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set HTTP config (headers/proxy/user-agent)
    pub fn with_http_config(mut self, http: HttpConfig) -> Self {
        self.http_config = http;
        self
    }

    /// Set request timeout on the canonical config-first HTTP surface.
    ///
    /// This complements the legacy `with_timeout(u64)` seconds-based field by
    /// configuring the shared `HttpConfig` directly for new config-first code.
    pub fn with_http_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http_config.timeout = Some(timeout);
        self.timeout = Some(timeout.as_secs());
        self
    }

    /// Set connection timeout on the canonical config-first HTTP surface.
    pub fn with_connect_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http_config.connect_timeout = Some(timeout);
        self
    }

    /// Control whether streaming requests disable compression.
    pub fn with_http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.http_config.stream_disable_compression = disable;
        self
    }

    /// Set a Bearer token provider for enterprise auth (e.g., Vertex AI).
    pub fn with_token_provider(
        mut self,
        provider: std::sync::Arc<dyn crate::auth::TokenProvider>,
    ) -> Self {
        self.token_provider = Some(provider);
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

    /// Override providerMetadata namespace key (`google` or `vertex`).
    pub fn with_provider_metadata_key(mut self, key: impl Into<String>) -> Self {
        self.provider_metadata_key = Some(key.into());
        self
    }

    /// Install HTTP interceptors for requests created by clients built from this config.
    pub fn with_http_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors;
        self
    }

    /// Append a single HTTP interceptor on the canonical config-first HTTP surface.
    pub fn with_http_interceptor(mut self, interceptor: Arc<dyn HttpInterceptor>) -> Self {
        self.http_interceptors.push(interceptor);
        self
    }

    /// Install model-level middlewares for chat requests created by clients built from this config.
    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.model_middlewares = middlewares;
        self
    }
}

/// Tool configuration for any Tool specified in the request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolConfig {
    /// Optional. Function calling config.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_calling_config: Option<FunctionCallingConfig>,

    /// Optional. Configuration for grounding retrieval (location context, etc.).
    #[serde(skip_serializing_if = "Option::is_none", rename = "retrievalConfig")]
    pub retrieval_config: Option<RetrievalConfig>,
}

/// Configuration for grounding retrieval (e.g. Google Maps grounding).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RetrievalConfig {
    #[serde(skip_serializing_if = "Option::is_none", rename = "latLng")]
    pub lat_lng: Option<LatLng>,
}

/// Latitude/longitude pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatLng {
    pub latitude: f64,
    pub longitude: f64,
}

/// Configuration for specifying function calling behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallingConfig {
    /// Optional. Specifies the mode in which function calling should execute.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<FunctionCallingMode>,
    /// Optional. A set of function names that, when provided, limits the functions the model will call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_function_names: Option<Vec<String>>,
}

/// Defines the execution behavior for function calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FunctionCallingMode {
    #[serde(rename = "MODE_UNSPECIFIED")]
    Unspecified,
    #[serde(rename = "AUTO")]
    Auto,
    #[serde(rename = "ANY")]
    Any,
    #[serde(rename = "NONE")]
    None,
}

// ================================================================================================
// Embedding options & extension helpers (Gemini-specific convenience APIs)
// ================================================================================================

/// Gemini-specific embedding configuration options
///
/// This struct provides type-safe configuration for Gemini embedding requests,
/// including task type optimization, context titles, and custom dimensions.
///
/// # Example
/// ```rust,no_run
/// use siumai::providers::gemini::types::GeminiEmbeddingOptions;
/// use siumai::types::EmbeddingTaskType;
///
/// let options = GeminiEmbeddingOptions::new()
///     .with_task_type(EmbeddingTaskType::RetrievalQuery)
///     .with_title("Search Context")
///     .with_output_dimensionality(768);
/// ```
#[derive(Debug, Clone, Default)]
pub struct GeminiEmbeddingOptions {
    /// Task type for optimization (Gemini-specific feature)
    pub task_type: Option<crate::types::EmbeddingTaskType>,
    /// Title for additional context (helps with embedding quality)
    pub title: Option<String>,
    /// Custom output dimensions (128-3072, must be supported by model)
    pub output_dimensionality: Option<u32>,
}

impl GeminiEmbeddingOptions {
    /// Create new Gemini embedding options with default values
    pub fn new() -> Self {
        Self::default()
    }
    /// Set task type for optimization
    pub fn with_task_type(mut self, task_type: crate::types::EmbeddingTaskType) -> Self {
        self.task_type = Some(task_type);
        self
    }
    /// Set title for additional context
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }
    /// Set custom output dimensions
    pub fn with_output_dimensionality(mut self, dimensions: u32) -> Self {
        self.output_dimensionality = Some(dimensions);
        self
    }
    /// Apply these options to an EmbeddingRequest
    pub fn apply_to_request(
        self,
        mut request: crate::types::EmbeddingRequest,
    ) -> crate::types::EmbeddingRequest {
        if let Some(task_type) = self.task_type {
            request = request.with_task_type(task_type);
        }
        if let Some(title) = self.title {
            request = request.with_title(title);
        }
        if let Some(dims) = self.output_dimensionality {
            request.dimensions = Some(dims);
        }
        request
    }
}

/// Extension trait for EmbeddingRequest to add Gemini-specific configuration
pub trait GeminiEmbeddingRequestExt {
    /// Configure this request with Gemini-specific options
    fn with_gemini_config(self, config: GeminiEmbeddingOptions) -> Self;
    /// Quick method to set Gemini task type
    fn with_gemini_task_type(self, task_type: crate::types::EmbeddingTaskType) -> Self;
    /// Quick method to set Gemini title
    fn with_gemini_title(self, title: impl Into<String>) -> Self;
    /// Quick method to set Gemini output dimensions
    fn with_gemini_dimensions(self, dimensions: u32) -> Self;
}

impl GeminiEmbeddingRequestExt for crate::types::EmbeddingRequest {
    fn with_gemini_config(self, config: GeminiEmbeddingOptions) -> Self {
        config.apply_to_request(self)
    }
    fn with_gemini_task_type(self, task_type: crate::types::EmbeddingTaskType) -> Self {
        self.with_task_type(task_type)
    }
    fn with_gemini_title(self, title: impl Into<String>) -> Self {
        self.with_title(title)
    }
    fn with_gemini_dimensions(self, dimensions: u32) -> Self {
        let mut request = self;
        request.dimensions = Some(dimensions);
        request
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::Arc, time::Duration};

    #[test]
    fn gemini_config_provider_specific_fluent_setters() {
        let config = GeminiConfig::new("test-key")
            .with_model("gemini-2.5-pro".to_string())
            .with_top_k(8)
            .with_candidate_count(2)
            .with_json_schema(serde_json::json!({ "type": "object" }))
            .with_reasoning_budget(2048)
            .with_thought_summaries(true);

        let generation_config = config.generation_config.expect("generation config");
        assert_eq!(config.model, "gemini-2.5-pro");
        assert_eq!(generation_config.top_k, Some(8));
        assert_eq!(generation_config.candidate_count, Some(2));
        assert_eq!(
            generation_config.response_mime_type.as_deref(),
            Some("application/json")
        );
        assert_eq!(
            generation_config.response_schema,
            Some(serde_json::json!({ "type": "object" }))
        );
        let thinking = generation_config.thinking_config.expect("thinking config");
        assert_eq!(thinking.thinking_budget, Some(2048));
        assert_eq!(thinking.include_thoughts, Some(true));
    }

    #[test]
    fn gemini_config_reasoning_disable_maps_to_thinking_disabled() {
        let config = GeminiConfig::new("test-key").with_reasoning(false);
        let thinking = config
            .generation_config
            .expect("generation config")
            .thinking_config
            .expect("thinking config");
        assert_eq!(thinking.thinking_budget, Some(0));
        assert_eq!(thinking.include_thoughts, Some(false));
    }

    #[test]
    fn gemini_config_http_convenience_helpers() {
        let config = GeminiConfig::new("test-key")
            .with_http_timeout(Duration::from_secs(20))
            .with_connect_timeout(Duration::from_secs(4))
            .with_http_stream_disable_compression(true)
            .with_http_interceptor(Arc::new(
                crate::execution::http::interceptor::LoggingInterceptor,
            ));

        assert_eq!(config.http_config.timeout, Some(Duration::from_secs(20)));
        assert_eq!(config.timeout, Some(20));
        assert_eq!(
            config.http_config.connect_timeout,
            Some(Duration::from_secs(4))
        );
        assert!(config.http_config.stream_disable_compression);
        assert_eq!(config.http_interceptors.len(), 1);
    }
}
