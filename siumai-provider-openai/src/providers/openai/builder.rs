//! `OpenAI` Provider Builder
//!
//! This module provides the OpenAI-specific builder implementation that follows
//! the design pattern established in the main builder module.
//!
//! This builder uses composition with `ProviderCore` to eliminate code duplication
//! and ensure consistency across all provider builders.

use crate::builder::{BuilderBase, ProviderCore};
use crate::error::LlmError;
use crate::params::{OpenAiParams, ResponseFormat, ToolChoice};
use crate::retry_api::RetryOptions;
use crate::types::*;
use std::sync::Arc;

use super::OpenAiClient;
use super::vendors::OpenAiVendorId;

/// OpenAI-specific builder for configuring `OpenAI` clients.
///
/// This builder provides OpenAI-specific configuration options while
/// using `ProviderCore` for common HTTP, tracing, and retry configuration.
///
/// Retry: call `.with_retry(RetryOptions::backoff())` to enable automatic retries
/// for chat operations via the unified retry facade.
///
/// # Example
/// ```rust,no_run
/// use siumai::builder::LlmBuilder;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let client = LlmBuilder::new()
///         .openai()
///         .api_key("your-api-key")
///         .model("gpt-4")
///         .temperature(0.7)
///         .max_tokens(1000)
///         .build()
///         .await?;
///
///     Ok(())
/// }
/// ```
pub struct OpenAiBuilder {
    /// Core configuration shared across all providers (composition over inheritance)
    pub(crate) core: ProviderCore,
    /// OpenAI API key
    api_key: Option<String>,
    /// Base URL for OpenAI API
    base_url: Option<String>,
    /// Organization ID
    organization: Option<String>,
    /// Project ID
    project: Option<String>,
    /// Common parameters (temperature, max_tokens, etc.)
    common_params: CommonParams,
    /// OpenAI-specific parameters
    openai_params: OpenAiParams,
    /// Whether to use the OpenAI Responses API instead of Chat Completions
    use_responses_api: bool,
    /// Responses API chaining id
    responses_previous_response_id: Option<String>,
    /// Default provider options (open map, Vercel-aligned).
    default_provider_options_map: ProviderOptionsMap,
}

#[cfg(feature = "openai")]
impl OpenAiBuilder {
    pub fn new(base: BuilderBase) -> Self {
        Self {
            core: ProviderCore::new(base),
            api_key: None,
            base_url: None,
            organization: None,
            project: None,
            common_params: CommonParams::default(),
            openai_params: OpenAiParams::default(),
            // Vercel AI SDK alignment: OpenAI defaults to Responses API for text generation.
            use_responses_api: true,
            responses_previous_response_id: None,
            default_provider_options_map: ProviderOptionsMap::default(),
        }
    }

    /// Sets the API key
    pub fn api_key<S: Into<String>>(mut self, key: S) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets the base URL
    pub fn base_url<S: Into<String>>(mut self, url: S) -> Self {
        self.base_url = Some(url.into());
        self
    }

    pub fn with_base_url<S: Into<String>>(mut self, url: S) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Sets the organization ID
    pub fn organization<S: Into<String>>(mut self, org: S) -> Self {
        self.organization = Some(org.into());
        self
    }

    pub fn with_organization<S: Into<String>>(mut self, org: S) -> Self {
        self.organization = Some(org.into());
        self
    }

    /// Sets the project ID
    pub fn project<S: Into<String>>(mut self, project: S) -> Self {
        self.project = Some(project.into());
        self
    }

    pub fn with_project<S: Into<String>>(mut self, project: S) -> Self {
        self.project = Some(project.into());
        self
    }

    /// Sets the model
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.common_params.model = model.into();
        self
    }

    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.common_params.model = model.into();
        self
    }

    /// Sets the temperature
    pub const fn temperature(mut self, temp: f64) -> Self {
        self.common_params.temperature = Some(temp);
        self
    }

    pub const fn with_temperature(mut self, temp: f64) -> Self {
        self.common_params.temperature = Some(temp);
        self
    }

    /// Sets the maximum number of tokens
    pub const fn max_tokens(mut self, tokens: u32) -> Self {
        self.common_params.max_tokens = Some(tokens);
        self
    }

    pub const fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.common_params.max_tokens = Some(tokens);
        self
    }

    /// Sets `top_p`
    pub const fn top_p(mut self, top_p: f64) -> Self {
        self.common_params.top_p = Some(top_p);
        self
    }

    pub const fn with_top_p(mut self, top_p: f64) -> Self {
        self.common_params.top_p = Some(top_p);
        self
    }

    /// Sets the stop sequences
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.common_params.stop_sequences = Some(sequences);
        self
    }

    pub fn with_stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.common_params.stop_sequences = Some(sequences);
        self
    }

    /// Sets the random seed
    pub const fn seed(mut self, seed: u64) -> Self {
        self.common_params.seed = Some(seed);
        self
    }

    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.common_params.seed = Some(seed);
        self
    }

    // OpenAI-specific parameters

    /// Sets the response format
    pub fn response_format(mut self, format: ResponseFormat) -> Self {
        self.openai_params.response_format = Some(format);
        self
    }

    pub fn with_response_format(mut self, format: ResponseFormat) -> Self {
        self.openai_params.response_format = Some(format);
        self
    }

    /// Sets the tool choice strategy
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.openai_params.tool_choice = Some(choice);
        self
    }

    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.openai_params.tool_choice = Some(choice);
        self
    }

    /// Sets the frequency penalty
    pub const fn frequency_penalty(mut self, penalty: f32) -> Self {
        self.openai_params.frequency_penalty = Some(penalty);
        self
    }

    pub const fn with_frequency_penalty(mut self, penalty: f32) -> Self {
        self.openai_params.frequency_penalty = Some(penalty);
        self
    }

    /// Sets the presence penalty
    pub const fn presence_penalty(mut self, penalty: f32) -> Self {
        self.openai_params.presence_penalty = Some(penalty);
        self
    }

    pub const fn with_presence_penalty(mut self, penalty: f32) -> Self {
        self.openai_params.presence_penalty = Some(penalty);
        self
    }

    /// Sets the user ID
    pub fn user<S: Into<String>>(mut self, user: S) -> Self {
        self.openai_params.user = Some(user.into());
        self
    }

    pub fn with_user<S: Into<String>>(mut self, user: S) -> Self {
        self.openai_params.user = Some(user.into());
        self
    }

    /// Enables parallel tool calls
    pub const fn parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.openai_params.parallel_tool_calls = Some(enabled);
        self
    }

    pub const fn with_parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.openai_params.parallel_tool_calls = Some(enabled);
        self
    }

    // ========================================================================
    // Common Configuration Methods (delegated to ProviderCore)
    // ========================================================================

    /// Control whether to disable compression for streaming (SSE) requests.
    /// Default is true for stability. Set to false to allow compression.
    pub fn http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.core = self.core.http_stream_disable_compression(disable);
        self
    }

    /// Set custom tracing configuration
    pub fn tracing(mut self, config: crate::observability::tracing::TracingConfig) -> Self {
        self.core = self.core.tracing(config);
        self
    }

    /// Enable debug tracing (development-friendly configuration)
    pub fn debug_tracing(mut self) -> Self {
        self.core = self.core.debug_tracing();
        self
    }

    /// Enable minimal tracing (info level, LLM only)
    pub fn minimal_tracing(mut self) -> Self {
        self.core = self.core.minimal_tracing();
        self
    }

    /// Enable production-ready JSON tracing
    pub fn json_tracing(mut self) -> Self {
        self.core = self.core.json_tracing();
        self
    }

    /// Enable pretty-printed formatting for JSON bodies and headers in tracing
    pub fn pretty_json(mut self, pretty: bool) -> Self {
        self.core = self.core.pretty_json(pretty);
        self
    }

    /// Control masking of sensitive values (API keys, tokens) in tracing logs
    pub fn mask_sensitive_values(mut self, mask: bool) -> Self {
        self.core = self.core.mask_sensitive_values(mask);
        self
    }

    /// Set unified retry options for chat operations
    pub fn with_retry(mut self, options: RetryOptions) -> Self {
        self.core = self.core.with_retry(options);
        self
    }

    /// Add a custom HTTP interceptor
    pub fn with_http_interceptor(
        mut self,
        interceptor: std::sync::Arc<dyn crate::execution::http::interceptor::HttpInterceptor>,
    ) -> Self {
        self.core = self.core.with_http_interceptor(interceptor);
        self
    }

    /// Enable a built-in logging interceptor for HTTP debugging
    pub fn http_debug(mut self, enabled: bool) -> Self {
        self.core = self.core.http_debug(enabled);
        self
    }

    /// Set request timeout
    pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
        self.core = self.core.timeout(timeout);
        self
    }

    pub fn with_http_config(mut self, config: HttpConfig) -> Self {
        self.core.http_config = config;
        self
    }

    /// Set connection timeout
    pub fn connect_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.core = self.core.connect_timeout(timeout);
        self
    }

    /// Set custom HTTP client
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.core = self.core.with_http_client(client);
        self
    }

    /// Set a custom HTTP transport (Vercel-style "custom fetch" parity).
    pub fn with_http_transport(
        mut self,
        transport: Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.core = self.core.with_http_transport(transport);
        self
    }

    /// Alias for `with_http_transport(...)` (Vercel-aligned: `fetch`).
    pub fn fetch(
        self,
        transport: Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.with_http_transport(transport)
    }

    /// Use OpenAI WebSocket mode for streaming Responses requests (`POST /responses` with `stream=true`).
    ///
    /// This injects an `OpenAiWebSocketTransport` as the builder's HTTP transport:
    /// - streaming `/responses` is routed through WebSocket
    /// - all other requests still use HTTP
    ///
    /// Requires the `openai-websocket` feature.
    #[cfg(feature = "openai-websocket")]
    pub fn use_openai_websocket_transport(
        self,
        transport: super::OpenAiWebSocketTransport,
    ) -> Self {
        self.fetch(Arc::new(transport))
    }

    /// Build a single-connection OpenAI WebSocket session.
    ///
    /// This is the recommended entry point for agentic workflows (sequential tool loops) because it:
    /// - enforces one in-flight stream at a time
    /// - keeps `previous_response_id` continuation unambiguous (connection-local)
    /// - supports connection warm-up (`generate=false`)
    ///
    /// Requires the `openai-websocket` feature.
    #[cfg(feature = "openai-websocket")]
    pub async fn use_openai_websocket_session(
        self,
    ) -> Result<super::OpenAiWebSocketSession, LlmError> {
        super::OpenAiWebSocketSession::from_builder(self).await
    }

    /// Build a single-connection OpenAI WebSocket session with a custom recovery configuration.
    ///
    /// Requires the `openai-websocket` feature.
    #[cfg(feature = "openai-websocket")]
    pub async fn use_openai_websocket_session_with_recovery(
        self,
        recovery: super::OpenAiWebSocketRecoveryConfig,
    ) -> Result<super::OpenAiWebSocketSession, LlmError> {
        Ok(super::OpenAiWebSocketSession::from_builder(self)
            .await?
            .with_recovery_config(recovery))
    }

    /// Use the OpenAI Responses API instead of Chat Completions.
    ///
    /// When enabled, the client routes chat requests to `/responses` and sets
    /// the required request shape automatically.
    ///
    /// Note: this is implemented as a client-level default via `providerOptions.openai`,
    /// so a request can still override it.
    pub fn use_responses_api(mut self, enabled: bool) -> Self {
        self.use_responses_api = enabled;
        self
    }

    pub fn with_use_responses_api(mut self, enabled: bool) -> Self {
        self.use_responses_api = enabled;
        self
    }

    /// Set previous response id for Responses API chaining.
    pub fn responses_previous_response_id<S: Into<String>>(mut self, id: S) -> Self {
        self.responses_previous_response_id = Some(id.into());
        self
    }

    pub fn with_responses_previous_response_id<S: Into<String>>(mut self, id: S) -> Self {
        self.responses_previous_response_id = Some(id.into());
        self
    }

    /// Set OpenAI default provider options (open JSON object).
    ///
    /// These defaults are merged into each request's `provider_options_map` with
    /// "request overrides defaults" semantics.
    pub fn provider_options(mut self, options: serde_json::Value) -> Self {
        let mut overrides = ProviderOptionsMap::new();
        overrides.insert("openai", options);
        self.default_provider_options_map.merge_overrides(overrides);
        self
    }

    pub fn with_provider_options(mut self, options: serde_json::Value) -> Self {
        let mut overrides = ProviderOptionsMap::new();
        overrides.insert("openai", options);
        self.default_provider_options_map.merge_overrides(overrides);
        self
    }

    /// Merge typed default OpenAI provider options.
    pub fn with_openai_options(
        self,
        options: crate::provider_options::openai::OpenAiOptions,
    ) -> Self {
        self.with_provider_options(
            serde_json::to_value(options).expect("OpenAI options should serialize"),
        )
    }

    /// Replace the full provider options map (advanced usage).
    pub fn provider_options_map(mut self, map: ProviderOptionsMap) -> Self {
        self.default_provider_options_map = map;
        self
    }

    pub fn with_provider_options_map(mut self, map: ProviderOptionsMap) -> Self {
        self.default_provider_options_map = map;
        self
    }

    /// Switch into an OpenAI-compatible vendor builder (OpenAI-like providers).
    ///
    /// This keeps the "entry point is `openai`" mental model while using the OpenAI-compatible
    /// adapter system under the hood (SiliconFlow/DeepSeek/OpenRouter/etc.).
    ///
    /// The returned builder supports the same common params (`model`, `temperature`, ...)
    /// and retains HTTP settings/interceptors configured on this OpenAI builder.
    pub fn compatible(
        self,
        provider_id: impl AsRef<str>,
    ) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        let mut b = crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(
            self.core.base,
            provider_id.as_ref(),
        )
        .with_http_config(self.core.http_config)
        .http_debug(self.core.http_debug);

        if let Some(key) = self.api_key {
            b = b.api_key(key);
        }
        if let Some(url) = self.base_url {
            b = b.base_url(url);
        }

        // Common params
        if !self.common_params.model.is_empty() {
            b = b.model(self.common_params.model);
        }
        if let Some(t) = self.common_params.temperature {
            b = b.temperature(t);
        }
        if let Some(max_tokens) = self.common_params.max_tokens {
            b = b.max_tokens(max_tokens);
        }
        if let Some(top_p) = self.common_params.top_p {
            b = b.top_p(top_p);
        }
        if let Some(stops) = self.common_params.stop_sequences {
            b = b.stop(stops);
        }
        if let Some(seed) = self.common_params.seed {
            b = b.seed(seed);
        }

        // Retry + interceptors (builder-owned)
        if let Some(retry) = self.core.retry_options {
            b = b.with_retry(retry);
        }
        if let Some(transport) = self.core.http_transport {
            b = b.with_http_transport(transport);
        }
        for it in self.core.http_interceptors {
            b = b.with_http_interceptor(it);
        }

        b
    }

    /// Switch into an OpenAI-compatible vendor builder using a typed vendor id.
    pub fn vendor(
        self,
        vendor: OpenAiVendorId,
    ) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        self.compatible(vendor.as_str())
    }

    pub fn siliconflow(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        self.vendor(OpenAiVendorId::SILICONFLOW)
    }

    pub fn deepseek(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        self.vendor(OpenAiVendorId::DEEPSEEK)
    }

    pub fn openrouter(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        self.vendor(OpenAiVendorId::OPENROUTER)
    }

    pub fn together(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        self.vendor(OpenAiVendorId::TOGETHER)
    }

    pub fn fireworks(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        self.vendor(OpenAiVendorId::FIREWORKS)
    }

    pub fn github_copilot(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        self.vendor(OpenAiVendorId::GITHUB_COPILOT)
    }

    pub fn perplexity(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        self.vendor(OpenAiVendorId::PERPLEXITY)
    }

    pub fn mistral(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        self.vendor(OpenAiVendorId::MISTRAL)
    }

    pub fn jina(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        self.vendor(OpenAiVendorId::JINA)
    }

    pub fn voyageai(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        self.vendor(OpenAiVendorId::VOYAGEAI)
    }

    pub fn infini(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        self.vendor(OpenAiVendorId::INFINI)
    }

    // Note: Built-in tools should be configured via OpenAiOptions + hosted_tools::openai

    /// Convert the builder into the canonical OpenAI config.
    pub fn into_config(self) -> Result<crate::providers::openai::OpenAiConfig, LlmError> {
        let api_key = self
            .api_key
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .ok_or(LlmError::MissingApiKey(
                "OpenAI API key not provided".to_string(),
            ))?;

        let base_url = crate::utils::builder_helpers::resolve_base_url_with_env(
            self.base_url,
            Some("OPENAI_BASE_URL"),
            "https://api.openai.com/v1",
        );

        let model_id = self.common_params.model.clone();
        let http_interceptors = self.core.get_http_interceptors();
        let model_middlewares = self.core.get_auto_middlewares("openai", &model_id);
        let mut provider_options_map = self.default_provider_options_map;

        if self.use_responses_api {
            let mut openai_obj = serde_json::json!({ "responsesApi": { "enabled": true } });
            if let Some(id) = self.responses_previous_response_id
                && let Some(obj) = openai_obj
                    .get_mut("responsesApi")
                    .and_then(|v| v.as_object_mut())
            {
                obj.insert(
                    "previousResponseId".to_string(),
                    serde_json::Value::String(id),
                );
            }

            let mut overrides = ProviderOptionsMap::new();
            overrides.insert("openai", openai_obj);
            provider_options_map.merge_overrides(overrides);
        }

        Ok(crate::providers::openai::OpenAiConfig {
            api_key: secrecy::SecretString::from(api_key),
            base_url,
            organization: self.organization,
            project: self.project,
            common_params: self.common_params,
            openai_params: self.openai_params,
            provider_options_map,
            http_config: self.core.http_config.clone(),
            http_transport: self.core.http_transport.clone(),
            http_interceptors,
            model_middlewares,
        })
    }

    /// Builds the `OpenAI` client
    pub async fn build(self) -> Result<OpenAiClient, LlmError> {
        let http_client_override = self.core.base.http_client.clone();
        let tracing_config = self.core.tracing_config.clone();
        let retry_options = self.core.retry_options.clone();
        let config = self.into_config()?;

        let http_interceptors = config.http_interceptors.clone();
        let model_middlewares = config.model_middlewares.clone();

        let mut client = if let Some(http_client) = http_client_override {
            OpenAiClient::new(config, http_client)
                .with_http_interceptors(http_interceptors)
                .with_model_middlewares(model_middlewares)
        } else {
            OpenAiClient::from_config(config)?
        };

        if let Some(tracing_config) = tracing_config {
            client.set_tracing_config(Some(tracing_config));
        }
        if let Some(retry_options) = retry_options {
            client.set_retry_options(Some(retry_options));
        }

        Ok(client)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unsafe_code)]

    use super::*;
    use std::sync::{Arc, Mutex, MutexGuard};
    use std::time::Duration;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn lock_env() -> MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    struct EnvGuard {
        key: &'static str,
        previous: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var(key).ok();
            unsafe {
                std::env::set_var(key, value);
            }
            Self { key, previous }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => unsafe {
                    std::env::set_var(self.key, value);
                },
                None => unsafe {
                    std::env::remove_var(self.key);
                },
            }
        }
    }

    #[test]
    fn openai_builder_into_config_converges_on_openai_config() {
        let config = OpenAiBuilder::new(BuilderBase::default())
            .api_key("test-key")
            .base_url("https://example.com/v1")
            .organization("org-1")
            .project("proj-1")
            .model("gpt-4o-mini")
            .temperature(0.7)
            .max_tokens(256)
            .top_p(0.85)
            .stop_sequences(vec!["END".to_string()])
            .seed(99)
            .response_format(ResponseFormat::JsonObject)
            .tool_choice(ToolChoice::String("required".to_string()))
            .frequency_penalty(0.5)
            .presence_penalty(-0.25)
            .parallel_tool_calls(true)
            .responses_previous_response_id("resp_123")
            .provider_options(serde_json::json!({ "custom": { "enabled": true } }))
            .timeout(Duration::from_secs(12))
            .http_debug(true)
            .into_config()
            .expect("into_config ok");

        assert_eq!(config.base_url, "https://example.com/v1");
        assert_eq!(config.organization.as_deref(), Some("org-1"));
        assert_eq!(config.project.as_deref(), Some("proj-1"));
        assert_eq!(config.common_params.model, "gpt-4o-mini");
        assert_eq!(config.common_params.temperature, Some(0.7));
        assert_eq!(config.common_params.max_tokens, Some(256));
        assert_eq!(config.common_params.top_p, Some(0.85));
        assert_eq!(
            config.common_params.stop_sequences,
            Some(vec!["END".to_string()])
        );
        assert_eq!(config.common_params.seed, Some(99));
        assert!(matches!(
            config.openai_params.response_format,
            Some(ResponseFormat::JsonObject)
        ));
        assert!(matches!(
            config.openai_params.tool_choice,
            Some(ToolChoice::String(ref choice)) if choice == "required"
        ));
        assert_eq!(config.openai_params.frequency_penalty, Some(0.5));
        assert_eq!(config.openai_params.presence_penalty, Some(-0.25));
        assert_eq!(config.openai_params.parallel_tool_calls, Some(true));
        assert_eq!(config.http_config.timeout, Some(Duration::from_secs(12)));
        let openai_options = config
            .provider_options_map
            .get("openai")
            .expect("openai options");
        assert_eq!(
            openai_options["responsesApi"]["enabled"],
            serde_json::json!(true)
        );
        assert_eq!(
            openai_options["responsesApi"]["previousResponseId"],
            serde_json::json!("resp_123")
        );
        assert_eq!(openai_options["custom"]["enabled"], serde_json::json!(true));
        assert_eq!(config.http_interceptors.len(), 1);
    }

    #[test]
    fn openai_builder_into_config_matches_manual_openai_config() {
        let builder_config = OpenAiBuilder::new(BuilderBase::default())
            .api_key("test-key")
            .base_url("https://example.com/v1")
            .organization("org-1")
            .project("proj-1")
            .model("gpt-4o-mini")
            .temperature(0.7)
            .max_tokens(256)
            .top_p(0.85)
            .stop_sequences(vec!["END".to_string()])
            .seed(99)
            .response_format(ResponseFormat::JsonObject)
            .tool_choice(ToolChoice::String("required".to_string()))
            .frequency_penalty(0.5)
            .presence_penalty(-0.25)
            .parallel_tool_calls(true)
            .responses_previous_response_id("resp_123")
            .provider_options(serde_json::json!({ "custom": { "enabled": true } }))
            .timeout(Duration::from_secs(12))
            .http_debug(true)
            .into_config()
            .expect("builder config");

        let mut http_config = crate::types::HttpConfig::default();
        http_config.timeout = Some(Duration::from_secs(12));
        let manual_config = crate::providers::openai::OpenAiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_organization("org-1")
            .with_project("proj-1")
            .with_model("gpt-4o-mini")
            .with_temperature(0.7)
            .with_max_tokens(256)
            .with_top_p(0.85)
            .with_stop_sequences(vec!["END".to_string()])
            .with_seed(99)
            .with_response_format(ResponseFormat::JsonObject)
            .with_tool_choice(ToolChoice::String("required".to_string()))
            .with_frequency_penalty(0.5)
            .with_presence_penalty(-0.25)
            .with_parallel_tool_calls(true)
            .with_use_responses_api(true)
            .with_responses_previous_response_id("resp_123")
            .with_provider_options(serde_json::json!({ "custom": { "enabled": true } }))
            .with_http_config(http_config)
            .with_http_interceptors(vec![Arc::new(
                crate::execution::http::interceptor::LoggingInterceptor,
            )])
            .with_model_middlewares(crate::execution::middleware::build_auto_middlewares_vec(
                "openai",
                "gpt-4o-mini",
            ));

        assert_eq!(builder_config.base_url, manual_config.base_url);
        assert_eq!(builder_config.organization, manual_config.organization);
        assert_eq!(builder_config.project, manual_config.project);
        assert_eq!(
            builder_config.common_params.model,
            manual_config.common_params.model
        );
        assert_eq!(
            builder_config.common_params.temperature,
            manual_config.common_params.temperature
        );
        assert_eq!(
            builder_config.common_params.max_tokens,
            manual_config.common_params.max_tokens
        );
        assert_eq!(
            builder_config.common_params.top_p,
            manual_config.common_params.top_p
        );
        assert_eq!(
            builder_config.common_params.stop_sequences,
            manual_config.common_params.stop_sequences
        );
        assert_eq!(
            builder_config.common_params.seed,
            manual_config.common_params.seed
        );
        assert_eq!(
            serde_json::to_value(&builder_config.openai_params).expect("serialize builder params"),
            serde_json::to_value(&manual_config.openai_params).expect("serialize manual params")
        );
        assert_eq!(
            builder_config.http_config.timeout,
            manual_config.http_config.timeout
        );
        assert_eq!(
            builder_config.provider_options_map.get("openai"),
            manual_config.provider_options_map.get("openai")
        );
        assert_eq!(
            builder_config.http_interceptors.len(),
            manual_config.http_interceptors.len()
        );
        assert_eq!(
            builder_config.model_middlewares.len(),
            manual_config.model_middlewares.len()
        );
    }

    #[test]
    fn openai_builder_with_aliases_matches_primary_builder_surface() {
        let alias_config = OpenAiBuilder::new(BuilderBase::default())
            .api_key("test-key")
            .with_base_url("https://example.com/v1")
            .with_organization("org-1")
            .with_project("proj-1")
            .with_model("gpt-4o-mini")
            .with_temperature(0.7)
            .with_max_tokens(256)
            .with_top_p(0.85)
            .with_stop_sequences(vec!["END".to_string()])
            .with_seed(99)
            .with_response_format(ResponseFormat::JsonObject)
            .with_tool_choice(ToolChoice::String("required".to_string()))
            .with_frequency_penalty(0.5)
            .with_presence_penalty(-0.25)
            .with_parallel_tool_calls(true)
            .with_use_responses_api(true)
            .with_responses_previous_response_id("resp_123")
            .with_provider_options(serde_json::json!({ "custom": { "enabled": true } }))
            .with_http_config(crate::types::HttpConfig {
                timeout: Some(Duration::from_secs(12)),
                ..Default::default()
            })
            .into_config()
            .expect("alias config");

        let primary_config = OpenAiBuilder::new(BuilderBase::default())
            .api_key("test-key")
            .base_url("https://example.com/v1")
            .organization("org-1")
            .project("proj-1")
            .model("gpt-4o-mini")
            .temperature(0.7)
            .max_tokens(256)
            .top_p(0.85)
            .stop_sequences(vec!["END".to_string()])
            .seed(99)
            .response_format(ResponseFormat::JsonObject)
            .tool_choice(ToolChoice::String("required".to_string()))
            .frequency_penalty(0.5)
            .presence_penalty(-0.25)
            .parallel_tool_calls(true)
            .use_responses_api(true)
            .responses_previous_response_id("resp_123")
            .provider_options(serde_json::json!({ "custom": { "enabled": true } }))
            .timeout(Duration::from_secs(12))
            .into_config()
            .expect("primary config");

        assert_eq!(alias_config.base_url, primary_config.base_url);
        assert_eq!(alias_config.organization, primary_config.organization);
        assert_eq!(alias_config.project, primary_config.project);
        assert_eq!(
            alias_config.common_params.model,
            primary_config.common_params.model
        );
        assert_eq!(
            alias_config.common_params.temperature,
            primary_config.common_params.temperature
        );
        assert_eq!(
            alias_config.common_params.max_tokens,
            primary_config.common_params.max_tokens
        );
        assert_eq!(
            alias_config.common_params.top_p,
            primary_config.common_params.top_p
        );
        assert_eq!(
            alias_config.common_params.stop_sequences,
            primary_config.common_params.stop_sequences
        );
        assert_eq!(
            alias_config.common_params.seed,
            primary_config.common_params.seed
        );
        assert_eq!(
            serde_json::to_value(&alias_config.openai_params).expect("serialize alias params"),
            serde_json::to_value(&primary_config.openai_params).expect("serialize primary params")
        );
        assert_eq!(
            alias_config.provider_options_map.get("openai"),
            primary_config.provider_options_map.get("openai")
        );
    }

    #[test]
    fn openai_builder_into_config_reads_base_url_from_env() {
        let _lock = lock_env();
        let _guard = EnvGuard::set("OPENAI_BASE_URL", "https://example.com/env/v1/");

        let config = OpenAiBuilder::new(BuilderBase::default())
            .api_key("test-key")
            .into_config()
            .expect("into_config with env base url");

        assert_eq!(config.base_url, "https://example.com/env/v1");
    }
}
