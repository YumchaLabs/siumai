use crate::LlmError;
use crate::builder::BuilderBase;
use crate::execution::http::interceptor::{HttpInterceptor, LoggingInterceptor};
use crate::execution::http::transport::HttpTransport;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::providers::openai_compatible::{RequestBodyTransformer, ResponseMetadataExtractor};
use crate::retry_api::RetryOptions;
use crate::standards::openai::compat::provider_registry::ProviderConfig;
use std::collections::BTreeMap;
use std::sync::Arc;

mod reasoning;

/// OpenAI-compatible builder for configuring OpenAI-compatible providers.
///
/// Retry: call `.with_retry(RetryOptions::backoff())` to enable unified retry
/// for chat operations across OpenAI-compatible providers.
///
/// This unified builder supports all OpenAI-compatible providers (SiliconFlow, DeepSeek,
/// OpenRouter, etc.) using the adapter system for proper parameter handling and field mapping.
///
/// # Example
/// ```rust,no_run
/// use siumai::builder::LlmBuilder;
/// use std::time::Duration;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // SiliconFlow
///     let client = LlmBuilder::new()
///         .with_timeout(Duration::from_secs(60))
///         .siliconflow()
///         .api_key("your-api-key")
///         .model("deepseek-chat")
///         .temperature(0.7)
///         .build()
///         .await?;
///
///     Ok(())
/// }
/// ```
#[derive(Clone)]
pub struct OpenAiCompatibleBuilder {
    /// Base builder with HTTP configuration
    base: BuilderBase,
    /// Provider identifier (siliconflow, deepseek, openrouter, etc.)
    provider_id: String,
    /// Optional explicit provider config for generic OpenAI-compatible providers.
    provider_config_override: Option<ProviderConfig>,
    /// API key for the provider
    api_key: Option<String>,
    /// Custom base URL (overrides provider default)
    base_url: Option<String>,
    /// Common parameters
    common_params: crate::types::CommonParams,
    /// HTTP configuration
    http_config: crate::types::HttpConfig,
    /// Optional custom HTTP transport (Vercel-style "custom fetch" parity).
    http_transport: Option<Arc<dyn HttpTransport>>,
    /// Optional async Bearer token provider for auth flows that do not use static API keys.
    token_provider: Option<Arc<dyn crate::auth::TokenProvider>>,
    /// Provider-specific configuration
    provider_specific_config: std::collections::HashMap<String, serde_json::Value>,
    /// Unified retry options
    retry_options: Option<RetryOptions>,
    /// Optional HTTP interceptors applied to chat requests
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Additional model middlewares appended after provider auto-middlewares.
    extra_model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    /// Optional public response-metadata extractor, mirroring AI SDK's compat provider hook.
    response_metadata_extractor: Option<Arc<dyn ResponseMetadataExtractor>>,
    /// Optional provider-level URL query parameters appended to all compat request routes.
    query_params: BTreeMap<String, String>,
    /// Whether streaming chat requests should include `stream_options.include_usage`.
    include_usage: Option<bool>,
    /// Whether compat chat should keep JSON Schema structured outputs instead of downgrading to
    /// `response_format = { "type": "json_object" }`.
    supports_structured_outputs: Option<bool>,
    /// Optional public request-body transformer, mirroring AI SDK's `transformRequestBody`.
    request_body_transformer: Option<Arc<dyn RequestBodyTransformer>>,
    /// Explicit API-key auth requirement override.
    auth_required: Option<bool>,
    /// Enable lightweight HTTP debug logging interceptor
    http_debug: bool,
}

impl OpenAiCompatibleBuilder {
    /// Create a new OpenAI-compatible builder
    pub fn new(base: BuilderBase, provider_id: &str) -> Self {
        Self {
            base: base.clone(),
            provider_id: provider_id.to_string(),
            api_key: None,
            base_url: None,
            common_params: {
                let mut cp = crate::types::CommonParams::default();
                if let Some(m) =
                    crate::providers::openai_compatible::default_models::get_default_chat_model(
                        provider_id,
                    )
                {
                    cp.model = m.to_string();
                }
                cp
            },
            provider_config_override: None,
            http_config: crate::defaults::http::config_default(),
            http_transport: None,
            token_provider: None,
            provider_specific_config: std::collections::HashMap::new(),
            retry_options: None,
            // Inherit interceptors/debug from unified builder
            http_interceptors: base.http_interceptors.clone(),
            extra_model_middlewares: Vec::new(),
            response_metadata_extractor: None,
            query_params: BTreeMap::new(),
            include_usage: None,
            supports_structured_outputs: None,
            request_body_transformer: None,
            auth_required: None,
            http_debug: base.http_debug,
        }
    }

    /// Set the API key
    pub fn api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set a custom base URL
    ///
    /// This allows you to override the default provider base URL, which is useful for:
    /// - Self-deployed OpenAI-compatible servers
    /// - Providers with multiple service endpoints
    /// - Custom proxy or gateway configurations
    ///
    /// # Arguments
    /// * `base_url` - The custom base URL to use (e.g., `<https://my-server.com/v1>`)
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::prelude::*;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    ///     // Use a self-deployed OpenAI-compatible server
    ///     let client = LlmBuilder::new()
    ///         .deepseek()
    ///         .api_key("your-api-key")
    ///         .base_url("https://my-deepseek-server.com/v1")
    ///         .model("deepseek-chat")
    ///         .build()
    ///         .await?;
    ///
    ///     // Use an alternative endpoint for a provider
    ///     let client2 = LlmBuilder::new()
    ///         .openrouter()
    ///         .api_key("your-api-key")
    ///         .base_url("https://openrouter.ai/api/v1")
    ///         .model("openai/gpt-4")
    ///         .build()
    ///         .await?;
    /// #   Ok(())
    /// # }
    /// ```
    pub fn base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Install an explicit provider configuration.
    ///
    /// This is mainly used by the AI SDK-style generic `openai-compatible` package surface where
    /// `name` and `baseURL` define a provider that is intentionally not one of Siumai's built-in
    /// provider presets.
    pub fn with_provider_config(mut self, provider_config: ProviderConfig) -> Self {
        self.provider_id = provider_config.id.clone();
        self.provider_config_override = Some(provider_config);
        self
    }

    /// Control whether API-key style auth is required at client construction time.
    pub fn with_auth_required(mut self, required: bool) -> Self {
        self.auth_required = Some(required);
        self
    }

    /// Alias for `with_auth_required(...)`.
    pub fn auth_required(self, required: bool) -> Self {
        self.with_auth_required(required)
    }

    /// Set the model
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.common_params.model = model.into();
        self
    }

    /// Set temperature
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.common_params.temperature = Some(temperature);
        self
    }

    /// Set max tokens
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.common_params.max_tokens = Some(max_tokens);
        self
    }

    /// Set top_p
    pub fn top_p(mut self, top_p: f64) -> Self {
        self.common_params.top_p = Some(top_p);
        self
    }

    /// Set request timeout
    pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http_config.timeout = Some(timeout);
        self
    }

    /// Set connection timeout
    pub fn connect_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http_config.connect_timeout = Some(timeout);
        self
    }

    /// Control whether to disable compression for streaming (SSE) requests.
    pub fn http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.http_config.stream_disable_compression = disable;
        self
    }

    /// Set user agent
    pub fn user_agent<S: Into<String>>(mut self, user_agent: S) -> Self {
        self.http_config.user_agent = Some(user_agent.into());
        self
    }

    /// Set proxy URL
    pub fn proxy<S: Into<String>>(mut self, proxy: S) -> Self {
        self.http_config.proxy = Some(proxy.into());
        self
    }

    /// Set custom headers
    pub fn custom_headers(mut self, headers: std::collections::HashMap<String, String>) -> Self {
        self.http_config.headers = headers;
        self
    }

    /// Add a custom header
    pub fn header<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.http_config.headers.insert(key.into(), value.into());
        self
    }

    /// Set unified retry options for chat operations
    pub fn with_retry(mut self, options: RetryOptions) -> Self {
        self.retry_options = Some(options);
        self
    }

    /// Add a custom HTTP interceptor (builder collects and installs them on build).
    pub fn with_http_interceptor(mut self, interceptor: Arc<dyn HttpInterceptor>) -> Self {
        self.http_interceptors.push(interceptor);
        self
    }

    /// Enable a built-in logging interceptor for HTTP debugging (no sensitive data).
    pub fn http_debug(mut self, enabled: bool) -> Self {
        self.http_debug = enabled;
        self
    }

    /// Set stop sequences
    pub fn stop<S: Into<String>>(mut self, stop: Vec<S>) -> Self {
        self.common_params.stop_sequences = Some(stop.into_iter().map(|s| s.into()).collect());
        self
    }

    /// Set seed for deterministic outputs
    pub fn seed(mut self, seed: u64) -> Self {
        self.common_params.seed = Some(seed);
        self
    }

    /// Set HTTP configuration
    pub fn with_http_config(mut self, config: crate::types::HttpConfig) -> Self {
        self.http_config = config;
        self
    }

    /// Set custom HTTP client
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.base.http_client = Some(client);
        self
    }

    /// Set a custom HTTP transport (Vercel-style "custom fetch" parity).
    pub fn with_http_transport(mut self, transport: Arc<dyn HttpTransport>) -> Self {
        self.http_transport = Some(transport);
        self
    }

    /// Set an async Bearer token provider.
    pub fn with_token_provider(
        mut self,
        token_provider: Arc<dyn crate::auth::TokenProvider>,
    ) -> Self {
        self.token_provider = Some(token_provider);
        self
    }

    /// Alias for `with_token_provider(...)`.
    pub fn token_provider(self, token_provider: Arc<dyn crate::auth::TokenProvider>) -> Self {
        self.with_token_provider(token_provider)
    }

    /// Append extra model middlewares after provider auto-middlewares.
    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.extra_model_middlewares = middlewares;
        self
    }

    /// Install a public response-metadata extractor for this OpenAI-compatible provider.
    ///
    /// This mirrors AI SDK's `metadataExtractor` setting and is merged on top of the built-in
    /// provider adapter metadata policy.
    pub fn with_metadata_extractor(
        mut self,
        extractor: Arc<dyn ResponseMetadataExtractor>,
    ) -> Self {
        self.response_metadata_extractor = Some(extractor);
        self
    }

    /// Alias for `with_metadata_extractor(...)`.
    pub fn metadata_extractor(self, extractor: Arc<dyn ResponseMetadataExtractor>) -> Self {
        self.with_metadata_extractor(extractor)
    }

    /// Control whether streaming chat requests should include `stream_options.include_usage`.
    ///
    /// This mirrors AI SDK's `includeUsage` provider setting for `openai-compatible`.
    pub fn with_include_usage(mut self, include_usage: bool) -> Self {
        self.include_usage = Some(include_usage);
        self
    }

    /// Alias for `with_include_usage(...)`.
    pub fn include_usage(self, include_usage: bool) -> Self {
        self.with_include_usage(include_usage)
    }

    /// Replace the provider-level URL query parameter map.
    ///
    /// This mirrors AI SDK's `queryParams` provider setting for `openai-compatible`.
    pub fn with_query_params<K, V, I>(mut self, query_params: I) -> Self
    where
        K: Into<String>,
        V: Into<String>,
        I: IntoIterator<Item = (K, V)>,
    {
        self.query_params = query_params
            .into_iter()
            .map(|(key, value)| (key.into(), value.into()))
            .collect();
        self
    }

    /// Alias for `with_query_params(...)`.
    pub fn query_params<K, V, I>(self, query_params: I) -> Self
    where
        K: Into<String>,
        V: Into<String>,
        I: IntoIterator<Item = (K, V)>,
    {
        self.with_query_params(query_params)
    }

    /// Insert or replace a single provider-level URL query parameter.
    pub fn with_query_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.query_params.insert(key.into(), value.into());
        self
    }

    /// Alias for `with_query_param(...)`.
    pub fn query_param(self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.with_query_param(key, value)
    }

    /// Control whether compat chat keeps JSON Schema structured outputs.
    ///
    /// `false` mirrors AI SDK's `supportsStructuredOutputs = false` policy for providers that only
    /// accept `response_format = { \"type\": \"json_object\" }`.
    pub fn with_supports_structured_outputs(mut self, supports: bool) -> Self {
        self.supports_structured_outputs = Some(supports);
        self
    }

    /// Alias for `with_supports_structured_outputs(...)`.
    pub fn supports_structured_outputs(self, supports: bool) -> Self {
        self.with_supports_structured_outputs(supports)
    }

    /// Install a public request-body transformer for chat requests.
    ///
    /// This mirrors AI SDK's `transformRequestBody` provider setting.
    pub fn with_request_body_transformer(
        mut self,
        transformer: Arc<dyn RequestBodyTransformer>,
    ) -> Self {
        self.request_body_transformer = Some(transformer);
        self
    }

    /// Alias for `with_request_body_transformer(...)`.
    pub fn request_body_transformer(self, transformer: Arc<dyn RequestBodyTransformer>) -> Self {
        self.with_request_body_transformer(transformer)
    }

    /// Alias for `with_http_transport(...)` (Vercel-aligned: `fetch`).
    pub fn fetch(self, transport: Arc<dyn HttpTransport>) -> Self {
        self.with_http_transport(transport)
    }

    /// Convert the builder into the canonical OpenAI-compatible config.
    pub fn into_config(
        self,
    ) -> Result<crate::providers::openai_compatible::OpenAiCompatibleConfig, LlmError> {
        let (provider_config, is_synthetic_generic_provider) = if let Some(provider_config) =
            self.provider_config_override
        {
            (provider_config, false)
        } else if let Some(provider_config) =
            crate::providers::openai_compatible::config::get_provider_config(&self.provider_id)
        {
            (provider_config, false)
        } else if let Some(base_url) = self.base_url.clone() {
            (
                crate::providers::openai_compatible::config::generic_provider_config(
                    &self.provider_id,
                    &self.provider_id,
                    &base_url,
                ),
                true,
            )
        } else {
            return Err(LlmError::ConfigurationError(format!(
                "Unknown OpenAI-compatible provider id: {}; provide `base_url` for a generic OpenAI-compatible provider",
                self.provider_id
            )));
        };
        let auth_required = self.auth_required.unwrap_or(!is_synthetic_generic_provider);

        fn headers_have_authorization(headers: &std::collections::HashMap<String, String>) -> bool {
            headers
                .keys()
                .any(|key| key.eq_ignore_ascii_case("authorization"))
        }

        let has_external_auth = self.token_provider.is_some()
            || headers_have_authorization(&self.http_config.headers)
            || headers_have_authorization(&self.base.default_headers);
        let allow_empty_api_key =
            has_external_auth || provider_config.id == "vertex-maas" || !auth_required;
        let api_key = if allow_empty_api_key {
            self.api_key.unwrap_or_default()
        } else {
            crate::utils::builder_helpers::get_api_key_with_envs(
                self.api_key,
                &self.provider_id,
                provider_config.api_key_env.as_deref(),
                &provider_config.api_key_env_aliases,
            )?
        };
        let canonical_provider_id = provider_config.id.clone();
        let mut adapter: Box<dyn crate::providers::openai_compatible::ProviderAdapter> = Box::new(
            crate::standards::openai::compat::provider_registry::ConfigurableAdapter::new(
                provider_config,
            ),
        );
        if !self.provider_specific_config.is_empty() {
            adapter = Box::new(
                crate::standards::openai::compat::adapter::ParamMergingAdapter::new(
                    adapter,
                    self.provider_specific_config,
                ),
            );
        }
        let adapter: Arc<dyn crate::providers::openai_compatible::ProviderAdapter> =
            Arc::from(adapter);

        if canonical_provider_id == "vertex-maas" && self.base_url.is_none() {
            return Err(LlmError::ConfigurationError(
                "Google Vertex MaaS requires `base_url` on the OpenAI-compatible builder; use GoogleVertexMaasProviderSettings with project/location or the unified Provider::vertex_maas() builder for project/location construction".to_string(),
            ));
        }

        let base_url = crate::utils::builder_helpers::resolve_base_url(
            self.base_url.clone(),
            adapter.base_url(),
        );

        let mut config = crate::providers::openai_compatible::OpenAiCompatibleConfig::new(
            &canonical_provider_id,
            &api_key,
            &base_url,
            adapter,
        );

        let effective_model_raw = crate::utils::builder_helpers::get_effective_model(
            &self.common_params.model,
            &canonical_provider_id,
        );
        let effective_model = crate::providers::openai_compatible::normalize_model_id(
            &canonical_provider_id,
            &effective_model_raw,
        );

        if !effective_model.is_empty() {
            config = config.with_model(&effective_model);
        }

        config = config.with_common_params(self.common_params);
        if let Some(extractor) = self.response_metadata_extractor.clone() {
            config = config.with_metadata_extractor(extractor);
        }
        if !self.query_params.is_empty() {
            config = config.with_query_params(self.query_params);
        }
        let include_usage = self.include_usage.or_else(|| {
            matches!(
                canonical_provider_id.as_str(),
                "alibaba" | "deepseek" | "moonshotai" | "qwen" | "xai"
            )
            .then_some(true)
        });
        if let Some(include_usage) = include_usage {
            config = config.with_include_usage(include_usage);
        }
        if let Some(supports) = self.supports_structured_outputs {
            config = config.with_supports_structured_outputs(supports);
        }
        if let Some(transformer) = self.request_body_transformer.clone() {
            config = config.with_request_body_transformer(transformer);
        }
        config = config.with_auth_required(auth_required);

        let mut final_http_config = self.http_config;
        if let Some(timeout) = self.base.timeout {
            final_http_config.timeout = Some(timeout);
        }
        if let Some(connect_timeout) = self.base.connect_timeout {
            final_http_config.connect_timeout = Some(connect_timeout);
        }
        if let Some(user_agent) = self.base.user_agent {
            final_http_config.user_agent = Some(user_agent);
        }
        if let Some(proxy) = self.base.proxy {
            final_http_config.proxy = Some(proxy);
        }
        for (key, value) in self.base.default_headers {
            final_http_config.headers.insert(key, value);
        }

        config = config.with_http_config(final_http_config);
        if let Some(transport) = self.http_transport.clone() {
            config = config.with_http_transport(transport);
        }
        if let Some(token_provider) = self.token_provider.clone() {
            config = config.with_token_provider(token_provider);
        }

        let model_id = config.model.clone();
        let mut interceptors = self.http_interceptors;
        if self.http_debug {
            interceptors.push(Arc::new(LoggingInterceptor));
        }
        let mut middlewares =
            crate::execution::middleware::build_auto_middlewares_vec(&self.provider_id, &model_id);
        middlewares.extend(self.extra_model_middlewares);

        Ok(config
            .with_http_interceptors(interceptors)
            .with_model_middlewares(middlewares))
    }

    /// Build the OpenAI-compatible client
    pub async fn build(
        self,
    ) -> Result<crate::providers::openai_compatible::OpenAiCompatibleClient, LlmError> {
        let http_client_override = self.base.http_client.clone();
        let retry_options = self.retry_options.clone();
        let config = self.into_config()?;

        let mut client = if let Some(http_client) = http_client_override {
            let http_interceptors = config.http_interceptors.clone();
            let model_middlewares = config.model_middlewares.clone();
            crate::providers::openai_compatible::OpenAiCompatibleClient::with_http_client(
                config,
                http_client,
            )
            .await?
            .with_http_interceptors(http_interceptors)
            .with_model_middlewares(model_middlewares)
        } else {
            crate::providers::openai_compatible::OpenAiCompatibleClient::from_config(config).await?
        };

        client.set_retry_options(retry_options);
        Ok(client)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::middleware::language_model::LanguageModelMiddleware;
    use std::sync::Arc;
    use std::time::Duration;

    #[derive(Clone, Default)]
    struct NoopMiddleware;

    impl LanguageModelMiddleware for NoopMiddleware {}

    #[test]
    fn builder_shell_keeps_provider_reasoning_mapping_split() {
        let source = include_str!("builder.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap_or_default();
        let reasoning = include_str!("builder/reasoning.rs");

        assert!(
            source.contains("mod reasoning;"),
            "OpenAI-compatible builder shell should keep provider reasoning mapping in a dedicated module"
        );

        for marker in [
            "fn provider_thinking_value",
            "pub fn with_thinking(",
            "pub fn with_thinking_budget(",
            "pub fn reasoning(",
            "pub fn reasoning_budget(",
        ] {
            assert!(
                !source.contains(marker),
                "OpenAI-compatible builder shell should not own `{marker}`"
            );
            assert!(
                reasoning.contains(marker),
                "builder reasoning module should own `{marker}`"
            );
        }
    }

    #[test]
    fn openai_compatible_builder_into_config_converges() {
        let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
            .api_key("test-key")
            .model("deepseek-chat")
            .temperature(0.4)
            .max_tokens(256)
            .top_p(0.9)
            .stop(vec!["END"])
            .seed(7)
            .reasoning(true)
            .reasoning_budget(2048)
            .timeout(Duration::from_secs(15))
            .connect_timeout(Duration::from_secs(5))
            .http_stream_disable_compression(true)
            .user_agent("siumai-test/1.0")
            .proxy("http://127.0.0.1:8080")
            .custom_headers(std::collections::HashMap::from([(
                "x-one".to_string(),
                "1".to_string(),
            )]))
            .header("x-two", "2")
            .with_http_interceptor(Arc::new(
                crate::execution::http::interceptor::LoggingInterceptor,
            ))
            .http_debug(true)
            .into_config()
            .expect("into_config ok");

        assert_eq!(config.provider_id, "deepseek");
        assert_eq!(config.model, "deepseek-chat");
        assert_eq!(config.common_params.temperature, Some(0.4));
        assert_eq!(config.common_params.max_tokens, Some(256));
        assert_eq!(config.common_params.top_p, Some(0.9));
        assert_eq!(
            config.common_params.stop_sequences,
            Some(vec!["END".to_string()])
        );
        assert_eq!(config.common_params.seed, Some(7));
        let mut params = serde_json::json!({});
        config
            .adapter
            .transform_request_params(
                &mut params,
                &config.model,
                crate::providers::openai_compatible::RequestType::Chat,
            )
            .expect("transform request params");
        assert_eq!(
            params["thinking"],
            serde_json::json!({
                "type": "enabled"
            })
        );
        assert!(params.get("enable_reasoning").is_none());
        assert!(params.get("reasoning_budget").is_none());
        assert_eq!(config.http_config.timeout, Some(Duration::from_secs(15)));
        assert_eq!(
            config.http_config.connect_timeout,
            Some(Duration::from_secs(5))
        );
        assert!(config.http_config.stream_disable_compression);
        assert_eq!(
            config.http_config.user_agent.as_deref(),
            Some("siumai-test/1.0")
        );
        assert_eq!(
            config.http_config.proxy.as_deref(),
            Some("http://127.0.0.1:8080")
        );
        assert_eq!(
            config.http_config.headers.get("x-one"),
            Some(&"1".to_string())
        );
        assert_eq!(
            config.http_config.headers.get("x-two"),
            Some(&"2".to_string())
        );
        assert_eq!(config.http_interceptors.len(), 2);
    }

    #[test]
    fn openai_compatible_builder_maps_qwen_reasoning_to_alibaba_thinking_fields() {
        let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "qwen")
            .api_key("test-key")
            .model("qwen-plus")
            .reasoning(true)
            .reasoning_budget(2048)
            .into_config()
            .expect("into_config ok");

        assert_eq!(config.provider_id, "qwen");

        let mut params = serde_json::json!({});
        config
            .adapter
            .transform_request_params(
                &mut params,
                &config.model,
                crate::providers::openai_compatible::RequestType::Chat,
            )
            .expect("transform request params");

        assert_eq!(params["enable_thinking"], serde_json::json!(true));
        assert_eq!(params["thinking_budget"], serde_json::json!(2048));
        assert!(params.get("enable_reasoning").is_none());
        assert!(params.get("reasoning_budget").is_none());
    }

    #[test]
    fn openai_compatible_builder_maps_xai_reasoning_to_effort() {
        let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "xai")
            .api_key("test-key")
            .model("grok-4")
            .reasoning(true)
            .reasoning_budget(2048)
            .into_config()
            .expect("into_config ok");

        assert_eq!(config.provider_id, "xai");

        let mut params = serde_json::json!({});
        config
            .adapter
            .transform_request_params(
                &mut params,
                &config.model,
                crate::providers::openai_compatible::RequestType::Chat,
            )
            .expect("transform request params");

        assert_eq!(params["reasoning_effort"], serde_json::json!("high"));
        assert!(params.get("enable_reasoning").is_none());
        assert!(params.get("reasoning_budget").is_none());
        assert!(params.get("enable_thinking").is_none());
        assert!(params.get("thinking_budget").is_none());
    }

    #[test]
    fn openai_compatible_builder_allows_external_authorization_without_api_key() {
        let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
            .model("deepseek-chat")
            .header("Authorization", "Bearer external-token")
            .into_config()
            .expect("authorization header should satisfy compat auth");

        assert_eq!(config.provider_id, "deepseek");
        assert!(config.api_key.is_empty());
        assert_eq!(
            config
                .http_config
                .headers
                .get("Authorization")
                .map(String::as_str),
            Some("Bearer external-token")
        );
    }

    #[test]
    fn openai_compatible_builder_carries_token_provider_without_api_key() {
        let token_provider = Arc::new(crate::auth::StaticTokenProvider::new("test-token"));
        let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
            .model("deepseek-chat")
            .with_token_provider(token_provider)
            .into_config()
            .expect("token provider should satisfy compat auth");

        assert_eq!(config.provider_id, "deepseek");
        assert!(config.api_key.is_empty());
        assert!(config.token_provider.is_some());
    }

    #[test]
    fn openai_compatible_builder_generic_provider_defaults_to_optional_auth() {
        let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "local-gateway")
            .base_url("http://localhost:11434/v1")
            .model("llama3.2")
            .into_config()
            .expect("generic compat config");

        assert_eq!(config.provider_id, "local-gateway");
        assert!(config.api_key.is_empty());
        assert!(!config.auth_required);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn openai_compatible_builder_generic_provider_honors_explicit_auth_requirement() {
        let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "private-gateway")
            .base_url("https://gateway.example.com/v1")
            .model("gateway-model")
            .with_auth_required(true)
            .header("Authorization", "Bearer external-token")
            .into_config()
            .expect("generic compat config with explicit auth");

        assert_eq!(config.provider_id, "private-gateway");
        assert!(config.api_key.is_empty());
        assert!(config.auth_required);
        assert_eq!(
            config
                .http_config
                .headers
                .get("Authorization")
                .map(String::as_str),
            Some("Bearer external-token")
        );
        assert!(config.validate().is_ok());
    }

    #[test]
    fn openai_compatible_builder_installs_metadata_extractor() {
        let extractor: Arc<dyn ResponseMetadataExtractor> = Arc::new(|raw: &serde_json::Value| {
            raw.get("test_field").map(|value| {
                std::collections::HashMap::from([(
                    "test-provider".to_string(),
                    serde_json::json!({ "value": value }),
                )])
            })
        });

        let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
            .api_key("test-key")
            .model("deepseek-chat")
            .with_metadata_extractor(extractor)
            .into_config()
            .expect("into_config ok");

        let metadata = config
            .adapter
            .extract_response_provider_metadata(&serde_json::json!({
                "test_field": "test-value"
            }))
            .expect("metadata");
        let provider = metadata.get("test-provider").expect("provider metadata");
        assert_eq!(
            provider.get("value"),
            Some(&serde_json::json!("test-value"))
        );
    }

    #[test]
    fn openai_compatible_builder_records_include_usage_setting() {
        let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
            .api_key("test-key")
            .model("deepseek-chat")
            .with_include_usage(true)
            .into_config()
            .expect("into_config ok");

        assert_eq!(config.include_usage, Some(true));
    }

    #[test]
    fn openai_compatible_builder_defaults_ai_sdk_usage_streaming_providers() {
        for provider_id in ["alibaba", "deepseek", "moonshotai", "qwen", "xai"] {
            let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), provider_id)
                .api_key("test-key")
                .model("test-model")
                .into_config()
                .expect("into_config ok");

            assert_eq!(config.include_usage, Some(true), "{provider_id}");
        }
    }

    #[test]
    fn openai_compatible_builder_respects_explicit_defaulted_include_usage() {
        for provider_id in ["alibaba", "deepseek", "moonshotai", "qwen", "xai"] {
            let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), provider_id)
                .api_key("test-key")
                .model("test-model")
                .with_include_usage(false)
                .into_config()
                .expect("into_config ok");

            assert_eq!(config.include_usage, Some(false), "{provider_id}");
        }
    }

    #[test]
    fn openai_compatible_builder_records_query_params_setting() {
        let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
            .api_key("test-key")
            .model("deepseek-chat")
            .with_query_params([("api-version", "2025-04-01"), ("tenant", "acme")])
            .into_config()
            .expect("into_config ok");

        assert_eq!(
            config.query_params.get("api-version").map(String::as_str),
            Some("2025-04-01")
        );
        assert_eq!(
            config.query_params.get("tenant").map(String::as_str),
            Some("acme")
        );
    }

    #[test]
    fn openai_compatible_builder_records_structured_outputs_policy() {
        let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
            .api_key("test-key")
            .model("deepseek-chat")
            .with_supports_structured_outputs(false)
            .into_config()
            .expect("into_config ok");

        assert_eq!(config.supports_structured_outputs, Some(false));
    }

    #[test]
    fn openai_compatible_builder_installs_request_body_transformer() {
        let transformer: Arc<dyn RequestBodyTransformer> = Arc::new(
            |body: &mut serde_json::Value,
             _model: &str,
             request_type: crate::providers::openai_compatible::RequestType| {
                assert!(matches!(
                    request_type,
                    crate::providers::openai_compatible::RequestType::Chat
                ));
                body["custom"] = serde_json::json!(true);
                Ok(())
            },
        );

        let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
            .api_key("test-key")
            .model("deepseek-chat")
            .with_request_body_transformer(transformer)
            .into_config()
            .expect("into_config ok");

        let hook = config
            .request_body_transformer
            .as_ref()
            .expect("request body transformer");
        let mut body = serde_json::json!({});
        hook.transform_request_body(
            &mut body,
            "deepseek-chat",
            crate::providers::openai_compatible::RequestType::Chat,
        )
        .expect("transform body");
        assert_eq!(body.get("custom"), Some(&serde_json::json!(true)));
    }

    #[test]
    fn openai_compatible_builder_into_config_matches_manual_compatible_config() {
        let builder_config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
            .api_key("test-key")
            .model("deepseek-chat")
            .temperature(0.4)
            .max_tokens(256)
            .top_p(0.9)
            .stop(vec!["END"])
            .seed(7)
            .reasoning(true)
            .reasoning_budget(2048)
            .timeout(Duration::from_secs(15))
            .connect_timeout(Duration::from_secs(5))
            .http_stream_disable_compression(true)
            .user_agent("siumai-test/1.0")
            .proxy("http://127.0.0.1:8080")
            .custom_headers(std::collections::HashMap::from([(
                "x-one".to_string(),
                "1".to_string(),
            )]))
            .header("x-two", "2")
            .with_http_interceptor(Arc::new(
                crate::execution::http::interceptor::LoggingInterceptor,
            ))
            .http_debug(true)
            .with_model_middlewares(vec![Arc::new(NoopMiddleware)])
            .into_config()
            .expect("builder config");

        let provider = crate::providers::openai_compatible::get_provider_config("deepseek")
            .expect("provider config");
        let adapter =
            Arc::new(crate::providers::openai_compatible::ConfigurableAdapter::new(provider));
        let manual_config = crate::providers::openai_compatible::OpenAiCompatibleConfig::new(
            "deepseek",
            "test-key",
            "https://api.deepseek.com",
            adapter,
        )
        .with_model("deepseek-chat")
        .with_temperature(0.4)
        .with_max_tokens(256)
        .with_top_p(0.9)
        .with_stop_sequences(vec!["END".to_string()])
        .with_seed(7)
        .with_reasoning(true)
        .with_reasoning_budget(2048)
        .with_timeout(Duration::from_secs(15))
        .with_connect_timeout(Duration::from_secs(5))
        .with_http_stream_disable_compression(true)
        .with_user_agent("siumai-test/1.0")
        .with_proxy("http://127.0.0.1:8080")
        .with_http_headers(std::collections::HashMap::from([(
            "x-one".to_string(),
            "1".to_string(),
        )]))
        .with_http_header("x-two", "2")
        .with_http_interceptor(Arc::new(
            crate::execution::http::interceptor::LoggingInterceptor,
        ))
        .with_http_interceptor(Arc::new(
            crate::execution::http::interceptor::LoggingInterceptor,
        ))
        .with_model_middlewares({
            let mut middlewares = crate::execution::middleware::build_auto_middlewares_vec(
                "deepseek",
                "deepseek-chat",
            );
            middlewares.push(Arc::new(NoopMiddleware));
            middlewares
        });

        assert_eq!(builder_config.provider_id, manual_config.provider_id);
        assert_eq!(builder_config.base_url, manual_config.base_url);
        assert_eq!(builder_config.model, manual_config.model);
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
        let mut builder_params = serde_json::json!({});
        builder_config
            .adapter
            .transform_request_params(
                &mut builder_params,
                &builder_config.model,
                crate::providers::openai_compatible::RequestType::Chat,
            )
            .expect("builder transform request params");
        let mut manual_params = serde_json::json!({});
        manual_config
            .adapter
            .transform_request_params(
                &mut manual_params,
                &manual_config.model,
                crate::providers::openai_compatible::RequestType::Chat,
            )
            .expect("manual transform request params");
        assert_eq!(builder_params, manual_params);
        assert_eq!(
            builder_config.http_config.timeout,
            manual_config.http_config.timeout
        );
        assert_eq!(
            builder_config.http_config.connect_timeout,
            manual_config.http_config.connect_timeout
        );
        assert_eq!(
            builder_config.http_config.stream_disable_compression,
            manual_config.http_config.stream_disable_compression
        );
        assert_eq!(
            builder_config.http_config.user_agent,
            manual_config.http_config.user_agent
        );
        assert_eq!(
            builder_config.http_config.proxy,
            manual_config.http_config.proxy
        );
        assert_eq!(
            builder_config.http_config.headers,
            manual_config.http_config.headers
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

    #[tokio::test]
    async fn openai_compatible_builder_build_preserves_http_client_override_and_retry_options() {
        let client = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
            .api_key("test-key")
            .model("deepseek-chat")
            .with_http_config(crate::types::HttpConfig {
                proxy: Some("not-a-url".to_string()),
                ..Default::default()
            })
            .with_http_client(reqwest::Client::new())
            .with_retry(RetryOptions::default())
            .build()
            .await
            .expect("build client with explicit http client");

        assert!(client.retry_options().is_some());
    }

    #[test]
    fn openai_compatible_builder_falls_back_to_provider_config_default_model() {
        let mistral = OpenAiCompatibleBuilder::new(BuilderBase::default(), "mistral")
            .api_key("test-key")
            .into_config()
            .expect("mistral config");
        let cohere = OpenAiCompatibleBuilder::new(BuilderBase::default(), "cohere")
            .api_key("test-key")
            .into_config()
            .expect("cohere config");

        assert_eq!(mistral.model, "mistral-large-latest");
        assert_eq!(mistral.common_params.model, mistral.model);
        assert_eq!(cohere.model, "command-r-plus");
        assert_eq!(cohere.common_params.model, cohere.model);
    }
}
