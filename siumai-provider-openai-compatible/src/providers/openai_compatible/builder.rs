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
mod tests;
