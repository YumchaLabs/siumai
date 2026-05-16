use std::sync::Arc;

use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::http::transport::HttpTransport;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::retry_api::RetryOptions;

/// Options for creating a provider registry handle.
#[derive(Default, Clone)]
pub struct ProviderBuildOverrides {
    /// Optional pre-built HTTP client override.
    pub http_client: Option<reqwest::Client>,
    /// Optional custom HTTP transport override.
    pub http_transport: Option<Arc<dyn HttpTransport>>,
    /// Optional HTTP configuration override.
    pub http_config: Option<crate::types::HttpConfig>,
    /// Optional API key override.
    pub api_key: Option<String>,
    /// Optional base URL override.
    pub base_url: Option<String>,
    /// Optional unified reasoning enable flag for registry-built language models.
    pub reasoning_enabled: Option<bool>,
    /// Optional unified reasoning budget for registry-built language models.
    pub reasoning_budget: Option<i32>,
}

impl ProviderBuildOverrides {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn api_key<S: Into<String>>(api_key: S) -> Self {
        Self::new().with_api_key(api_key)
    }

    pub fn base_url<S: Into<String>>(base_url: S) -> Self {
        Self::new().with_base_url(base_url)
    }

    pub fn api_key_base_url<K: Into<String>, U: Into<String>>(api_key: K, base_url: U) -> Self {
        Self::api_key(api_key).with_base_url(base_url)
    }

    pub fn http_transport(transport: Arc<dyn HttpTransport>) -> Self {
        Self::new().with_http_transport(transport)
    }

    pub fn api_key_fetch<S: Into<String>>(api_key: S, transport: Arc<dyn HttpTransport>) -> Self {
        Self::api_key(api_key).with_http_transport(transport)
    }

    pub fn api_key_base_url_fetch<K: Into<String>, U: Into<String>>(
        api_key: K,
        base_url: U,
        transport: Arc<dyn HttpTransport>,
    ) -> Self {
        Self::api_key_base_url(api_key, base_url).with_http_transport(transport)
    }

    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = Some(client);
        self
    }

    pub fn with_http_transport(mut self, transport: Arc<dyn HttpTransport>) -> Self {
        self.http_transport = Some(transport);
        self
    }

    pub fn with_http_config(mut self, config: crate::types::HttpConfig) -> Self {
        self.http_config = Some(config);
        self
    }

    pub fn fetch(self, transport: Arc<dyn HttpTransport>) -> Self {
        self.with_http_transport(transport)
    }

    pub fn with_fetch(self, transport: Arc<dyn HttpTransport>) -> Self {
        self.fetch(transport)
    }

    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn with_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub fn with_reasoning(mut self, enabled: bool) -> Self {
        self.reasoning_enabled = Some(enabled);
        self
    }

    pub fn with_reasoning_budget(mut self, budget: i32) -> Self {
        self.reasoning_budget = Some(budget);
        self
    }

    pub(crate) fn merged_with(&self, provider_override: Option<&ProviderBuildOverrides>) -> Self {
        if let Some(provider_override) = provider_override {
            Self {
                http_client: provider_override
                    .http_client
                    .clone()
                    .or_else(|| self.http_client.clone()),
                http_transport: provider_override
                    .http_transport
                    .clone()
                    .or_else(|| self.http_transport.clone()),
                http_config: merge_http_config(
                    self.http_config.as_ref(),
                    provider_override.http_config.as_ref(),
                ),
                api_key: provider_override
                    .api_key
                    .clone()
                    .or_else(|| self.api_key.clone()),
                base_url: provider_override
                    .base_url
                    .clone()
                    .or_else(|| self.base_url.clone()),
                reasoning_enabled: provider_override
                    .reasoning_enabled
                    .or(self.reasoning_enabled),
                reasoning_budget: provider_override.reasoning_budget.or(self.reasoning_budget),
            }
        } else {
            self.clone()
        }
    }
}

fn merge_http_config(
    base: Option<&crate::types::HttpConfig>,
    provider_override: Option<&crate::types::HttpConfig>,
) -> Option<crate::types::HttpConfig> {
    match (base, provider_override) {
        (None, None) => None,
        (Some(base), None) => Some(base.clone()),
        (None, Some(provider_override)) => Some(provider_override.clone()),
        (Some(base), Some(provider_override)) => {
            let mut merged = base.clone();
            if provider_override.timeout.is_some() {
                merged.timeout = provider_override.timeout;
            }
            if provider_override.connect_timeout.is_some() {
                merged.connect_timeout = provider_override.connect_timeout;
            }
            merged.headers.extend(provider_override.headers.clone());
            if provider_override.proxy.is_some() {
                merged.proxy = provider_override.proxy.clone();
            }
            if provider_override.user_agent.is_some() {
                merged.user_agent = provider_override.user_agent.clone();
            }
            merged.stream_disable_compression = provider_override.stream_disable_compression;
            Some(merged)
        }
    }
}

/// Build-time context for ProviderFactory client construction.
///
/// This struct carries all cross-cutting configuration needed to build
/// concrete provider clients in a unified way (HTTP config, auth, tracing,
/// interceptors, retry options, etc.).
#[derive(Default, Clone)]
pub struct BuildContext {
    /// HTTP interceptors applied at the registry / builder level.
    pub http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Unified retry options (optional).
    pub retry_options: Option<RetryOptions>,
    /// Optional model-level middlewares (applied before provider mapping).
    pub model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    /// Optional pre-built HTTP client. When present, factories should prefer
    /// this client over constructing a new one from `http_config`.
    pub http_client: Option<reqwest::Client>,
    /// Optional custom HTTP transport (Vercel-style "custom fetch" parity).
    pub http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    /// Optional HTTP configuration (timeouts, headers, proxy, user-agent, etc.).
    /// When no custom client is supplied, factories may use this to build one.
    pub http_config: Option<crate::types::HttpConfig>,
    /// Optional API key override. When `None`, factories may fall back to
    /// environment variables or other defaults.
    pub api_key: Option<String>,
    /// Optional base URL override for the provider.
    pub base_url: Option<String>,
    /// Optional organization identifier (e.g., OpenAI `organization`).
    pub organization: Option<String>,
    /// Optional project identifier (e.g., OpenAI `project`).
    pub project: Option<String>,
    /// Optional location/region identifier (e.g., Google Vertex `location`).
    pub location: Option<String>,
    /// Optional tracing configuration for providers that support it.
    pub tracing_config: Option<crate::observability::tracing::TracingConfig>,
    /// Optional Google-family token provider (Gemini / Vertex / Anthropic-on-Vertex).
    #[cfg(any(feature = "google", feature = "google-vertex"))]
    pub google_token_provider: Option<std::sync::Arc<dyn crate::auth::TokenProvider>>,
    /// Backward-compatible alias for older Gemini-focused call sites.
    #[cfg(any(feature = "google", feature = "google-vertex"))]
    pub gemini_token_provider: Option<std::sync::Arc<dyn crate::auth::TokenProvider>>,
    /// Optional common parameters (model id, temperature, max_tokens, etc.).
    /// When `None`, factories may construct minimal defaults based on `model_id`.
    pub common_params: Option<crate::types::CommonParams>,
    /// Optional unified reasoning enable flag propagated from top-level builders.
    pub reasoning_enabled: Option<bool>,
    /// Optional unified reasoning budget propagated from top-level builders.
    pub reasoning_budget: Option<i32>,
    /// Optional canonical provider id override (for adapter-style providers).
    pub provider_id: Option<String>,
}

impl BuildContext {
    #[cfg(any(feature = "google", feature = "google-vertex"))]
    pub fn resolved_google_token_provider(
        &self,
    ) -> Option<std::sync::Arc<dyn crate::auth::TokenProvider>> {
        self.google_token_provider
            .clone()
            .or_else(|| self.gemini_token_provider.clone())
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_registry_context(
    provider_id: &str,
    http_interceptors: &[Arc<dyn HttpInterceptor>],
    retry_options: &Option<RetryOptions>,
    http_client: &Option<reqwest::Client>,
    http_transport: &Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    http_config: &Option<crate::types::HttpConfig>,
    api_key: &Option<String>,
    base_url: &Option<String>,
    reasoning_enabled: Option<bool>,
    reasoning_budget: Option<i32>,
) -> BuildContext {
    BuildContext {
        http_interceptors: http_interceptors.to_vec(),
        retry_options: retry_options.clone(),
        http_client: http_client.clone(),
        http_transport: http_transport.clone(),
        http_config: http_config.clone(),
        api_key: api_key.clone(),
        base_url: base_url.clone(),
        reasoning_enabled,
        reasoning_budget,
        provider_id: Some(provider_id.to_string()),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::ProviderBuildOverrides;
    use crate::types::HttpConfig;
    use std::time::Duration;

    #[test]
    fn provider_build_overrides_constructors_match_fluent_chain() {
        let api_key_base_url =
            ProviderBuildOverrides::api_key_base_url("test-key", "https://example.com/custom");
        assert_eq!(api_key_base_url.api_key.as_deref(), Some("test-key"));
        assert_eq!(
            api_key_base_url.base_url.as_deref(),
            Some("https://example.com/custom")
        );

        let api_key_only = ProviderBuildOverrides::api_key("test-key");
        assert_eq!(api_key_only.api_key.as_deref(), Some("test-key"));
        assert!(api_key_only.base_url.is_none());

        let base_url_only = ProviderBuildOverrides::base_url("https://example.com/custom");
        assert!(base_url_only.api_key.is_none());
        assert_eq!(
            base_url_only.base_url.as_deref(),
            Some("https://example.com/custom")
        );
    }

    #[test]
    fn merged_with_merges_http_config_over_global_defaults() {
        let mut base_http_config = HttpConfig {
            timeout: Some(Duration::from_secs(30)),
            connect_timeout: Some(Duration::from_secs(5)),
            proxy: Some("http://global-proxy".to_string()),
            user_agent: Some("global-agent".to_string()),
            stream_disable_compression: true,
            ..Default::default()
        };
        base_http_config
            .headers
            .insert("authorization".to_string(), "Bearer global".to_string());
        base_http_config
            .headers
            .insert("x-global-header".to_string(), "keep-me".to_string());

        let mut provider_http_config = HttpConfig::empty();
        provider_http_config.timeout = Some(Duration::from_secs(90));
        provider_http_config
            .headers
            .insert("authorization".to_string(), "Bearer provider".to_string());
        provider_http_config
            .headers
            .insert("x-provider-header".to_string(), "set-me".to_string());
        provider_http_config.user_agent = Some("provider-agent".to_string());
        provider_http_config.stream_disable_compression = false;

        let merged = ProviderBuildOverrides::default()
            .with_api_key("global-key")
            .with_base_url("https://example.com/global")
            .with_http_config(base_http_config)
            .merged_with(Some(
                &ProviderBuildOverrides::default()
                    .with_api_key("provider-key")
                    .with_http_config(provider_http_config),
            ));

        assert_eq!(merged.api_key.as_deref(), Some("provider-key"));
        assert_eq!(
            merged.base_url.as_deref(),
            Some("https://example.com/global")
        );

        let merged_http_config = merged.http_config.expect("merged http config");
        assert_eq!(merged_http_config.timeout, Some(Duration::from_secs(90)));
        assert_eq!(
            merged_http_config.connect_timeout,
            Some(Duration::from_secs(5))
        );
        assert_eq!(
            merged_http_config
                .headers
                .get("authorization")
                .map(String::as_str),
            Some("Bearer provider")
        );
        assert_eq!(
            merged_http_config
                .headers
                .get("x-global-header")
                .map(String::as_str),
            Some("keep-me")
        );
        assert_eq!(
            merged_http_config
                .headers
                .get("x-provider-header")
                .map(String::as_str),
            Some("set-me")
        );
        assert_eq!(
            merged_http_config.proxy.as_deref(),
            Some("http://global-proxy")
        );
        assert_eq!(
            merged_http_config.user_agent.as_deref(),
            Some("provider-agent")
        );
        assert!(!merged_http_config.stream_disable_compression);
    }
}
