//! `xAI` configuration.
//!
//! Provider-owned config-first surface for the xAI wrapper.

use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::http::transport::HttpTransport;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::provider_options::{XaiOptions, XaiSearchParameters};
use crate::types::{CommonParams, HttpConfig};
use secrecy::{ExposeSecret, SecretString};
use std::collections::HashMap;
use std::sync::Arc;

/// `xAI` configuration (provider layer).
#[derive(Clone)]
pub struct XaiConfig {
    /// API key (securely stored).
    pub api_key: SecretString,
    /// Base URL for the xAI API.
    pub base_url: String,
    /// Common parameters shared across providers.
    pub common_params: CommonParams,
    /// HTTP configuration.
    pub http_config: HttpConfig,
    /// Optional custom HTTP transport.
    pub http_transport: Option<Arc<dyn HttpTransport>>,
    /// Optional HTTP interceptors applied to all requests built from this config.
    pub http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional model-level middlewares applied before provider mapping (chat only).
    pub model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    /// Provider-specific request defaults merged into chat requests.
    pub provider_specific_config: HashMap<String, serde_json::Value>,
}

impl std::fmt::Debug for XaiConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("XaiConfig");
        ds.field("base_url", &self.base_url)
            .field("common_params", &self.common_params)
            .field("http_config", &self.http_config);

        if !self.api_key.expose_secret().is_empty() {
            ds.field("has_api_key", &true);
        }
        if self.http_transport.is_some() {
            ds.field("has_http_transport", &true);
        }
        if !self.provider_specific_config.is_empty() {
            ds.field("provider_specific_config", &self.provider_specific_config);
        }

        ds.finish()
    }
}

impl XaiConfig {
    /// Default xAI API base URL.
    pub const DEFAULT_BASE_URL: &'static str = "https://api.x.ai/v1";

    /// Create a new config with the given API key.
    pub fn new<S: Into<String>>(api_key: S) -> Self {
        Self {
            api_key: SecretString::from(api_key.into()),
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            common_params: CommonParams::default(),
            http_config: HttpConfig::default(),
            http_transport: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
            provider_specific_config: HashMap::new(),
        }
    }

    /// Create config from `XAI_API_KEY`.
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("XAI_API_KEY")
            .map_err(|_| LlmError::MissingApiKey("xAI API key not provided".to_string()))?;
        Ok(Self::new(api_key))
    }

    pub fn with_base_url<S: Into<String>>(mut self, url: S) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.common_params.model = model.into();
        self
    }

    pub const fn with_temperature(mut self, temperature: f64) -> Self {
        self.common_params.temperature = Some(temperature);
        self
    }

    pub const fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.common_params.max_tokens = Some(max_tokens);
        self
    }

    pub const fn with_top_p(mut self, top_p: f64) -> Self {
        self.common_params.top_p = Some(top_p);
        self
    }

    pub fn with_stop_sequences(mut self, stop: Vec<String>) -> Self {
        self.common_params.stop_sequences = Some(stop);
        self
    }

    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.common_params.seed = Some(seed);
        self
    }

    pub fn with_http_config(mut self, http: HttpConfig) -> Self {
        self.http_config = http;
        self
    }

    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http_config.timeout = Some(timeout);
        self
    }

    pub fn with_connect_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http_config.connect_timeout = Some(timeout);
        self
    }

    pub fn with_http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.http_config.stream_disable_compression = disable;
        self
    }

    pub fn with_http_transport(mut self, transport: Arc<dyn HttpTransport>) -> Self {
        self.http_transport = Some(transport);
        self
    }

    pub fn with_http_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors;
        self
    }

    pub fn with_http_interceptor(mut self, interceptor: Arc<dyn HttpInterceptor>) -> Self {
        self.http_interceptors.push(interceptor);
        self
    }

    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.model_middlewares = middlewares;
        self
    }

    pub fn with_provider_specific_config(
        mut self,
        params: HashMap<String, serde_json::Value>,
    ) -> Self {
        self.provider_specific_config.extend(params);
        self
    }

    pub fn with_provider_specific_param(
        mut self,
        key: impl Into<String>,
        value: serde_json::Value,
    ) -> Self {
        self.provider_specific_config.insert(key.into(), value);
        self
    }

    /// Merge typed xAI options into config defaults.
    pub fn with_xai_options(mut self, options: XaiOptions) -> Self {
        if let Ok(serde_json::Value::Object(obj)) = serde_json::to_value(options) {
            self.provider_specific_config
                .extend(obj.into_iter().filter(|(_, value)| !value.is_null()));
        }
        self
    }

    /// Enable provider-native reasoning mode when supported.
    pub fn with_reasoning(mut self, enable: bool) -> Self {
        self.provider_specific_config.insert(
            "enable_reasoning".to_string(),
            serde_json::Value::Bool(enable),
        );
        self
    }

    /// Set the reasoning token budget.
    pub fn with_reasoning_budget(mut self, budget: i32) -> Self {
        let clamped_budget = budget.clamp(128, 32768) as u32;
        self.provider_specific_config.insert(
            "reasoning_budget".to_string(),
            serde_json::Value::Number(serde_json::Number::from(clamped_budget)),
        );
        self.provider_specific_config.insert(
            "enable_reasoning".to_string(),
            serde_json::Value::Bool(true),
        );
        self
    }

    /// Set xAI-specific reasoning effort.
    pub fn with_reasoning_effort(self, effort: impl Into<String>) -> Self {
        self.with_xai_options(XaiOptions::new().with_reasoning_effort(effort))
    }

    /// Set xAI-specific search parameters.
    pub fn with_search_parameters(self, params: XaiSearchParameters) -> Self {
        self.with_xai_options(XaiOptions::new().with_search(params))
    }

    /// Enable xAI search defaults.
    pub fn with_default_search(self) -> Self {
        self.with_search_parameters(XaiSearchParameters::default())
    }

    pub fn validate(&self) -> Result<(), LlmError> {
        if self.api_key.expose_secret().is_empty() {
            return Err(LlmError::ConfigurationError(
                "API key cannot be empty".to_string(),
            ));
        }
        if self.base_url.is_empty() {
            return Err(LlmError::ConfigurationError(
                "Base URL cannot be empty".to_string(),
            ));
        }
        if !self.base_url.starts_with("http://") && !self.base_url.starts_with("https://") {
            return Err(LlmError::ConfigurationError(
                "Base URL must start with http:// or https://".to_string(),
            ));
        }
        if self.common_params.model.is_empty() {
            return Err(LlmError::ConfigurationError(
                "Model is required for xAI config".to_string(),
            ));
        }
        Ok(())
    }

    /// Build a provider-owned config from an OpenAI-compatible config.
    pub fn from_compatible_config(
        config: siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleConfig,
    ) -> Result<Self, LlmError> {
        use siumai_provider_openai_compatible::providers::openai_compatible::RequestType;

        let mut provider_specific_config = HashMap::new();
        let mut request_defaults = serde_json::json!({});
        config.adapter.transform_request_params(
            &mut request_defaults,
            &config.model,
            RequestType::Chat,
        )?;
        if let Some(obj) = request_defaults.as_object() {
            provider_specific_config.extend(obj.iter().map(|(k, v)| (k.clone(), v.clone())));
        }

        Ok(Self {
            api_key: SecretString::from(config.api_key),
            base_url: config.base_url,
            common_params: config.common_params,
            http_config: config.http_config,
            http_transport: config.http_transport,
            http_interceptors: config.http_interceptors,
            model_middlewares: config.model_middlewares,
            provider_specific_config,
        })
    }

    /// Convert to the OpenAI-compatible config used by the execution backend.
    pub fn into_compatible_config(
        self,
    ) -> Result<
        siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleConfig,
        LlmError,
    > {
        use siumai_provider_openai_compatible::providers::openai_compatible::{
            ConfigurableAdapter, OpenAiCompatibleConfig,
        };
        use siumai_provider_openai_compatible::providers::openai_compatible::{
            ProviderAdapter, get_provider_config,
        };

        self.validate()?;

        let provider = get_provider_config("xai").ok_or_else(|| {
            LlmError::ConfigurationError("OpenAI-compatible provider config not found: xai".into())
        })?;

        let mut adapter: Box<dyn ProviderAdapter> = Box::new(ConfigurableAdapter::new(provider));
        if !self.provider_specific_config.is_empty() {
            adapter = Box::new(
                siumai_provider_openai_compatible::providers::openai_compatible::adapter::ParamMergingAdapter::new(
                    adapter,
                    self.provider_specific_config.clone(),
                ),
            );
        }
        let adapter: Arc<dyn ProviderAdapter> = Arc::from(adapter);

        let mut config = OpenAiCompatibleConfig::new(
            "xai",
            self.api_key.expose_secret(),
            &self.base_url,
            adapter,
        )
        .with_common_params(self.common_params.clone())
        .with_model(&self.common_params.model)
        .with_http_config(self.http_config.clone())
        .with_http_interceptors(self.http_interceptors.clone())
        .with_model_middlewares(self.model_middlewares.clone());

        if let Some(transport) = self.http_transport.clone() {
            config = config.with_http_transport(transport);
        }

        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[derive(Clone, Default)]
    struct NoopInterceptor;

    impl HttpInterceptor for NoopInterceptor {}

    #[test]
    fn xai_config_reasoning_and_search_defaults_roundtrip_into_compatible_config() {
        let mut http_config = HttpConfig::default();
        http_config.timeout = Some(Duration::from_secs(9));

        let config = XaiConfig::new("test-key")
            .with_model("grok-4")
            .with_reasoning(true)
            .with_reasoning_budget(4096)
            .with_reasoning_effort("high")
            .with_default_search()
            .with_http_config(http_config.clone());

        let compat = config
            .clone()
            .into_compatible_config()
            .expect("into_compatible_config");
        let mut params = serde_json::json!({});
        compat
            .adapter
            .transform_request_params(
                &mut params,
                &compat.model,
                siumai_provider_openai_compatible::providers::openai_compatible::RequestType::Chat,
            )
            .expect("transform request params");

        assert_eq!(compat.model, "grok-4");
        assert_eq!(compat.http_config.timeout, Some(Duration::from_secs(9)));
        assert_eq!(params["enable_reasoning"], serde_json::json!(true));
        assert_eq!(params["reasoning_budget"], serde_json::json!(4096));
        assert_eq!(params["reasoning_effort"], serde_json::json!("high"));
        assert_eq!(
            params["search_parameters"]["return_citations"],
            serde_json::json!(true)
        );
        assert_eq!(
            params["search_parameters"]["max_search_results"],
            serde_json::json!(5)
        );
    }

    #[test]
    fn xai_config_http_convenience_helpers_update_http_state() {
        let config = XaiConfig::new("test-key")
            .with_model("grok-4")
            .with_timeout(Duration::from_secs(9))
            .with_connect_timeout(Duration::from_secs(3))
            .with_http_stream_disable_compression(true)
            .with_http_interceptor(Arc::new(NoopInterceptor));

        assert_eq!(config.http_config.timeout, Some(Duration::from_secs(9)));
        assert_eq!(
            config.http_config.connect_timeout,
            Some(Duration::from_secs(3))
        );
        assert!(config.http_config.stream_disable_compression);
        assert_eq!(config.http_interceptors.len(), 1);
    }
}
