//! `DeepSeek` configuration.
//!
//! Provider-owned config-first surface for the DeepSeek wrapper.

use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::http::transport::HttpTransport;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::provider_options::DeepSeekOptions;
use crate::types::{CommonParams, HttpConfig};
use secrecy::{ExposeSecret, SecretString};
use std::collections::HashMap;
use std::sync::Arc;

/// `DeepSeek` configuration (provider layer).
#[derive(Clone)]
pub struct DeepSeekConfig {
    /// API key (securely stored).
    pub api_key: SecretString,
    /// Base URL for the DeepSeek API.
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
    /// Provider-specific request defaults merged into Chat requests.
    pub provider_specific_config: HashMap<String, serde_json::Value>,
}

impl std::fmt::Debug for DeepSeekConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("DeepSeekConfig");
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

impl DeepSeekConfig {
    /// Default DeepSeek API base URL.
    pub const DEFAULT_BASE_URL: &'static str = "https://api.deepseek.com";

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

    /// Create config from `DEEPSEEK_API_KEY`.
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("DEEPSEEK_API_KEY")
            .map_err(|_| LlmError::MissingApiKey("DeepSeek API key not provided".to_string()))?;
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
        self.provider_specific_config
            .extend(params.into_iter().filter(|(_, value)| !value.is_null()));
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

    pub fn with_deepseek_options(mut self, options: DeepSeekOptions) -> Self {
        if let Ok(serde_json::Value::Object(obj)) = serde_json::to_value(options) {
            self.provider_specific_config
                .extend(obj.into_iter().filter(|(_, value)| !value.is_null()));
        }
        self
    }

    pub fn with_deepseek_reasoning(self, enable: bool) -> Self {
        self.with_deepseek_options(DeepSeekOptions::new().with_reasoning(enable))
    }

    pub fn with_deepseek_reasoning_budget(self, budget: i32) -> Self {
        self.with_deepseek_options(DeepSeekOptions::new().with_reasoning_budget(budget))
    }

    pub fn with_reasoning(mut self, enable: bool) -> Self {
        self.provider_specific_config.insert(
            "enable_reasoning".to_string(),
            serde_json::Value::Bool(enable),
        );
        self
    }

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
                "Model is required for DeepSeek config".to_string(),
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

        let provider = get_provider_config("deepseek").ok_or_else(|| {
            LlmError::ConfigurationError(
                "OpenAI-compatible provider config not found: deepseek".into(),
            )
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
            "deepseek",
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
    fn deepseek_config_reasoning_defaults_roundtrip_into_compatible_config() {
        let cfg = DeepSeekConfig::new("test-key")
            .with_model("deepseek-chat")
            .with_reasoning(true)
            .with_reasoning_budget(2048);

        let compat = cfg
            .into_compatible_config()
            .expect("into_compatible_config");
        let mut body = serde_json::json!({});
        compat
            .adapter
            .transform_request_params(
                &mut body,
                &compat.model,
                siumai_provider_openai_compatible::providers::openai_compatible::RequestType::Chat,
            )
            .expect("transform request params");

        assert_eq!(body["enable_reasoning"], serde_json::json!(true));
        assert_eq!(body["reasoning_budget"], serde_json::json!(2048));
    }

    #[test]
    fn deepseek_config_http_convenience_helpers_update_http_state() {
        let cfg = DeepSeekConfig::new("test-key")
            .with_model("deepseek-chat")
            .with_timeout(Duration::from_secs(9))
            .with_connect_timeout(Duration::from_secs(3))
            .with_http_stream_disable_compression(true)
            .with_http_interceptor(Arc::new(NoopInterceptor));

        assert_eq!(cfg.http_config.timeout, Some(Duration::from_secs(9)));
        assert_eq!(
            cfg.http_config.connect_timeout,
            Some(Duration::from_secs(3))
        );
        assert!(cfg.http_config.stream_disable_compression);
        assert_eq!(cfg.http_interceptors.len(), 1);
    }
}
