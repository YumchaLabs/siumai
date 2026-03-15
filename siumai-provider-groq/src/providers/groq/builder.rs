//! `Groq` Builder Implementation
//!
//! Thin wrapper around the OpenAI-compatible vendor builder:
//! `LlmBuilder::new().openai().compatible("groq")`.

use crate::builder::BuilderBase;
use crate::error::LlmError;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::provider_options::{
    GroqOptions, GroqReasoningEffort, GroqReasoningFormat, GroqServiceTier,
};
use crate::retry_api::RetryOptions;
use siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleBuilder;
use std::collections::HashMap;
use std::sync::Arc;

/// `Groq` client builder.
///
/// This builder intentionally reuses the OpenAI-compatible vendor implementation.
#[derive(Clone)]
pub struct GroqBuilder {
    inner: OpenAiCompatibleBuilder,
    http_client_override: Option<reqwest::Client>,
    retry_options: Option<RetryOptions>,
    extra_model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    provider_specific_config: HashMap<String, serde_json::Value>,
}

impl GroqBuilder {
    pub fn new(base: BuilderBase) -> Self {
        Self {
            inner: OpenAiCompatibleBuilder::new(base, "groq"),
            http_client_override: None,
            retry_options: None,
            extra_model_middlewares: Vec::new(),
            provider_specific_config: HashMap::new(),
        }
    }

    pub fn api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.inner = self.inner.api_key(api_key);
        self
    }

    pub fn base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        let custom = base_url.into();
        self.inner = self
            .inner
            .base_url(super::GroqConfig::normalize_base_url(&custom));
        self
    }

    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.inner = self.inner.model(model);
        self
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.inner = self.inner.temperature(temperature);
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.inner = self.inner.max_tokens(max_tokens);
        self
    }

    pub fn top_p(mut self, top_p: f64) -> Self {
        self.inner = self.inner.top_p(top_p);
        self
    }

    pub fn stop_sequences(mut self, stop: Vec<String>) -> Self {
        self.inner = self.inner.stop(stop);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.inner = self.inner.seed(seed);
        self
    }

    pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
        self.inner = self.inner.timeout(timeout);
        self
    }

    pub fn connect_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.inner = self.inner.connect_timeout(timeout);
        self
    }

    pub fn http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.inner = self.inner.http_stream_disable_compression(disable);
        self
    }

    pub fn with_http_config(mut self, config: crate::types::HttpConfig) -> Self {
        self.inner = self.inner.with_http_config(config);
        self
    }

    pub fn custom_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.inner = self.inner.custom_headers(headers);
        self
    }

    pub fn header<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.inner = self.inner.header(key, value);
        self
    }

    pub fn with_retry(mut self, options: RetryOptions) -> Self {
        self.retry_options = Some(options.clone());
        self.inner = self.inner.with_retry(options);
        self
    }

    pub fn with_http_interceptor(
        mut self,
        interceptor: Arc<dyn crate::execution::http::interceptor::HttpInterceptor>,
    ) -> Self {
        self.inner = self.inner.with_http_interceptor(interceptor);
        self
    }

    pub fn http_debug(mut self, enabled: bool) -> Self {
        self.inner = self.inner.http_debug(enabled);
        self
    }

    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http_client_override = Some(client.clone());
        self.inner = self.inner.with_http_client(client);
        self
    }

    pub fn with_http_transport(
        mut self,
        transport: Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.inner = self.inner.with_http_transport(transport);
        self
    }

    /// Alias for `with_http_transport(...)` (Vercel-aligned: `fetch`).
    pub fn fetch(
        self,
        transport: Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.with_http_transport(transport)
    }

    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.extra_model_middlewares = middlewares;
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

    pub fn with_groq_options(mut self, options: GroqOptions) -> Self {
        if let Ok(serde_json::Value::Object(obj)) = serde_json::to_value(options) {
            self.provider_specific_config
                .extend(obj.into_iter().filter(|(_, value)| !value.is_null()));
        }
        self
    }

    pub fn logprobs(self, enabled: bool) -> Self {
        self.with_groq_options(GroqOptions::new().with_logprobs(enabled))
    }

    pub fn top_logprobs(self, count: u32) -> Self {
        self.with_groq_options(GroqOptions::new().with_top_logprobs(count))
    }

    pub fn service_tier(self, tier: GroqServiceTier) -> Self {
        self.with_groq_options(GroqOptions::new().with_service_tier(tier))
    }

    pub fn reasoning_effort(self, effort: GroqReasoningEffort) -> Self {
        self.with_groq_options(GroqOptions::new().with_reasoning_effort(effort))
    }

    pub fn reasoning_format(self, format: GroqReasoningFormat) -> Self {
        self.with_groq_options(GroqOptions::new().with_reasoning_format(format))
    }

    pub fn into_config(self) -> Result<super::GroqConfig, LlmError> {
        let compatible = self.inner.into_config()?;
        let mut config = super::GroqConfig::new(compatible.api_key)
            .with_base_url(compatible.base_url)
            .with_model(compatible.common_params.model.clone())
            .with_http_interceptors(compatible.http_interceptors)
            .with_model_middlewares(compatible.model_middlewares);
        config.common_params = compatible.common_params;
        config.http_config = compatible.http_config;
        config.http_transport = compatible.http_transport;
        if !self.extra_model_middlewares.is_empty() {
            let mut middlewares = config.model_middlewares.clone();
            middlewares.extend(self.extra_model_middlewares);
            config = config.with_model_middlewares(middlewares);
        }
        if !self.provider_specific_config.is_empty() {
            config = config.with_provider_specific_config(self.provider_specific_config);
        }
        Ok(config)
    }

    pub async fn build(self) -> Result<super::GroqClient, LlmError> {
        let http_client_override = self.http_client_override.clone();
        let retry_options = self.retry_options.clone();
        let config = self.into_config()?;
        let mut client = if let Some(http_client) = http_client_override {
            super::GroqClient::with_http_client(config, http_client).await?
        } else {
            super::GroqClient::from_config(config).await?
        };
        client.set_retry_options(retry_options);
        Ok(client)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::middleware::language_model::LanguageModelMiddleware;
    use crate::provider_options::{GroqReasoningEffort, GroqReasoningFormat, GroqServiceTier};
    use std::sync::Arc;
    use std::time::Duration;

    #[derive(Clone, Default)]
    struct NoopMiddleware;

    impl LanguageModelMiddleware for NoopMiddleware {}

    #[test]
    fn groq_builder_into_config_converges_on_groq_config() {
        let config = GroqBuilder::new(BuilderBase::default())
            .api_key("test-key")
            .base_url("https://api.groq.com")
            .model("llama-3.3-70b-versatile")
            .temperature(0.1)
            .max_tokens(128)
            .top_p(0.8)
            .stop_sequences(vec!["END".to_string()])
            .seed(7)
            .custom_headers(HashMap::from([("x-test".to_string(), "1".to_string())]))
            .timeout(Duration::from_secs(9))
            .http_debug(true)
            .logprobs(true)
            .top_logprobs(2)
            .service_tier(GroqServiceTier::Flex)
            .reasoning_effort(GroqReasoningEffort::Default)
            .reasoning_format(GroqReasoningFormat::Parsed)
            .into_config()
            .expect("into_config ok");

        assert_eq!(config.base_url, "https://api.groq.com/openai/v1");
        assert_eq!(config.common_params.model, "llama-3.3-70b-versatile");
        assert_eq!(config.common_params.temperature, Some(0.1));
        assert_eq!(config.common_params.max_tokens, Some(128));
        assert_eq!(config.common_params.top_p, Some(0.8));
        assert_eq!(
            config.common_params.stop_sequences,
            Some(vec!["END".to_string()])
        );
        assert_eq!(config.common_params.seed, Some(7));
        assert_eq!(
            config.http_config.headers.get("x-test"),
            Some(&"1".to_string())
        );
        assert_eq!(config.http_config.timeout, Some(Duration::from_secs(9)));
        assert_eq!(config.http_interceptors.len(), 1);
        assert_eq!(
            config.provider_specific_config.get("logprobs"),
            Some(&serde_json::json!(true))
        );
        assert_eq!(
            config.provider_specific_config.get("top_logprobs"),
            Some(&serde_json::json!(2))
        );
        assert_eq!(
            config.provider_specific_config.get("service_tier"),
            Some(&serde_json::json!("flex"))
        );
        assert_eq!(
            config.provider_specific_config.get("reasoning_effort"),
            Some(&serde_json::json!("default"))
        );
        assert_eq!(
            config.provider_specific_config.get("reasoning_format"),
            Some(&serde_json::json!("parsed"))
        );
    }

    #[test]
    fn groq_builder_into_config_matches_manual_groq_config() {
        let builder_config = GroqBuilder::new(BuilderBase::default())
            .api_key("test-key")
            .base_url("https://api.groq.com")
            .model("llama-3.3-70b-versatile")
            .temperature(0.1)
            .max_tokens(128)
            .top_p(0.8)
            .stop_sequences(vec!["END".to_string()])
            .seed(7)
            .custom_headers(HashMap::from([("x-test".to_string(), "1".to_string())]))
            .timeout(Duration::from_secs(9))
            .http_debug(true)
            .logprobs(true)
            .top_logprobs(2)
            .service_tier(GroqServiceTier::Flex)
            .reasoning_effort(GroqReasoningEffort::Default)
            .reasoning_format(GroqReasoningFormat::Parsed)
            .with_model_middlewares(vec![Arc::new(NoopMiddleware)])
            .into_config()
            .expect("builder config");

        let manual_config = crate::providers::groq::GroqConfig::new("test-key")
            .with_base_url("https://api.groq.com/openai/v1")
            .with_model("llama-3.3-70b-versatile")
            .with_temperature(0.1)
            .with_max_tokens(128)
            .with_top_p(0.8)
            .with_stop_sequences(vec!["END".to_string()])
            .with_seed(7)
            .with_http_config({
                let mut c = crate::types::HttpConfig::default();
                c.timeout = Some(Duration::from_secs(9));
                c
            })
            .with_header("x-test", "1")
            .with_logprobs(true)
            .with_top_logprobs(2)
            .with_service_tier(GroqServiceTier::Flex)
            .with_reasoning_effort(GroqReasoningEffort::Default)
            .with_reasoning_format(GroqReasoningFormat::Parsed)
            .with_http_interceptors(vec![Arc::new(
                crate::execution::http::interceptor::LoggingInterceptor,
            )])
            .with_model_middlewares({
                let mut middlewares = crate::execution::middleware::build_auto_middlewares_vec(
                    "groq",
                    "llama-3.3-70b-versatile",
                );
                middlewares.push(Arc::new(NoopMiddleware));
                middlewares
            });

        assert_eq!(builder_config.base_url, manual_config.base_url);
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
            builder_config.http_config.headers,
            manual_config.http_config.headers
        );
        assert_eq!(
            builder_config.http_config.timeout,
            manual_config.http_config.timeout
        );
        assert_eq!(
            builder_config.http_interceptors.len(),
            manual_config.http_interceptors.len()
        );
        assert_eq!(
            builder_config.model_middlewares.len(),
            manual_config.model_middlewares.len()
        );
        assert_eq!(
            builder_config.provider_specific_config,
            manual_config.provider_specific_config
        );
    }

    #[tokio::test]
    async fn groq_builder_build_preserves_http_client_override_and_retry_options() {
        let client = GroqBuilder::new(BuilderBase::default())
            .api_key("test-key")
            .model("llama-3.3-70b-versatile")
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
}
